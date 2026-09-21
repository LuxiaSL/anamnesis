"""Fitting a model's calibration: what its states do because of position, and what is left.

Every corrected feature this instrument computes rests on two model-specific
artifacts, and this module produces both:

* the **positional means** — the mean state at each position of each layer, over a
  fixed set of content-diverse prompts. Subtracting it removes the part of a state
  that is a function of *where a token is* rather than of how it was processed.
  Without it, a position effect reads as a signature.
* the **residual basis** — components for the residual stream at the layers a
  preset names, fitted over the same prompts.

The basis is fitted on positionally corrected states, at each layer separately.
Fitting on raw states and applying the result to corrected ones is a fit-and-apply
mismatch: the basis then describes a distribution the projection never sees, and the
components it spends on position are wasted. A **pooled** fit is the other shape —
one basis over uncorrected states, the shape the banked pooled bases were fitted
under. The two have different readers:
:func:`anamnesis.extraction.calibration.load_pca_model` projects onto the pooled one,
and :mod:`anamnesis.extraction.feature_pipeline` reads the per-layer one on the
recompute path.

**A calibration generates at the preset's own decode policy.** The artifacts are a
mean and a basis over a distribution of states, so a pass that samples at a
nucleus mass or a temperature the checkpoint is never run at describes a
distribution no signature is ever computed over — and nothing downstream can see
that it happened, because a mean is a mean whatever produced it.
:func:`generation_settings` therefore takes every decode value from the preset row.
An explicit argument overrides one, which is how a shortened trial pass asks for
fewer tokens.

The prompt set is fixed and lives here. It is not a corpus, it is a ruler: fifty
prompts spread across science, history, craft, economics and mechanics, whose only
job is to wash content out of the average so that what remains is position.
Changing it changes every number downstream of it.

The fit takes an iterable of :class:`PromptStates` rather than a loaded model, so
the arithmetic runs — and is tested — on a machine with no weights on it.
:func:`generate_prompt_states` is the one function here that runs a model, and it
is where the model runtime is imported.
"""

from __future__ import annotations

import gc
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from anamnesis.config import GenerationConfig, ModelPreset, resolve_preset
from anamnesis.extraction.calibration import (
    POSITION_COUNTS_KEY,
    POSITIONAL_MEANS_KEY,
    load_positional_means,
)

if TYPE_CHECKING:  # the annotation alone, so reading a fit costs no model runtime
    from anamnesis.extraction.model_loader import LoadedModel

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]
I64 = NDArray[np.int64]

POSITION_COUNT_FLOOR = 5
"""A position whose mean is an average of this many states or fewer is left at
zero: a mean over one or two prompts is not a mean, it is one of the states."""

PROMPT_HEADROOM = 200
"""Positions reserved above the token budget for the prompt itself, since a
position is absolute and a generated token sits after its prompt."""

CALIBRATION_PROMPTS: tuple[str, ...] = (
    "Explain how photosynthesis works in plants.",
    "What are the main causes of the French Revolution?",
    "Describe the process of making traditional Japanese ramen.",
    "How do electric vehicles compare to gasoline cars?",
    "What is the significance of the Rosetta Stone?",
    "Explain the concept of supply and demand in economics.",
    "How does the human immune system fight infections?",
    "Describe the architecture of Gothic cathedrals.",
    "What are the principles of object-oriented programming?",
    "How do tides work and what causes them?",
    "Explain the theory of plate tectonics.",
    "What makes a good leader?",
    "How do birds navigate during migration?",
    "Describe the water cycle and its importance.",
    "What is quantum entanglement?",
    "How do vaccines work?",
    "Explain the causes and effects of inflation.",
    "What are the different types of clouds?",
    "How does a combustion engine work?",
    "Describe the life cycle of a star.",
    "What is machine learning and how does it differ from traditional programming?",
    "How do earthquakes happen?",
    "Explain the basics of music theory.",
    "What are renewable energy sources?",
    "How does the stock market work?",
    "Describe the process of fermentation.",
    "What are the effects of sleep deprivation?",
    "How do submarines work?",
    "Explain the concept of natural selection.",
    "What is the significance of pi in mathematics?",
    "How do 3D printers work?",
    "Describe the history of the internet.",
    "What causes aurora borealis?",
    "How do computers store and retrieve data?",
    "Explain the process of osmosis.",
    "What are the major types of rocks?",
    "How do airplanes fly?",
    "Describe the structure of DNA.",
    "What is cryptocurrency and how does blockchain work?",
    "How do telescopes work?",
    "Explain the greenhouse effect.",
    "What are the stages of grief?",
    "How does sonar work?",
    "Describe the Silk Road and its importance.",
    "What is dark matter?",
    "How do coral reefs form?",
    "Explain the basics of game theory.",
    "What are the layers of the atmosphere?",
    "How does a nuclear reactor work?",
    "Describe the process of cheese making.",
)


class CalibrationFitError(RuntimeError):
    """A fit that would write an artifact nothing can be projected onto.

    Raised rather than returned: an empty basis is not a partial result, it is a
    file that would make every residual-PCA feature downstream of it zero.
    """


def generation_settings(
    preset: str | ModelPreset,
    *,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> GenerationConfig:
    """The decode policy one calibration pass generates under.

    Every value defaults to the preset row's, because the artifacts describe the
    distribution of states a model produces and that distribution is a function of
    how it decodes. A value given here wins over the preset's.

    Attention weights and logits are switched off: a calibration reads hidden
    states only, and those two are what make a capture expensive.
    """
    overrides: dict[str, Any] = {
        name: value
        for name, value in (
            ("max_new_tokens", max_new_tokens),
            ("temperature", temperature),
            ("top_p", top_p),
        )
        if value is not None
    }
    return GenerationConfig.from_preset(
        preset, output_attentions=False, output_logits=False, **overrides
    )


@dataclass(frozen=True)
class PromptStates:
    """One prompt's hidden states as numpy, in the order a forward pass produced them.

    ``prefill`` is indexed ``[layer, position, unit]`` over the prompt's own
    positions; ``steps[i]`` is ``[layer, unit]`` for the ``i``-th generated token,
    whose absolute position is ``prompt_length + i``. Both index layers the way
    ``hidden_states`` does — entry 0 is the embedding output, so layer *l* of the
    network is entry *l + 1*, and an off-by-one here corrupts every layer-indexed
    number that follows.

    A frozen dataclass rather than a model with validators: the invariants are
    array shapes, and they are checked here at construction.
    """

    prompt_length: int
    prefill: F32
    steps: tuple[F32, ...]

    def __post_init__(self) -> None:
        if self.prompt_length <= 0:
            raise ValueError(f"prompt_length {self.prompt_length} is not a length")
        if self.prefill.ndim != 3:
            raise ValueError(
                f"prefill is {self.prefill.ndim}-dimensional; "
                "it is indexed [layer, position, unit]"
            )
        for index, step in enumerate(self.steps):
            if step.ndim != 2:
                raise ValueError(
                    f"step {index} is {step.ndim}-dimensional; it is indexed [layer, unit]"
                )


@dataclass(frozen=True)
class CalibrationFit:
    """What one calibration pass produced, before any of it is written.

    ``basis`` is a single mapping with ``components`` and ``mean`` for a pooled
    fit, or one such mapping per layer index for a corrected fit. ``means_refitted``
    says whether :attr:`positional_means` came out of this pass or off disk, which
    is what decides whether writing them would move a correction that banked
    signatures were computed under.
    """

    positional_means: F32
    position_counts: I64
    basis: dict[str, Any] | dict[int, dict[str, Any]]
    means_refitted: bool

    @property
    def furthest_position(self) -> int:
        """The last position any prompt reached, as a coverage read on the means."""
        covered = self.position_counts.sum(axis=0) > 0
        return int(np.max(np.where(covered))) if covered.any() else 0


def generate_prompt_states(
    loaded: LoadedModel, prompts: Sequence[str], settings: GenerationConfig
) -> Iterator[PromptStates]:
    """Generate each prompt under ``settings`` and hand its states back as numpy.

    Seeded by prompt index so a calibration is reproducible, and a checkpoint with
    no chat template takes the bare prompt — a base model's calibration must match
    the bare prompts its generations will use.

    The model runtime is imported inside this function: everything else in this
    module is numpy and scikit-learn, and a machine with no accelerator reads the
    prompt set, the decode policy and the fit.

    One prompt's states are materialised at a time and released before the next,
    so the peak is one generation's hidden states rather than the pass's.

    Raises
    ------
    CalibrationFitError
        When ``settings`` asks for no hidden states, which are the only substrate a
        calibration reads.
    """
    import torch

    if not settings.output_hidden_states:
        raise CalibrationFitError(
            "the decode policy asks for no hidden states, and a calibration reads nothing else"
        )
    device = next(loaded.model.parameters()).device
    for index, text in enumerate(prompts):
        if loaded.tokenizer.chat_template is None:
            result = loaded.tokenizer(text, return_tensors="pt")["input_ids"]
        else:
            result = loaded.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
        input_ids = (result if torch.is_tensor(result) else result["input_ids"]).to(device)
        torch.manual_seed(index)
        with torch.no_grad():
            out = loaded.model.generate(
                input_ids,
                max_new_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
                top_p=settings.top_p,
                do_sample=settings.do_sample,
                eos_token_id=list(settings.eos_token_ids),
                output_hidden_states=settings.output_hidden_states,
                output_attentions=settings.output_attentions,
                output_logits=settings.output_logits,
                return_dict_in_generate=settings.return_dict_in_generate,
            )
        hidden = out.hidden_states
        yield PromptStates(
            prompt_length=int(input_ids.shape[1]),
            prefill=np.stack([layer[0].cpu().float().numpy() for layer in hidden[0]]),
            steps=tuple(
                np.stack([layer[0, -1].cpu().float().numpy() for layer in step])
                for step in hidden[1:]
            ),
        )
        del out, hidden
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if (index + 1) % 10 == 0:
            logger.info(f"calibration {index + 1}/{len(prompts)}")


def _accumulate_positions(
    sums: NDArray[np.float64], counts: I64, prompt: PromptStates
) -> None:
    """Add one prompt's states into the running per-position, per-layer totals.

    Accumulation is float64 over float32 states: a sum of thousands of activations
    in float32 loses low-order bits, and the mean is subtracted from single states.
    """
    depth, max_positions = counts.shape
    layers = min(prompt.prefill.shape[0], depth)
    positions = min(prompt.prefill.shape[1], max_positions)
    sums[:layers, :positions] += prompt.prefill[:layers, :positions].astype(np.float64)
    counts[:layers, :positions] += 1
    for index, step in enumerate(prompt.steps):
        absolute = prompt.prompt_length + index
        if absolute >= max_positions:
            break
        layers = min(step.shape[0], depth)
        sums[:layers, absolute] += step[:layers].astype(np.float64)
        counts[:layers, absolute] += 1


def _basis_samples(
    prompt: PromptStates, pca_layers: Sequence[int], pooled: bool
) -> list[tuple[int, int, F32]]:
    """``(layer, absolute position, state)`` at the sampled steps of one prompt.

    A pooled fit keeps the three sample points as they fall, including when a short
    generation makes two of them the same step; the per-layer fit takes the
    distinct ones. Both are the shape their banked artifacts were fitted under, so
    neither is normalised into the other.
    """
    n_steps = len(prompt.steps)
    if n_steps <= 0:
        return []
    midpoint = max(1, n_steps // 2)
    chosen = [1, midpoint, n_steps] if pooled else sorted({1, midpoint, n_steps})
    out: list[tuple[int, int, F32]] = []
    for step in chosen:
        if step > n_steps:
            continue
        states = prompt.steps[step - 1]
        absolute = prompt.prompt_length + step - 1
        for layer in pca_layers:
            if layer + 1 >= states.shape[0]:
                continue
            out.append((layer, absolute, states[layer + 1]))
    return out


def means_from_totals(sums: NDArray[np.float64], counts: I64) -> F32:
    """Per-position means, with a position under the count floor left at zero.

    Zero rather than a partial average, because subtracting a mean taken over two
    prompts removes one of those prompts' own states from a third.
    """
    covered = counts > POSITION_COUNT_FLOOR
    safe = np.where(covered, counts, 1)
    means: F32 = np.where(
        covered[:, :, np.newaxis], sums / safe[:, :, np.newaxis], 0.0
    ).astype(np.float32)
    return means


def _fit_one(matrix: NDArray[np.float64], n_components: int) -> dict[str, Any]:
    """One PCA over a sample matrix, as the mapping both readers accept."""
    fitted = PCA(n_components=min(n_components, *matrix.shape)).fit(matrix)
    return {
        "components": fitted.components_.astype(np.float32),
        "mean": fitted.mean_.astype(np.float32),
        "explained_variance_ratio": fitted.explained_variance_ratio_,
    }


def fit_pooled_basis(
    samples: Sequence[tuple[int, int, F32]], n_components: int
) -> dict[str, Any]:
    """One basis over every sampled state, uncorrected.

    Raises
    ------
    CalibrationFitError
        When no states were sampled.
    """
    if not samples:
        raise CalibrationFitError(
            "no basis samples were collected; the residual-PCA features would be empty"
        )
    matrix = np.stack([state for _, _, state in samples]).astype(np.float64)
    basis = _fit_one(matrix, n_components)
    logger.info(
        f"pooled basis {basis['components'].shape} over {matrix.shape[0]} samples, "
        f"explaining {basis['explained_variance_ratio'].sum():.3f}"
    )
    return basis


def fit_per_layer_basis(
    samples: Sequence[tuple[int, int, F32]],
    means: F32,
    pca_layers: Sequence[int],
    n_components: int,
) -> dict[int, dict[str, Any]]:
    """One basis per layer, over positionally corrected states.

    A sample beyond the reach of the means carries no correction and is dropped
    rather than fitted raw, which would mix two distributions in one basis.

    Raises
    ------
    CalibrationFitError
        When a named layer ends up with no corrected samples.
    """
    corrected: dict[int, list[F32]] = {int(layer): [] for layer in pca_layers}
    for layer, absolute, state in samples:
        if absolute < means.shape[1]:
            corrected[int(layer)].append(state - means[layer + 1, absolute])
    basis: dict[int, dict[str, Any]] = {}
    for layer in pca_layers:
        rows = corrected[int(layer)]
        if not rows:
            raise CalibrationFitError(
                f"no corrected samples at layer {layer}; the fit would be empty"
            )
        matrix = np.stack(rows).astype(np.float64)
        basis[int(layer)] = _fit_one(matrix, n_components)
        logger.info(
            f"layer {layer}: {matrix.shape[0]} corrected samples -> "
            f"{basis[int(layer)]['components'].shape}, explaining "
            f"{basis[int(layer)]['explained_variance_ratio'].sum():.3f}"
        )
    return basis


def fit_calibration(
    states: Iterable[PromptStates],
    *,
    preset: str | ModelPreset,
    settings: GenerationConfig,
    n_components: int,
    pooled: bool = False,
    existing_means: F32 | None = None,
) -> CalibrationFit:
    """Both artifacts, from one pass over a prompt set's states.

    ``existing_means`` reuses a correction already on disk instead of measuring a
    new one, which is what lets a basis be refitted and compared without moving
    the means underneath the signatures already computed against them. When it is
    given, the position accumulator is skipped entirely.

    ``n_components`` is an upper bound: a fit over fewer samples or narrower states
    than that keeps what it can.

    Raises
    ------
    CalibrationFitError
        From the basis fit, when it would have nothing to fit over.
    """
    row = resolve_preset(preset)
    depth = row.num_layers + 1
    max_positions = settings.max_new_tokens + PROMPT_HEADROOM
    sums = np.zeros((depth, max_positions, row.hidden_dim), dtype=np.float64)
    counts: I64 = np.zeros((depth, max_positions), dtype=np.int64)
    # A sample is kept with the position it came from, because correcting it needs
    # that position and which correction to apply is not known until the means are.
    samples: list[tuple[int, int, F32]] = []

    for prompt in states:
        if existing_means is None:
            _accumulate_positions(sums, counts, prompt)
        samples.extend(_basis_samples(prompt, row.pca_layers, pooled))

    means = means_from_totals(sums, counts) if existing_means is None else existing_means
    basis: dict[str, Any] | dict[int, dict[str, Any]] = (
        fit_pooled_basis(samples, n_components)
        if pooled
        else fit_per_layer_basis(samples, means, row.pca_layers, n_components)
    )
    return CalibrationFit(
        positional_means=means,
        position_counts=counts,
        basis=basis,
        means_refitted=existing_means is None,
    )


def read_existing_means(means_path: Path, refit: bool) -> F32 | None:
    """The means already at ``means_path``, unless a refit was asked for.

    A refit overwrites the correction every signature in the directory was
    computed under, so it happens only when a caller says so.
    """
    if refit or not means_path.is_file():
        return None
    means = load_positional_means(means_path.parent)
    if means is not None:
        logger.info(
            f"reusing the positional means at {means_path}: a correction a bank was "
            "computed under does not move"
        )
    return means


def write_calibration(fit: CalibrationFit, means_path: Path, basis_path: Path) -> None:
    """Write the artifacts this pass produced, leaving reused means alone.

    Means that came off disk are not rewritten: the bytes would be the same, and
    the timestamp would say a correction moved when it did not.
    """
    basis_path.parent.mkdir(parents=True, exist_ok=True)
    if fit.means_refitted:
        means_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            means_path,
            **{
                POSITIONAL_MEANS_KEY: fit.positional_means,
                POSITION_COUNTS_KEY: fit.position_counts,
            },
        )
        logger.info(
            f"positional means {fit.positional_means.shape} -> {means_path} "
            f"(furthest position {fit.furthest_position})"
        )
    with open(basis_path, "wb") as handle:
        pickle.dump(fit.basis, handle)
    logger.info(f"basis -> {basis_path}")
