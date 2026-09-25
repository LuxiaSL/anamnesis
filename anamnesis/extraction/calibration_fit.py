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

The prompt set is data, in ``anamnesis/prompts/calibration_prompts.json`` beside
the topic sets, read by :func:`calibration_prompts`. It is not a corpus, it is a
ruler: prompts spread across science, history, craft, economics and mechanics, whose
only job is to wash content out of the average so that what remains is position. Its
exact contents are load-bearing — every banked positional mean was fitted over them
and every corrected feature subtracts those means — so the file's bytes are pinned by
digest in ``tests/test_prompt_sets.py`` rather than left to be edited under a fit
that has already run.

**A calibration can be asked to cover a position, and refuses to be written short of
it.** Coverage is :func:`anamnesis.extraction.calibration.positions_calibrated`: the
rows a fit filled, not the width it allocated. A row no generation reached is written
as zeros and corrects nothing, so a table that is wide enough and short of data
disables the correction over its tail with no error anywhere downstream.
:func:`require_coverage` is the refusal, and :func:`write_calibration` applies it
before a byte is written. How far a generation reaches is its prompt length plus its
token budget, and the ruler's prompts are short, so the budget is what a late position
needs: :func:`budget_to_reach` sizes it for a required position, and
:func:`required_position` reads that position off the replay manifest of the bank the
calibration will correct. ``suppress_eos`` keeps a generation going past its
end-of-turn token as well, for a checkpoint whose answers stop early; the text it adds
is text the model does not produce in a real generation. Every write leaves a
:class:`CalibrationBuildReceipt` beside the basis saying what was asked for, what was
reached, and the digests of the files it describes.

**A calibration samples tokens first and reads states second**, the way a signature is
computed. :func:`generate_calibration_tokens` samples each prompt with no states
captured, into a replay manifest; :func:`replay_prompt_states` then runs one forward
pass over each whole sequence and reads every layer's state at every position. The
states the means are taken over are therefore computed the way the states they
correct are — a teacher-forced pass over banked ids — and sampling, which is most of
the cost, carries none of the capture. The manifest is written beside the basis, so a
basis or a table width can be refitted over the same text without sampling again.

The fit takes an iterable of :class:`PromptStates` rather than a loaded model, so
the arithmetic runs — and is tested — on a machine with no weights on it. The two
functions that run a model import the model runtime inside themselves.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from sklearn.decomposition import PCA

from anamnesis.config import GenerationConfig, ModelPreset, resolve_preset
from anamnesis.config.paths import prompts_path
from anamnesis.extraction.calibration import (
    POSITION_COUNT_FLOOR,
    POSITION_COUNTS_KEY,
    POSITIONAL_MEANS_KEY,
    load_positional_means,
    positions_calibrated,
)
from anamnesis.extraction.replay.manifest import (
    ReplayManifest,
    entry_from_ids,
    manifest_from_entries,
)
from anamnesis.provenance import digest_of_shas, file_sha

if TYPE_CHECKING:  # the annotation alone, so reading a fit costs no model runtime
    from anamnesis.extraction.model_loader import LoadedModel

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]
I64 = NDArray[np.int64]

PROMPT_HEADROOM = 200
"""Positions reserved above the token budget for the prompt itself, since a
position is absolute and a generated token sits after its prompt."""

CALIBRATION_PROMPT_SET = "calibration_prompts.json"
"""The prompt-set file the ruler is read from, shipped in the package."""

CALIBRATION_PROMPTS_KEY = "prompts"
"""The list of prompt strings inside that file."""

BUILD_RECEIPT_SUFFIX = ".build.json"
"""What replaces the basis file's suffix to name the receipt beside it, so a second
basis written into one directory under another name keeps its own receipt."""

BASIS_STEPS_PER_PROMPT = 32
"""Generated positions per prompt a per-layer basis is fitted over, evenly spaced from the
first generated state to the last. A basis of fifty components over a few thousand
units needs thousands of samples to pin anything past its leading directions, and
the replay reads every position anyway, so sampling densely costs no generation."""

TOKENS_SUFFIX = ".tokens.json"
"""What replaces the basis file's suffix to name the replay manifest of the sequences
the calibration was fitted over, keyed by prompt index."""


class CalibrationFitError(RuntimeError):
    """A fit that would write an artifact nothing can be projected onto.

    Raised rather than returned: an empty basis is not a partial result, it is a
    file that would make every residual-PCA feature downstream of it zero.
    """


class CoverageShortfall(CalibrationFitError):
    """A fit whose filled positions stop short of the position it was asked to cover.

    Raised before anything is written: a thin table that reaches disk is read as a
    calibration, and its zero rows switch the correction off over every position
    past the last filled one.
    """


def calibration_prompts(path: Path | None = None) -> tuple[str, ...]:
    """The ruler, read from the prompt-set file the package ships.

    The package copy is the answer, rather than
    :func:`anamnesis.config.paths.resolve_prompts_path`'s fallback to a Phase-0 tree:
    the ruler's bytes are pinned by digest, and a set that varied with which data
    tree was mounted would move every positional mean fitted against it.

    Raises
    ------
    CalibrationFitError
        When the file is absent, unreadable, or holds a set that is empty or repeats
        a prompt. A repeat is a refusal rather than a shrug because a prompt counted
        twice is weighted twice in every position's mean.
    """
    source = prompts_path(CALIBRATION_PROMPT_SET) if path is None else Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        prompts = tuple(str(text) for text in payload[CALIBRATION_PROMPTS_KEY])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CalibrationFitError(
            f"no usable calibration prompt set at {source}: {exc}"
        ) from exc
    if not prompts:
        raise CalibrationFitError(f"the calibration prompt set at {source} is empty")
    if len(set(prompts)) != len(prompts):
        raise CalibrationFitError(
            f"the calibration prompt set at {source} repeats a prompt, which would "
            "weight it twice in every position's mean"
        )
    return prompts


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
    basis_steps_per_prompt: int | None = None
    """Steps per prompt a per-layer basis was fitted over; ``None`` for a pooled fit."""

    @property
    def furthest_position(self) -> int:
        """The last position any prompt reached, as a coverage read on the means."""
        covered = self.position_counts.sum(axis=0) > 0
        return int(np.max(np.where(covered))) if covered.any() else 0

    @property
    def positions_calibrated(self) -> int:
        """How many leading positions the means fill, by
        :func:`anamnesis.extraction.calibration.positions_calibrated`.

        Read from this pass's counts when it measured the means, and from the means
        themselves when they came off disk, since a reused table has no counts here.
        """
        counts = self.position_counts if self.means_refitted else None
        return positions_calibrated(self.positional_means, counts)


def encode_prompt(tokenizer: Any, text: str, *, chat_template: bool = True) -> list[int]:
    """One ruler prompt as token ids: a user turn in the chat template, or bare.

    ``chat_template=False``, or a tokenizer with no template, takes the bare prompt —
    a base model's calibration must match the bare prompts its generations will use,
    and a base checkpoint can ship a tokenizer that carries a template it was never
    trained on.
    """
    import torch

    if not chat_template or tokenizer.chat_template is None:
        result = tokenizer(text, return_tensors="pt")["input_ids"]
    else:
        result = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
    ids = result if torch.is_tensor(result) else result["input_ids"]
    return [int(token) for token in ids[0].tolist()]


def generate_calibration_tokens(
    loaded: LoadedModel,
    prompts: Sequence[str],
    settings: GenerationConfig,
    *,
    suppress_eos: bool = False,
    chat_template: bool = True,
) -> ReplayManifest:
    """Sample each prompt under ``settings`` and keep only the token ids.

    The manifest is keyed by prompt index. Each generation is seeded by that index,
    so a calibration is reproducible and one prompt's text does not depend on which
    prompts ran before it. No hidden state, attention weight or logit is kept while
    sampling: the states are read afterwards by :func:`replay_prompt_states`.

    The key-value cache is asked for explicitly. A checkpoint whose configuration
    turns it off otherwise recomputes the whole prefix at every step, which makes a
    generation quadratic in its length and a long calibration unaffordable.

    ``suppress_eos`` generates with no stop token, so every generation runs to
    ``settings.max_new_tokens`` and continues past any end-of-turn token it samples.
    """
    import torch

    device = next(loaded.model.parameters()).device
    entries = {}
    started = time.perf_counter()
    for index, text in enumerate(prompts):
        prompt = encode_prompt(loaded.tokenizer, text, chat_template=chat_template)
        input_ids = torch.tensor([prompt], device=device)
        torch.manual_seed(index)
        with torch.no_grad():
            out = loaded.model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
                top_p=settings.top_p,
                do_sample=settings.do_sample,
                eos_token_id=None if suppress_eos else list(settings.eos_token_ids),
                use_cache=True,
            )
        sequence = out[0] if torch.is_tensor(out) else out.sequences[0]
        entries[index] = entry_from_ids(sequence.tolist(), len(prompt))
        # Every prompt, with a rate and what is left: on a CPU one prompt can take a
        # minute, and a pass that reports every tenth one looks hung for ten.
        elapsed = time.perf_counter() - started
        left = elapsed / (index + 1) * (len(prompts) - index - 1)
        logger.info(
            f"calibration tokens {index + 1}/{len(prompts)}: {entries[index].n_gen} generated, "
            f"{elapsed:.0f}s elapsed, about {left:.0f}s left"
        )
    return manifest_from_entries(entries)


def replay_prompt_states(
    loaded: LoadedModel, manifest: ReplayManifest
) -> Iterator[PromptStates]:
    """Every layer's state at every position of each banked sequence, one forward each.

    The positions read are the ones step-by-step decoding computes: the prompt, then
    one state per generated token except the last, whose state no decode step ever
    computes because no token is sampled from it. So the forward runs over the
    sequence without its final token, and ``steps[i]`` is absolute position
    ``prompt_length + i`` exactly as it is for a generation.

    Sequences come out in the manifest's id order. One sequence's states are
    materialised at a time and released before the next.
    """
    import torch

    device = next(loaded.model.parameters()).device
    for gen_id in manifest.gen_ids():
        entry = manifest.entry(gen_id)
        ids = torch.tensor([entry.input_ids[:-1]], device=device)
        with torch.no_grad():
            out = loaded.model(ids, output_hidden_states=True, use_cache=False)
        states = np.stack([layer[0].float().cpu().numpy() for layer in out.hidden_states])
        del out
        length = entry.prompt_length
        yield PromptStates(
            prompt_length=length,
            prefill=states[:, :length],
            steps=tuple(states[:, position] for position in range(length, states.shape[1])),
        )
        del states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_prompt_states(
    loaded: LoadedModel,
    prompts: Sequence[str],
    settings: GenerationConfig,
    *,
    suppress_eos: bool = False,
    chat_template: bool = True,
) -> Iterator[PromptStates]:
    """Sample the prompts, then replay them: :func:`generate_calibration_tokens`
    followed by :func:`replay_prompt_states`.

    Raises
    ------
    CalibrationFitError
        When ``settings`` asks for no hidden states, which are the only substrate a
        calibration reads.
    """
    if not settings.output_hidden_states:
        raise CalibrationFitError(
            "the decode policy asks for no hidden states, and a calibration reads nothing else"
        )
    manifest = generate_calibration_tokens(
        loaded, prompts, settings, suppress_eos=suppress_eos, chat_template=chat_template
    )
    yield from replay_prompt_states(loaded, manifest)


def required_position(manifest: ReplayManifest) -> int:
    """The last position a replay over ``manifest`` reads a state at.

    A sequence of ``n`` ids has states read through position ``n - 2``, the last
    one a decode step computes, so this is the position a calibration correcting
    that bank must cover.

    Raises
    ------
    ValueError
        When the manifest holds no sequences.
    """
    if not manifest.entries:
        raise ValueError("the replay manifest holds no sequences, so it requires no position")
    return max(len(entry.input_ids) for entry in manifest.entries.values()) - 2


def budget_to_reach(required_through: int, prompt_lengths: Sequence[int]) -> int:
    """The token budget at which every prompt reaches ``required_through``.

    A prompt of length ``L`` generating ``N`` tokens has states through position
    ``L + N - 2``, so the shortest prompt needs ``required_through - L + 2``. A
    generation that stops at its end-of-turn token earlier than its budget still
    falls short, and :func:`require_coverage` reads what was actually reached.

    Raises
    ------
    ValueError
        When no prompt lengths are given.
    """
    if not prompt_lengths:
        raise ValueError("no prompts to size a budget for")
    return max(1, int(required_through) - min(int(length) for length in prompt_lengths) + 2)


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


def basis_steps(n_steps: int, pooled: bool, per_prompt: int = BASIS_STEPS_PER_PROMPT) -> list[int]:
    """The generated steps, counted from one, that one prompt contributes to a basis.

    A pooled fit keeps its three sample points as they fall — first, middle, last —
    including when a short generation makes two of them the same step, because that
    is the shape the banked pooled bases were fitted under. A per-layer fit takes
    ``per_prompt`` distinct steps spread evenly from the first to the last, or every
    step of a generation shorter than that.
    """
    if n_steps <= 0:
        return []
    if pooled:
        return [1, max(1, n_steps // 2), n_steps]
    if n_steps <= per_prompt:
        return list(range(1, n_steps + 1))
    return sorted({int(step) for step in np.linspace(1, n_steps, per_prompt).round()})


def basis_samples(
    prompt: PromptStates,
    pca_layers: Sequence[int],
    pooled: bool,
    per_prompt: int = BASIS_STEPS_PER_PROMPT,
) -> list[tuple[int, int, F32]]:
    """``(layer, absolute position, state)`` at the steps :func:`basis_steps` picks."""
    n_steps = len(prompt.steps)
    chosen = basis_steps(n_steps, pooled, per_prompt)
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


ComponentCounts = int | Mapping[int, int]
"""Components a per-layer basis keeps: one count for every layer, or a count per layer."""


def components_at(n_components: ComponentCounts, layer: int) -> int:
    """The count ``n_components`` sets for ``layer``.

    Raises
    ------
    CalibrationFitError
        When a per-layer mapping has no count for ``layer``.
    """
    if isinstance(n_components, int):
        return n_components
    if int(layer) not in n_components:
        raise CalibrationFitError(
            f"no component count for layer {layer}; the per-layer counts name "
            f"{sorted(n_components)}"
        )
    return int(n_components[int(layer)])


def fit_per_layer_basis(
    samples: Sequence[tuple[int, int, F32]],
    means: F32,
    pca_layers: Sequence[int],
    n_components: ComponentCounts,
) -> dict[int, dict[str, Any]]:
    """One basis per layer, over positionally corrected states.

    ``n_components`` is one count for every layer or a count per layer; each layer
    keeps its own, and the extraction projects each layer onto the rows its basis
    holds.

    A sample beyond the reach of the means carries no correction and is dropped
    rather than fitted raw, which would mix two distributions in one basis. The
    reach is :func:`anamnesis.extraction.calibration.positions_calibrated`, not the
    table's width: a zero row past the last filled one subtracts nothing, so a
    sample there would be fitted raw.

    Raises
    ------
    CalibrationFitError
        When a named layer ends up with no corrected samples.
    """
    corrected: dict[int, list[F32]] = {int(layer): [] for layer in pca_layers}
    reach = positions_calibrated(means)
    for layer, absolute, state in samples:
        if absolute < reach:
            corrected[int(layer)].append(state - means[layer + 1, absolute])
    basis: dict[int, dict[str, Any]] = {}
    for layer in pca_layers:
        rows = corrected[int(layer)]
        if not rows:
            raise CalibrationFitError(
                f"no corrected samples at layer {layer}; the fit would be empty"
            )
        matrix = np.stack(rows).astype(np.float64)
        basis[int(layer)] = _fit_one(matrix, components_at(n_components, layer))
        logger.info(
            f"layer {layer}: {matrix.shape[0]} corrected samples -> "
            f"{basis[int(layer)]['components'].shape}, explaining "
            f"{basis[int(layer)]['explained_variance_ratio'].sum():.3f}"
        )
    return basis


def subspace_agreement(components_a: F32, components_b: F32) -> NDArray[np.float64]:
    """How far two bases agree on their leading ``k`` directions, for every ``k``.

    Entry ``k - 1`` is the smallest principal-angle cosine between the spans of the
    first ``k`` components of each basis: 1 when the two top-``k`` subspaces coincide,
    near 0 when one of them holds a direction the other lacks. Components are rows and
    orthonormal, as a PCA returns them.

    This is strict at every cut: two components with nearly equal variance trade
    ranks between fits, and the curve drops at that ``k`` even when both bases span the
    same directions. It says where a basis's *ordering* is stable, which is less than
    what it determines — :func:`determined_components` counts that instead.

    Raises
    ------
    ValueError
        When the two bases do not have the same width.
    """
    a = np.asarray(components_a, dtype=np.float64)
    b = np.asarray(components_b, dtype=np.float64)
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"bases of width {a.shape[1]} and {b.shape[1]} cannot be compared")
    depth = min(a.shape[0], b.shape[0])
    return np.array(
        [np.linalg.svd(a[:k] @ b[:k].T, compute_uv=False).min() for k in range(1, depth + 1)]
    )


def principal_cosines(components_a: F32, components_b: F32) -> NDArray[np.float64]:
    """The principal-angle cosines between the two full bases, largest first.

    Where :func:`subspace_agreement` asks whether each leading run is shared in order,
    this asks which directions the two bases share at all: a direction that has moved
    to another rank still counts here.
    """
    a = np.asarray(components_a, dtype=np.float64)
    b = np.asarray(components_b, dtype=np.float64)
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"bases of width {a.shape[1]} and {b.shape[1]} cannot be compared")
    return np.linalg.svd(a @ b.T, compute_uv=False)


def determined_components(cosines: NDArray[np.float64], threshold: float = 0.9) -> int:
    """How many directions two bases share: the :func:`principal_cosines` at or above ``threshold``.

    Over two fits to disjoint halves of a calibration's samples, this is how many
    components the samples determine — a floor, since each half has half the samples.
    """
    return int(np.count_nonzero(np.asarray(cosines) >= threshold))


def fit_calibration(
    states: Iterable[PromptStates],
    *,
    preset: str | ModelPreset,
    settings: GenerationConfig,
    n_components: ComponentCounts,
    pooled: bool = False,
    existing_means: F32 | None = None,
    max_positions: int | None = None,
    basis_steps_per_prompt: int = BASIS_STEPS_PER_PROMPT,
) -> CalibrationFit:
    """Both artifacts, from one pass over a prompt set's states.

    ``max_positions`` is the width of the means table, and defaults to the token
    budget plus :data:`PROMPT_HEADROOM`. A state past it is not counted. The width
    is an allocation, not a coverage: which rows are filled is decided by how far
    the generations reach, and is what :func:`require_coverage` reads.

    ``existing_means`` reuses a correction already on disk instead of measuring a
    new one, which is what lets a basis be refitted and compared without moving
    the means underneath the signatures already computed against them. When it is
    given, the position accumulator is skipped entirely.

    ``n_components`` is an upper bound: a fit over fewer samples or narrower states
    than that keeps what it can. A per-layer mapping sets each layer's own bound, and
    only a per-layer basis can hold one.

    Raises
    ------
    CalibrationFitError
        From the basis fit, when it would have nothing to fit over, or when
        ``max_positions`` is not a positive width.
    """
    row = resolve_preset(preset)
    if pooled and not isinstance(n_components, int):
        raise CalibrationFitError(
            "a pooled basis is one basis for every layer, so it takes one component count, "
            "not a count per layer"
        )
    depth = row.num_layers + 1
    if max_positions is None:
        max_positions = settings.max_new_tokens + PROMPT_HEADROOM
    if max_positions <= 0:
        raise CalibrationFitError(f"max_positions {max_positions} is not a table width")
    sums = np.zeros((depth, max_positions, row.hidden_dim), dtype=np.float64)
    counts: I64 = np.zeros((depth, max_positions), dtype=np.int64)
    # A sample is kept with the position it came from, because correcting it needs
    # that position and which correction to apply is not known until the means are.
    samples: list[tuple[int, int, F32]] = []

    for prompt in states:
        if existing_means is None:
            _accumulate_positions(sums, counts, prompt)
        samples.extend(basis_samples(prompt, row.pca_layers, pooled, basis_steps_per_prompt))

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
        basis_steps_per_prompt=None if pooled else basis_steps_per_prompt,
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


def require_coverage(fit: CalibrationFit, required_through: int) -> None:
    """Refuse a fit whose filled rows do not reach position ``required_through``.

    Raises
    ------
    CoverageShortfall
        When ``required_through`` is outside the table, or when the last filled
        row is before it. The message names both, and the ways to close the gap.
    """
    width = int(fit.positional_means.shape[1])
    if required_through < 0:
        raise CoverageShortfall(f"required_through {required_through} is not a position")
    if required_through >= width:
        raise CoverageShortfall(
            f"position {required_through} is required, and the means table is "
            f"{width} positions wide; raise max_positions past it"
        )
    reached = fit.positions_calibrated
    if reached <= required_through:
        raise CoverageShortfall(
            f"the positional means fill positions 0..{reached - 1} of a table "
            f"{width} wide, and position {required_through} is required; rows "
            f"{reached}..{required_through} are zeros and would correct nothing. "
            f"Raise the token budget, add prompts, or suppress the stop token so "
            f"generations reach it"
        )


def write_calibration(
    fit: CalibrationFit,
    means_path: Path,
    basis_path: Path,
    *,
    required_through: int | None = None,
    tokens: ReplayManifest | None = None,
) -> None:
    """Write the artifacts this pass produced, leaving reused means alone.

    Means that came off disk are not rewritten: the bytes would be the same, and
    the timestamp would say a correction moved when it did not.

    With ``required_through``, :func:`require_coverage` runs first, so a fit that
    falls short writes nothing. ``tokens``, the sequences the fit was taken over, is
    written beside the basis at :func:`tokens_path`.

    Raises
    ------
    CoverageShortfall
        From :func:`require_coverage`.
    """
    if required_through is not None:
        require_coverage(fit, required_through)
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
            f"(filled through position {fit.positions_calibrated - 1})"
        )
    with open(basis_path, "wb") as handle:
        pickle.dump(fit.basis, handle)
    logger.info(f"basis -> {basis_path}")
    if tokens is not None:
        target = tokens_path(basis_path)
        target.write_text(json.dumps(tokens.model_dump()))
        logger.info(f"calibration tokens ({tokens.n_ok} sequences) -> {target}")


class BuildCoverage(BaseModel):
    """What a calibration's means cover, as measured when it was written."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    table_positions: int = Field(description="Width of the means table")
    positions_calibrated: int = Field(
        description="Leading positions the means fill; position p is covered when p < this"
    )
    trailing_zero_rows: int = Field(description="Rows past the last filled one")
    measured_by: Literal["pos_counts", "means"] = Field(
        description="Whether coverage was read from this pass's counts or from reused means"
    )
    required_through: int | None = Field(
        description="The position the build was required to cover, if one was named"
    )


class CalibrationBuildReceipt(BaseModel):
    """What one calibration build was asked for, what it reached, and what it wrote.

    Written beside the basis by :func:`write_build_receipt`. The digests name the
    bytes on disk when the receipt was written — both artifacts, including means
    that were reused rather than refitted — and ``calibration_sha256`` is
    :func:`anamnesis.provenance.digest_of_shas` over them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(description="The preset row the build ran under")
    model_id: str = Field(description="The checkpoint the states were generated from")
    prompt_set_sha256: str = Field(description="SHA-256 of the prompts used, in order")
    n_prompts: int
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    eos_token_ids: list[int] = Field(description="The preset's stop tokens")
    suppress_eos: bool = Field(description="Whether generation ignored the stop tokens")
    chat_template: bool = Field(
        description="Whether prompts were encoded as a user turn in the tokenizer's chat "
                    "template where it has one, rather than bare"
    )
    pooled: bool
    n_components: int | dict[int, int] = Field(
        description="Components kept: one count for every layer, or a count per PCA layer"
    )
    means_refitted: bool
    coverage: BuildCoverage
    files: dict[str, str] = Field(description="Filename to SHA-256 of each artifact")
    calibration_sha256: str
    basis_steps_per_prompt: int | None = Field(
        default=None,
        description="Generated positions per prompt the per-layer basis was fitted over; "
                    "absent for a pooled fit, which keeps three",
    )
    tokens_sha256: str | None = Field(
        default=None,
        description="SHA-256 of the replay manifest the fit was taken over, when one was written",
    )


def prompts_digest(prompts: Sequence[str]) -> str:
    """SHA-256 over the prompts, in order, as a JSON list of strings."""
    return hashlib.sha256(
        json.dumps(list(prompts), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def build_receipt_path(basis_path: Path) -> Path:
    """Where the receipt for a basis written at ``basis_path`` goes."""
    return Path(basis_path).with_suffix(BUILD_RECEIPT_SUFFIX)


def tokens_path(basis_path: Path) -> Path:
    """Where the replay manifest of a basis written at ``basis_path`` goes."""
    return Path(basis_path).with_suffix(TOKENS_SUFFIX)


def build_receipt(
    fit: CalibrationFit,
    *,
    model: str,
    model_id: str,
    prompts: Sequence[str],
    settings: GenerationConfig,
    suppress_eos: bool,
    chat_template: bool,
    pooled: bool,
    n_components: ComponentCounts,
    means_path: Path,
    basis_path: Path,
    required_through: int | None = None,
    tokens_written: bool = False,
) -> CalibrationBuildReceipt:
    """The receipt for a fit whose artifacts are at ``means_path`` and ``basis_path``.

    ``tokens_written`` says :func:`write_calibration` wrote the sequences beside the
    basis in this pass, and their digest is then recorded; a file left there by an
    earlier pass is not this fit's and is not read.

    Raises
    ------
    FileNotFoundError
        When an artifact is absent: a receipt names bytes, so it is taken after
        :func:`write_calibration`.
    """
    width = int(fit.positional_means.shape[1])
    reached = fit.positions_calibrated
    files = {
        Path(means_path).name: file_sha(means_path),
        Path(basis_path).name: file_sha(basis_path),
    }
    return CalibrationBuildReceipt(
        model=model,
        model_id=model_id,
        prompt_set_sha256=prompts_digest(prompts),
        n_prompts=len(prompts),
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        do_sample=settings.do_sample,
        eos_token_ids=list(settings.eos_token_ids),
        suppress_eos=suppress_eos,
        chat_template=chat_template,
        pooled=pooled,
        n_components=n_components if isinstance(n_components, int) else dict(n_components),
        means_refitted=fit.means_refitted,
        coverage=BuildCoverage(
            table_positions=width,
            positions_calibrated=reached,
            trailing_zero_rows=width - reached,
            measured_by="pos_counts" if fit.means_refitted else "means",
            required_through=required_through,
        ),
        files=files,
        calibration_sha256=digest_of_shas(files),
        tokens_sha256=file_sha(tokens_path(basis_path)) if tokens_written else None,
        basis_steps_per_prompt=fit.basis_steps_per_prompt,
    )


def write_build_receipt(receipt: CalibrationBuildReceipt, basis_path: Path) -> Path:
    """Write ``receipt`` beside the basis it describes, returning where it went."""
    target = build_receipt_path(basis_path)
    target.write_text(json.dumps(receipt.model_dump(mode="json"), indent=2) + "\n")
    logger.info(
        f"build receipt -> {target} (positions calibrated "
        f"{receipt.coverage.positions_calibrated} of {receipt.coverage.table_positions})"
    )
    return target
