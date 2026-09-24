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
before a byte is written. Instruct checkpoints stop at their end-of-turn token, so
their generations rarely reach late positions at all; ``suppress_eos`` in
:func:`generate_prompt_states` keeps each generation going to its token budget, which
is what makes a late position reachable. Every write leaves a
:class:`CalibrationBuildReceipt` beside the basis saying what was asked for, what was
reached, and the digests of the files it describes.

The fit takes an iterable of :class:`PromptStates` rather than a loaded model, so
the arithmetic runs — and is tested — on a machine with no weights on it.
:func:`generate_prompt_states` is the one function here that runs a model, and it
is where the model runtime is imported.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal, Sequence

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


def generate_prompt_states(
    loaded: LoadedModel,
    prompts: Sequence[str],
    settings: GenerationConfig,
    *,
    suppress_eos: bool = False,
) -> Iterator[PromptStates]:
    """Generate each prompt under ``settings`` and hand its states back as numpy.

    Seeded by prompt index so a calibration is reproducible, and a checkpoint with
    no chat template takes the bare prompt — a base model's calibration must match
    the bare prompts its generations will use.

    ``suppress_eos`` generates with no stop token, so every generation runs to
    ``settings.max_new_tokens`` and continues past any end-of-turn token it
    samples. Without it, an instruct checkpoint ends where its answer ends, and
    positions past the longest answer get no data however large the budget is.

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
                eos_token_id=None if suppress_eos else list(settings.eos_token_ids),
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
    max_positions: int | None = None,
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
    than that keeps what it can.

    Raises
    ------
    CalibrationFitError
        From the basis fit, when it would have nothing to fit over, or when
        ``max_positions`` is not a positive width.
    """
    row = resolve_preset(preset)
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
) -> None:
    """Write the artifacts this pass produced, leaving reused means alone.

    Means that came off disk are not rewritten: the bytes would be the same, and
    the timestamp would say a correction moved when it did not.

    With ``required_through``, :func:`require_coverage` runs first, so a fit that
    falls short writes nothing.

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
    pooled: bool
    n_components: int
    means_refitted: bool
    coverage: BuildCoverage
    files: dict[str, str] = Field(description="Filename to SHA-256 of each artifact")
    calibration_sha256: str


def prompts_digest(prompts: Sequence[str]) -> str:
    """SHA-256 over the prompts, in order, as a JSON list of strings."""
    return hashlib.sha256(
        json.dumps(list(prompts), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def build_receipt_path(basis_path: Path) -> Path:
    """Where the receipt for a basis written at ``basis_path`` goes."""
    return Path(basis_path).with_suffix(BUILD_RECEIPT_SUFFIX)


def build_receipt(
    fit: CalibrationFit,
    *,
    model: str,
    model_id: str,
    prompts: Sequence[str],
    settings: GenerationConfig,
    suppress_eos: bool,
    pooled: bool,
    n_components: int,
    means_path: Path,
    basis_path: Path,
    required_through: int | None = None,
) -> CalibrationBuildReceipt:
    """The receipt for a fit whose artifacts are at ``means_path`` and ``basis_path``.

    Raises
    ------
    FileNotFoundError
        When either artifact is absent: a receipt names bytes, so it is taken after
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
        pooled=pooled,
        n_components=n_components,
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
