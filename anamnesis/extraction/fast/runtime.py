"""The runtime a fast-lane pass needs, resolved once for every reader of it.

Two entry points run the lane: ``anamnesis.scripts.run_gpu_replay`` banks through
it, and ``anamnesis.scripts.qualify_box`` measures whether it agrees with the
numeric anchor on this machine. They have to be talking about the same machine
state, and the state is not one setting — it is the pinned arithmetic, the feature
schema the selected spans resolve to, the calibration the features are corrected
by, the hooked capture surface and the lane built over all four. Two entry points
resolving that separately can qualify one configuration and run another, and the
only symptom is a qualification receipt that describes a machine the run does not
reproduce. :func:`resolve_fast_lane` is the one resolution, so there is nothing
for them to disagree about.

:func:`require_lane_arithmetic` is separable because its refusal is worth reaching
before anything else is read. It is idempotent and :func:`resolve_fast_lane` calls
it as its first act, so a caller that wants to refuse before it opens a manifest
asks for it early and a caller that does not still gets it.

What is *not* here is each command's own selection policy: a qualification pairs
rows and needs two distinct spans, a replay takes a whole bank and needs one.
Those are different questions about what to run, asked by the commands, and the
ids they arrive at come in as an argument.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from anamnesis.config import ModelConfig, ModelPreset
from anamnesis.extraction.calibration import CALIBRATION_ARTIFACT_NAMES, load_calibration
from anamnesis.extraction.replay_config import native_replay_configs
from anamnesis.provenance import digest_of_shas, file_sha

if TYPE_CHECKING:
    from anamnesis.config import ExtractionConfig, FeaturePipelineConfig
    from anamnesis.extraction.fast.schema import GpuFeatureSchema

F32 = NDArray[np.float32]

WORKSPACE_ENV = "CUBLAS_WORKSPACE_CONFIG"
"""The cuBLAS workspace variable the lane identity names. cuBLAS reads it when it
initialises, so it cannot be set from inside the process that uses it."""

WORKSPACE_VALUE = ":16:8"
"""The one value the lane is defined at. Another value is another lane."""

DEFAULT_DEVICE = "cuda:0"
"""The device a lane runs on unless a caller names another. A verdict or a bank is
about the device it ran on, so the device is never inferred from what is free."""


def require_lane_arithmetic() -> None:
    """Pin the arithmetic that fixes the lane's last digits, refusing what cannot be.

    Deterministic algorithms on, TF32 off on both the matmul and the cuDNN path: a
    TF32 reduction and a deterministic kernel choice each change the trailing digits
    of every feature, so both are part of the lane identity rather than a
    performance preference. The cuBLAS workspace setting is the same kind of fact
    and the same identity, but it has to be in the environment before Python starts,
    which is why this refuses rather than sets it.

    Calling it twice is harmless: the torch settings are assignments and the
    environment is only read.

    Raises
    ------
    ValueError
        When :data:`WORKSPACE_ENV` is unset or holds anything but
        :data:`WORKSPACE_VALUE`.
    """
    import torch

    if os.environ.get(WORKSPACE_ENV) != WORKSPACE_VALUE:
        raise ValueError(
            f"set {WORKSPACE_ENV}={WORKSPACE_VALUE} before Python startup; it is part "
            "of the lane identity, so a verdict or a bank produced without it is "
            "about a different lane"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@dataclass(frozen=True)
class LaneSpan:
    """One banked generation as the lane replays it: its ids, and where the prompt ends.

    ``end`` is the full sequence length rather than a field, because a span is the
    whole banked sequence and a second copy of its length could disagree with it.
    """

    gen_id: int
    input_ids: list[int]
    prompt_length: int

    @property
    def end(self) -> int:
        """One past the last banked position — the length of the whole sequence."""
        return len(self.input_ids)

    @property
    def n_steps(self) -> int:
        """Incremental steps the span replays: every generated position but the first.

        The first generated token is produced by the prefill, so a span of *n*
        generated tokens takes *n - 1* incremental forwards, and the feature
        schema's time-series widths are counted in those.
        """
        return self.end - self.prompt_length - 1


@dataclass(frozen=True)
class FastLaneRuntime:
    """Everything a fast-lane pass reads, resolved and mutually consistent.

    The schema, the calibration and the lane are resolved from one another here, so
    a caller cannot hold a lane built for one calibration beside a schema resolved
    against a different basis.
    """

    extraction: ExtractionConfig
    families: FeaturePipelineConfig
    spans: tuple[LaneSpan, ...]
    schemas: dict[int, GpuFeatureSchema]
    feature_names: tuple[str, ...]
    positional_means: F32
    pca_components: Any
    pca_mean: Any
    calibration_files: dict[str, str]
    calibration_sha256: str
    model_files: dict[str, str] | None
    loaded: Any
    lane: Any

    @property
    def gen_ids(self) -> tuple[int, ...]:
        """The generations this runtime covers, in the order they were asked for."""
        return tuple(span.gen_id for span in self.spans)


def resolve_lane_spans(
    entries: Mapping[str, Any], gen_ids: Sequence[int], *, positions_calibrated: int
) -> tuple[LaneSpan, ...]:
    """The manifest rows for ``gen_ids``, each checked against what the lane supports.

    A span needs a prompt, at least two generated tokens (one prefilled and one
    incremental step), and a last position the positional means cover — an
    uncalibrated position would be corrected against nothing and the feature would
    be a different quantity under the same name.

    Parameters
    ----------
    entries
        A replay manifest's ``entries``, keyed by generation id as a string.
    gen_ids
        Which of them to replay, in the order the caller means to run them.
    positions_calibrated
        Width of the positional-means table: the number of positions calibration
        covers.

    Raises
    ------
    ValueError
        When an id is not in the manifest, or its span falls outside what the lane
        and the calibration support.
    """
    spans: list[LaneSpan] = []
    for gen_id in gen_ids:
        row = entries.get(str(gen_id))
        if row is None:
            raise ValueError(f"generation {gen_id} is not in the manifest")
        span = LaneSpan(
            gen_id=int(gen_id),
            input_ids=list(row["input_ids"]),
            prompt_length=int(row["prompt_length"]),
        )
        if (
            not 0 < span.prompt_length < span.end - 1
            or span.end - 2 >= positions_calibrated
        ):
            raise ValueError(f"generation {gen_id} outside supported span/calibration")
        spans.append(span)
    return tuple(spans)


def weight_file_digests(model_path: str | Path) -> dict[str, str]:
    """Digests of the checkpoint bytes: its config, and every safetensors shard.

    A receipt that named the checkpoint directory would say nothing a reader can
    check, because a directory is replaced in place. These name the bytes.

    Raises
    ------
    ValueError
        When the directory holds no safetensors file. There is then nothing to
        digest, and a bank stamped with a path instead of with bytes has no
        provenance at all.
    """
    root = Path(model_path)
    weights = sorted(root.glob("*.safetensors"))
    if not weights:
        raise ValueError("local safetensors checkpoint required for explicit provenance")
    return {path.name: file_sha(path) for path in [root / "config.json", *weights]}


def resolve_fast_lane(
    *,
    preset: ModelPreset,
    model_path: str,
    calib_dir: Path,
    entries: Mapping[str, Any],
    gen_ids: Sequence[int],
    device: str = DEFAULT_DEVICE,
    require_local_weights: bool = False,
) -> FastLaneRuntime:
    """Pin the arithmetic, resolve the schema and calibration, load the lane.

    The order is the cheap refusals first: nothing loads a model until the
    calibration is complete, every selected span is supported, and all of them
    resolve to one feature schema. Mixing two schemas inside one pass is refused
    rather than reconciled — the vectors would carry the same names for different
    widths.

    ``require_local_weights`` adds the checkpoint digests to the runtime, for a
    caller that stamps them into a bank's receipt. A qualification does not, since
    its verdict is about the machine rather than about a corpus.

    Parameters
    ----------
    preset
        The model row. Depth, head counts, dtype and the layer plan all come from
        it, so a lane cannot describe a model other than the one it loads.
    model_path
        Local checkpoint directory, overriding the preset's identifier because a
        replay's provenance is the bytes on this disk.
    calib_dir
        Directory holding the positional means and the residual basis.
    entries
        A replay manifest's ``entries``.
    gen_ids
        The generations to cover, already selected by the caller's own policy.
    device
        Device the model and the lane run on; part of what a verdict is about.

    Raises
    ------
    ValueError
        From :func:`require_lane_arithmetic`, from an incomplete calibration
        directory, from :func:`resolve_lane_spans`, from
        :func:`weight_file_digests`, or when the selected spans do not share one
        feature schema.
    """
    require_lane_arithmetic()
    extraction, families = native_replay_configs(preset)
    positional_means, pca_components, pca_mean = load_calibration(calib_dir, True)
    if positional_means is None or pca_components is None or pca_mean is None:
        raise ValueError("complete positional/PCA calibration required")
    calibration_files = {
        name: file_sha(Path(calib_dir) / name) for name in CALIBRATION_ARTIFACT_NAMES
    }
    calibration_sha256 = digest_of_shas(calibration_files)
    spans = resolve_lane_spans(
        entries, gen_ids, positions_calibrated=int(positional_means.shape[1])
    )
    model_files = weight_file_digests(model_path) if require_local_weights else None

    from anamnesis.extraction.fast.schema import resolve_gpu_schema

    schemas = {
        span.gen_id: resolve_gpu_schema(
            preset.num_layers, span.n_steps, extraction, families, pca_components
        )
        for span in spans
    }
    names = {schema.feature_names for schema in schemas.values()}
    if len(names) != 1:
        raise ValueError("selected spans have different feature schemas; do not mix")
    feature_names = names.pop()

    from anamnesis.extraction.fast.features import GpuFeatureLane
    from anamnesis.extraction.model_loader import load_model

    # This lane banks features and no raw tensors, so it hooks exactly the surfaces
    # its reducers read and nothing else: :class:`anamnesis.extraction.fast.features.GpuFeatureLane`
    # takes pre-RoPE keys, values, queries and gate activations for
    # ``preset.sampled_layers``, and no consumer of this runtime reads o_proj
    # outputs, so no attention-output hook is registered. Capturing every layer
    # belongs to :mod:`anamnesis.extraction.replay.cell`, which banks the tensors
    # themselves and therefore has to keep depth available as an axis to measure
    # later. A capture nothing reads costs device memory per step and buys nothing.
    sampled = list(preset.sampled_layers)
    loaded = load_model(
        ModelConfig.from_preset(preset, model_id=model_path, device_map=device),
        sampled_layers=preset.sampled_layers,
        register_gate_hooks=True,
        key_layers=sampled,
        value_layers=sampled,
        query_layers=sampled,
    )
    lane = GpuFeatureLane(
        extraction,
        families,
        list(feature_names),
        positional_means,
        pca_components,
        pca_mean,
        device=device,
        calibration_sha256=calibration_sha256,
        replay_path="full",
    )
    return FastLaneRuntime(
        extraction=extraction,
        families=families,
        spans=spans,
        schemas=schemas,
        feature_names=feature_names,
        positional_means=positional_means,
        pca_components=pca_components,
        pca_mean=pca_mean,
        calibration_files=calibration_files,
        calibration_sha256=calibration_sha256,
        model_files=model_files,
        loaded=loaded,
        lane=lane,
    )
