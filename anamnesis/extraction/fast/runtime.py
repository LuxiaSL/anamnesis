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

A process that keeps one model resident and harvests from its own generations
needs the same resolution without a manifest and without a second load.
:func:`prepare_fast_lane` is that: the arithmetic and the calibration, bound to a
model the caller already holds (checked by :func:`check_loaded_model`) or to one
:func:`load_lane_model` loads. :func:`resolve_fast_lane` binds through the same
step, and :func:`anamnesis.extraction.fast.harvest.harvest_loaded` runs one span
against the result.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from anamnesis.config import ModelConfig, ModelPreset, resolve_preset
from anamnesis.extraction.calibration import (
    CALIBRATION_ARTIFACT_NAMES,
    load_position_counts,
    load_positional_means,
    positions_calibrated,
    read_pca_basis,
    resolve_pca_model,
)
from anamnesis.extraction.replay_config import native_replay_configs
from anamnesis.provenance import digest_of_shas, file_sha

if TYPE_CHECKING:
    from anamnesis.config import ExtractionConfig, FeaturePipelineConfig
    from anamnesis.extraction.fast.features import GpuFeatureLane
    from anamnesis.extraction.fast.schema import GpuFeatureSchema
    from anamnesis.extraction.model_loader import LoadedModel

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
        How many leading positions the positional means fill, from
        :func:`anamnesis.extraction.calibration.positions_calibrated`. The table's
        width is not this number: rows past the last filled one are zeros and
        correct nothing.

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
        if not span_is_supported(span, positions_calibrated):
            raise ValueError(f"generation {gen_id} outside supported span/calibration")
        spans.append(span)
    return tuple(spans)


def span_is_supported(span: LaneSpan, positions_calibrated: int) -> bool:
    """Whether the lane can replay ``span`` against a calibration this wide.

    A span needs a prompt, at least two generated tokens, and a last replayed
    position among the rows the positional means fill.
    """
    return (
        0 < span.prompt_length < span.end - 1 and span.end - 2 < positions_calibrated
    )


def weight_file_digests(model_path: str | Path) -> dict[str, str]:
    """Digests of the checkpoint bytes: its config, and every safetensors shard.

    A receipt that named the checkpoint directory would say nothing a reader can
    check, because a directory is replaced in place. These name the bytes.

    Each file is read once per process for as long as its size and mtime hold, so
    a process that stamps many runs against one checkpoint pays for it once.

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
    return {
        path.name: _cached_file_sha(path) for path in [root / "config.json", *weights]
    }


@dataclass(frozen=True)
class LaneCalibration:
    """The battery a preset declares, and the calibration its features are corrected by.

    Everything a lane needs except a model, so a caller can refuse an incomplete
    calibration or an unsupported span before paying for a load.
    """

    preset: ModelPreset
    extraction: ExtractionConfig
    families: FeaturePipelineConfig
    positional_means: F32
    pca_components: Any
    pca_mean: Any
    calibration_files: dict[str, str]
    calibration_sha256: str
    position_counts: NDArray[np.int64] | None = None

    @property
    def positions_calibrated(self) -> int:
        """How many leading positions the means fill: the positions a span may reach.

        :func:`anamnesis.extraction.calibration.positions_calibrated`, read from the
        fit's counts when the archive carries them. The table's width is not this
        number: rows past the last filled one are zeros and correct nothing.
        """
        return positions_calibrated(self.positional_means, self.position_counts)

    def schema(self, n_steps: int) -> GpuFeatureSchema:
        """The feature names and family slices a span of ``n_steps`` resolves to.

        The names depend on the step count only for the shortest spans, where a
        family emits zeros under its full names or drops a windowed series, so two
        spans past that threshold share one schema.
        """
        from anamnesis.extraction.fast.schema import resolve_gpu_schema

        return resolve_gpu_schema(
            self.preset.num_layers,
            n_steps,
            self.extraction,
            self.families,
            self.pca_components,
        )

    def build_lane(self, feature_names: Sequence[str], device: str) -> GpuFeatureLane:
        """A full-path lane over this calibration, emitting exactly ``feature_names``."""
        from anamnesis.extraction.fast.features import GpuFeatureLane

        return GpuFeatureLane(
            self.extraction,
            self.families,
            list(feature_names),
            self.positional_means,
            self.pca_components,
            self.pca_mean,
            device=device,
            calibration_sha256=self.calibration_sha256,
            replay_path="full",
        )


def read_lane_calibration(preset: ModelPreset, calib_dir: Path) -> LaneCalibration:
    """The preset's native battery and a complete calibration, digested.

    The basis is read by :func:`anamnesis.extraction.calibration.read_pca_basis`, so
    either shape the lane projects onto is accepted: the pooled basis the banked
    calibrations hold, and the per-layer basis
    :mod:`anamnesis.extraction.calibration_fit` writes by default.

    Raises
    ------
    ValueError
        When the directory lacks the positional means or the residual basis, or the
        basis stores no mean to centre on. A lane with any of them missing would
        correct its features against nothing and emit them under the corrected names.
    """
    extraction, families = native_replay_configs(preset)
    positional_means = load_positional_means(Path(calib_dir))
    pca_path = resolve_pca_model(Path(calib_dir))
    if positional_means is None or pca_path is None:
        raise ValueError("complete positional/PCA calibration required")
    pca_components, pca_mean = read_pca_basis(pca_path).float32_arrays()
    if pca_mean is None:
        raise ValueError(f"{pca_path} holds a basis with no mean to centre on")
    calibration_files = {
        name: file_sha(Path(calib_dir) / name) for name in CALIBRATION_ARTIFACT_NAMES
    }
    return LaneCalibration(
        preset=preset,
        extraction=extraction,
        families=families,
        positional_means=positional_means,
        pca_components=pca_components,
        pca_mean=pca_mean,
        calibration_files=calibration_files,
        calibration_sha256=digest_of_shas(calibration_files),
        position_counts=load_position_counts(calib_dir),
    )


def load_lane_model(preset: ModelPreset, model_path: str, device: str) -> LoadedModel:
    """Load a checkpoint with exactly the capture surface the lane reads.

    This lane banks features and no raw tensors, so it hooks exactly the surfaces
    its reducers read: pre-RoPE keys, values, queries and gate activations at
    ``preset.sampled_layers``. No consumer here reads o_proj outputs, so no
    attention-output hook is registered — a capture nothing reads costs device
    memory every step and buys nothing. :mod:`anamnesis.extraction.replay.cell`
    banks the tensors themselves and so must keep every layer available instead.

    A process that keeps one model resident loads it here once and hands the result
    to :func:`prepare_fast_lane`, so the lane reads the surface it expects without
    loading a second copy.
    """
    from anamnesis.extraction.model_loader import load_model

    sampled = list(preset.sampled_layers)
    return load_model(
        ModelConfig.from_preset(preset, model_id=model_path, device_map=device),
        sampled_layers=sampled,
        register_gate_hooks=True,
        key_layers=sampled,
        value_layers=sampled,
        query_layers=sampled,
    )


def _declared_device(device: str) -> Any:
    """``device`` as the lane compares it: a CUDA device always carries its index."""
    import torch

    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def check_loaded_model(loaded: LoadedModel, preset: ModelPreset, device: str) -> None:
    """Refuse a loaded model the lane would read wrongly or not at all.

    A model handed in by a caller did not come through :func:`load_lane_model`, so
    each fact that function guarantees is checked here instead: the architecture
    the lane is written for, the depth and widths the preset declares (a feature
    named for layer 20 of one model is a different quantity in another), eager
    attention, eval mode, every parameter on the declared device, and a forward
    hook on each projection the reducers read at each sampled layer.

    Raises
    ------
    ValueError
        Naming the first fact that does not hold.
    """
    from anamnesis.extraction.model_loader import decoder_layers

    model = loaded.model
    config = model.config
    if config.model_type != "llama":
        raise ValueError(
            f"the fast lane reads a dense Llama; this model is {config.model_type!r}"
        )
    for field, have, want in (
        ("num_hidden_layers", config.num_hidden_layers, preset.num_layers),
        ("hidden_size", config.hidden_size, preset.hidden_dim),
        ("num_attention_heads", config.num_attention_heads, preset.num_attention_heads),
        ("num_key_value_heads", config.num_key_value_heads, preset.num_kv_heads),
    ):
        if have != want:
            raise ValueError(
                f"model {field}={have} but preset {preset.name!r} declares {want}; "
                "the layer plan and the calibration describe a different model"
            )
    if config._attn_implementation != "eager":
        raise ValueError("GPU lane requires eager attention")
    if model.training:
        raise ValueError("GPU lane requires an eval-mode model")
    declared = _declared_device(device)
    if any(p.device != declared for p in model.parameters()):
        raise ValueError(f"the lane requires a model entirely on {declared}")
    layers = decoder_layers(model)
    for layer in preset.sampled_layers:
        block = layers[layer]
        for name, module in (
            ("k_proj", block.self_attn.k_proj),
            ("v_proj", block.self_attn.v_proj),
            ("q_proj", block.self_attn.q_proj),
            ("gate_proj", block.mlp.gate_proj),
        ):
            if not module._forward_hooks:
                raise ValueError(
                    f"layer {layer} {name} carries no capture hook; load the model "
                    "with load_lane_model so the lane reads the surface it expects"
                )


_WEIGHT_SHAS: dict[tuple[str, int, int], str] = {}
"""File digests keyed by (resolved path, size, mtime), for :func:`weight_file_digests`."""


def _cached_file_sha(path: Path) -> str:
    """:func:`file_sha`, computed once per file for as long as its size and mtime hold.

    A checkpoint is tens to hundreds of gigabytes, so a process that stamps many
    runs against one resident model would otherwise re-read it for every stamp. A
    file rewritten in place changes its mtime, so the digest cannot go stale.
    """
    stat = path.stat()
    key = (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
    digest = _WEIGHT_SHAS.get(key)
    if digest is None:
        digest = _WEIGHT_SHAS[key] = file_sha(path)
    return digest


@dataclass
class PreparedLane:
    """A calibration bound to a resident model: what a process harvests against.

    Built once by :func:`prepare_fast_lane`. A lane is keyed by the feature schema
    it emits, and a schema depends on a span's step count only for the shortest
    spans, so :meth:`lane` builds at most a handful of lanes over the life of a
    process and reuses them.
    """

    calibration: LaneCalibration
    loaded: LoadedModel
    device: str
    model_files: dict[str, str] | None
    _schemas: dict[int, GpuFeatureSchema] = field(
        default_factory=dict, repr=False, compare=False
    )
    _lanes: dict[tuple[str, ...], GpuFeatureLane] = field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def preset(self) -> ModelPreset:
        """The model row the calibration and the model were both checked against."""
        return self.calibration.preset

    def lane(self, n_steps: int) -> tuple[GpuFeatureLane, GpuFeatureSchema]:
        """The lane and schema for a span of ``n_steps`` incremental steps."""
        schema = self._schemas.get(n_steps)
        if schema is None:
            schema = self._schemas[n_steps] = self.calibration.schema(n_steps)
        lane = self._lanes.get(schema.feature_names)
        if lane is None:
            lane = self.calibration.build_lane(schema.feature_names, self.device)
            self._lanes[schema.feature_names] = lane
        return lane, schema

    def provenance(self) -> dict[str, Any]:
        """What a harvest ran against: preset, calibration digests, checkpoint digests.

        Per-span identity — the lane id, the token digest — is on each result's
        receipt; this is the part every span in the process shares.
        """
        return dict(
            preset=self.preset.name,
            device=str(self.device),
            calibration_files=dict(self.calibration.calibration_files),
            calibration_sha256=self.calibration.calibration_sha256,
            model_files_sha256=self.model_files,
            model_config=self.loaded.model.config.to_dict(),
            lane_runtime_source_sha256=file_sha(Path(__file__)),
        )


def prepare_fast_lane(
    preset: str | ModelPreset,
    calib_dir: Path,
    *,
    device: str = DEFAULT_DEVICE,
    loaded: LoadedModel | None = None,
    model_path: str | None = None,
    model_files: Mapping[str, str] | None = None,
    require_local_weights: bool = False,
) -> PreparedLane:
    """Pin the arithmetic, read the calibration, and bind a model to them.

    With ``loaded``, the model is the caller's and is checked rather than loaded:
    :func:`check_loaded_model` refuses one whose architecture, depth, widths,
    attention kernel, placement or capture hooks the lane cannot read. Without it,
    the checkpoint at ``model_path`` is loaded by :func:`load_lane_model`.

    Parameters
    ----------
    preset
        A registry name or a :class:`~anamnesis.config.ModelPreset`. A row that is
        not in the registry can be passed directly.
    calib_dir
        Directory holding the positional means and the residual basis.
    device
        Device the model and the lane run on; part of what a result is about.
    loaded
        A model already resident in this process.
    model_path
        Local checkpoint directory. Required to load a model, and to digest the
        checkpoint when ``require_local_weights`` asks for digests.
    model_files
        Checkpoint digests the caller already holds, keyed by file name as
        :func:`weight_file_digests` keys them. Taken as given in place of reading
        the checkpoint.
    require_local_weights
        Stamp the checkpoint's digests into :meth:`PreparedLane.provenance`. They
        are computed once per file and reused while the file's size and mtime hold.

    Raises
    ------
    ValueError
        From :func:`require_lane_arithmetic`, :func:`read_lane_calibration`,
        :func:`check_loaded_model` or :func:`weight_file_digests`; or when neither a
        model nor a path to load one is given.
    """
    require_lane_arithmetic()
    row = resolve_preset(preset)
    calibration = read_lane_calibration(row, Path(calib_dir))
    if loaded is None and model_path is None:
        raise ValueError("pass a loaded model or a model_path to load one from")
    digests: dict[str, str] | None = None
    if model_files is not None:
        digests = dict(model_files)
    elif require_local_weights:
        if model_path is None:
            raise ValueError(
                "require_local_weights digests a checkpoint directory; pass model_path "
                "or the digests themselves as model_files"
            )
        digests = weight_file_digests(model_path)
    if loaded is None:
        assert model_path is not None
        loaded = load_lane_model(row, model_path, device)
    return _bind(calibration, loaded, device, digests)


def _bind(
    calibration: LaneCalibration,
    loaded: LoadedModel,
    device: str,
    model_files: dict[str, str] | None,
) -> PreparedLane:
    """A calibration and a model the lane can read, as one :class:`PreparedLane`."""
    check_loaded_model(loaded, calibration.preset, device)
    return PreparedLane(
        calibration=calibration, loaded=loaded, device=device, model_files=model_files
    )


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
        :func:`weight_file_digests`, from :func:`check_loaded_model`, or when the
        selected spans do not share one feature schema.
    """
    require_lane_arithmetic()
    calibration = read_lane_calibration(preset, calib_dir)
    spans = resolve_lane_spans(
        entries, gen_ids, positions_calibrated=calibration.positions_calibrated
    )
    model_files = weight_file_digests(model_path) if require_local_weights else None
    schemas = {span.gen_id: calibration.schema(span.n_steps) for span in spans}
    names = {schema.feature_names for schema in schemas.values()}
    if len(names) != 1:
        raise ValueError("selected spans have different feature schemas; do not mix")
    feature_names = names.pop()

    prepared = _bind(
        calibration, load_lane_model(preset, model_path, device), device, model_files
    )
    lane, _ = prepared.lane(spans[0].n_steps)
    return FastLaneRuntime(
        extraction=calibration.extraction,
        families=calibration.families,
        spans=spans,
        schemas=schemas,
        feature_names=feature_names,
        positional_means=calibration.positional_means,
        pca_components=calibration.pca_components,
        pca_mean=calibration.pca_mean,
        calibration_files=calibration.calibration_files,
        calibration_sha256=calibration.calibration_sha256,
        model_files=model_files,
        loaded=prepared.loaded,
        lane=lane,
    )
