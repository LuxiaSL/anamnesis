"""The block-output family of a statistics row, reduced for many rows at once, batchmate-invariant.

:meth:`ExternalStatsSpan.finish <anamnesis.extraction.fast.external_stats.ExternalStatsSpan.finish>`
walks every layer's block output one row at a time: the activation norms (mean, deviation
and trajectory), the deltas between adjacent layers (norm mean and deviation, cosine mean),
the spectral smoothness at the sampled layers, the residual trajectory at the trajectory
layers, and the PCA projections at the PCA layers. :func:`block_output_features` computes
the same features for ``B`` rows of one step count in one pass over ``[B, steps, layers,
hidden]`` tensors.

**Batchmate invariance.** A row's features here are a function of that row's tensors and
the step count, never of its batchmates. Every reduction over a row's own steps or hidden
dimension is a primitive of :mod:`anamnesis.extraction.fast.invariant_ops`, which reduces
each row in a fixed order whatever the batch holds. Everything else is elementwise, an
index into a row's own steps, or a constant that depends only on the step count. The PCA
projections and the smoothness quadratic form are written as :func:`row_dot` products, not
matrix products, because a batched GEMM may choose its algorithm from the batch size; the
short-time spectrum of the operators is a direct DFT through :func:`row_dot` for the same
reason. A row reduced alone and the same row inside any batch therefore have the same bytes.

**Agreement with the per-row loop.** The arithmetic is the per-row loop's, operator for
operator, with each reduction's order replaced by the primitives' fixed order and the FFT
by its DFT. The features agree with :class:`ExternalStatsSpan`'s to rounding, not to the
byte. The names, their placeholders for short spans (residual trajectory below three
steps, windowed operators below four, the spectrum below eight) and the zeroing of a
trajectory whose first raw block output is null are the per-row loop's.

**The batched row.** Every family is reduced batched: the attention family in
:mod:`anamnesis.extraction.fast.batched_attention`, the block outputs here, and the key,
value, query, gate, per-head, cross-layer, output and routing families in
:mod:`anamnesis.extraction.fast.batched_families`. :func:`finish_rows` assembles each row's
vector in the lane's schema for the whole batch at once and writes each row's receipt. The
arithmetic differs from the statistics span's, so a batched row's lane id is
:func:`hybrid_lane_id`, which folds the statistics lane id together with the three batched
modules' and the primitives' source digests, the batched families, the per-row exceptions
(none), the effective-dimension method, the gate top-k's tie rule, the eigensolver's path,
the bucket rule and the row-reduction backend, under its own prefix
:data:`HYBRID_LANE_PREFIX`.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from anamnesis.extraction.fast import invariant_ops
from anamnesis.extraction.fast.external import (
    READOUT_SOURCE_SHA256,
    ExternalCaptureError,
    external_lane_id,
)
from anamnesis.extraction.fast.external_stats import (
    CAPTURE_KIND,
    STATS_SOURCE_SHA256,
    stats_lane_id,
)
from anamnesis.extraction.fast.features import GpuFeatureLane, GpuFeatureResult
from anamnesis.extraction.fast.invariant_ops import (
    row_dot,
    row_mean,
    row_norm,
    row_std,
    row_sum,
    row_sumsq,
)
from anamnesis.extraction.fast.ops import _stft_constants, _unit_abscissa, trajectory_indices

BATCHED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
"""This module's source digest, part of every hybrid row's lane id."""

INVARIANT_OPS_SOURCE_SHA256 = hashlib.sha256(Path(invariant_ops.__file__).read_bytes()).hexdigest()
"""The primitives' source digest, part of every hybrid row's lane id."""

FAMILIES_SOURCE_SHA256 = hashlib.sha256(
    Path(__file__).with_name("batched_families.py").read_bytes()).hexdigest()
"""The source digest of :mod:`anamnesis.extraction.fast.batched_families`, part of every
hybrid row's lane id."""

ATTENTION_SOURCE_SHA256 = hashlib.sha256(
    Path(__file__).with_name("batched_attention.py").read_bytes()).hexdigest()
"""The source digest of :mod:`anamnesis.extraction.fast.batched_attention`, part of every
hybrid row's lane id."""

BATCHED_FAMILIES = ("attention", "block_outputs", "keys", "per_head_keys", "values", "queries",
                    "gates", "cross_layer", "outputs", "routing")
"""The families a hybrid row reduces batched: all of them."""

PER_ROW_EXCEPTIONS: tuple[str, ...] = ()
"""Features the batched reduction computes per row: none."""

TOPK_TIES = "lowest-index"
"""How the gate's top-k breaks ties at the k-th value: the lowest column index first
(:func:`anamnesis.extraction.fast.batched_families.gate_topk_overlap_rows`)."""

def eigvalsh_path(device: torch.device | str) -> str:
    """How the spectral Laplacians' eigenvalues are solved on ``device``
    (:func:`anamnesis.extraction.fast.batched_attention.eigvalsh_rows`): ``batched`` (one call
    over a chunk's rows and sampled layers, never a batch of one) on CUDA, ``per-matrix`` on
    CPU."""
    return "batched" if torch.device(device).type == "cuda" else "per-matrix"

EFF_DIM = "norms"
"""How a hybrid row computes every effective dimension: from Frobenius norms."""

BUCKET_RULE = "exact-steps"
"""Rows are batched only with rows of the same step count; nothing is padded."""

HYBRID_LANE_PREFIX = "engine-stats-b1-"
"""Every hybrid row's lane id starts with this; no per-row statistics id does."""

WORKING_BYTES = 1 << 30
"""The float64 working set one chunk of layers may take; chunking does not change a byte."""


_DEVICE_POSITIONAL: dict[tuple[int, str], tuple[weakref.ref, Tensor]] = {}


def device_positional_means(pm: np.ndarray, device: torch.device | str) -> Tensor:
    """A lane's positional means as one tensor on ``device``, uploaded once and kept for as long
    as the host table lives. Lanes that share one table (one per capped step count) share the
    device copy. Each row's slice is then gathered on the device: a copy, so its bytes are the
    host table's."""
    key = (id(pm), str(torch.device(device)))
    hit = _DEVICE_POSITIONAL.get(key)
    if hit is not None and hit[0]() is pm:
        return hit[1]
    for stale in [k for k, (ref, _) in _DEVICE_POSITIONAL.items() if ref() is None]:
        del _DEVICE_POSITIONAL[stale]
    table = torch.as_tensor(np.ascontiguousarray(pm), device=device)
    _DEVICE_POSITIONAL[key] = (weakref.ref(pm), table)
    return table


def row_backend(device: torch.device | str) -> str:
    """The implementation the primitives run on a device: ``triton`` or ``torch``."""
    device = torch.device(device)
    return "triton" if device.type == "cuda" and invariant_ops.HAVE_TRITON else "torch"


def hybrid_lane_id(lane: GpuFeatureLane, capture_identity: Mapping[str, Any]) -> str:
    """The lane id of a row whose families were reduced batched."""
    return HYBRID_LANE_PREFIX + hashlib.sha256(json.dumps(
        dict(stats_lane_id=stats_lane_id(lane, capture_identity), **batched_receipt(lane.device)),
        sort_keys=True, default=str).encode()).hexdigest()


def batched_receipt(device: torch.device | str) -> dict[str, Any]:
    """What a hybrid row's receipt records of its batched reduction on ``device``, and its
    lane id folds in."""
    return dict(families=list(BATCHED_FAMILIES), source_sha256=BATCHED_SOURCE_SHA256,
                families_source_sha256=FAMILIES_SOURCE_SHA256,
                attention_source_sha256=ATTENTION_SOURCE_SHA256,
                invariant_ops_source_sha256=INVARIANT_OPS_SOURCE_SHA256,
                bucket_rule=BUCKET_RULE, row_backend=row_backend(device), eff_dim=EFF_DIM,
                topk_ties=TOPK_TIES, eigvalsh=eigvalsh_path(device), per_row=list(PER_ROW_EXCEPTIONS))


@dataclass(frozen=True)
class BlockOutputs:
    """The block-output family of ``B`` rows: ``names`` and ``values [B, len(names)]``
    (float64; each value is the one the per-row loop would emit before the collector's
    float32 cast)."""

    names: tuple[str, ...]
    values: Tensor

    def row(self, index: int) -> Tensor:
        return self.values[index]


@lru_cache(maxsize=None)
def _dft_basis(device: torch.device) -> tuple[Tensor, Tensor]:
    """The 64-point real DFT's cosine and sine rows, ``[33, 64]`` each, float64."""
    t = torch.arange(64, dtype=torch.float64)
    k = torch.arange(33, dtype=torch.float64)
    angle = 2 * math.pi * ((k[:, None] * t[None, :]) % 64) / 64
    return angle.cos().to(device), angle.sin().to(device)


def _slope_rows(y: Tensor) -> Tensor:
    """``ops.slope`` of each row of ``y [B, m]`` with every point kept."""
    m = y.shape[-1]
    if m < 2:
        return y.new_zeros(y.shape[:-1])
    w, count, xc, denominator = _unit_abscissa(m, y.device)
    ym = row_sum(y * w) / count
    return row_sum(xc * (y - ym[..., None]) * w) / denominator


def _temporal_rows(x: Tensor, n_windows: int, include_stft: bool) -> Tensor:
    """``ops.temporal`` of each row of ``x [B, m]``: ``[B, 3 * n_windows (+ 5)]`` float32."""
    x = x.float().double()
    batch, n = x.shape
    if n < 4:
        windowed = x.new_zeros(batch, 3 * n_windows)
    else:
        width = n // n_windows
        pieces = []
        for i in range(n_windows):
            w = x[:, i * width : (i + 1) * width if i < n_windows - 1 else n]
            if not w.shape[1]:
                raise ValueError("empty temporal window (unsupported reference configuration)")
            w = w.contiguous()
            pieces.extend((row_mean(w), row_std(w), _slope_rows(w)))
        windowed = torch.stack(pieces, dim=1)
    if not include_stft:
        return windowed.float()
    if n < 8:
        spectral = x.new_zeros(batch, 5)
    else:
        if n < 64:
            x = torch.nn.functional.pad(x, (0, 64 - n))
        extended = torch.nn.functional.pad(x, (32, 32 + (-x.shape[1]) % 32))
        window, window_sum, freq, bands = _stft_constants(x.device)
        frames = (extended.unfold(1, 64, 32) * window).unsqueeze(2)  # [B, F, 1, 64]
        cos_basis, sin_basis = _dft_basis(x.device)
        shape = (batch, frames.shape[1], 33, 64)
        re = row_dot(frames.expand(shape), cos_basis.expand(shape)) / window_sum
        im = row_dot(frames.expand(shape), sin_basis.expand(shape)) / window_sum
        power = row_mean((re.square() + im.square()).transpose(1, 2).contiguous())  # [B, 33]
        total = row_sum(power)
        p = power / total.clamp_min(1e-30)[:, None]
        centroid = row_sum(p * freq)
        bandwidth = row_sum(p * (freq - centroid[:, None]).square()).sqrt()
        band_mass = [row_sum(p[:, int(b[0]) : int(b[-1]) + 1].contiguous()) for b in bands]
        spectral = torch.stack((centroid, bandwidth, *band_mass), dim=1)
        spectral = torch.where((total < 1e-12)[:, None], spectral.new_zeros(()), spectral)
    return torch.cat((windowed, spectral), dim=1).float()


def _operator_names(prefix: str, n_windows: int, include_stft: bool) -> list[str]:
    from anamnesis.extraction.feature_families.operators import _stft_names, _windowed_names

    names = list(_windowed_names(prefix, n_windows))
    if include_stft:
        names.extend(_stft_names(prefix))
    return names


def _rowcos(a: Tensor, b: Tensor) -> Tensor:
    """``ops.rowcos`` over the last dimension."""
    return row_dot(a, b) / (row_norm(a) * row_norm(b)).clamp_min(1e-12)


def _cos(a: Tensor, b: Tensor) -> Tensor:
    """``ops.cos`` over the last dimension."""
    a, b = a.double(), b.double()
    na, nb = row_norm(a), row_norm(b)
    value = row_dot(a, b) / (na * nb).clamp_min(1e-30)
    return torch.where((na < 1e-12) | (nb < 1e-12), torch.zeros_like(value), value)


class _Columns:
    """Named ``[B]`` columns, kept as views until one stack at the end."""

    def __init__(self) -> None:
        self.names: list[str] = []
        self.columns: list[Tensor] = []

    def put(self, name: str, value: Tensor) -> None:
        self.names.append(name)
        self.columns.append(value)

    def block(self, names: Sequence[str], values: Tensor) -> None:
        """``values [B, len(names)]`` as one column per name; cast to float64 once."""
        values = values.double()
        for i, name in enumerate(names):
            self.put(name, values[:, i])


def _residual_trajectory(cols: _Columns, layer: int, corrected: Tensor, raw_first_norm: Tensor,
                         n_windows: int, include_stft: bool) -> None:
    """``FamilyReducer.residual_trajectory`` for ``corrected [B, steps, hidden]``."""
    from anamnesis.extraction.feature_families.residual_stream import _trajectory_feature_names

    batch, steps = corrected.shape[:2]
    if steps < 3:
        names = _trajectory_feature_names(layer, n_windows, include_stft)
        cols.block(names, corrected.new_zeros(batch, len(names), dtype=torch.float64))
        return
    prefix = f"res_traj_L{layer}"
    x = corrected.double()
    vel = x[:, 1:] - x[:, :-1]
    vn = row_norm(vel)
    angles = _cos(vel[:, :-1].float(), vel[:, 1:].float()).clamp(-1, 1).acos()
    accel = row_norm(vel[:, 1:] - vel[:, :-1])
    names = [prefix + "_velocity_norm_mean", prefix + "_velocity_norm_std",
             prefix + "_direction_change_mean", prefix + "_direction_change_std",
             prefix + "_directness",
             prefix + "_acceleration_norm_mean", prefix + "_acceleration_norm_std"]
    values = [row_mean(vn), row_std(vn), row_mean(angles), row_std(angles),
              row_norm(x[:, -1] - x[:, 0]) / row_sum(vn).clamp_min(1e-12),
              row_mean(accel), row_std(accel)]
    names += _operator_names(prefix + "_velocity_norm", n_windows, include_stft)
    names += _operator_names(prefix + "_direction_change", n_windows, include_stft)
    block = torch.cat([torch.stack(values, dim=1),
                       _temporal_rows(vn, n_windows, include_stft).double(),
                       _temporal_rows(angles, n_windows, include_stft).double()], dim=1)
    block = torch.where((raw_first_norm < 1e-6)[:, None], block.new_zeros(()), block)
    cols.block(names, block)


def block_output_features(lane: GpuFeatureLane, hidden: Tensor, starts: Sequence[int],
                          laplacians: Mapping[int, tuple[Sequence[int], Tensor]]) -> BlockOutputs:
    """The block-output family of ``B`` rows of one step count.

    ``hidden [B, steps, layers, width]`` is each row's block outputs in the engine's layout
    (the signed row's ``depth``); ``starts[b]`` is row ``b``'s prompt length, which selects
    its positional means; ``laplacians[layer]`` is, at each sampled layer, the spectral
    steps (the same for every row of one step count) and ``[B, k, k]``, each row's
    normalised Laplacian as its attention reduction left it.

    Raises
    ------
    ExternalCaptureError
        When the shapes disagree with each other, the positional means do not cover a row,
        or a Laplacian is missing.
    """
    if hidden.ndim != 4:
        raise ExternalCaptureError(f"hidden must be [rows, steps, layers, width], got {tuple(hidden.shape)}")
    batch, steps, layers, width = hidden.shape
    if len(starts) != batch:
        raise ExternalCaptureError(f"{batch} rows but {len(starts)} starts")
    pm = lane.pm
    if pm.ndim != 3 or pm.shape[0] < layers + 1 or pm.shape[2] != width or any(
            pm.shape[1] < int(s) + steps for s in starts):
        raise ExternalCaptureError("the lane's positional means do not cover this capture")
    if not hidden.is_floating_point():
        raise ExternalCaptureError("hidden must be floating point")
    config, families = lane.config, lane.families
    sampled = set(config.sampled_layers)
    missing = sorted(sampled - set(laplacians))
    if missing:
        raise ExternalCaptureError(f"no spectral Laplacian for sampled layers {missing}")
    n_windows, include_stft = families.temporal_n_windows, families.enable_stft
    device = lane.device
    traj = trajectory_indices(steps, config.trajectory_points)
    pca_index = trajectory_indices(steps, config.pca_temporal_samples)
    k = config.pca_components

    cols = _Columns()
    per_layer = max(1, batch * steps * width * 8 * 4)
    chunk = max(1, min(layers, WORKING_BYTES // per_layer))
    table = device_positional_means(pm, device)
    positions = torch.as_tensor([[int(s) + t for t in range(steps)] for s in starts],
                                dtype=torch.long, device=device)  # [B, steps]
    previous: Tensor | None = None
    for first in range(0, layers, chunk):
        last = min(layers, first + chunk)
        h = hidden[:, :, first:last].to(device).float()
        positional = table[1 + first:1 + last][:, positions].permute(1, 2, 0, 3)  # [B, steps, chunk, width]
        corrected = h - positional  # [B, steps, chunk, width]
        if corrected.dtype not in invariant_ops.SUPPORTED_DTYPES:
            raise ExternalCaptureError(f"block outputs corrected to {corrected.dtype}")
        norms = row_norm(corrected).float().transpose(1, 2).contiguous()  # [B, chunk, steps]
        mean, dev = row_mean(norms).double(), row_std(norms).double()
        traj_values = norms[:, :, traj].double()
        if previous is not None:
            # The first layer of this chunk differs from the last of the one before.
            prev_all = torch.cat([previous, corrected[:, :, :-1]], dim=2)
            delta = corrected - prev_all
            delta_first = first
        else:
            prev_all = corrected[:, :, :-1]
            delta = corrected[:, :, 1:] - prev_all
            delta_first = first + 1
        if delta.shape[2]:
            dn = row_norm(delta).float().transpose(1, 2).contiguous()
            dc = _rowcos(delta.double(), prev_all.double()).float().transpose(1, 2).contiguous()
            dn_mean, dn_std = row_mean(dn).double(), row_std(dn).double()
            dc_mean = row_mean(dc).double()
        for j, layer in enumerate(range(first, last)):
            cols.put(f"activation_norm_mean_L{layer}", mean[:, j])
            cols.put(f"activation_norm_std_L{layer}", dev[:, j])
            for i in range(len(traj)):
                cols.put(f"activation_norm_traj{i}_L{layer}", traj_values[:, j, i])
            if layer >= delta_first:
                d = layer - delta_first
                cols.put(f"delta_norm_mean_L{layer - 1}", dn_mean[:, d])
                cols.put(f"delta_norm_std_L{layer - 1}", dn_std[:, d])
                cols.put(f"delta_cosine_mean_L{layer - 1}", dc_mean[:, d])
            x = corrected[:, :, j]  # [B, steps, width]
            if layer in sampled:
                spectral_steps, lap = laplacians[layer]
                lap = lap.to(device).double()
                signal = row_norm(x[:, list(spectral_steps)].double())  # [B, k]
                shape = lap.shape
                lap_signal = row_dot(lap.transpose(1, 2).contiguous(),
                                     signal[:, None, :].expand(shape).contiguous())
                cols.put(f"spectral_smoothness_L{layer}",
                         row_dot(lap_signal, signal) / row_sumsq(signal).clamp_min(1e-12))
            if layer in families.trajectory_layers:
                _residual_trajectory(cols, layer, x, row_norm(h[:, 0, j]), n_windows, include_stft)
            if layer in config.pca_layers:
                components = lane.pca[layer] if isinstance(lane.pca, dict) else lane.pca
                pca_mean = lane.pca_mean[layer] if isinstance(lane.pca_mean, dict) else lane.pca_mean
                centred = x[:, pca_index].double() - pca_mean  # [B, T, width]
                shape = (batch, len(pca_index), k, width)
                projections = row_dot(centred[:, :, None, :].expand(shape),
                                      components[:k][None, None].expand(shape))
                for ti in range(len(pca_index)):
                    for ci in range(k):
                        cols.put(f"pca_L{layer}_t{ti}_c{ci}", projections[:, ti, ci])
        previous = corrected[:, :, -1:]
    values = (torch.stack(cols.columns, dim=1) if cols.columns
              else torch.zeros(batch, 0, dtype=torch.float64, device=device))
    return BlockOutputs(tuple(cols.names), values)


@dataclass(frozen=True)
class RowSpan:
    """What a batched row's receipt names of its span: the prompt length ``start``, the span
    end ``end`` (the length of its token ids) and the token ids up to it."""

    start: int
    end: int
    token_ids: Sequence[int]


_ORDER: dict[tuple[int, int], tuple[tuple[str, ...], tuple[str, ...], Tensor]] = {}


def _schema_order(names: tuple[str, ...], schema: tuple[str, ...], device) -> Tensor:
    """The index of each schema name in ``names``, kept for the pair."""
    key = (id(names), id(schema))
    hit = _ORDER.get(key)
    if hit is not None and hit[0] is names and hit[1] is schema and hit[2].device == torch.device(device):
        return hit[2]
    if len(set(names)) != len(names) or set(names) != set(schema) or len(names) != len(schema):
        missing, extra = set(schema) - set(names), set(names) - set(schema)
        raise ValueError(
            f"GPU feature schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    where = {name: i for i, name in enumerate(names)}
    order = torch.tensor([where[name] for name in schema], dtype=torch.long, device=device)
    if len(_ORDER) > 64:
        _ORDER.clear()
    _ORDER[key] = (names, schema, order)
    return order


def finish_rows(lane: GpuFeatureLane, spans: Sequence[RowSpan], names: tuple[str, ...],
                values: Tensor, mean_logprob: Tensor, knnlm: Tensor | None,
                capture_identity: Mapping[str, Any], *, num_layers: int,
                routing_names: Sequence[str] | None = None, routing_values: Tensor | None = None,
                retained_attention_summary_bytes: int = 0) -> list[GpuFeatureResult | ExternalCaptureError]:
    """Each row's ``GpuFeatureResult``, assembled for the whole batch at once.

    ``values [B, len(names)]`` holds every lane feature (float64, each the value the per-row
    reduction would emit before its float32 cast); they are put in the lane's schema order by
    one gather and cast once, and the routing family's float32 ``routing_values [B, ...]``
    follows them. ``num_layers`` is the capture's layer count, every one of which the attention
    reduction read. ``mean_logprob [B]`` and ``knnlm [B, hidden]`` (or ``None``) are per row.
    The receipt's lane id is :func:`hybrid_lane_id`; the fields common to the batch are
    computed once. A row with a nonfinite feature gets the :class:`ExternalCaptureError` that
    refuses it, in its place.

    Raises
    ------
    ExternalCaptureError
        When ``values`` does not match ``names`` and ``spans``.
    """
    batch = len(spans)
    if values.ndim != 2 or values.shape != (batch, len(names)):
        raise ExternalCaptureError(
            f"{batch} rows and {len(names)} batched names for values {tuple(values.shape)}")
    schema = tuple(lane.names)
    order = _schema_order(names, schema, values.device)
    vector = values.index_select(1, order).float()
    all_names = list(schema)
    routed = routing_names is not None
    if routed:
        vector = torch.cat([vector, routing_values.to(vector.device, vector.dtype)], dim=1)
        all_names += list(routing_names)
    features = vector.cpu().numpy()
    finite = np.isfinite(features).all(axis=1)
    logprob = mean_logprob.double().cpu().tolist()
    knn = knnlm.float().cpu().numpy() if knnlm is not None else None
    names_out = tuple(all_names)
    common = dict(
        lane_id=hybrid_lane_id(lane, capture_identity),
        stats_lane_id=stats_lane_id(lane, capture_identity),
        external_lane_id=external_lane_id(lane, capture_identity),
        fast_lane_id=lane.lane_id,
        lane_source_sha256=dict(lane.source_sha256),
        readout_source_sha256=dict(READOUT_SOURCE_SHA256),
        stats_source_sha256=STATS_SOURCE_SHA256,
        batched=batched_receipt(lane.device),
        capture=dict(capture_identity),
        capture_kind=CAPTURE_KIND,
        reductions=dict({family: "batched" for family in BATCHED_FAMILIES},
                        logits="stats", routing="batched" if routed else "none"),
        replay_path="external-stats-hybrid",
        feature_schema_sha256=hashlib.sha256(json.dumps(all_names).encode()).hexdigest(),
        calibration_sha256=lane.calibration_sha256,
        channel="probe-free",
        certified=False,
        batch_size=batch,
        attention_layers_reduced=int(num_layers),
        retained_attention_summary_bytes=int(retained_attention_summary_bytes),
        routing_family=routed,
    )
    results: list[GpuFeatureResult | ExternalCaptureError] = []
    for b, span in enumerate(spans):
        if not finite[b]:
            results.append(ExternalCaptureError("the reduction produced a nonfinite feature"))
            continue
        receipt = dict(common, replay_id=str(uuid.uuid4()), span_start=span.start,
                       span_end=span.end,
                       input_tokens_sha256=hashlib.sha256(np.asarray(
                           list(span.token_ids[:span.end]), dtype="<i8").tobytes()).hexdigest())
        receipt["capture"] = dict(capture_identity)
        receipt["lane_source_sha256"] = dict(lane.source_sha256)
        receipt["readout_source_sha256"] = dict(READOUT_SOURCE_SHA256)
        receipt["batched"] = dict(common["batched"])
        receipt["reductions"] = dict(common["reductions"])
        results.append(GpuFeatureResult(features[b], names_out, float(logprob[b]),
                                        knn[b].copy() if knn is not None else None, receipt))
    return results


__all__ = (
    "ATTENTION_SOURCE_SHA256",
    "BATCHED_FAMILIES",
    "BATCHED_SOURCE_SHA256",
    "BUCKET_RULE",
    "BlockOutputs",
    "EFF_DIM",
    "FAMILIES_SOURCE_SHA256",
    "HYBRID_LANE_PREFIX",
    "INVARIANT_OPS_SOURCE_SHA256",
    "PER_ROW_EXCEPTIONS",
    "RowSpan",
    "TOPK_TIES",
    "batched_receipt",
    "block_output_features",
    "eigvalsh_path",
    "finish_rows",
    "hybrid_lane_id",
    "row_backend",
)
