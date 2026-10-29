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

**The hybrid row.** :class:`HybridStatsSpan` is an :class:`ExternalStatsSpan` whose
``finish`` takes every family but attention computed batched: the block outputs here, and
the key, value, query, gate, per-head, cross-layer, output and routing families in
:mod:`anamnesis.extraction.fast.batched_families`. The attention family it reduces exactly
as the statistics span does. Its arithmetic differs from the statistics span's, so its lane
id is :func:`hybrid_lane_id`, which folds the statistics lane id together with both batched
modules' and the primitives' source digests, the batched families, the per-row exceptions,
the effective-dimension method, the bucket rule and the row-reduction backend, under its
own prefix :data:`HYBRID_LANE_PREFIX`.
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
    ExternalStatsSpan,
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

BATCHED_FAMILIES = ("block_outputs", "keys", "per_head_keys", "values", "queries", "gates",
                    "cross_layer", "outputs", "routing")
"""The families a hybrid row reduces batched; the attention family is reduced per row."""

PER_ROW_EXCEPTIONS = ("gate_topk_overlap",)
"""Features the batched reduction computes per row, inside the batch, for a stated reason
(:mod:`anamnesis.extraction.fast.batched_families`)."""

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
    payload = dict(stats_lane_id=stats_lane_id(lane, capture_identity),
                   batched_source_sha256=BATCHED_SOURCE_SHA256,
                   invariant_ops_source_sha256=INVARIANT_OPS_SOURCE_SHA256,
                   batched_families=list(BATCHED_FAMILIES), bucket_rule=BUCKET_RULE,
                   row_backend=row_backend(lane.device),
                   families_source_sha256=FAMILIES_SOURCE_SHA256, eff_dim=EFF_DIM,
                   per_row=list(PER_ROW_EXCEPTIONS))
    return HYBRID_LANE_PREFIX + hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


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
class BatchedRow:
    """One row's share of a batched reduction: every lane feature computed batched
    (``names`` and float64 ``values``, one per name), the row's mean log probability, and
    its routing family's names and float32 values (``None`` for a row without routing)."""

    names: tuple[str, ...]
    values: Tensor
    mean_logprob: float
    routing_names: tuple[str, ...] | None = None
    routing_values: Tensor | None = None


class HybridStatsSpan(ExternalStatsSpan):
    """A statistics span whose every family but attention is handed over, computed batched.

    :meth:`finish` takes the row's :class:`BatchedRow` (the block-output, key, value,
    query, gate, per-head, cross-layer, output and routing families) and adds the
    attention family its own :meth:`attention_stats` reduced; the receipt's fields are
    the statistics span's, plus the hybrid lane id and which families were batched.
    """

    def _validate_shapes(self, hidden: Tensor, keys: Mapping[int, Tensor],
                         values: Mapping[int, Tensor], queries: Mapping[int, Tensor],
                         gates: Mapping[int, Tensor], logit_stats: Tensor,
                         token_ids: Sequence[int]) -> None:
        """``ExternalSpan._validate``'s checks that read shapes and settings only. The
        finiteness of the surfaces is checked once for the whole batch, before the batched
        reduction, by :func:`~anamnesis.extraction.fast.engine_reducer.reduce_signed_rows_hybrid`."""
        lane, steps = self.lane, self.steps
        if not torch.are_deterministic_algorithms_enabled():
            raise ExternalCaptureError("the reduction runs with deterministic algorithms on")
        if lane.device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
            raise ExternalCaptureError("the reduction runs with TF32 off")
        if not isinstance(hidden, Tensor) or hidden.ndim != 3 or hidden.shape[:2] != (self.num_layers, steps):
            raise ExternalCaptureError(
                f"hidden must be [{self.num_layers}, {steps}, width], got "
                f"{tuple(hidden.shape) if isinstance(hidden, Tensor) else type(hidden).__name__}")
        if (lane.pm.ndim != 3 or lane.pm.shape[0] < self.num_layers + 1
                or lane.pm.shape[1] < self.start + steps or lane.pm.shape[2] != hidden.shape[2]):
            raise ExternalCaptureError("the lane's positional means do not cover this capture")
        sampled = set(lane.config.sampled_layers)
        if not sampled <= set(range(self.num_layers)):
            raise ExternalCaptureError("a sampled layer lies outside the capture")
        for name, mapping, ndim in (("keys", keys, 3), ("values", values, 3), ("queries", queries, 3),
                                    ("gates", gates, 2)):
            if set(mapping) != sampled:
                raise ExternalCaptureError(f"{name} must hold exactly the sampled layers")
            for layer, x in mapping.items():
                if x.ndim != ndim or x.shape[0] != steps:
                    raise ExternalCaptureError(f"{name} layer {layer} has shape {tuple(x.shape)}")
                if not x.is_floating_point():
                    raise ExternalCaptureError("the capture holds a nonfloating or nonfinite tensor")
        if not hidden.is_floating_point():
            raise ExternalCaptureError("the capture holds a nonfloating or nonfinite tensor")
        if (not isinstance(logit_stats, Tensor) or tuple(logit_stats.shape) != (steps, 6)
                or logit_stats.dtype not in (torch.float32, torch.float64)):
            raise ExternalCaptureError(f"logit_stats must be finite float32 or float64 [{steps}, 6]")
        if len(token_ids) < self.end:
            raise ExternalCaptureError("token_ids do not cover the span")

    def finish(self, *, hidden: Tensor, keys: Mapping[int, Tensor],
               values: Mapping[int, Tensor], queries: Mapping[int, Tensor],
               gates: Mapping[int, Tensor], logit_stats: Tensor,
               token_ids: Sequence[int], capture_identity: Mapping[str, Any],
               batched: BatchedRow, batch_size: int = 1) -> GpuFeatureResult:
        """The row's features: its attention family and ``batched``, assembled in the
        lane's schema, with the routing family after them. ``hidden`` (``[layers, steps,
        width]``, a view is enough), the per-layer surfaces and ``logit_stats`` are the
        row's own, checked here for shape; their values were reduced into ``batched``.

        Raises
        ------
        ExternalCaptureError
            When a surface is mis-shaped, an attention layer was not reduced, ``batched``
            is mis-shaped, or a feature is nonfinite.
        """
        if self.finished:
            raise ExternalCaptureError("a span is finished once")
        if not isinstance(batched, BatchedRow):
            raise ExternalCaptureError("a hybrid span is finished with its BatchedRow")
        if batched.values.ndim != 1 or batched.values.shape[0] != len(batched.names):
            raise ExternalCaptureError(
                f"{len(batched.names)} batched names for values {tuple(batched.values.shape)}")
        self.finished = True
        lane, steps, start, end = self.lane, self.steps, self.start, self.end
        self._validate_shapes(hidden, keys, values, queries, gates, logit_stats, token_ids)
        if self.attention_reducer.seen != set(range(self.num_layers)):
            missing = sorted(set(range(self.num_layers)) - self.attention_reducer.seen)
            raise ExternalCaptureError(f"attention layers {missing} were not reduced")
        out, attention = self.collector, self.attention_reducer
        # The per-head summaries were handed to the batched reduction.
        attention.per_head.clear()
        out.put_block(list(batched.names), batched.values.to(lane.device))
        vector = out.finish(list(lane.names))
        names = list(lane.names)
        routed = batched.routing_names is not None
        if routed:
            vector = torch.cat([vector, batched.routing_values.to(vector.device, vector.dtype)])
            names += list(batched.routing_names)
        features = vector.cpu().numpy()
        if not np.isfinite(features).all():
            raise ExternalCaptureError("the reduction produced a nonfinite feature")
        knnlm = (hidden[-1][steps - 1].float().cpu().numpy().copy()
                 if lane.config.enable_knnlm_baseline else None)
        receipt = dict(
            lane_id=hybrid_lane_id(lane, capture_identity),
            stats_lane_id=stats_lane_id(lane, capture_identity),
            external_lane_id=external_lane_id(lane, capture_identity),
            fast_lane_id=lane.lane_id,
            lane_source_sha256=dict(lane.source_sha256),
            readout_source_sha256=dict(READOUT_SOURCE_SHA256),
            stats_source_sha256=STATS_SOURCE_SHA256,
            batched=dict(families=list(BATCHED_FAMILIES),
                         source_sha256=BATCHED_SOURCE_SHA256,
                         families_source_sha256=FAMILIES_SOURCE_SHA256,
                         invariant_ops_source_sha256=INVARIANT_OPS_SOURCE_SHA256,
                         bucket_rule=BUCKET_RULE, row_backend=row_backend(lane.device),
                         eff_dim=EFF_DIM, per_row=list(PER_ROW_EXCEPTIONS)),
            capture=dict(capture_identity),
            capture_kind=CAPTURE_KIND,
            reductions=dict({family: "batched" for family in BATCHED_FAMILIES},
                            attention="stats", logits="stats",
                            routing="batched" if routed else "none"),
            replay_id=str(uuid.uuid4()),
            replay_path="external-stats-hybrid",
            span_start=start,
            span_end=end,
            input_tokens_sha256=hashlib.sha256(
                np.asarray(list(token_ids[:end]), dtype="<i8").tobytes()).hexdigest(),
            feature_schema_sha256=hashlib.sha256(json.dumps(names).encode()).hexdigest(),
            calibration_sha256=lane.calibration_sha256,
            channel="probe-free",
            certified=False,
            batch_size=int(batch_size),
            attention_layers_reduced=len(attention.seen),
            retained_attention_summary_bytes=attention.retained_bytes,
            routing_family=routed,
        )
        return GpuFeatureResult(features, tuple(names), float(batched.mean_logprob), knnlm, receipt)


__all__ = (
    "BATCHED_FAMILIES",
    "BATCHED_SOURCE_SHA256",
    "BUCKET_RULE",
    "BatchedRow",
    "BlockOutputs",
    "EFF_DIM",
    "FAMILIES_SOURCE_SHA256",
    "HYBRID_LANE_PREFIX",
    "HybridStatsSpan",
    "INVARIANT_OPS_SOURCE_SHA256",
    "PER_ROW_EXCEPTIONS",
    "block_output_features",
    "hybrid_lane_id",
    "row_backend",
)
