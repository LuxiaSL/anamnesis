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

BUCKET_FLOOR = 8
"""Step counts below this have feature schemas of their own (the fast lane's per-step cap), so
rows below it are batched only with rows of their exact step count."""

BUCKET_RULES = ("exact-steps", "pow2", "pow4", "tail256")
"""The step-count bucket rules (:func:`bucket_width`):

* ``exact-steps``: rows are batched only with rows of the same step count; nothing is padded.
* ``pow2``: a row of ``n >= BUCKET_FLOOR`` steps is padded to the smallest power of two at or
  above ``n``.
* ``pow4``: to the smallest ``8 * 4**i`` at or above ``n`` (8, 32, 128, 512, ...).
* ``tail256``: to 256 steps (the signing decoder's tail) up to 256, and to the smallest power of
  two above that.

Below :data:`BUCKET_FLOOR` every rule is ``exact-steps``."""

BUCKET_RULE = "pow2"
"""The default bucket rule."""


def bucket_width(steps: int, rule: str = BUCKET_RULE) -> int:
    """The canonical step width a row of ``steps`` steps is padded to under ``rule``."""
    steps = int(steps)
    if rule not in BUCKET_RULES:
        raise ValueError(f"bucket rule must be one of {BUCKET_RULES}, not {rule!r}")
    if steps < 1:
        raise ValueError(f"a row needs at least one step, not {steps}")
    if rule == "exact-steps" or steps < BUCKET_FLOOR:
        return steps
    pow2 = 1 << (steps - 1).bit_length()
    if rule == "pow2":
        return pow2
    if rule == "tail256":
        return max(256, pow2)
    width = BUCKET_FLOOR
    while width < steps:
        width *= 4
    return width

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


def hybrid_lane_id(lane: GpuFeatureLane, capture_identity: Mapping[str, Any],
                   bucket_rule: str = BUCKET_RULE) -> str:
    """The lane id of a row whose families were reduced batched under ``bucket_rule``."""
    return HYBRID_LANE_PREFIX + hashlib.sha256(json.dumps(
        dict(stats_lane_id=stats_lane_id(lane, capture_identity),
             **batched_receipt(lane.device, bucket_rule)),
        sort_keys=True, default=str).encode()).hexdigest()


def batched_receipt(device: torch.device | str, bucket_rule: str = BUCKET_RULE) -> dict[str, Any]:
    """What a hybrid row's receipt records of its batched reduction on ``device`` under
    ``bucket_rule``, and its lane id folds in."""
    if bucket_rule not in BUCKET_RULES:
        raise ValueError(f"bucket rule must be one of {BUCKET_RULES}, not {bucket_rule!r}")
    return dict(families=list(BATCHED_FAMILIES), source_sha256=BATCHED_SOURCE_SHA256,
                families_source_sha256=FAMILIES_SOURCE_SHA256,
                attention_source_sha256=ATTENTION_SOURCE_SHA256,
                invariant_ops_source_sha256=INVARIANT_OPS_SOURCE_SHA256,
                bucket_rule=bucket_rule, bucket_floor=BUCKET_FLOOR,
                row_backend=row_backend(device), eff_dim=EFF_DIM,
                topk_ties=TOPK_TIES, eigvalsh=eigvalsh_path(device), per_row=list(PER_ROW_EXCEPTIONS))


class Counts:
    """Each row's length along one axis of a padded batch: ``host`` (a tuple of ints, one per
    row) and ``dev`` (the same, a long tensor on the batch's device).

    A batch of rows padded to one canonical width keeps each row's real values at the front
    of the padded axis. A reduction over that axis passes :meth:`lengths`, so the padded
    positions are masked to exact zeros at fixed positions; every per-row offset (a window,
    a half, a trajectory point) is gathered to the front of its own buffer first, so a row's
    values sit at the positions, and in the tiles, they occupy in an unpadded row of its own
    length. Derived counts (:meth:`minus`, :meth:`repeat`, :meth:`derive`) are computed on
    the device, never copied to it."""

    __slots__ = ("host", "dev", "_lo", "_hi", "_cache")

    def __init__(self, host: Sequence[int], dev: Tensor) -> None:
        self.host = tuple(int(c) for c in host)
        self.dev = dev
        self._lo = min(self.host) if self.host else 0
        self._hi = max(self.host) if self.host else 0
        self._cache: dict[Any, Any] = {}

    @classmethod
    def of(cls, host: Sequence[int], device: torch.device | str) -> "Counts":
        host = tuple(int(c) for c in host)
        return cls(host, torch.as_tensor(host, dtype=torch.long, device=device))

    @property
    def lo(self) -> int:
        return self._lo

    @property
    def hi(self) -> int:
        return self._hi

    def uniform(self, width: int) -> bool:
        """Whether every row fills ``width``."""
        return self._lo == self._hi == int(width)

    def derive(self, host: Sequence[int], dev: Tensor) -> "Counts":
        return Counts(host, dev)

    def minus(self, k: int) -> "Counts":
        """Each count less ``k``, at least zero."""
        key = ("minus", int(k))
        if key not in self._cache:
            self._cache[key] = Counts([max(0, c - k) for c in self.host],
                                      (self.dev - k).clamp_min(0))
        return self._cache[key]

    def repeat(self, r: int) -> "Counts":
        """Each count repeated ``r`` times in place (rows ``[B]`` to ``[B * r]``, row-major)."""
        key = ("repeat", int(r))
        if key not in self._cache:
            self._cache[key] = Counts([c for c in self.host for _ in range(r)],
                                      self.dev.repeat_interleave(r))
        return self._cache[key]

    def memo(self, key: Any, build) -> Any:
        """A value derived from these counts, built once."""
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    def lengths(self, x: Tensor) -> Tensor | None:
        """The ``lengths`` argument of a primitive reducing ``x``'s last dimension, whose
        first dimension is the rows; ``None`` when every row fills it."""
        if self.uniform(x.shape[-1]):
            return None
        if x.shape[0] != len(self.host):
            raise ValueError(f"{len(self.host)} counts for a batch of {x.shape[0]} rows")
        return self.dev.view(-1, *([1] * (x.ndim - 2))).expand(x.shape[:-1])

    def mask(self, width: int) -> Tensor:
        """``[rows, width]``: which positions of each row are real."""
        key = ("mask", int(width))
        if key not in self._cache:
            cols = torch.arange(width, device=self.dev.device)
            self._cache[key] = cols[None, :] < self.dev[:, None]
        return self._cache[key]


def uniform_counts(rows: int, width: int, device: torch.device | str) -> Counts:
    """Counts of ``rows`` rows that each fill ``width``."""
    return Counts([width] * rows, torch.full((rows,), int(width), dtype=torch.long, device=device))


def take_steps(x: Tensor, index: Tensor | Sequence[int], dim: int = 1) -> Tensor:
    """``x`` indexed along ``dim`` by ``index``: one list for every row (a plain index), or a
    long tensor ``[rows, m]`` of each row's own positions (a gather, clamped into range)."""
    if not isinstance(index, Tensor):
        return x[(slice(None),) * dim + (list(index),)]
    idx = index.clamp(0, x.shape[dim] - 1)
    moved = x.movedim(dim, 1)
    view = idx.view(idx.shape[0], idx.shape[1], *([1] * (moved.ndim - 2)))
    out = moved.gather(1, view.expand(moved.shape[0], idx.shape[1], *moved.shape[2:]))
    return out.movedim(1, dim)


def window_steps(x: Tensor, start: Tensor, width: int, dim: int = 1) -> Tensor:
    """Each row's ``width`` positions along ``dim`` from its own ``start [rows]``, at the front."""
    index = start[:, None] + torch.arange(width, device=start.device)[None, :]
    return take_steps(x, index, dim)


def per_row_index(counts: Counts, key: Any, picks) -> tuple[Tensor | list[int], Counts]:
    """Each row's positions ``picks(n)`` for its count ``n``: one list when every row has the
    same count, else ``[rows, m]`` (padded with the last position); and how many each row has."""
    def build():
        lists = [list(picks(n)) for n in counts.host]
        sizes = [len(p) for p in lists]
        device = counts.dev.device
        if len(set(counts.host)) <= 1:
            return lists[0] if lists else [], Counts.of(sizes, device)
        width = max(sizes)
        padded = [p + [p[-1] if p else 0] * (width - len(p)) for p in lists]
        return (torch.as_tensor(padded, dtype=torch.long, device=device).reshape(len(lists), width),
                Counts.of(sizes, device))
    return counts.memo(("index", key), build)


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


@lru_cache(maxsize=None)
def _abscissa_table(width: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """``ops._unit_abscissa(m)`` for every ``m < width``: the centred abscissa ``[width,
    width]`` (row ``m`` holds the ``m`` terms, then zeros), the clamped counts and the clamped
    squared sums ``[width]``; row ``m`` is the unpadded constants' bytes."""
    xc = torch.zeros(width, width, dtype=torch.float64, device=device)
    count = torch.ones(width, dtype=torch.float64, device=device)
    denominator = torch.ones(width, dtype=torch.float64, device=device)
    for m in range(2, width):
        _, c, x, d = _unit_abscissa(m, device)
        xc[m, :m] = x
        count[m] = c
        denominator[m] = d
    return xc, count, denominator


def _slope_rows(y: Tensor, counts: Counts | None = None) -> Tensor:
    """``ops.slope`` of each row of ``y [B, m]`` with every point kept; with ``counts``, of
    each row's first ``counts`` points (zero below two)."""
    m = y.shape[-1]
    if counts is None or counts.uniform(m):
        if m < 2:
            return y.new_zeros(y.shape[:-1])
        w, count, xc, denominator = _unit_abscissa(m, y.device)
        ym = row_sum(y * w) / count
        return row_sum(xc * (y - ym[..., None]) * w) / denominator
    if counts.hi < 2:
        return y.new_zeros(y.shape[:-1])
    table, count_t, den_t = _abscissa_table(1 << m.bit_length(), y.device)
    n = counts.dev
    xc = table[n][:, :m]
    lengths = counts.lengths(y)
    ym = row_sum(y, lengths=lengths) / count_t[n]
    value = row_sum(xc * (y - ym[..., None]), lengths=lengths) / den_t[n]
    return torch.where(n < 2, value.new_zeros(()), value)


def _windowed_rows(x: Tensor, n_windows: int, counts: Counts) -> Tensor:
    """The windowed operators of each row's first ``counts`` values: ``[B, 3 * n_windows]``,
    each window gathered to the front of its own buffer; zero for a row below four."""
    n = counts.dev
    width = n // n_windows
    widths = [c // n_windows for c in counts.host]
    pieces = []
    for i in range(n_windows):
        last = i == n_windows - 1
        sizes = [c - i * w if last else w for c, w in zip(counts.host, widths, strict=True)]
        size_dev = n - i * width if last else width
        segment = window_steps(x, width * i, max(1, max(sizes)))
        seg_counts = counts.derive(sizes, size_dev)
        lengths = seg_counts.lengths(segment)
        pieces.extend((row_mean(segment, lengths=lengths), row_std(segment, lengths=lengths),
                       _slope_rows(segment, seg_counts)))
    windowed = torch.stack(pieces, dim=1)
    return torch.where((n < 4)[:, None], windowed.new_zeros(()), windowed)


def _spectrum_rows(x: Tensor) -> Tensor:
    """The short-time spectrum of each row of ``x [B, m]`` over all ``m`` values, frames
    ``[B, F, 64]`` reduced by a direct DFT; returns the power per bin ``[B, 33, F]`` and the
    STFT constants."""
    n = x.shape[1]
    if n < 64:
        x = torch.nn.functional.pad(x, (0, 64 - n))
    extended = torch.nn.functional.pad(x, (32, 32 + (-x.shape[1]) % 32))
    window, window_sum, freq, bands = _stft_constants(x.device)
    frames = (extended.unfold(1, 64, 32) * window).unsqueeze(2)  # [B, F, 1, 64]
    cos_basis, sin_basis = _dft_basis(x.device)
    shape = (x.shape[0], frames.shape[1], 33, 64)
    re = row_dot(frames.expand(shape), cos_basis.expand(shape)) / window_sum
    im = row_dot(frames.expand(shape), sin_basis.expand(shape)) / window_sum
    return (re.square() + im.square()).transpose(1, 2).contiguous(), freq, bands


def _spectral_summary(power: Tensor, freq: Tensor, bands) -> Tensor:
    total = row_sum(power)
    p = power / total.clamp_min(1e-30)[:, None]
    centroid = row_sum(p * freq)
    bandwidth = row_sum(p * (freq - centroid[:, None]).square()).sqrt()
    band_mass = [row_sum(p[:, int(b[0]) : int(b[-1]) + 1].contiguous()) for b in bands]
    spectral = torch.stack((centroid, bandwidth, *band_mass), dim=1)
    return torch.where((total < 1e-12)[:, None], spectral.new_zeros(()), spectral)


def _temporal_rows(x: Tensor, n_windows: int, include_stft: bool,
                   counts: Counts | None = None) -> Tensor:
    """``ops.temporal`` of each row of ``x [B, m]``: ``[B, 3 * n_windows (+ 5)]`` float32.

    With ``counts``, of each row's first ``counts`` values: every window is gathered to the
    front of its own buffer, the spectrum's values past a row's count are exact zeros (as the
    unpadded row's zero padding is) and its frame mean runs over the row's own frames, and
    the placeholders (windows below four values, the spectrum below eight) are per row."""
    x = x.float().double()
    batch, n = x.shape
    if counts is not None and not counts.uniform(n):
        windowed = _windowed_rows(x, n_windows, counts)
        if not include_stft:
            return windowed.float()
        live = counts.dev
        x = torch.where(counts.mask(n), x, x.new_zeros(()))
        power, freq, bands = _spectrum_rows(x)
        frames = (live.clamp_min(64) + 31) // 32 + 1
        power = row_mean(power, lengths=frames[:, None].expand(power.shape[:2]))  # [B, 33]
        spectral = _spectral_summary(power, freq, bands)
        spectral = torch.where((live < 8)[:, None], spectral.new_zeros(()), spectral)
        return torch.cat((windowed, spectral), dim=1).float()
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
        power, freq, bands = _spectrum_rows(x)
        spectral = _spectral_summary(row_mean(power), freq, bands)  # [B, 33] power
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
                         n_windows: int, include_stft: bool, counts: Counts) -> None:
    """``FamilyReducer.residual_trajectory`` for ``corrected [B, steps, hidden]``, each row
    over its first ``counts`` steps."""
    from anamnesis.extraction.feature_families.residual_stream import _trajectory_feature_names

    batch, steps = corrected.shape[:2]
    if counts.hi < 3:
        names = _trajectory_feature_names(layer, n_windows, include_stft)
        cols.block(names, corrected.new_zeros(batch, len(names), dtype=torch.float64))
        return
    if counts.lo < 3:
        raise ExternalCaptureError("rows below three steps share a batch with longer rows")
    prefix = f"res_traj_L{layer}"
    x = corrected.double()
    vel = x[:, 1:] - x[:, :-1]
    vn = row_norm(vel)
    angles = _cos(vel[:, :-1].float(), vel[:, 1:].float()).clamp(-1, 1).acos()
    accel = row_norm(vel[:, 1:] - vel[:, :-1])
    one, two = counts.minus(1), counts.minus(2)
    l1, l2 = one.lengths(vn), two.lengths(angles)
    last = x[:, -1] if counts.uniform(steps) else take_steps(x, (counts.dev - 1)[:, None])[:, 0]
    names = [prefix + "_velocity_norm_mean", prefix + "_velocity_norm_std",
             prefix + "_direction_change_mean", prefix + "_direction_change_std",
             prefix + "_directness",
             prefix + "_acceleration_norm_mean", prefix + "_acceleration_norm_std"]
    values = [row_mean(vn, lengths=l1), row_std(vn, lengths=l1), row_mean(angles, lengths=l2),
              row_std(angles, lengths=l2),
              row_norm(last - x[:, 0]) / row_sum(vn, lengths=l1).clamp_min(1e-12),
              row_mean(accel, lengths=l2), row_std(accel, lengths=l2)]
    names += _operator_names(prefix + "_velocity_norm", n_windows, include_stft)
    names += _operator_names(prefix + "_direction_change", n_windows, include_stft)
    block = torch.cat([torch.stack(values, dim=1),
                       _temporal_rows(vn, n_windows, include_stft, one).double(),
                       _temporal_rows(angles, n_windows, include_stft, two).double()], dim=1)
    block = torch.where((raw_first_norm < 1e-6)[:, None], block.new_zeros(()), block)
    cols.block(names, block)


def spectral_smoothness(x: Tensor, index: Tensor | Sequence[int], k: Counts, lap: Tensor) -> Tensor:
    """The smoothness of each row's step norms over its spectral graph: ``x [B, steps, width]``,
    the spectral steps ``index`` (one list, or ``[B, K]``), how many each row has, and its
    normalised Laplacian ``lap [B, K, K]``, zero-padded past the row's own size."""
    signal = row_norm(take_steps(x, index).double())  # [B, K]
    if not k.uniform(signal.shape[1]):
        signal = torch.where(k.mask(signal.shape[1]), signal, signal.new_zeros(()))
    shape = lap.shape
    lap_t = lap.transpose(1, 2).contiguous()
    lap_signal = row_dot(lap_t, signal[:, None, :].expand(shape).contiguous(),
                         lengths=k.lengths(lap_t))
    lengths = k.lengths(signal)
    return (row_dot(lap_signal, signal, lengths=lengths)
            / row_sumsq(signal, lengths=lengths).clamp_min(1e-12))


def block_output_features(lane: GpuFeatureLane, hidden: Tensor, starts: Sequence[int],
                          laplacians: Mapping[int, tuple],
                          counts: Counts | None = None) -> BlockOutputs:
    """The block-output family of ``B`` rows.

    ``hidden [B, steps, layers, width]`` is each row's block outputs in the engine's layout
    (the signed row's ``depth``); ``starts[b]`` is row ``b``'s prompt length, which selects
    its positional means; ``counts`` is each row's own step count (default: every row fills
    ``steps``), the rest of its steps being padding; ``laplacians[layer]`` is, at each sampled
    layer, the spectral steps (one list, or each row's ``[B, K]``), ``[B, K, K]``, each row's
    normalised Laplacian as its attention reduction left it, and how many spectral steps each
    row has; the counts may be left out when every row has the same spectral steps.

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
    device = lane.device
    counts = counts if counts is not None else uniform_counts(batch, steps, device)
    if len(counts.host) != batch or counts.hi > steps or counts.lo < 1:
        raise ExternalCaptureError(f"step counts {counts.host} for {batch} rows of {steps} steps")
    pm = lane.pm
    if pm.ndim != 3 or pm.shape[0] < layers + 1 or pm.shape[2] != width or any(
            pm.shape[1] < int(s) + n for s, n in zip(starts, counts.host, strict=True)):
        raise ExternalCaptureError("the lane's positional means do not cover this capture")
    if not hidden.is_floating_point():
        raise ExternalCaptureError("hidden must be floating point")
    config, families = lane.config, lane.families
    sampled = set(config.sampled_layers)
    missing = sorted(sampled - set(laplacians))
    if missing:
        raise ExternalCaptureError(f"no spectral Laplacian for sampled layers {missing}")
    n_windows, include_stft = families.temporal_n_windows, families.enable_stft
    traj, _ = per_row_index(counts, ("traj", config.trajectory_points),
                            lambda n: trajectory_indices(n, config.trajectory_points))
    pca_index, _ = per_row_index(counts, ("traj", config.pca_temporal_samples),
                                 lambda n: trajectory_indices(n, config.pca_temporal_samples))
    k = config.pca_components

    cols = _Columns()
    per_layer = max(1, batch * steps * width * 8 * 4)
    chunk = max(1, min(layers, WORKING_BYTES // per_layer))
    table = device_positional_means(pm, device)
    start_t = torch.as_tensor([int(s) for s in starts], dtype=torch.long, device=device)
    step_t = torch.arange(steps, device=device)[None, :]
    if not counts.uniform(steps):  # a padded step reads its row's last real position
        step_t = torch.minimum(step_t, (counts.dev - 1)[:, None])
    positions = start_t[:, None] + step_t  # [B, steps]
    previous: Tensor | None = None
    for first in range(0, layers, chunk):
        last = min(layers, first + chunk)
        h = hidden[:, :, first:last].to(device).float()
        positional = table[1 + first:1 + last][:, positions].permute(1, 2, 0, 3)  # [B, steps, chunk, width]
        corrected = h - positional  # [B, steps, chunk, width]
        if corrected.dtype not in invariant_ops.SUPPORTED_DTYPES:
            raise ExternalCaptureError(f"block outputs corrected to {corrected.dtype}")
        norms = row_norm(corrected).float().transpose(1, 2).contiguous()  # [B, chunk, steps]
        lengths = counts.lengths(norms)
        mean = row_mean(norms, lengths=lengths).double()
        dev = row_std(norms, lengths=lengths).double()
        traj_values = take_steps(norms, traj, dim=2).double()
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
            lengths = counts.lengths(dn)
            dn_mean = row_mean(dn, lengths=lengths).double()
            dn_std = row_std(dn, lengths=lengths).double()
            dc_mean = row_mean(dc, lengths=lengths).double()
        for j, layer in enumerate(range(first, last)):
            cols.put(f"activation_norm_mean_L{layer}", mean[:, j])
            cols.put(f"activation_norm_std_L{layer}", dev[:, j])
            for i in range(traj_values.shape[2]):
                cols.put(f"activation_norm_traj{i}_L{layer}", traj_values[:, j, i])
            if layer >= delta_first:
                d = layer - delta_first
                cols.put(f"delta_norm_mean_L{layer - 1}", dn_mean[:, d])
                cols.put(f"delta_norm_std_L{layer - 1}", dn_std[:, d])
                cols.put(f"delta_cosine_mean_L{layer - 1}", dc_mean[:, d])
            x = corrected[:, :, j]  # [B, steps, width]
            if layer in sampled:
                entry = laplacians[layer]
                if len(entry) == 2:  # (steps, lap): every row has the same spectral steps
                    spectral_steps, lap = entry
                    k_counts = uniform_counts(batch, len(spectral_steps), device)
                else:
                    spectral_steps, lap, k_counts = entry
                cols.put(f"spectral_smoothness_L{layer}",
                         spectral_smoothness(x, spectral_steps, k_counts, lap.to(device).double()))
            if layer in families.trajectory_layers:
                _residual_trajectory(cols, layer, x, row_norm(h[:, 0, j]), n_windows, include_stft,
                                     counts)
            if layer in config.pca_layers:
                components = lane.pca[layer] if isinstance(lane.pca, dict) else lane.pca
                pca_mean = lane.pca_mean[layer] if isinstance(lane.pca_mean, dict) else lane.pca_mean
                centred = take_steps(x, pca_index).double() - pca_mean  # [B, T, width]
                samples = centred.shape[1]
                shape = (batch, samples, k, width)
                projections = row_dot(centred[:, :, None, :].expand(shape),
                                      components[:k][None, None].expand(shape))
                for ti in range(samples):
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
                retained_attention_summary_bytes: int | Sequence[int] = 0,
                bucket_rule: str = BUCKET_RULE,
                bucket_width: int | None = None) -> list[GpuFeatureResult | ExternalCaptureError]:
    """Each row's ``GpuFeatureResult``, assembled for the whole batch at once.

    ``values [B, len(names)]`` holds every lane feature (float64, each the value the per-row
    reduction would emit before its float32 cast); they are put in the lane's schema order by
    one gather and cast once, and the routing family's float32 ``routing_values [B, ...]``
    follows them. ``num_layers`` is the capture's layer count, every one of which the attention
    reduction read. ``mean_logprob [B]`` and ``knnlm [B, hidden]`` (or ``None``) are per row.
    The receipt's lane id is :func:`hybrid_lane_id` under ``bucket_rule``, and the receipt
    names the batch's canonical step width ``bucket_width``; the fields common to the batch
    are computed once. ``retained_attention_summary_bytes`` is one count for every row, or
    one per row. A row with a nonfinite feature gets the :class:`ExternalCaptureError` that
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
        lane_id=hybrid_lane_id(lane, capture_identity, bucket_rule),
        stats_lane_id=stats_lane_id(lane, capture_identity),
        external_lane_id=external_lane_id(lane, capture_identity),
        fast_lane_id=lane.lane_id,
        lane_source_sha256=dict(lane.source_sha256),
        readout_source_sha256=dict(READOUT_SOURCE_SHA256),
        stats_source_sha256=STATS_SOURCE_SHA256,
        batched=batched_receipt(lane.device, bucket_rule),
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
        bucket_width=int(bucket_width) if bucket_width is not None else None,
        routing_family=routed,
    )
    retained = ([int(retained_attention_summary_bytes)] * batch
                if isinstance(retained_attention_summary_bytes, int)
                else [int(r) for r in retained_attention_summary_bytes])
    if len(retained) != batch:
        raise ExternalCaptureError(f"{len(retained)} retained-summary sizes for {batch} rows")
    results: list[GpuFeatureResult | ExternalCaptureError] = []
    for b, span in enumerate(spans):
        if not finite[b]:
            results.append(ExternalCaptureError("the reduction produced a nonfinite feature"))
            continue
        receipt = dict(common, replay_id=str(uuid.uuid4()), span_start=span.start,
                       span_end=span.end, retained_attention_summary_bytes=retained[b],
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
    "BUCKET_FLOOR",
    "BUCKET_RULE",
    "BUCKET_RULES",
    "BlockOutputs",
    "Counts",
    "EFF_DIM",
    "FAMILIES_SOURCE_SHA256",
    "HYBRID_LANE_PREFIX",
    "INVARIANT_OPS_SOURCE_SHA256",
    "PER_ROW_EXCEPTIONS",
    "RowSpan",
    "TOPK_TIES",
    "batched_receipt",
    "block_output_features",
    "bucket_width",
    "eigvalsh_path",
    "finish_rows",
    "hybrid_lane_id",
    "per_row_index",
    "row_backend",
    "spectral_smoothness",
    "take_steps",
    "uniform_counts",
    "window_steps",
)
