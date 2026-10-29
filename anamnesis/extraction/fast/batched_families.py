"""The key, value, query, gate, per-head, output and routing families of a statistics row,
reduced for many rows at once, batchmate-invariant.

After its block outputs, :meth:`ExternalStatsSpan.finish
<anamnesis.extraction.fast.external_stats.ExternalStatsSpan.finish>` reduces one row at a
time: at each sampled layer the key, per-head, value, query and gate families
(:class:`~anamnesis.extraction.fast.families.FamilyReducer` and
:meth:`~anamnesis.extraction.fast.attention.AttentionReducer.head_features`), then the
cross-layer features of :meth:`FamilyReducer.finish` (effective dimensions, epochs, gate
agreement, spread diversity, key and value CKA), the output family from ``logit_stats``
and the routing family. :func:`kvqg_features`, :func:`outputs_features` and
:func:`routing_features` compute the same features for ``B`` rows of one step count, in
one pass over ``[B, steps, ...]`` tensors.

**Batchmate invariance.** As in :mod:`anamnesis.extraction.fast.batched_stats`: every
reduction over a row's own steps, heads, sources or hidden dimension is a primitive of
:mod:`anamnesis.extraction.fast.invariant_ops`; everything else is elementwise, an index
into a row's own tensors, an order-free maximum or minimum, or a constant that depends
only on the step count and the lane. The running means of the novelty and cache-alignment
series are an inclusive scan by doubling, which is elementwise at each of its
``ceil(log2(steps))`` stages. Gram matrices are :func:`row_sum` over elementwise products,
never matrix products. A row reduced alone and the same row inside any batch therefore
have the same bytes.

**Effective dimension by Frobenius norms.** The per-row lane takes each matrix's singular
values; here every effective dimension is computed from two Frobenius norms, ``sum s**2 =
|A|_F**2`` and ``sum s**4 = |G|_F**2`` with ``G`` the smaller Gram matrix
(:data:`~anamnesis.extraction.fast.features.EFF_DIM_METHODS` ``norms``). The two agree to
rounding. The key and value CKA uses the same identity: ``|C_b^T C_a|_F**2`` is the
Frobenius inner product of the two layers' ``steps x steps`` Gram matrices.

**The gate's top-k, ties broken by the lowest index.** The per-row lane's top-k Jaccard
(:func:`~anamnesis.extraction.fast.families.gate_topk_overlap`) takes each step's top-k
columns from NumPy's ``argsort``, whose order among tied values is an implementation
detail; bf16 gates often tie at the k-th value. Here the top-k is defined: the ``k``
columns of largest ``|SiLU|``, ties at the k-th value broken by the lowest column index,
as a stable sort descending by value would take them (:data:`TOPK_TIES`). That set is a
function of the step's values alone: the k-th largest value, which columns exceed it, and
an exact integer count of the tied columns before each. It runs on the device, and the
Jaccard of adjacent steps is an exact integer ratio averaged by :func:`row_mean`. Where a
step has a tie at the k-th value that NumPy broke otherwise, the feature differs from the
per-row lane's; elsewhere it agrees to the rounding of the mean.

**Agreement with the per-row path.** The arithmetic is the per-row reducers', operator for
operator, with each reduction's order replaced by the primitives' fixed order, the
cumulative sum by the scan, and the matrix products by row sums. The features agree with
:class:`ExternalStatsSpan`'s to rounding, not to the byte; the effective dimensions agree
to the rounding of a different algorithm. The names and the placeholders for short spans
(per family, below two or four steps; epochs below ``epoch_window_size +
epoch_stride`` steps; the surprise boundary count below ``surprise_window``) are the
per-row reducers'.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch
from torch import Tensor

from anamnesis.extraction.fast.batched_stats import (
    TOPK_TIES,
    WORKING_BYTES,
    Counts,
    _Columns,
    _cos,
    _operator_names,
    _rowcos,
    _temporal_rows,
    per_row_index,
    take_steps,
    uniform_counts,
    window_steps,
)
from anamnesis.extraction.fast.external import ExternalCaptureError
from anamnesis.extraction.fast.features import GpuFeatureLane
from anamnesis.extraction.fast.invariant_ops import (
    row_dot,
    row_mean,
    row_norm,
    row_std,
    row_sum,
    row_sumsq,
)
from anamnesis.extraction.fast.ops import trajectory_indices
from anamnesis.extraction.feature_families.attn_res import EPS, QNAMES, SUMM, committed_pairs

def gate_topk_overlap_rows(pre_silu: Tensor, counts: Counts | None = None) -> Tensor:
    """The gate's top-k Jaccard of each row of ``pre_silu [B, steps, width]``: ``[B]``
    float64, with :data:`TOPK_TIES`; with ``counts``, over each row's first ``counts`` steps.

    The SiLU is the per-row lane's, ``a * (1 / (1 + exp(-clip(a, -88, 88))))`` in float64,
    and ``k = min(100, width // 10)``; ``k == 0`` keeps every column, as the per-row lane's
    slice does. Adjacent steps' sets share ``s`` columns of a union of ``2k - s``.
    """
    batch, steps, width = pre_silu.shape
    if steps < 2:
        return torch.zeros(batch, dtype=torch.float64, device=pre_silu.device)
    a = pre_silu.double()
    a = (a * (1 / (1 + torch.exp(-a.clamp(-88, 88))))).abs()
    k = min(100, width // 10)
    if k == 0:
        member = torch.ones_like(a, dtype=torch.bool)
        k = width
    else:
        kth = a.topk(k, dim=-1, sorted=False).values.amin(dim=-1, keepdim=True)
        above = a > kth
        tied = a == kth
        room = k - above.sum(dim=-1, keepdim=True)  # tied columns the set still takes
        member = above | (tied & (tied.to(torch.int32).cumsum(dim=-1) <= room))
    del a
    shared = (member[:, :-1] & member[:, 1:]).sum(dim=-1)  # [B, steps - 1], exact
    union = 2 * k - shared
    overlaps = torch.where(union > 0, shared.double() / union.clamp_min(1).double(),
                           torch.zeros((), dtype=torch.float64, device=shared.device))
    return row_mean(overlaps, lengths=counts.minus(1).lengths(overlaps) if counts else None)


@dataclass(frozen=True)
class BatchedFeatures:
    """Named features of ``B`` rows: ``names`` and ``values [B, len(names)]``."""

    names: tuple[str, ...]
    values: Tensor

    def row(self, index: int) -> Tensor:
        return self.values[index]


def _stack(cols: _Columns, batch: int, device) -> BatchedFeatures:
    values = (torch.stack(cols.columns, dim=1) if cols.columns
              else torch.zeros(batch, 0, dtype=torch.float64, device=device))
    return BatchedFeatures(tuple(cols.names), values)


def _zeros(cols: _Columns, names: Sequence[str], batch: int, device) -> None:
    cols.block(list(names), torch.zeros(batch, len(names), dtype=torch.float64, device=device))


def _time_mean(x: Tensor, counts: Counts | None = None) -> Tensor:
    """The mean over dimension 1 (a row's steps) of ``x [B, steps, ...]``, over each row's
    first ``counts`` steps."""
    moved = x.movedim(1, -1).contiguous()
    return row_mean(moved, lengths=counts.lengths(moved) if counts is not None else None)


def _head_mean(x: Tensor) -> Tensor:
    """``x [..., heads, dim]`` averaged over its heads in float32: ``[..., dim]``."""
    return row_mean(x.float().transpose(-1, -2).contiguous())


def _running_mean(x: Tensor) -> Tensor:
    """The running mean along dimension 1 of ``x [B, steps, dim]``: an inclusive scan by
    doubling, each stage elementwise, then a division by the count. A position's value reads
    only the positions before it, so padding past a row's steps leaves its steps unchanged."""
    steps = x.shape[1]
    total, d = x, 1
    while d < steps:
        total = torch.cat([total[:, :d], total[:, d:] + total[:, :-d]], dim=1)
        d *= 2
    count = torch.arange(1, steps + 1, dtype=x.dtype, device=x.device)
    return total / count[:, None]


def _novelty(x: Tensor) -> Tensor:
    return 1 - _rowcos(x[:, 1:], _running_mean(x)[:, :-1])


def _drift(x: Tensor) -> Tensor:
    return 1 - _rowcos(x[:, :-1], x[:, 1:])


def _halves(x: Tensor, counts: Counts) -> tuple[Tensor, Counts, Tensor, Counts]:
    """Each row's first ``n // 2`` steps and the rest of its ``n``, each at the front of its own
    buffer along dimension 1, with their counts."""
    steps = x.shape[1]
    if counts.uniform(steps):
        mid = steps // 2
        return (x[:, :mid], counts.derive([mid] * len(counts.host), counts.dev // 2),
                x[:, mid:], counts.derive([steps - mid] * len(counts.host), counts.dev - counts.dev // 2))
    mid = counts.dev // 2
    first = counts.derive([n // 2 for n in counts.host], mid)
    second = counts.derive([n - n // 2 for n in counts.host], counts.dev - mid)
    return x, first, window_steps(x, mid, max(1, second.hi)), second


def _half_drift(x: Tensor, counts: Counts) -> Tensor:
    a, ca, b, cb = _halves(x, counts)
    return 1 - _cos(_time_mean(a, ca).float(), _time_mean(b, cb).float())


def _spread(x: Tensor, center: Tensor, counts: Counts) -> Tensor:
    """The mean over steps of one minus each step's cosine to ``center [B, dim]``."""
    series = 1 - _rowcos(x, center[:, None].expand_as(x))
    return row_mean(series, lengths=counts.lengths(series))


def _corr(a: Tensor, b: Tensor, min_n: int = 3, counts: Counts | None = None) -> Tensor:
    """``ops.corr`` of each row of ``a`` and ``b`` (``[B, n]``); with ``counts``, of each
    row's first ``counts`` values."""
    n = min(a.shape[1], b.shape[1])
    if counts is None or counts.uniform(n):
        if n < min_n:
            return a.new_zeros(a.shape[0], dtype=torch.float64)
        a, b = a[:, :n].double(), b[:, :n].double()
        ac, bc = a - row_mean(a)[:, None], b - row_mean(b)[:, None]
        value = row_dot(ac, bc) / (row_norm(ac) * row_norm(bc)).clamp_min(1e-30)
        flat = (row_std(a) < 1e-12) | (row_std(b) < 1e-12)
        return torch.where(flat, value.new_zeros(()), value.clamp(-1, 1))
    a, b = a[:, :n].double(), b[:, :n].double()
    lengths = counts.lengths(a)
    ac, bc = a - row_mean(a, lengths=lengths)[:, None], b - row_mean(b, lengths=lengths)[:, None]
    value = row_dot(ac, bc, lengths=lengths) / (
        row_norm(ac, lengths=lengths) * row_norm(bc, lengths=lengths)).clamp_min(1e-30)
    flat = ((row_std(a, lengths=lengths) < 1e-12) | (row_std(b, lengths=lengths) < 1e-12)
            | (counts.dev < min_n))
    return torch.where(flat, value.new_zeros(()), value.clamp(-1, 1))


def _gram(x: Tensor, counts: Counts | None = None) -> Tensor:
    """``x x^T`` of each row of ``x [B, m, n]``: ``[B, m, m]``, each entry a :func:`row_sum`
    over an elementwise product (over each row's first ``counts`` of ``n``), chunked over the
    first index to bound the working set."""
    batch, m, n = x.shape
    chunk = max(1, min(m, WORKING_BYTES // max(1, batch * m * n * x.element_size())))
    parts = []
    for first in range(0, m, chunk):
        last = min(m, first + chunk)
        product = x[:, first:last, None, :] * x[:, None, :, :]
        parts.append(row_sum(product, lengths=counts.lengths(product) if counts else None))
    return torch.cat(parts, dim=1)


def _square_counts(counts: Counts) -> Counts:
    return counts.memo("square", lambda: counts.derive([c * c for c in counts.host],
                                                       counts.dev * counts.dev))


def _front_square(gram: Tensor, counts: Counts) -> Tensor:
    """Each row's leading ``n x n`` block of ``gram [B, m, m]``, row-major at the front of a
    flat ``[B, m * m]``: the layout of the unpadded row's flattened ``n x n`` matrix."""
    batch, m = gram.shape[:2]
    if counts.uniform(m):
        return gram.reshape(batch, -1)

    def build() -> Tensor:
        p = torch.arange(m * m, device=counts.dev.device)[None, :]
        n = counts.dev[:, None].clamp_min(1)
        return ((p // n) * m + p % n).clamp_max(m * m - 1)
    index = counts.memo(("front-square", m), build)
    return gram.reshape(batch, -1).gather(1, index)


def _eff_dims(x: Tensor, counts: Counts | None = None) -> Tensor:
    """The effective dimension of each row's ``[m, n]`` matrix from its Frobenius norms,
    with ``features.EFF_DIM_METHODS``' ``norms`` guards: ``|A|_F**4 / |G|_F**2 / min(m, n)``,
    ``G`` the smaller Gram matrix; zero where ``|A|_F**2 <= 1e-12``. With ``counts``, each
    row's matrix is its first ``counts`` of ``m`` rows, and its smaller Gram matrix is chosen
    by its own ``m``."""
    batch, m, n = x.shape
    if counts is None or counts.uniform(m):
        total = row_sumsq(x.reshape(batch, m * n))
        gram = _gram(x.transpose(1, 2).contiguous() if n <= m else x)
        fourth = row_sumsq(gram.reshape(batch, -1))
        return torch.where(total > 1e-12, total.square() / fourth.clamp_min(1e-30) / min(m, n),
                           total.new_zeros(()))
    flat = x.reshape(batch, m * n)
    total = row_sumsq(flat, lengths=counts.derive([c * n for c in counts.host],
                                                  counts.dev * n).lengths(flat))
    fourth = total.new_zeros(batch)
    for columns, (index, sub) in zip((True, False), _gram_split(counts, n), strict=True):
        if index is None:
            continue
        part = x.index_select(0, index)
        if columns:  # the Gram matrix over the n columns
            gram = _gram(part.transpose(1, 2).contiguous(), sub)
            fourth[index] = row_sumsq(gram.reshape(len(sub.host), -1))
        else:  # the Gram matrix over the row's own steps
            front = _front_square(_gram(part), sub)
            fourth[index] = row_sumsq(front, lengths=_square_counts(sub).lengths(front))
    smaller = torch.minimum(counts.dev, torch.full_like(counts.dev, n)).double()
    return torch.where(total > 1e-12, total.square() / fourth.clamp_min(1e-30) / smaller,
                       total.new_zeros(()))


def _gram_split(counts: Counts, n: int):
    """The rows whose effective dimension takes the Gram matrix over ``n`` columns (``n <=``
    their count) and the rest, each as ``(index or None, their counts)``."""
    def build():
        out = []
        for members in ([b for b, c in enumerate(counts.host) if n <= c],
                        [b for b, c in enumerate(counts.host) if n > c]):
            if not members:
                out.append((None, None))
                continue
            index = torch.as_tensor(members, dtype=torch.long, device=counts.dev.device)
            out.append((index, counts.derive([counts.host[b] for b in members],
                                             counts.dev.index_select(0, index))))
        return tuple(out)
    return counts.memo(("gram-split", int(n)), build)


def _moments(cols: _Columns, prefix: str, x: Tensor, counts: Counts | None = None) -> None:
    lengths = counts.lengths(x) if counts is not None else None
    cols.put(prefix + "_mean", row_mean(x, lengths=lengths))
    cols.put(prefix + "_std", row_std(x, lengths=lengths))


def _operators(cols: _Columns, prefix: str, x: Tensor, n_windows: int, include_stft: bool,
               counts: Counts | None = None) -> None:
    cols.block(_operator_names(prefix, n_windows, include_stft),
               _temporal_rows(x, n_windows, include_stft, counts))


def _surface(name: str, x: Tensor, ndim: int, batch: int, steps: int, sampled: int) -> None:
    if x.ndim != ndim or tuple(x.shape[:3]) != (batch, steps, sampled):
        raise ExternalCaptureError(
            f"{name} must be [{batch}, {steps}, {sampled}, ...], got {tuple(x.shape)}")
    if not x.is_floating_point():
        raise ExternalCaptureError(f"{name} must be floating point")


def kvqg_features(lane: GpuFeatureLane, keys: Tensor, values: Tensor, queries: Tensor,
                  gates: Tensor, per_head: Mapping[int, tuple[Tensor, Tensor]],
                  counts: Counts | None = None) -> BatchedFeatures:
    """The key, per-head, value, query and gate families of ``B`` rows, with
    :meth:`FamilyReducer.finish`'s cross-layer features.

    ``keys`` and ``values`` are ``[B, steps, sampled, kv heads, head dim]``, ``queries``
    ``[B, steps, sampled, query heads, head dim]`` and ``gates [B, steps, sampled, gate
    width]`` (pre-SiLU), at the lane's sampled layers in order; ``counts`` is each row's own
    step count (default: every row fills ``steps``), the rest of its steps being padding.
    ``per_head[layer]`` is the attention reduction's per-head entropy and sink, ``[B, steps,
    query heads]`` each, as the per-row attention reducer retains them.

    Raises
    ------
    ExternalCaptureError
        When the surfaces' shapes disagree with each other or the lane, or a sampled
        layer's per-head summaries are missing.
    """
    config, families, device = lane.config, lane.families, lane.device
    sampled = list(config.sampled_layers)
    batch, steps = keys.shape[:2]
    for name, x, ndim in (("keys", keys, 5), ("values", values, 5), ("queries", queries, 5),
                          ("gates", gates, 4)):
        _surface(name, x, ndim, batch, steps, len(sampled))
    missing = [layer for layer in sampled if layer not in per_head]
    if missing:
        raise ExternalCaptureError(f"no per-head summaries for sampled layers {missing}")
    counts = counts if counts is not None else uniform_counts(batch, steps, device)
    if len(counts.host) != batch or counts.hi > steps or counts.lo < 1:
        raise ExternalCaptureError(f"step counts {counts.host} for {batch} rows of {steps} steps")
    if counts.lo < 4 <= counts.hi or counts.lo < 2 <= counts.hi:
        raise ExternalCaptureError("rows below four steps share a batch with longer rows")
    short = counts.hi < 2  # placeholders below two steps (and, below four, per head)
    n_windows, include_stft = families.temporal_n_windows, families.enable_stft
    one = counts.minus(1)
    cols = _Columns()

    key_x: dict[int, Tensor] = {}
    value_x: dict[int, Tensor] = {}
    eff_dims: list[tuple[str, Tensor]] = []
    epochs: list[tuple[Tensor, Tensor, Tensor]] = []
    value_spreads: list[Tensor] = []
    query_spreads: list[Tensor] = []
    gate_means: dict[int, Tensor] = {}
    gate_sparsity: dict[int, Tensor] = {}
    window, stride = config.epoch_window_size, config.epoch_stride
    epoch_rows = counts.dev >= window + stride  # rows with epochs
    any_epochs = counts.hi >= window + stride
    novelty_traj, _ = per_row_index(one, ("traj", 5), trajectory_indices)
    for j, layer in enumerate(sampled):
        k_raw = keys[:, :, j].to(device).float()
        # Keys: FamilyReducer.key.
        matrix32 = _head_mean(k_raw)  # [B, steps, dim]
        x = matrix32.double()
        key_x[layer] = x
        if short:
            _zeros(cols, [f"kv_{name}_L{layer}" for name in (
                "key_spread", "key_eff_dim", "key_drift", "key_novelty_mean",
                "key_novelty_std", *[f"key_novelty_traj{i}" for i in range(5)])],
                batch, device)
        else:
            cols.put(f"kv_key_spread_L{layer}",
                     _spread(x, _time_mean(matrix32, counts).double(), counts))
            eff_dims.append((f"kv_key_eff_dim_L{layer}", x))
            cols.put(f"kv_key_drift_L{layer}", _half_drift(x, counts))
            nov = _novelty(x).float()
            lengths = one.lengths(nov)
            cols.put(f"kv_key_novelty_mean_L{layer}", row_mean(nov, lengths=lengths))
            cols.put(f"kv_key_novelty_std_L{layer}", row_std(nov, lengths=lengths))
            picked = take_steps(nov, novelty_traj)
            for i in range(picked.shape[1]):
                cols.put(f"kv_key_novelty_traj{i}_L{layer}", picked[:, i])
            if any_epochs:
                starts = range(0, steps - window + 1, stride)
                windows = torch.stack([matrix32[:, s:s + window] for s in starts], dim=1)
                centers = row_mean(windows.transpose(2, 3).contiguous())  # [B, E, dim]
                boundaries = (1 - _cos(centers[:, :-1], centers[:, 1:])).float()
                if counts.uniform(steps):
                    spread_b = row_std(boundaries)
                    threshold = (row_mean(boundaries).double()
                                 + 1.5 * spread_b.double().clamp_min(1e-6))
                    count = row_mean((boundaries > threshold.float()[:, None]).double())
                    peak = boundaries.amax(dim=1)
                else:  # each row's own epochs: (n - window) // stride + 1 of them
                    edges = _epoch_counts(counts, window, stride)
                    lengths = edges.lengths(boundaries)
                    spread_b = row_std(boundaries, lengths=lengths)
                    threshold = (row_mean(boundaries, lengths=lengths).double()
                                 + 1.5 * spread_b.double().clamp_min(1e-6))
                    count = row_mean((boundaries > threshold.float()[:, None]).double(),
                                     lengths=lengths)
                    inside = edges.mask(boundaries.shape[1])
                    peak = torch.where(inside, boundaries,
                                       boundaries.new_full((), -math.inf)).amax(dim=1)
                epochs.append((count, peak.double(), spread_b.double()))

        # Per-head keys: AttentionReducer.head_features.
        prefix = f"ph_L{layer}"
        ent, sink = per_head[layer]
        if counts.hi < 4:
            from anamnesis.extraction.feature_families.per_head import _per_head_names

            _zeros(cols, _per_head_names(layer), batch, device)
        else:
            ent, sink = ent.to(device).double(), sink.to(device).double()
            means = _time_mean(ent, counts)  # [B, heads]
            cols.put(prefix + "_head_entropy_mean", row_mean(means))
            cols.put(prefix + "_head_entropy_std", row_std(means))
            cols.put(prefix + "_head_entropy_min", means.amin(dim=1))
            cols.put(prefix + "_head_entropy_max", means.amax(dim=1))
            a, ca, b, cb = _halves(ent, counts)
            cols.put(prefix + "_head_role_stability",
                     _corr(_time_mean(a, ca), _time_mean(b, cb), min_n=2))
            cols.put(prefix + "_sink_head_std", row_std(_time_mean(sink, counts)))
            k = k_raw.double()  # [B, steps, kv heads, dim]
            center = _time_mean(k, counts)  # [B, kv heads, dim]
            sims = row_dot(k, center[:, None].expand_as(k)) / (
                row_norm(k) * row_norm(center)[:, None]).clamp_min(1e-12)
            spread_h = _time_mean(1 - sims, counts)  # [B, kv heads]
            cols.put(prefix + "_kv_key_spread_head_mean", row_mean(spread_h))
            cols.put(prefix + "_kv_key_spread_head_std", row_std(spread_h))

        # Values: FamilyReducer.value.
        v_raw = values[:, :, j].to(device).float()
        x = _head_mean(v_raw).double()
        value_x[layer] = x
        if short:
            from anamnesis.extraction.feature_families.value_geometry import _value_layer_names

            _zeros(cols, _value_layer_names(layer, n_windows, include_stft), batch, device)
        else:
            nov, dr = _novelty(x), _drift(x)
            sp = _spread(x, _time_mean(x, counts), counts)
            value_spreads.append(sp)
            norms = row_norm(v_raw.double())  # [B, steps, kv heads]
            mean = row_mean(norms)
            cv = torch.where(mean > 1e-12, row_std(norms) / mean.clamp_min(1e-30),
                             mean.new_zeros(()))
            ln, lc = one.lengths(nov), counts.lengths(cv)
            for name, value in (("spread", sp), ("drift_halfcentroid", _half_drift(x, counts)),
                                ("novelty_mean", row_mean(nov, lengths=ln)),
                                ("novelty_std", row_std(nov, lengths=ln)),
                                ("crosshead_normcv_mean", row_mean(cv, lengths=lc)),
                                ("crosshead_normcv_std", row_std(cv, lengths=lc)),
                                ("key_drift_corr", _corr(dr, _drift(key_x[layer]), counts=one))):
                cols.put(f"value_L{layer}_{name}", value)
            eff_dims.append((f"value_L{layer}_eff_dim", x))
            _operators(cols, f"value_L{layer}_novelty", nov, n_windows, include_stft, one)
            _operators(cols, f"value_L{layer}_drift", dr, n_windows, include_stft, one)

        # Queries: FamilyReducer.query.
        q = _head_mean(queries[:, :, j].to(device)).double()
        if short:
            from anamnesis.extraction.feature_families.qk_geometry import _qk_layer_names

            _zeros(cols, _qk_layer_names(layer, n_windows, include_stft), batch, device)
        else:
            k = key_x[layer]
            self_align = _rowcos(q, k)
            cache_align = _rowcos(q, _running_mean(k))
            nov = _novelty(q)
            sp = _spread(q, _time_mean(q, counts), counts)
            query_spreads.append(sp)
            ln, lc = one.lengths(nov), counts.lengths(self_align)
            for name, value in (("q_spread", sp), ("q_drift_halfcentroid", _half_drift(q, counts)),
                                ("q_novelty_mean", row_mean(nov, lengths=ln)),
                                ("q_novelty_std", row_std(nov, lengths=ln)),
                                ("qk_self_align_mean", row_mean(self_align, lengths=lc)),
                                ("qk_self_align_std", row_std(self_align, lengths=lc)),
                                ("qk_cache_align_mean", row_mean(cache_align, lengths=lc)),
                                ("qk_cache_align_std", row_std(cache_align, lengths=lc))):
                cols.put(f"qk_L{layer}_{name}", value)
            eff_dims.append((f"qk_L{layer}_q_eff_dim", q))
            _operators(cols, f"qk_L{layer}_q_novelty", nov, n_windows, include_stft, one)
            _operators(cols, f"qk_L{layer}_qk_self_align", self_align, n_windows, include_stft,
                       counts)

        # Gates: FamilyReducer.gate (no features below two steps).
        if not short:
            x = gates[:, :, j].to(device).double()
            g = x / (1 + torch.exp(-x.clamp(-88, 88)))
            del x
            sp = row_mean((g.abs() > families.gate_sparsity_threshold).double())  # [B, steps]
            s2 = row_sumsq(g)
            dim = torch.where(s2 > 1e-12, row_sum(g.abs()).square() / s2.clamp_min(1e-30),
                              s2.new_zeros(()))
            dr = 1 - _cos(g[:, :-1].float(), g[:, 1:].float())
            prefix = f"gate_L{layer}"
            _moments(cols, prefix + "_sparsity", sp, counts)
            _moments(cols, prefix + "_eff_dim", dim, counts)
            _moments(cols, prefix + "_drift", dr, one)
            gate_means[layer] = _time_mean(g, counts).float()
            gate_sparsity[layer] = row_mean(sp, lengths=counts.lengths(sp))
            del g
            cols.put(prefix + "_topk_overlap_mean",
                     gate_topk_overlap_rows(gates[:, :, j].to(device).float(), counts))
            _operators(cols, prefix + "_sparsity", sp, n_windows, include_stft, counts)
            _operators(cols, prefix + "_drift", dr, n_windows, include_stft, one)

    # FamilyReducer.finish.
    for name, x in eff_dims:
        cols.put(name, _eff_dims(x, counts))
    for index, name in enumerate(("epoch_n_transitions", "epoch_max_transition",
                                  "epoch_regularity")):
        if epochs:
            series = torch.stack([e[index] for e in epochs], dim=1)  # [B, layers]
            stats = (row_mean(series), series.amax(dim=1), row_std(series))
            for suffix, value in zip(("_mean", "_max", "_std"), stats, strict=True):
                if not counts.uniform(steps):  # rows without epochs take the placeholder
                    value = torch.where(epoch_rows, value, value.new_zeros(()))
                cols.put(name + suffix, value)
        else:
            _zeros(cols, [name + "_mean", name + "_max", name + "_std"], batch, device)
    if len(gate_means) >= 2:
        cols.put("gate_cross_layer_sparsity_diversity",
                 row_std(torch.stack(list(gate_sparsity.values()), dim=1)))
        pairs = list(combinations(sorted(gate_means), 2))
        a = torch.stack([gate_means[p] for p, _ in pairs], dim=1)
        b = torch.stack([gate_means[p] for _, p in pairs], dim=1)
        cols.put("gate_cross_layer_agreement", row_mean(_cos(a, b)))
    elif gate_means:
        _zeros(cols, ["gate_cross_layer_sparsity_diversity", "gate_cross_layer_agreement"],
               batch, device)
    for name, spreads in (("value_crosslayer_spread_diversity", value_spreads),
                          ("q_crosslayer_spread_diversity", query_spreads)):
        if len(spreads) >= 2:
            cols.put(name, row_std(torch.stack(spreads, dim=1)))
        else:
            _zeros(cols, [name], batch, device)
    squares = _square_counts(counts)
    for label, matrices in (("key", key_x), ("value", value_x)):
        available = [layer for layer in sampled if not short]
        if not available:
            _zeros(cols, [f"kv_{label}_cka_{s}" for s in ("early_late", "adjacent_mean",
                                                          "overall_mean")], batch, device)
            continue
        # |C_b^T C_a|_F**2 = <C_a C_a^T, C_b C_b^T>_F and |C^T C|_F = |C C^T|_F.
        grams = {}
        for layer in available:
            x = matrices[layer]
            centered = x - _time_mean(x, counts)[:, None]
            grams[layer] = _front_square(_gram(centered), counts)
        lengths = squares.lengths(next(iter(grams.values())))
        self_norm = {layer: row_norm(gram, lengths=lengths) for layer, gram in grams.items()}
        pair_values = {}
        for a_layer, b_layer in combinations(available, 2):
            num = row_dot(grams[b_layer], grams[a_layer], lengths=lengths)
            value = num / (self_norm[a_layer] * self_norm[b_layer]).clamp_min(1e-12)
            pair_values[(a_layer, b_layer)] = value
            cols.put(f"kv_{label}_cka_L{a_layer}_L{b_layer}", value)
        third = max(1, len(available) // 3)
        early, late = set(available[:third]), set(available[-third:])
        groups = {
            "early_late": [v for (a, b), v in pair_values.items() if a in early and b in late],
            "adjacent_mean": [pair_values[(a, b)]
                              for a, b in zip(available[:-1], available[1:], strict=True)],
            "overall_mean": list(pair_values.values()),
        }
        for name, group in groups.items():
            if group:
                cols.put(f"kv_{label}_cka_{name}", row_mean(torch.stack(group, dim=1)))
            else:
                _zeros(cols, [f"kv_{label}_cka_{name}"], batch, device)
    return _stack(cols, batch, device)


def _epoch_counts(counts: Counts, window: int, stride: int) -> Counts:
    """Each row's epoch boundaries: one fewer than its ``(n - window) // stride + 1`` epochs
    (zero for a row without epochs)."""
    def build() -> Counts:
        host = [max(0, (c - window) // stride) if c >= window else 0 for c in counts.host]
        dev = torch.where(counts.dev >= window, (counts.dev - window) // stride,
                          counts.dev.new_zeros(()))
        return counts.derive(host, dev)
    return counts.memo(("epoch-edges", window, stride), build)


def outputs_features(lane: GpuFeatureLane, logit_stats: Tensor,
                     counts: Counts | None = None) -> tuple[BatchedFeatures, Tensor]:
    """The output family of ``B`` rows from ``logit_stats [B, steps, 6]``, each over its
    first ``counts`` steps (default: every step); returns the features and each row's mean
    log probability ``[B]`` (float64). Twin of ``external_stats._outputs_from_stats``.

    Raises
    ------
    ExternalCaptureError
        When ``logit_stats`` is not ``[B, steps, 6]`` float32 or float64.
    """
    if (logit_stats.ndim != 3 or logit_stats.shape[2] != 6
            or logit_stats.dtype not in (torch.float32, torch.float64)):
        raise ExternalCaptureError(
            f"logit_stats must be float32 or float64 [rows, steps, 6], got "
            f"{tuple(logit_stats.shape)} {logit_stats.dtype}")
    config, device = lane.config, lane.device
    stats = logit_stats.to(device)
    batch, n = stats.shape[:2]
    counts = counts if counts is not None else uniform_counts(batch, n, device)
    ent, top, topfive, rank, chosen_prob = (stats[:, :, i].float().contiguous() for i in range(5))
    logprob = stats[:, :, 5].double().contiguous()
    surprise = -chosen_prob.clamp_min(1e-10).log()
    traj, _ = per_row_index(counts, ("traj", config.trajectory_points),
                            lambda m: trajectory_indices(m, config.trajectory_points))
    lengths = counts.lengths(ent)
    cols = _Columns()
    for name, values in (("logit_entropy", ent), ("top1_prob", top)):
        _moments(cols, name, values, counts)
        picked = take_steps(values, traj)
        for i in range(picked.shape[1]):
            cols.put(f"{name}_traj{i}", picked[:, i])
    cols.put("top5_mass_mean", row_mean(topfive, lengths=lengths))
    for name, values in (("chosen_rank", rank), ("surprise", surprise)):
        cols.put("mean_" + name, row_mean(values, lengths=lengths))
        cols.put("std_" + name, row_std(values, lengths=lengths))
    picked = take_steps(surprise, traj)
    for i in range(picked.shape[1]):
        cols.put(f"surprise_traj{i}", picked[:, i])
    window = config.surprise_window
    if counts.hi >= window:
        values = surprise.unfold(1, window, 1).contiguous()  # [B, n - window + 1, window]
        running_mean = row_mean(values.double())
        running_std = row_std(values).float()
        threshold = (running_mean + config.surprise_threshold_sigma
                     * running_std.double().clamp_min(1e-6))
        above = (surprise[:, window - 1:].double() > threshold).double()
        boundaries = row_sum(above, lengths=counts.minus(window - 1).lengths(above))
        if not counts.uniform(n):  # rows shorter than the window take the placeholder
            boundaries = torch.where(counts.dev >= window, boundaries, boundaries.new_zeros(()))
        cols.put("surprise_boundary_count", boundaries)
    else:
        _zeros(cols, ["surprise_boundary_count"], batch, device)
    return _stack(cols, batch, device), row_mean(logprob, lengths=counts.lengths(logprob))


def _point_quantities(w: Tensor) -> list[Tensor]:
    """``attn_res._point_quantities`` of ``w [B, steps, sources]``, in ``QNAMES`` order,
    each ``[B, steps]`` float64."""
    w = w.double().clamp_min(0.0)
    batch, steps, n_src = w.shape
    if n_src == 0:
        return [w.new_zeros(batch, max(steps, 1)) for _ in QNAMES]
    w = w / (row_sum(w) + EPS)[..., None]
    ent = -row_sum(w * torch.log(w + EPS)) / math.log(max(n_src, 2))
    eff = (1.0 / (row_sumsq(w) + EPS)) / n_src
    idx = torch.arange(n_src, dtype=torch.float64, device=w.device) / max(n_src - 1, 1)
    quantities = dict(entropy=ent, top1=w.amax(dim=2), eff_src=eff, anchor_w=w[..., 0],
                      partial_w=w[..., -1], recency_cent=row_sum(w * idx))
    return [quantities[name].contiguous() for name in QNAMES]


def _summaries(ts: Tensor, counts: Counts | None = None) -> Tensor:
    """``attn_res._summ`` of each row of ``ts [B, steps]``, over its first ``counts`` steps:
    ``[B, 3]``."""
    batch, steps = ts.shape
    zero = ts.new_zeros(batch)
    if counts is None or counts.uniform(steps):
        if steps == 0:
            return torch.stack([zero, zero, zero], dim=1)
        if steps < 2:
            return torch.stack([row_mean(ts), zero, zero], dim=1)
        h = steps // 2
        return torch.stack([row_mean(ts), row_std(ts),
                            (row_mean(ts[:, :h].contiguous())
                             - row_mean(ts[:, h:].contiguous())).abs()], dim=1)
    if counts.lo < 2:
        raise ExternalCaptureError("rows below two steps share a batch with longer rows")
    lengths = counts.lengths(ts)
    a, ca, b, cb = _halves(ts, counts)
    return torch.stack([row_mean(ts, lengths=lengths), row_std(ts, lengths=lengths),
                        (row_mean(a, lengths=ca.lengths(a))
                         - row_mean(b.contiguous(), lengths=cb.lengths(b))).abs()], dim=1)


def _front_blocks(series: Sequence[Tensor], counts: Counts) -> tuple[Tensor, Counts]:
    """Rows' ``[B, steps]`` series joined end to end along each row's own steps: the first
    ``counts`` of each, in order, at the front of ``[B, len(series) * steps]``."""
    joined = torch.cat(list(series), dim=1)
    blocks, steps = len(series), series[0].shape[1]
    total = counts.derive([blocks * c for c in counts.host], counts.dev * blocks)
    if counts.uniform(steps):
        return joined, total

    def build() -> Tensor:
        p = torch.arange(blocks * steps, device=counts.dev.device)[None, :]
        n = counts.dev[:, None].clamp_min(1)
        return ((p // n) * steps + p % n).clamp_max(blocks * steps - 1)
    index = counts.memo(("front-blocks", blocks, steps), build)
    return joined.gather(1, index), total


def routing_features(sampled_layers: Sequence[int], points: Sequence[tuple[str, int, Tensor]],
                     committed: Sequence[Tensor],
                     counts: Counts | None = None) -> tuple[list[str], Tensor]:
    """The routing family of ``B`` rows that share one routing plan: ``points`` in the
    routing order, each ``(tag, layer, weights [B, steps, sources])``, and ``committed``,
    the committed blocks in order, each ``[B, steps, width]``; each row over its first
    ``counts`` steps (default: every step). Returns the names and ``[B, names]`` float32.
    Twin of ``external_stats._routing_from_stats`` given every point's full weights; the
    depth profile reads each point's three quantities from them."""
    sampled = {int(layer) for layer in sampled_layers}
    first = points[0][2] if points else (committed[0] if committed else None)
    if first is None:
        raise ExternalCaptureError("routing needs at least one point or committed block")
    batch, device = first.shape[0], first.device
    steps = first.shape[1]
    counts = counts if counts is not None else uniform_counts(batch, steps, device)
    values: list[Tensor] = []
    names: list[str] = []
    by_layer: dict[int, list[Tensor]] = {}
    stats: list[tuple[int, Tensor, Tensor, Tensor]] = []
    final_vec: Tensor | None = None
    for tag, layer, w in points:
        q = _point_quantities(w)
        if tag == "final" or int(layer) in sampled:
            vec = torch.cat([_summaries(series, counts) for series in q], dim=1)  # [B, 18]
            if tag == "final":
                final_vec = vec
                continue
            by_layer.setdefault(int(layer), []).append(vec)
        named = dict(zip(QNAMES, q, strict=True))
        lengths = counts.lengths(named["entropy"])
        stats.append((int(layer), row_mean(named["entropy"], lengths=lengths),
                      row_mean(named["anchor_w"], lengths=lengths),
                      row_mean(named["partial_w"], lengths=lengths)))

    zeros18 = torch.zeros(batch, len(QNAMES) * 3, dtype=torch.float64, device=device)
    for layer in sampled_layers:
        vecs = by_layer.get(int(layer))
        values.append(row_mean(torch.stack(vecs, dim=2)) if vecs else zeros18)
        names += [f"attnres_L{layer}_{qn}_{sm}" for qn in QNAMES for sm in SUMM]

    zero = torch.zeros(batch, dtype=torch.float64, device=device)
    if len(stats) >= 2:
        layers = [s[0] for s in stats]
        flat = float(np.std(np.asarray(layers, dtype=np.float64))) <= EPS
        lay = torch.tensor(layers, dtype=torch.float64, device=device)[None].expand(batch, -1)
        lay_c = (lay - row_mean(lay.contiguous())[:, None]).contiguous()
        denominator = row_sumsq(lay_c)
        columns = [torch.stack([s[i] for s in stats], dim=1) for i in (1, 2, 3)]
        slopes = [zero if flat else
                  row_dot(lay_c, (c - row_mean(c)[:, None]).contiguous()) / denominator
                  for c in columns]
        values.append(torch.stack(slopes + [row_std(columns[0])], dim=1))
    else:
        values.append(torch.stack([zero, zero, zero, zero], dim=1))
    names += [f"attnres_depthslope_{cn}" for cn in ("entropy", "anchor_w", "partial_w")]
    names.append("attnres_xpoint_entropy_std")

    values.append(final_vec[:, 0::3] if final_vec is not None
                  else torch.zeros(batch, len(QNAMES), dtype=torch.float64, device=device))
    names += [f"attnres_final_{qn}_mean" for qn in QNAMES]

    n_pairs = committed_pairs(len(committed))
    cos_means = [zero] * n_pairs
    overall = [zero, zero]
    if len(committed) >= 2:
        cos_all = []
        for b in range(len(committed) - 1):
            a, c = committed[b].to(device).double(), committed[b + 1].to(device).double()
            if a.ndim == 3 and a.shape == c.shape and a.shape[1] > 0:
                cos = row_dot(a, c) / (row_norm(a) * row_norm(c) + EPS)  # [B, steps]
                cos_means[b] = row_mean(cos, lengths=counts.lengths(cos))
                cos_all.append(cos)
        if cos_all:
            allc, total = _front_blocks(cos_all, counts)
            lengths = total.lengths(allc)
            overall = [row_mean(allc, lengths=lengths), row_std(allc, lengths=lengths)]
    values.append(torch.stack(cos_means + overall, dim=1))
    names += [f"attnres_committed_cos_b{b}b{b + 1}_mean" for b in range(n_pairs)]
    names += ["attnres_committed_cos_overall_mean", "attnres_committed_cos_overall_std"]
    return names, torch.nan_to_num(torch.cat(values, dim=1).float())


__all__ = (
    "BatchedFeatures",
    "TOPK_TIES",
    "gate_topk_overlap_rows",
    "kvqg_features",
    "outputs_features",
    "routing_features",
)
