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

**What stays per row.** The gate's top-k Jaccard: its top-k comes from NumPy's
``argsort``, whose order among tied values is part of the feature. Each row's gate
surfaces move to the host in one copy for the whole batch, and each (row, layer) is
reduced there by :func:`~anamnesis.extraction.fast.families.gate_topk_overlap`, the
per-row lane's own function, on a thread pool. It reads only that row's values, so it is
batchmate-invariant as it stands and agrees with the per-row lane to the byte.

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
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch
from torch import Tensor

from anamnesis.extraction.fast.batched_stats import (
    PER_ROW_EXCEPTIONS,
    WORKING_BYTES,
    _Columns,
    _cos,
    _operator_names,
    _rowcos,
    _temporal_rows,
)
from anamnesis.extraction.fast.external import ExternalCaptureError
from anamnesis.extraction.fast.families import gate_topk_overlap
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

TOPK_WORKERS = min(32, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                   else os.cpu_count() or 1)
"""Threads of the pool that computes the gate's top-k Jaccard on the host, one task per
(row, sampled layer). NumPy releases the GIL in its ufuncs and sorts."""

_TOPK_POOL: ThreadPoolExecutor | None = None


def _topk_pool() -> ThreadPoolExecutor:
    global _TOPK_POOL
    if _TOPK_POOL is None:
        _TOPK_POOL = ThreadPoolExecutor(max_workers=TOPK_WORKERS,
                                        thread_name_prefix="batched-gate-topk")
    return _TOPK_POOL


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


def _time_mean(x: Tensor) -> Tensor:
    """The mean over dimension 1 (a row's steps) of ``x [B, steps, ...]``."""
    return row_mean(x.movedim(1, -1).contiguous())


def _head_mean(x: Tensor) -> Tensor:
    """``x [..., heads, dim]`` averaged over its heads in float32: ``[..., dim]``."""
    return row_mean(x.float().transpose(-1, -2).contiguous())


def _std_over_steps(x: Tensor) -> Tensor:
    return row_std(x.movedim(1, -1).contiguous())


def _running_mean(x: Tensor) -> Tensor:
    """The running mean along dimension 1 of ``x [B, steps, dim]``: an inclusive scan by
    doubling, each stage elementwise, then a division by the count."""
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


def _half_drift(x: Tensor) -> Tensor:
    mid = x.shape[1] // 2
    return 1 - _cos(_time_mean(x[:, :mid]).float(), _time_mean(x[:, mid:]).float())


def _spread(x: Tensor, center: Tensor) -> Tensor:
    """The mean over steps of one minus each step's cosine to ``center [B, dim]``."""
    return row_mean(1 - _rowcos(x, center[:, None].expand_as(x)))


def _corr(a: Tensor, b: Tensor, min_n: int = 3) -> Tensor:
    """``ops.corr`` of each row of ``a`` and ``b`` (``[B, n]``)."""
    n = min(a.shape[1], b.shape[1])
    if n < min_n:
        return a.new_zeros(a.shape[0], dtype=torch.float64)
    a, b = a[:, :n].double(), b[:, :n].double()
    ac, bc = a - row_mean(a)[:, None], b - row_mean(b)[:, None]
    value = row_dot(ac, bc) / (row_norm(ac) * row_norm(bc)).clamp_min(1e-30)
    flat = (row_std(a) < 1e-12) | (row_std(b) < 1e-12)
    return torch.where(flat, value.new_zeros(()), value.clamp(-1, 1))


def _gram(x: Tensor) -> Tensor:
    """``x x^T`` of each row of ``x [B, m, n]``: ``[B, m, m]``, each entry a :func:`row_sum`
    over an elementwise product, chunked over the first index to bound the working set."""
    batch, m, n = x.shape
    chunk = max(1, min(m, WORKING_BYTES // max(1, batch * m * n * x.element_size())))
    parts = []
    for first in range(0, m, chunk):
        last = min(m, first + chunk)
        parts.append(row_sum(x[:, first:last, None, :] * x[:, None, :, :]))
    return torch.cat(parts, dim=1)


def _eff_dims(x: Tensor) -> Tensor:
    """The effective dimension of each row's ``[m, n]`` matrix from its Frobenius norms,
    with ``features.EFF_DIM_METHODS``' ``norms`` guards: ``|A|_F**4 / |G|_F**2 / min(m, n)``,
    ``G`` the smaller Gram matrix; zero where ``|A|_F**2 <= 1e-12``."""
    batch, m, n = x.shape
    total = row_sumsq(x.reshape(batch, m * n))
    gram = _gram(x.transpose(1, 2).contiguous() if n <= m else x)
    fourth = row_sumsq(gram.reshape(batch, -1))
    return torch.where(total > 1e-12, total.square() / fourth.clamp_min(1e-30) / min(m, n),
                       total.new_zeros(()))


def _moments(cols: _Columns, prefix: str, x: Tensor) -> None:
    cols.put(prefix + "_mean", row_mean(x))
    cols.put(prefix + "_std", row_std(x))


def _operators(cols: _Columns, prefix: str, x: Tensor, n_windows: int, include_stft: bool) -> None:
    cols.block(_operator_names(prefix, n_windows, include_stft),
               _temporal_rows(x, n_windows, include_stft))


def _surface(name: str, x: Tensor, ndim: int, batch: int, steps: int, sampled: int) -> None:
    if x.ndim != ndim or tuple(x.shape[:3]) != (batch, steps, sampled):
        raise ExternalCaptureError(
            f"{name} must be [{batch}, {steps}, {sampled}, ...], got {tuple(x.shape)}")
    if not x.is_floating_point():
        raise ExternalCaptureError(f"{name} must be floating point")


def kvqg_features(lane: GpuFeatureLane, keys: Tensor, values: Tensor, queries: Tensor,
                  gates: Tensor, per_head: Mapping[int, tuple[Tensor, Tensor]]) -> BatchedFeatures:
    """The key, per-head, value, query and gate families of ``B`` rows of one step count,
    with :meth:`FamilyReducer.finish`'s cross-layer features.

    ``keys`` and ``values`` are ``[B, steps, sampled, kv heads, head dim]``, ``queries``
    ``[B, steps, sampled, query heads, head dim]`` and ``gates [B, steps, sampled, gate
    width]`` (pre-SiLU), at the lane's sampled layers in order. ``per_head[layer]`` is the
    attention reduction's per-head entropy and sink, ``[B, steps, query heads]`` each, as
    the per-row attention reducer retains them.

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
    n_windows, include_stft = families.temporal_n_windows, families.enable_stft
    cols = _Columns()

    # The named per-row exception, started first so the host works while the device does.
    topk: dict[str, list[Future]] = {}
    if steps >= 2:
        host = gates.float().cpu().numpy()
        pool = _topk_pool()
        for j, layer in enumerate(sampled):
            topk[f"gate_L{layer}_topk_overlap_mean"] = [
                pool.submit(gate_topk_overlap, host[b, :, j]) for b in range(batch)]

    key_x: dict[int, Tensor] = {}
    value_x: dict[int, Tensor] = {}
    eff_dims: list[tuple[str, Tensor]] = []
    epochs: list[tuple[Tensor, Tensor, Tensor]] = []
    value_spreads: list[Tensor] = []
    query_spreads: list[Tensor] = []
    gate_means: dict[int, Tensor] = {}
    gate_sparsity: dict[int, Tensor] = {}
    window, stride = config.epoch_window_size, config.epoch_stride
    for j, layer in enumerate(sampled):
        k_raw = keys[:, :, j].to(device).float()
        # Keys: FamilyReducer.key.
        matrix32 = _head_mean(k_raw)  # [B, steps, dim]
        x = matrix32.double()
        key_x[layer] = x
        if steps < 2:
            _zeros(cols, [f"kv_{name}_L{layer}" for name in (
                "key_spread", "key_eff_dim", "key_drift", "key_novelty_mean",
                "key_novelty_std", *[f"key_novelty_traj{i}" for i in range(5)])],
                batch, device)
        else:
            cols.put(f"kv_key_spread_L{layer}", _spread(x, _time_mean(matrix32).double()))
            eff_dims.append((f"kv_key_eff_dim_L{layer}", x))
            cols.put(f"kv_key_drift_L{layer}", _half_drift(x))
            nov = _novelty(x).float()
            cols.put(f"kv_key_novelty_mean_L{layer}", row_mean(nov))
            cols.put(f"kv_key_novelty_std_L{layer}", row_std(nov))
            for i, index in enumerate(trajectory_indices(nov.shape[1])):
                cols.put(f"kv_key_novelty_traj{i}_L{layer}", nov[:, index])
            if steps >= window + stride:
                starts = range(0, steps - window + 1, stride)
                windows = torch.stack([matrix32[:, s:s + window] for s in starts], dim=1)
                centers = row_mean(windows.transpose(2, 3).contiguous())  # [B, E, dim]
                boundaries = (1 - _cos(centers[:, :-1], centers[:, 1:])).float()
                spread_b = row_std(boundaries)
                threshold = row_mean(boundaries).double() + 1.5 * spread_b.double().clamp_min(1e-6)
                count = row_mean((boundaries > threshold.float()[:, None]).double())
                epochs.append((count, boundaries.amax(dim=1).double(), spread_b.double()))

        # Per-head keys: AttentionReducer.head_features.
        prefix = f"ph_L{layer}"
        ent, sink = per_head[layer]
        if steps < 4:
            from anamnesis.extraction.feature_families.per_head import _per_head_names

            _zeros(cols, _per_head_names(layer), batch, device)
        else:
            ent, sink = ent.to(device).double(), sink.to(device).double()
            means = _time_mean(ent)  # [B, heads]
            cols.put(prefix + "_head_entropy_mean", row_mean(means))
            cols.put(prefix + "_head_entropy_std", row_std(means))
            cols.put(prefix + "_head_entropy_min", means.amin(dim=1))
            cols.put(prefix + "_head_entropy_max", means.amax(dim=1))
            half = steps // 2
            cols.put(prefix + "_head_role_stability",
                     _corr(_time_mean(ent[:, :half]), _time_mean(ent[:, half:]), min_n=2))
            cols.put(prefix + "_sink_head_std", row_std(_time_mean(sink)))
            k = k_raw.double()  # [B, steps, kv heads, dim]
            center = _time_mean(k)  # [B, kv heads, dim]
            sims = row_dot(k, center[:, None].expand_as(k)) / (
                row_norm(k) * row_norm(center)[:, None]).clamp_min(1e-12)
            spread_h = _time_mean(1 - sims)  # [B, kv heads]
            cols.put(prefix + "_kv_key_spread_head_mean", row_mean(spread_h))
            cols.put(prefix + "_kv_key_spread_head_std", row_std(spread_h))

        # Values: FamilyReducer.value.
        v_raw = values[:, :, j].to(device).float()
        x = _head_mean(v_raw).double()
        value_x[layer] = x
        if steps < 2:
            from anamnesis.extraction.feature_families.value_geometry import _value_layer_names

            _zeros(cols, _value_layer_names(layer, n_windows, include_stft), batch, device)
        else:
            nov, dr = _novelty(x), _drift(x)
            sp = _spread(x, _time_mean(x))
            value_spreads.append(sp)
            norms = row_norm(v_raw.double())  # [B, steps, kv heads]
            mean = row_mean(norms)
            cv = torch.where(mean > 1e-12, row_std(norms) / mean.clamp_min(1e-30),
                             mean.new_zeros(()))
            for name, value in (("spread", sp), ("drift_halfcentroid", _half_drift(x)),
                                ("novelty_mean", row_mean(nov)), ("novelty_std", row_std(nov)),
                                ("crosshead_normcv_mean", row_mean(cv)),
                                ("crosshead_normcv_std", row_std(cv)),
                                ("key_drift_corr", _corr(dr, _drift(key_x[layer])))):
                cols.put(f"value_L{layer}_{name}", value)
            eff_dims.append((f"value_L{layer}_eff_dim", x))
            _operators(cols, f"value_L{layer}_novelty", nov, n_windows, include_stft)
            _operators(cols, f"value_L{layer}_drift", dr, n_windows, include_stft)

        # Queries: FamilyReducer.query.
        q = _head_mean(queries[:, :, j].to(device)).double()
        if steps < 2:
            from anamnesis.extraction.feature_families.qk_geometry import _qk_layer_names

            _zeros(cols, _qk_layer_names(layer, n_windows, include_stft), batch, device)
        else:
            k = key_x[layer]
            self_align = _rowcos(q, k)
            cache_align = _rowcos(q, _running_mean(k))
            nov = _novelty(q)
            sp = _spread(q, _time_mean(q))
            query_spreads.append(sp)
            for name, value in (("q_spread", sp), ("q_drift_halfcentroid", _half_drift(q)),
                                ("q_novelty_mean", row_mean(nov)), ("q_novelty_std", row_std(nov)),
                                ("qk_self_align_mean", row_mean(self_align)),
                                ("qk_self_align_std", row_std(self_align)),
                                ("qk_cache_align_mean", row_mean(cache_align)),
                                ("qk_cache_align_std", row_std(cache_align))):
                cols.put(f"qk_L{layer}_{name}", value)
            eff_dims.append((f"qk_L{layer}_q_eff_dim", q))
            _operators(cols, f"qk_L{layer}_q_novelty", nov, n_windows, include_stft)
            _operators(cols, f"qk_L{layer}_qk_self_align", self_align, n_windows, include_stft)

        # Gates: FamilyReducer.gate (no features below two steps).
        if steps >= 2:
            x = gates[:, :, j].to(device).double()
            g = x / (1 + torch.exp(-x.clamp(-88, 88)))
            del x
            sp = row_mean((g.abs() > families.gate_sparsity_threshold).double())  # [B, steps]
            s2 = row_sumsq(g)
            dim = torch.where(s2 > 1e-12, row_sum(g.abs()).square() / s2.clamp_min(1e-30),
                              s2.new_zeros(()))
            dr = 1 - _cos(g[:, :-1].float(), g[:, 1:].float())
            prefix = f"gate_L{layer}"
            _moments(cols, prefix + "_sparsity", sp)
            _moments(cols, prefix + "_eff_dim", dim)
            _moments(cols, prefix + "_drift", dr)
            gate_means[layer] = _time_mean(g).float()
            gate_sparsity[layer] = row_mean(sp)
            del g
            _operators(cols, prefix + "_sparsity", sp, n_windows, include_stft)
            _operators(cols, prefix + "_drift", dr, n_windows, include_stft)

    # FamilyReducer.finish.
    for name, x in eff_dims:
        cols.put(name, _eff_dims(x))
    for index, name in enumerate(("epoch_n_transitions", "epoch_max_transition",
                                  "epoch_regularity")):
        if epochs:
            series = torch.stack([e[index] for e in epochs], dim=1)  # [B, layers]
            cols.put(name + "_mean", row_mean(series))
            cols.put(name + "_max", series.amax(dim=1))
            cols.put(name + "_std", row_std(series))
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
    for label, matrices in (("key", key_x), ("value", value_x)):
        available = [layer for layer in sampled if steps >= 2]
        if not available:
            _zeros(cols, [f"kv_{label}_cka_{s}" for s in ("early_late", "adjacent_mean",
                                                          "overall_mean")], batch, device)
            continue
        # |C_b^T C_a|_F**2 = <C_a C_a^T, C_b C_b^T>_F and |C^T C|_F = |C C^T|_F.
        grams = {}
        for layer in available:
            x = matrices[layer]
            centered = x - _time_mean(x)[:, None]
            grams[layer] = _gram(centered).reshape(batch, -1)
        self_norm = {layer: row_norm(gram) for layer, gram in grams.items()}
        pair_values = {}
        for a_layer, b_layer in combinations(available, 2):
            num = row_dot(grams[b_layer], grams[a_layer])
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
    for name, futures in topk.items():
        cols.put(name, torch.tensor([f.result() for f in futures], dtype=torch.float64,
                                    device=device))
    return _stack(cols, batch, device)


def outputs_features(lane: GpuFeatureLane, logit_stats: Tensor) -> tuple[BatchedFeatures, Tensor]:
    """The output family of ``B`` rows from ``logit_stats [B, steps, 6]``; returns the
    features and each row's mean log probability ``[B]`` (float64). Twin of
    ``external_stats._outputs_from_stats``.

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
    ent, top, topfive, rank, chosen_prob = (stats[:, :, i].float().contiguous() for i in range(5))
    logprob = stats[:, :, 5].double().contiguous()
    surprise = -chosen_prob.clamp_min(1e-10).log()
    traj = trajectory_indices(n, config.trajectory_points)
    cols = _Columns()
    for name, values in (("logit_entropy", ent), ("top1_prob", top)):
        _moments(cols, name, values)
        for i, t in enumerate(traj):
            cols.put(f"{name}_traj{i}", values[:, t])
    cols.put("top5_mass_mean", row_mean(topfive))
    for name, values in (("chosen_rank", rank), ("surprise", surprise)):
        cols.put("mean_" + name, row_mean(values))
        cols.put("std_" + name, row_std(values))
    for i, t in enumerate(traj):
        cols.put(f"surprise_traj{i}", surprise[:, t])
    window = config.surprise_window
    if n >= window:
        values = surprise.unfold(1, window, 1).contiguous()  # [B, n - window + 1, window]
        running_mean = row_mean(values.double())
        running_std = row_std(values).float()
        threshold = (running_mean + config.surprise_threshold_sigma
                     * running_std.double().clamp_min(1e-6))
        cols.put("surprise_boundary_count",
                 row_sum((surprise[:, window - 1:].double() > threshold).double()))
    else:
        _zeros(cols, ["surprise_boundary_count"], batch, device)
    return _stack(cols, batch, device), row_mean(logprob)


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


def _summaries(ts: Tensor) -> Tensor:
    """``attn_res._summ`` of each row of ``ts [B, steps]``: ``[B, 3]``."""
    batch, steps = ts.shape
    zero = ts.new_zeros(batch)
    if steps == 0:
        return torch.stack([zero, zero, zero], dim=1)
    if steps < 2:
        return torch.stack([row_mean(ts), zero, zero], dim=1)
    h = steps // 2
    return torch.stack([row_mean(ts), row_std(ts),
                        (row_mean(ts[:, :h].contiguous()) - row_mean(ts[:, h:].contiguous())).abs()],
                       dim=1)


def routing_features(sampled_layers: Sequence[int], points: Sequence[tuple[str, int, Tensor]],
                     committed: Sequence[Tensor]) -> tuple[list[str], Tensor]:
    """The routing family of ``B`` rows that share one routing plan: ``points`` in the
    routing order, each ``(tag, layer, weights [B, steps, sources])``, and ``committed``,
    the committed blocks in order, each ``[B, steps, width]``. Returns the names and ``[B,
    names]`` float32. Twin of ``external_stats._routing_from_stats`` given every point's
    full weights; the depth profile reads each point's three quantities from them."""
    sampled = {int(layer) for layer in sampled_layers}
    first = points[0][2] if points else (committed[0] if committed else None)
    if first is None:
        raise ExternalCaptureError("routing needs at least one point or committed block")
    batch, device = first.shape[0], first.device
    values: list[Tensor] = []
    names: list[str] = []
    by_layer: dict[int, list[Tensor]] = {}
    stats: list[tuple[int, Tensor, Tensor, Tensor]] = []
    final_vec: Tensor | None = None
    for tag, layer, w in points:
        q = _point_quantities(w)
        if tag == "final" or int(layer) in sampled:
            vec = torch.cat([_summaries(series) for series in q], dim=1)  # [B, 18]
            if tag == "final":
                final_vec = vec
                continue
            by_layer.setdefault(int(layer), []).append(vec)
        named = dict(zip(QNAMES, q, strict=True))
        stats.append((int(layer), row_mean(named["entropy"]), row_mean(named["anchor_w"]),
                      row_mean(named["partial_w"])))

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
                cos_means[b] = row_mean(cos)
                cos_all.append(cos)
        if cos_all:
            allc = torch.cat(cos_all, dim=1)
            overall = [row_mean(allc), row_std(allc)]
    values.append(torch.stack(cos_means + overall, dim=1))
    names += [f"attnres_committed_cos_b{b}b{b + 1}_mean" for b in range(n_pairs)]
    names += ["attnres_committed_cos_overall_mean", "attnres_committed_cos_overall_std"]
    return names, torch.nan_to_num(torch.cat(values, dim=1).float())


__all__ = (
    "BatchedFeatures",
    "kvqg_features",
    "outputs_features",
    "routing_features",
)
