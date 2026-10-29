"""The attention family of a statistics row, reduced for many rows at once, batchmate-invariant.

:meth:`StatsAttentionReducer.consume_stats
<anamnesis.extraction.fast.external_stats.StatsAttentionReducer.consume_stats>` reduces one
row's attention statistics one layer at a time: at every layer the entropy summary and the
head agreement; at each sampled layer, from the head-mean rows, recency, sink, coverage and
lookback, the decay fit, the recency trajectory, the spectral graph and its eigenvalues, the
attention-flow regions with their operators and head diversity, and the per-head summaries
the per-head family reads later. :func:`attention_features` computes the same features for
``B`` rows of one step count in one pass over ``[B, steps, ...]`` tensors.

**Batchmate invariance.** As in :mod:`anamnesis.extraction.fast.batched_stats`: every
reduction over a row's own steps, heads, columns or spectral steps is a primitive of
:mod:`anamnesis.extraction.fast.invariant_ops`; everything else is elementwise, an index into
a row's own tensors, an exact integer count, or a constant that depends only on the step
count. The decay fit and the prompt-mass decay are masked least-squares slopes whose four
sums are primitives. The spectral graph's Gram matrix is :func:`row_sum` over elementwise
products, never a matrix product. The eigenvalues of each row's spectral Laplacian come from
one batched ``torch.linalg.eigvalsh`` over every row and sampled layer of the batch on CUDA,
and one call per matrix on CPU (:func:`eigvalsh_rows`); a test holds each matrix's
eigenvalues byte for byte alone and in a batch on both. Rows of different prompt lengths have different causal
lengths: a row's columns are its own ``mean_rows`` width plus the one zero column the
per-row reducer appends, every column past a step's causal length is an exact zero at a
fixed position, and the causal masks are built from the row's own prompt length. A row
reduced alone and the same row inside any batch therefore have the same bytes.

**Agreement with the per-row reducer.** The arithmetic is the per-row reducer's, operator
for operator, with each reduction's order replaced by the primitives' fixed order and the
Gram matrix's product by row sums. The features agree with
:class:`~anamnesis.extraction.fast.external_stats.ExternalStatsSpan`'s to rounding, not to
the byte. The eigen-derived features (Fiedler ratio, high-frequency energy ratio, spectral
entropy) agree to the rounding of a different solver path, and the high-frequency energy
ratio counts the eigenvalues above the median, so an eigenvalue within rounding of the median
can move it by that eigenvalue's share. The names and the placeholders for short spans (the
decay fit below ten steps, the attention-flow family below two) are the per-row reducer's.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from anamnesis.extraction.fast.batched_stats import (
    WORKING_BYTES,
    Counts,
    _Columns,
    _operator_names,
    _temporal_rows,
    per_row_index,
    take_steps,
    uniform_counts,
    window_steps,
)
from anamnesis.extraction.fast.external import ExternalCaptureError
from anamnesis.extraction.fast.features import GpuFeatureLane
from anamnesis.extraction.fast.invariant_ops import row_mean, row_norm, row_std, row_sum

FLOW = "attn_flow:"
"""The key prefix of a sampled chunk's attention-flow series, named ``attn_flow_L<layer>_<feature>``."""


@dataclass(frozen=True)
class AttentionFeatures:
    """The attention family of ``B`` rows: ``names`` and ``values [B, len(names)]`` (float64);
    at each sampled layer, ``laplacians[layer]`` (the spectral steps, one list or each row's
    ``[B, K]``, how many each row has, and each row's normalised Laplacian ``[B, K, K]``,
    zero-padded past its own size, which the smoothness feature reads) and
    ``per_head[layer]`` (each row's per-head entropy and sink, ``[B, steps, heads]``, which
    the per-head family reads); and ``retained_bytes``, the bytes of those summaries, per row."""

    names: tuple[str, ...]
    values: Tensor
    laplacians: dict[int, tuple[Tensor | list[int], Counts, Tensor]]
    per_head: dict[int, tuple[Tensor, Tensor]]
    retained_bytes: list[int]


def _masked_slope(y: Tensor, x: Tensor, mask: Tensor) -> Tensor:
    """``ops.slope(y, x, mask)`` of each row of ``[R, m]`` tensors: the masked least-squares
    slope, zero for a row with fewer than two kept points."""
    w = mask.double()
    count = row_sum(w)
    xm = row_sum(x * w) / count.clamp_min(1)
    ym = row_sum(y * w) / count.clamp_min(1)
    xc = x - xm[:, None]
    value = row_sum(xc * (y - ym[:, None]) * w) / row_sum(xc.square() * w).clamp_min(1e-30)
    return torch.where(count >= 2, value, value.new_zeros(()))


def _decay(y: Tensor, live: Tensor | None = None) -> Tensor:
    """``ops.decay`` of each row of ``y [R, m]``; with ``live [R, m]``, over its live points."""
    x = torch.arange(y.shape[1], dtype=torch.float64, device=y.device).expand_as(y)
    mask = y > 1e-10
    return -_masked_slope(y.clamp_min(1e-30).log(), x, mask if live is None else mask & live)


def _gram(x: Tensor) -> Tensor:
    """``x x^T`` of each row of ``x [R, k, W]``: ``[R, k, k]``, each entry a :func:`row_sum`
    over an elementwise product, chunked over the first index to bound the working set."""
    rows, k, width = x.shape
    chunk = max(1, min(k, WORKING_BYTES // max(1, rows * k * width * x.element_size())))
    parts = []
    for first in range(0, k, chunk):
        last = min(k, first + chunk)
        parts.append(row_sum(x[:, first:last, None, :] * x[:, None, :, :]))
    return torch.cat(parts, dim=1)


def eigvalsh_rows(lap: Tensor) -> Tensor:
    """The ascending eigenvalues of each symmetric matrix of ``lap [R, k, k]``, each matrix
    solved the same way whatever its batchmates (:func:`eigvalsh_path`).

    On CUDA, one batched ``torch.linalg.eigvalsh``, never a batch of one: the solver may take
    another path for a batch of one, and within a batch of several it treats each matrix
    independently, so a lone matrix is solved beside an identity. On CPU, one call per
    matrix: the batched LAPACK path's result depends on where each matrix sits in memory,
    so its bytes would follow its index in the batch.
    """
    if lap.device.type != "cuda":
        return torch.stack([torch.linalg.eigvalsh(m) for m in lap.unbind(0)])
    if lap.shape[0] >= 2:
        return torch.linalg.eigvalsh(lap)
    pad = torch.eye(lap.shape[-1], dtype=lap.dtype, device=lap.device).expand(1, -1, -1)
    return torch.linalg.eigvalsh(torch.cat([lap, pad]))[: lap.shape[0]]


def eigvalsh_sized(lap: Tensor, k: Counts) -> Tensor:
    """The ascending eigenvalues of each row's leading ``k x k`` block of ``lap [R, K, K]``,
    zero-padded to ``[R, K]``: rows of one size are solved together by :func:`eigvalsh_rows`."""
    size = lap.shape[-1]
    if k.uniform(size):
        return eigvalsh_rows(lap)
    out = lap.new_zeros(lap.shape[:2])
    by_size: dict[int, list[int]] = {}
    for r, n in enumerate(k.host):
        by_size.setdefault(n, []).append(r)
    for n, members in sorted(by_size.items()):
        index = torch.as_tensor(members, dtype=torch.long, device=lap.device)
        block = lap.index_select(0, index)[:, :n, :n].contiguous()
        out[index, :n] = eigvalsh_rows(block)
    return out


def _spectral_steps(steps: int, stride: int) -> list[int]:
    picked = list(range(0, steps, stride))
    return picked if len(picked) >= 3 else list(range(steps))


def _strided(counts: Counts, key: str, stride_of) -> tuple[Tensor | list[int], Counts]:
    """Each row's steps ``0, s, 2s, ...`` below its count, ``s = stride_of(count)``."""
    return per_row_index(counts, key, lambda n: list(range(0, n, stride_of(n))))


def _entropy_features(cols: _Columns, head_entropy: Tensor, mean_entropy: Tensor,
                      counts: Counts) -> None:
    """Every layer's entropy summary and head agreement."""
    batch, t, layers, heads = head_entropy.shape
    ent = head_entropy
    if counts.uniform(t):
        sample = ent[:, :: max(1, t // 60)] if t > 60 else ent
        flat_counts = None
    else:
        index, picked = _strided(counts, "entropy-sample",
                                 lambda n: max(1, n // 60) if n > 60 else 1)
        sample = take_steps(ent, index)
        flat_counts = picked.derive([c * heads for c in picked.host], picked.dev * heads)
    flat = sample.permute(0, 2, 1, 3).reshape(batch, layers, -1).contiguous()  # [B, L, t' * H]
    lengths = flat_counts.lengths(flat) if flat_counts is not None else None
    ent_mean, ent_std = row_mean(flat, lengths=lengths), row_std(flat, lengths=lengths)
    if counts.uniform(t):
        stride = max(1, t // 30) if t > 30 else 1
        h_mean = mean_entropy[:, ::stride].double()  # [B, t'', L]
        h_heads = row_mean(ent[:, ::stride].contiguous()).double()  # [B, t'', L]
        agreement_counts = None
    else:
        index, agreement_counts = _strided(counts, "agreement-sample",
                                           lambda n: max(1, n // 30) if n > 30 else 1)
        h_mean = take_steps(mean_entropy, index).double()
        h_heads = row_mean(take_steps(ent, index).contiguous()).double()
    agreement = (1 - (h_mean - h_heads).clamp_min(0) / max(math.log(heads), 1e-12)).float()
    agreement = agreement.transpose(1, 2).contiguous()  # [B, L, t'']
    lengths = agreement_counts.lengths(agreement) if agreement_counts is not None else None
    ag_mean, ag_std = row_mean(agreement, lengths=lengths), row_std(agreement, lengths=lengths)
    for layer in range(layers):
        cols.put(f"attn_entropy_mean_L{layer}", ent_mean[:, layer])
        cols.put(f"attn_entropy_std_L{layer}", ent_std[:, layer])
        cols.put(f"head_agreement_mean_L{layer}", ag_mean[:, layer])
        cols.put(f"head_agreement_std_L{layer}", ag_std[:, layer])


def _decay_steps(n: int) -> list[int]:
    return list(range(n // 4, n, max(1, n // 10)))[:10]


def _sampled_chunk(lane: GpuFeatureLane, mean_rows: Tensor, head_prompt: Tensor,
                   head_recency: Tensor, starts: Tensor, counts: Counts,
                   ) -> tuple[dict[str, Tensor], Tensor | list[int], Counts, Tensor]:
    """The sampled-layer attention features of ``R`` rows, one per (row, sampled layer):
    ``mean_rows [R, t, W]`` float32, ``head_prompt`` and ``head_recency [R, t, H]``, each
    row's prompt length ``starts [R]`` and its own step count ``counts``. Returns the named
    ``[R]`` series, the spectral steps (one list, or ``[R, K]``), how many each row has, and
    the normalised Laplacians ``[R, K, K]``, zero past each row's own size."""
    families = lane.families
    n_windows, include_stft = families.temporal_n_windows, families.enable_stft
    rows, t, width = mean_rows.shape
    device = mean_rows.device
    ragged = not counts.uniform(t)
    c = starts[:, None]  # [R, 1]
    pos = torch.arange(width + 1, device=device)  # the row's columns and the appended zero
    lengths = c + torch.arange(1, t + 1, device=device)[None, :]  # [R, t]
    valid = pos[None, None, :] < lengths[:, :, None]  # [R, t, W + 1]
    rows32 = torch.nn.functional.pad(mean_rows, (0, 1))
    mean = (rows32 * valid).double()  # [R, t, W + 1]
    out: dict[str, Tensor] = {}
    live = counts.lengths(lengths)  # [R] or None

    total = row_sum(mean).clamp_min(1e-12)
    cutoff = (lengths.double() * 0.8).long().clamp_min(1)
    rec = row_sum(mean, mask=pos[None, None, :] >= cutoff[:, :, None]) / total
    sink = mean[:, :, 0].contiguous()
    coverage = ((mean > 1 / lengths.double()[:, :, None]) & valid).sum(dim=-1) / lengths.double()
    prompt = row_sum(mean, lengths=c.expand(rows, t))
    after_prompt = pos[None, None, :] >= c[:, :, None]
    generated = row_sum(mean, mask=after_prompt.expand(rows, t, width + 1))
    out["cache_recency_bias"] = row_mean(rec, lengths=live)
    out["cache_sink_mass"] = row_mean(sink, lengths=live)
    out["cache_cache_coverage"] = row_mean(coverage.contiguous(), lengths=live)
    out["cache_lookback_ratio"] = (row_sum(prompt, lengths=live)
                                   / row_sum(generated, lengths=live).clamp_min(1e-12))

    if counts.hi < 10:
        fit = mean.new_zeros(rows)
    elif not ragged:
        sample_steps = _decay_steps(t)
        picked = mean[:, sample_steps, 1:]  # [R, m, W]
        distances = (lengths[:, sample_steps, None] - 1 - pos[None, None, 1:]).clamp_min(1).double()
        mask = (picked > 1e-10) & valid[:, sample_steps, 1:]
        fit = -_masked_slope(picked.clamp_min(1e-300).log().reshape(rows, -1),
                             distances.reshape(rows, -1), mask.reshape(rows, -1))
    else:
        index, picked_counts = per_row_index(counts, "decay", _decay_steps)
        if isinstance(index, list):
            index = torch.as_tensor(index, dtype=torch.long, device=device)[None].expand(rows, -1)
        picked = take_steps(mean, index)[:, :, 1:]  # [R, m, W]
        m = picked.shape[1]
        step_lengths = take_steps(lengths, index)  # [R, m]
        distances = (step_lengths[:, :, None] - 1 - pos[None, None, 1:]).clamp_min(1).double()
        kept = picked_counts.mask(m)[:, :, None]
        mask = (picked > 1e-10) & (pos[None, None, 1:] < step_lengths[:, :, None]) & kept
        fit = -_masked_slope(picked.clamp_min(1e-300).log().reshape(rows, -1),
                             distances.reshape(rows, -1), mask.reshape(rows, -1))
        fit = torch.where(counts.dev < 10, fit.new_zeros(()), fit)
    out["cache_attn_decay_rate"] = fit
    if not ragged:
        quarter = t // 4
        for i in range(4):
            series = rec[:, i * quarter : (i + 1) * quarter if i < 3 else t] if t >= 4 else rec
            out[f"cache_recency_traj{i}"] = row_mean(series.contiguous())
    else:
        n = counts.dev
        quarter = torch.where(n >= 4, n // 4, n.new_zeros(()))
        quarters = [q // 4 if q >= 4 else 0 for q in counts.host]
        for i in range(4):
            sizes = [(q if i < 3 else c - 3 * q) if c >= 4 else c
                     for q, c in zip(quarters, counts.host, strict=True)]
            size_dev = torch.where(n >= 4, quarter if i < 3 else n - 3 * quarter, n)
            series = window_steps(rec, quarter * i, max(1, max(sizes)))
            out[f"cache_recency_traj{i}"] = row_mean(
                series, lengths=counts.derive(sizes, size_dev).lengths(series))

    stride = lane.config.spectral_subsample_step
    if not ragged:
        steps: Tensor | list[int] = _spectral_steps(t, stride)
        k_counts = counts.derive([len(steps)] * rows, torch.full_like(counts.dev, len(steps)))
    else:
        steps, k_counts = per_row_index(counts, ("spectral", stride),
                                        lambda n: _spectral_steps(n, stride))
    matrix = take_steps(mean, steps)  # [R, K, W + 1]; zero past the last spectral step's length
    k = matrix.shape[1]
    sized = not k_counts.uniform(k)
    matrix = matrix / row_norm(matrix).clamp_min(1e-12)[:, :, None]
    graph = _gram(matrix).clamp_min(0)
    graph = (graph + graph.transpose(1, 2)) * 0.5
    graph.diagonal(dim1=1, dim2=2).zero_()
    if sized:  # past a row's own size, the graph has no edges
        inside = k_counts.mask(k)
        graph = torch.where(inside[:, :, None] & inside[:, None, :], graph, graph.new_zeros(()))
    degrees = row_sum(graph, lengths=k_counts.lengths(graph))
    eye = torch.eye(k, device=device, dtype=torch.float64)
    lap = torch.diag_embed(degrees) - graph + eye * 1e-10
    ev = eigvalsh_sized(lap, k_counts).clamp_min(0)  # [R, K], ascending, zero past the size
    zero = ev.new_zeros(rows)
    if not sized:
        energy = row_sum(ev)
        radius = ev[:, -1]
        fiedler = (torch.where(radius > 1e-12, ev[:, 1] / radius.clamp_min(1e-30), zero)
                   if k > 1 else zero)
        mid = k // 2
        median = ev[:, mid] if k % 2 else (ev[:, mid - 1] + ev[:, mid]) * 0.5
        p = ev / energy.clamp_min(1e-30)[:, None] + 1e-12
        p = p / row_sum(p)[:, None]
        spec = -row_sum(p * p.log()) / math.log(k) if k > 1 else zero
        hfer = row_sum(ev, mask=ev > median[:, None]) / energy.clamp_min(1e-30)
    else:
        kn = k_counts.dev
        lengths_k = k_counts.lengths(ev)
        inside = k_counts.mask(k)
        energy = row_sum(ev, lengths=lengths_k)
        radius = take_steps(ev, (kn - 1)[:, None])[:, 0]
        fiedler = torch.where((radius > 1e-12) & (kn > 1),
                              ev[:, min(1, k - 1)] / radius.clamp_min(1e-30), zero)
        mid = kn // 2
        upper = take_steps(ev, mid[:, None])[:, 0]
        lower = take_steps(ev, (mid - 1).clamp_min(0)[:, None])[:, 0]
        median = torch.where(kn % 2 == 1, upper, (lower + upper) * 0.5)
        p = ev / energy.clamp_min(1e-30)[:, None] + 1e-12
        p = p / row_sum(p, lengths=lengths_k)[:, None]
        log_k = torch.as_tensor([math.log(n) if n > 1 else 1.0 for n in k_counts.host],
                                dtype=torch.float64, device=device)
        spec = torch.where(kn > 1, -row_sum(p * p.log(), lengths=lengths_k) / log_k, zero)
        hfer = row_sum(ev, mask=(ev > median[:, None]) & inside) / energy.clamp_min(1e-30)
    out["spectral_fiedler"] = fiedler
    out["spectral_hfer"] = torch.where(energy > 1e-12, hfer, zero)
    out["spectral_spectral_entropy"] = torch.where(energy > 1e-12, spec, zero)
    inv = degrees.clamp_min(1e-12).rsqrt()
    normalized = eye - inv[:, :, None] * graph * inv[:, None, :]
    if sized:
        normalized = torch.where(inside[:, :, None] & inside[:, None, :], normalized,
                                 normalized.new_zeros(()))

    if counts.hi >= 2:
        if counts.lo < 2:
            raise ExternalCaptureError("rows below two steps share a batch with longer rows")
        sys = prompt / total
        out[FLOW + "prompt_mass_mean"] = row_mean(sys, lengths=live)
        out[FLOW + "prompt_mass_std"] = row_std(sys, lengths=live)
        out[FLOW + "prompt_decay_rate"] = _decay(sys, counts.mask(t) if ragged else None)
        third = ((lengths - c) // 3).clamp_min(1)[:, :, None]
        cc = c[:, :, None]
        p3 = pos[None, None, :]
        regions = dict(
            prompt=sys,
            early_gen=row_sum(mean, mask=(p3 >= cc) & (p3 < cc + third)) / total,
            mid_gen=row_sum(mean, mask=(p3 >= cc + third) & (p3 < cc + 2 * third)) / total,
            recent=row_sum(mean, mask=(p3 >= cc + 2 * third).expand(rows, t, width + 1)) / total,
        )
        for label, series in regions.items():
            out[f"{FLOW}region_{label}_mean"] = row_mean(series, lengths=live)
            out[f"{FLOW}region_{label}_std"] = row_std(series, lengths=live)
        out[FLOW + "head_diversity_prompt"] = row_mean(row_std(head_prompt.double()), lengths=live)
        out[FLOW + "head_diversity_recency"] = row_mean(row_std(head_recency.double()), lengths=live)
        for name, series in (("prompt_mass", sys), ("recency_bias", rec)):
            values = _temporal_rows(series, n_windows, include_stft, counts)
            for i, suffix in enumerate(_operator_names("", n_windows, include_stft)):
                out[f"{FLOW}{name}{suffix}"] = values[:, i]
    return out, steps, k_counts, normalized


def _check(name: str, x: Tensor, shape: Sequence[int | None]) -> None:
    if x.ndim != len(shape) or any(s is not None and x.shape[i] != s for i, s in enumerate(shape)):
        raise ExternalCaptureError(f"{name} must be {list(shape)}, got {list(x.shape)}")
    if not x.is_floating_point():
        raise ExternalCaptureError(f"{name} must be floating point")


def attention_features(lane: GpuFeatureLane, head_entropy: Tensor, mean_entropy: Tensor,
                       head_scalars: Tensor, mean_rows: Tensor,
                       starts: Sequence[int], counts: Counts | None = None) -> AttentionFeatures:
    """The attention family of ``B`` rows.

    The tensors are the engine's statistics surfaces, stacked: ``head_entropy [B, steps,
    layers, heads]``, ``mean_entropy [B, steps, layers]``, ``head_scalars [B, steps, sampled,
    3, heads]`` (prompt mass, recency mass, sink) and ``mean_rows [B, steps, sampled, W]``,
    with ``W`` at least each row's span end less one; ``starts[b]`` is row ``b``'s prompt
    length and ``counts`` each row's own step count (default: every row fills ``steps``), the
    rest of its steps being padding. Each is read in float32, as the per-row reduction reads
    it.

    Raises
    ------
    ExternalCaptureError
        When the shapes disagree with each other or the lane, or ``mean_rows`` does not
        cover a row's span.
    """
    config = lane.config
    sampled = list(config.sampled_layers)
    if head_entropy.ndim != 4:
        raise ExternalCaptureError(
            f"head_entropy must be [rows, steps, layers, heads], got {tuple(head_entropy.shape)}")
    batch, t, layers, heads = head_entropy.shape
    if len(starts) != batch:
        raise ExternalCaptureError(f"{batch} rows but {len(starts)} starts")
    if not set(sampled) <= set(range(layers)):
        raise ExternalCaptureError("a sampled layer lies outside the capture")
    _check("head_entropy", head_entropy, (batch, t, layers, heads))
    _check("mean_entropy", mean_entropy, (batch, t, layers))
    _check("head_scalars", head_scalars, (batch, t, len(sampled), 3, heads))
    _check("mean_rows", mean_rows, (batch, t, len(sampled), None))
    device = lane.device
    counts = counts if counts is not None else uniform_counts(batch, t, device)
    if len(counts.host) != batch or counts.hi > t or counts.lo < 1:
        raise ExternalCaptureError(f"step counts {counts.host} for {batch} rows of {t} steps")
    width = mean_rows.shape[3]
    if any(int(s) < 1 or width < int(s) + n for s, n in zip(starts, counts.host, strict=True)):
        raise ExternalCaptureError(f"mean_rows of width {width} do not cover every row's span")
    head_entropy = head_entropy.to(device).float()
    mean_entropy = mean_entropy.to(device).float()
    head_scalars = head_scalars.to(device).float()
    cols = _Columns()
    _entropy_features(cols, head_entropy, mean_entropy, counts)

    start_t = torch.as_tensor([int(s) for s in starts], dtype=torch.long, device=device)
    lengths = start_t[:, None] + torch.arange(1, t + 1, device=device)[None, :]  # [B, t]
    log_length = lengths.clamp_min(2).double().log()[:, :, None]
    per_layer = max(1, batch * t * (width + 1) * 8 * 6)
    chunk = max(1, min(len(sampled), WORKING_BYTES // per_layer))
    laplacians: dict[int, tuple[Tensor | list[int], Counts, Tensor]] = {}
    per_head: dict[int, tuple[Tensor, Tensor]] = {}
    retained = [0] * batch
    for first in range(0, len(sampled), chunk):
        js = list(range(first, min(len(sampled), first + chunk)))
        n = len(js)

        def per_row(x: Tensor) -> Tensor:
            """``[B, t, n, ...]`` -> ``[B * n, t, ...]``, row-major over (row, layer)."""
            return x.movedim(2, 1).reshape(batch * n, *x.shape[1:2], *x.shape[3:]).contiguous()

        series, steps, k_counts, normalized = _sampled_chunk(
            lane, per_row(mean_rows[:, :, js].to(device).float()),
            per_row(head_scalars[:, :, js, 0]), per_row(head_scalars[:, :, js, 1]),
            start_t.repeat_interleave(n), counts.repeat(n))
        normalized = normalized.reshape(batch, n, *normalized.shape[1:])
        if isinstance(steps, Tensor):
            steps = steps.reshape(batch, n, -1)
        sizes = k_counts.host
        size_dev = k_counts.dev.reshape(batch, n)
        for i, j in enumerate(js):
            layer = sampled[j]
            for name, value in series.items():
                if name.startswith(FLOW):
                    name = f"attn_flow_L{layer}_{name[len(FLOW):]}"
                else:
                    name = f"{name}_L{layer}"
                cols.put(name, value.reshape(batch, n)[:, i])
            layer_k = k_counts.derive(sizes[i::n], size_dev[:, i])
            laplacians[layer] = (steps[:, i] if isinstance(steps, Tensor) else steps, layer_k,
                                 normalized[:, i])
            per_ent = head_entropy[:, :, layer].double() / log_length
            per_sink = head_scalars[:, :, j, 2].double()
            per_head[layer] = (per_ent, per_sink)
            for b in range(batch):
                retained[b] += (layer_k.host[b] ** 2 + 2 * counts.host[b] * heads) * 8
        if counts.hi < 2:
            from anamnesis.extraction.feature_families.attention_flow import _attention_flow_names

            for j in js:
                names = _attention_flow_names(sampled[j], lane.families.temporal_n_windows,
                                              lane.families.enable_stft)
                cols.block(list(names), torch.zeros(batch, len(names), dtype=torch.float64,
                                                    device=device))
    values = torch.stack([c.double() for c in cols.columns], dim=1)
    return AttentionFeatures(tuple(cols.names), values, laplacians, per_head, retained)


__all__ = ("AttentionFeatures", "attention_features", "eigvalsh_rows", "eigvalsh_sized")
