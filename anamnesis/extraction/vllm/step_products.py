"""Scratch-to-product reduction for the second pass's selected rows.

The bounded scratch of normalized per-head probability rows is reduced
right after the launch, inside the capture step, to exactly what the
attention consumers keep. A sampled layer's span scratch gives the coverage of
every span row, the head-agreement entropy pair at the agreement rows, the
entropy series, the per-head summaries, and the head-mean rows the spectral,
decay, cache and flow consumers read. Every other layer's series scratch gives
the agreement pair and the entropy series. Nothing head-resolved survives the
step.

The entropy pair reproduces the reducer's arithmetic — double promotion,
per-head renormalization with the same clamp, ``ops.entropy`` — and it is
width-independent: keys beyond a row's valid length are exact zeros in
the scratch and contribute exactly zero to every sum and every entropy
term, so a scratch narrower or wider than the reference's materialized
row produces the same values. The head mean is taken in float32 before
any promotion, matching the reducer's declared order.
"""
from __future__ import annotations

import torch
from torch import Tensor

from anamnesis.extraction.fast.ops import entropy
from anamnesis.extraction.vllm.second_pass import coverage_of_selected_rows

SPAN_PRODUCT_KEYS = ('coverage', 'h_mean', 'h_heads', 'spectral_rows',
                     'decay_rows', 'row_sum_worst', 'entropy_rows', 'head_ent',
                     'head_sink', 'head_prompt', 'head_recency', 'span_rows')
"""What a sampled layer's span scratch reduces to. ``span_rows`` is the fp32
head-mean row of every span row, which the cache and flow families read."""

SERIES_PRODUCT_KEYS = ('h_mean', 'h_heads', 'entropy_rows', 'row_sum_worst')
"""What every other layer's series scratch reduces to."""


def agreement_pair(rows: Tensor) -> tuple[Tensor, Tensor]:
    """The reducer's h-pair from fp32 probability rows: [n, heads, keys].

    ``h_mean`` is the entropy of the head-mean of the renormalized rows;
    ``h_heads`` the mean over heads of each head's renormalized entropy.
    The downstream agreement statistic combines them against log(heads).
    """
    if rows.ndim != 3 or rows.dtype != torch.float32:
        raise ValueError('agreement rows must be fp32 [rows, heads, keys]')
    norm = rows.double()
    norm = norm / norm.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    h_mean = entropy(norm.mean(dim=1)).double()
    h_heads = entropy(norm).mean(dim=1).double()
    return h_mean, h_heads


def canonical_rows(rows: Tensor, width: int) -> Tensor:
    """One request run's scratch rows at the request's materialized width.

    The step scratch is as wide as the step's longest scheduled row, so the
    same span row meets a different trailing-zero extent under different
    schedules, and the per-head entropy reduction moves by an ulp with that
    extent. Slicing or padding
    every run to its own request's width ``c + t + 1`` before any reduction
    makes the reduction layout a property of the request, never of the
    schedule, and equal to the reference's materialized layout. Dropped
    columns and appended columns are exact zeros either way.
    """
    if rows.shape[-1] >= width:
        return rows[..., :width].contiguous()
    return torch.cat(
        [rows, rows.new_zeros(*rows.shape[:-1], width - rows.shape[-1])],
        dim=-1)


def entropy_of_rows(rows: Tensor, *, widths: Tensor) -> Tensor:
    """The reducer's entropy series values: ``ops.entropy`` on fp32 rows.

    The reducer computes ``entropy(a)`` over the masked, unrenormalized fp32
    weights at the request's materialized width; each run of rows reduces
    at that canonical width so the values cannot depend on the schedule.
    """
    if rows.ndim != 3 or rows.dtype != torch.float32:
        raise ValueError('entropy rows must be fp32 [rows, heads, keys]')
    if widths.shape != (rows.shape[0],):
        raise ValueError('widths must carry one entry per entropy row')
    pieces = []
    for start, stop, width in _width_runs(widths):
        pieces.append(entropy(canonical_rows(rows[start:stop], width)))
    if not pieces:
        return rows.new_zeros((0, rows.shape[1]))
    return torch.cat(pieces, dim=0)


def _width_runs(widths: Tensor):
    """Consecutive runs of equal canonical width, in slot order."""
    runs = []
    start = 0
    total = widths.shape[0]
    while start < total:
        stop = start
        width = int(widths[start])
        while stop < total and int(widths[stop]) == width:
            stop += 1
        runs.append((start, stop, width))
        start = stop
    return runs


def per_head_summaries(scratch: Tensor, *, lengths: Tensor,
                       prefixes: Tensor, widths: Tensor) -> dict[str, Tensor]:
    """The reducer's per-span-row per-head summaries from rounded rows.

    ``head_ent``/``head_sink`` are the reducer's normalized per-head entropy
    and sink shares; ``head_prompt``/``head_recency`` its per-head prompt and
    recency mass ratios. All four follow the reducer's own arithmetic —
    double promotion, the same clamps, the recency cutoff recomputed with
    the reducer's truncation, the prompt mass sliced at the run's prompt
    boundary — over rows brought to the request's canonical width first,
    so identical rows give identical bits under every schedule.
    """
    if scratch.ndim != 3 or scratch.dtype != torch.float32:
        raise ValueError('span scratch must be fp32 [rows, heads, keys]')
    rows = scratch.shape[0]
    for name, index in (('lengths', lengths), ('prefixes', prefixes),
                        ('widths', widths)):
        if index.shape != (rows,):
            raise ValueError(f'{name} must carry one entry per span row')
    lengths = lengths.to(device=scratch.device)
    prefixes = prefixes.to(device=scratch.device)
    heads = scratch.shape[1]
    parts = {key: [] for key in ('head_ent', 'head_sink', 'head_prompt',
                                 'head_recency')}
    start = 0
    runs = []
    while start < rows:
        stop = start
        width, prefix = int(widths[start]), int(prefixes[start])
        while stop < rows and int(widths[stop]) == width \
                and int(prefixes[stop]) == prefix:
            stop += 1
        runs.append((start, stop, width, prefix))
        start = stop
    for start, stop, width, prefix in runs:
        run = canonical_rows(scratch[start:stop], width)
        run_lengths = lengths[start:stop]
        pos = torch.arange(width, device=scratch.device)
        valid = pos[None, :] < run_lengths[:, None]
        a64 = run.double()
        norm = a64 / a64.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        safe = norm.clamp_min(1e-30)
        parts['head_ent'].append(
            -(safe * safe.log() * valid[:, None, :]).sum(dim=-1)
            / run_lengths.clamp_min(2).double().log()[:, None])
        parts['head_sink'].append(norm[:, :, 0])
        cutoff = (run_lengths.double() * 0.8).long().clamp_min(1)
        head_total = a64.sum(dim=-1).clamp_min(1e-12)
        parts['head_recency'].append(
            (a64 * (pos[None, None, :] >= cutoff[:, None, None])
             ).sum(dim=-1) / head_total)
        parts['head_prompt'].append(
            a64[:, :, :prefix].sum(dim=-1) / head_total)
    empty = scratch.new_zeros((0, heads), dtype=torch.float64)
    return {key: torch.cat(values, dim=0) if values else empty
            for key, values in parts.items()}


def row_sum_worst(rowsum: Tensor, slots: Tensor) -> Tensor:
    """Worst |sum - 1| over the selected rows, kept as step evidence."""
    selected = rowsum[slots >= 0]
    if selected.numel() == 0:
        raise ValueError('no selected rows carry a row sum')
    return (selected - 1.0).abs().max()


def _checked_index(index: Tensor, limit: int, name: str) -> Tensor:
    if index.ndim != 1 or index.dtype != torch.int64:
        raise ValueError(f'{name} must be a 1D int64 index tensor')
    if index.numel() and not 0 <= int(index.min()) <= int(index.max()) < limit:
        raise ValueError(f'{name} exceeds the scratch rows')
    return index


def span_products(scratch: Tensor, *, lengths: Tensor, agreement_index: Tensor,
                  spectral_index: Tensor, decay_index: Tensor, entropy_index: Tensor,
                  prefixes: Tensor, widths: Tensor, rowsum: Tensor,
                  slots: Tensor) -> dict[str, Tensor]:
    """Reduce a sampled layer's span scratch to :data:`SPAN_PRODUCT_KEYS`.

    ``lengths``, ``prefixes`` and ``widths`` carry one entry per scratch row;
    the ``*_index`` tensors select scratch rows for each series.
    """
    coverage = coverage_of_selected_rows(scratch, lengths)
    rows = scratch.shape[0]
    agreement_index = _checked_index(agreement_index, rows, 'agreement index')
    spectral_index = _checked_index(spectral_index, rows, 'spectral index')
    decay_index = _checked_index(decay_index, rows, 'decay index')
    entropy_index = _checked_index(entropy_index, rows, 'entropy index')
    mean32 = scratch.mean(dim=1)
    if agreement_index.numel():
        h_mean, h_heads = agreement_pair(scratch[agreement_index])
    else:
        h_mean = scratch.new_zeros((0,), dtype=torch.float64)
        h_heads = scratch.new_zeros((0,), dtype=torch.float64)
    products = dict(
        coverage=coverage,
        h_mean=h_mean,
        h_heads=h_heads,
        spectral_rows=mean32[spectral_index],
        decay_rows=mean32[decay_index],
        row_sum_worst=row_sum_worst(rowsum, slots),
        entropy_rows=entropy_of_rows(scratch[entropy_index],
                                     widths=widths[entropy_index]),
        span_rows=mean32,
    )
    products.update(per_head_summaries(scratch, lengths=lengths, prefixes=prefixes,
                                       widths=widths))
    return products


def series_products(scratch: Tensor, *, agreement_index: Tensor,
                    entropy_index: Tensor, widths: Tensor, rowsum: Tensor,
                    slots: Tensor) -> dict[str, Tensor]:
    """Reduce a series scratch to :data:`SERIES_PRODUCT_KEYS`.

    The scratch carries the entropy and agreement rows together; the agreement
    pair reduces the agreement subset and the entropy series the entropy
    subset, both from the same rounded rows.
    """
    rows = scratch.shape[0]
    agreement_index = _checked_index(agreement_index, rows, 'agreement index')
    entropy_index = _checked_index(entropy_index, rows, 'entropy index')
    if agreement_index.numel():
        h_mean, h_heads = agreement_pair(scratch[agreement_index])
    else:
        h_mean = scratch.new_zeros((0,), dtype=torch.float64)
        h_heads = scratch.new_zeros((0,), dtype=torch.float64)
    return dict(h_mean=h_mean, h_heads=h_heads,
                entropy_rows=entropy_of_rows(
                    scratch[entropy_index], widths=widths[entropy_index]),
                row_sum_worst=row_sum_worst(rowsum, slots))
