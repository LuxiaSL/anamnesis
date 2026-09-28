"""Selected-row schema for the attention statistics capture.

The attention reducer (``anamnesis.extraction.fast.attention``) consumes a
known set of query rows per replayed request: every span row for coverage
and the per-head summaries, strided subsets for head agreement and the
entropy series, the spectral Gram steps and the decay-fit rows. This module
states those selections once, in the reducer's own indexing, and folds them
into per-scheduler-step plans: which packed tokens of a step are selected,
which scratch slot each one owns, and the per-token region metadata
(absolute prompt boundary and recency cutoff) the statistics kernel needs.

Row indices are span-relative: row ``i`` is the query at absolute position
``prompt_length + i``, whose attention row holds ``prompt_length + i + 1``
valid keys. The final position ``end - 1`` produces no attention feature
and is never selected, matching the reducer's span.

The recency cutoff reproduces the reducer's arithmetic exactly: the row
length is promoted to float64, multiplied by 0.8, truncated toward zero
and clamped to at least one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The reducer's stride denominators for the strided series: head agreement
# samples about thirty rows, the entropy series about sixty.
AGREEMENT_TARGET = 30
ENTROPY_TARGET = 60
SPECTRAL_STRIDE = 10
DECAY_MINIMUM_STEPS = 10


def agreement_rows(steps: int) -> tuple[int, ...]:
    return tuple(range(0, steps, max(1, steps // AGREEMENT_TARGET)))


def entropy_rows(steps: int) -> tuple[int, ...]:
    return tuple(range(0, steps, max(1, steps // ENTROPY_TARGET)))


def series_rows(steps: int) -> tuple[int, ...]:
    """Entropy and agreement rows together, in row order.

    The two stride families are not nested: the strides are independent
    integer truncations (511 steps give stride 8 for entropy and 17 for
    agreement), so the union is the smallest selection that lets one
    materialized second pass serve both series on an unsampled layer.
    """
    return tuple(sorted(set(entropy_rows(steps)) | set(agreement_rows(steps))))


def spectral_rows(steps: int) -> tuple[int, ...]:
    rows = tuple(range(0, steps, SPECTRAL_STRIDE))
    if len(rows) < 3:
        rows = tuple(range(steps))
    return rows


def decay_rows(steps: int) -> tuple[int, ...]:
    if steps < DECAY_MINIMUM_STEPS:
        return ()
    return tuple(range(steps // 4, steps, max(1, steps // 10)))[:10]


@dataclass(frozen=True)
class RequestRowSchema:
    """Every row selection of one replayed request, span-relative."""

    prompt_length: int
    end: int
    steps: int
    agreement: tuple[int, ...]
    entropy: tuple[int, ...]
    series: tuple[int, ...]
    spectral: tuple[int, ...]
    decay: tuple[int, ...]
    spectral_width: int


def request_row_schema(*, prompt_length: int, end: int) -> RequestRowSchema:
    if type(prompt_length) is not int or type(end) is not int:
        raise ValueError('request span bounds must be plain integers')
    if not 0 <= prompt_length < end - 1:
        raise ValueError('request span requires 0 <= prompt < end - 1')
    steps = end - 1 - prompt_length
    spectral = spectral_rows(steps)
    return RequestRowSchema(
        prompt_length=prompt_length,
        end=end,
        steps=steps,
        agreement=agreement_rows(steps),
        entropy=entropy_rows(steps),
        series=series_rows(steps),
        spectral=spectral,
        decay=decay_rows(steps),
        spectral_width=prompt_length + spectral[-1] + 1,
    )


def recency_cutoffs(row_lengths: np.ndarray) -> np.ndarray:
    """The reducer's cutoff: trunc(float64(length) * 0.8), at least one."""
    cutoffs = (row_lengths.astype(np.float64) * 0.8).astype(np.int64)
    return np.maximum(cutoffs, 1).astype(np.int32)


@dataclass(frozen=True)
class StepRowPlan:
    """One scheduler step's packed-token metadata and slot assignments.

    Two scratches serve a step. The span scratch holds every selected span row
    scheduled this step and serves the sampled layers, which consume it whole
    for coverage and the per-head summaries. The series scratch holds only the
    entropy and agreement rows (their union) and serves every other layer.

    ``span_slots`` and ``series_slots`` map each packed token to its slot in
    that scratch, or -1. ``span_agreement``, ``span_spectral``,
    ``span_decay`` and ``span_entropy`` index into the span slots;
    ``series_agreement`` and ``series_entropy`` into the series slots. The
    ``*_members`` tuples name each slot's (request_id, span_row), and
    ``agreement_members`` the agreement rows in slot order, for the capture
    layer's reassembly. ``span_prefixes`` carries each span slot's prompt
    boundary, and the ``*_widths`` each slot's request width ``end``, the
    canonical width every row is reduced at.
    """

    total_tokens: int
    prefix_lengths: np.ndarray
    recency_cutoffs: np.ndarray
    span_slots: np.ndarray
    span_members: tuple[tuple[str, int], ...]
    span_lengths: np.ndarray
    span_agreement: np.ndarray
    span_spectral: np.ndarray
    span_decay: np.ndarray
    span_entropy: np.ndarray
    span_prefixes: np.ndarray
    span_widths: np.ndarray
    agreement_members: tuple[tuple[str, int], ...]
    series_slots: np.ndarray
    series_members: tuple[tuple[str, int], ...]
    series_lengths: np.ndarray
    series_agreement: np.ndarray
    series_entropy: np.ndarray
    series_widths: np.ndarray


def step_row_plan(items, schemas: dict[str, RequestRowSchema],
                  selected=None) -> StepRowPlan:
    """Fold one step's scheduled chunks into slot assignments.

    ``items`` carries the packed chunk order proven by the capture layer:
    each entry has ``request_id``, ``start``, ``count`` and ``offset``.
    Slots are assigned in packing order, so identical schedules produce
    identical plans by construction. Every scheduled request contributes
    metadata; with ``selected`` given, only the named requests contribute
    slots (the unretained occupants of a bounded group are scheduled and
    observed, but nothing of theirs is kept).
    """
    total = 0
    for item in items:
        if item['offset'] != total:
            raise ValueError('step items must arrive in packed order')
        if item['count'] <= 0:
            raise ValueError('a scheduled chunk cannot be empty')
        total += item['count']
    if total == 0:
        raise ValueError('a step plan requires scheduled tokens')
    prefix = np.empty(total, dtype=np.int32)
    lengths = np.empty(total, dtype=np.int64)
    span_slots = np.full(total, -1, dtype=np.int32)
    span_members: list[tuple[str, int]] = []
    agreement_members: list[tuple[str, int]] = []
    span_lengths: list[int] = []
    span_agreement: list[int] = []
    span_spectral: list[int] = []
    span_decay: list[int] = []
    series_slots = np.full(total, -1, dtype=np.int32)
    series_members: list[tuple[str, int]] = []
    series_lengths: list[int] = []
    series_agreement: list[int] = []
    series_entropy: list[int] = []
    span_entropy: list[int] = []
    span_prefixes: list[int] = []
    span_widths: list[int] = []
    series_widths: list[int] = []
    for item in items:
        request_id = item['request_id']
        schema = schemas[request_id]
        start, count, offset = item['start'], item['count'], item['offset']
        if start + count > schema.end:
            raise ValueError('a scheduled chunk exceeds the request span')
        positions = np.arange(start, start + count, dtype=np.int64)
        prefix[offset:offset + count] = schema.prompt_length
        lengths[offset:offset + count] = positions + 1
        if selected is not None and request_id not in selected:
            continue
        agreement_set = set(schema.agreement)
        entropy_set = set(schema.entropy)
        spectral_set = set(schema.spectral)
        decay_set = set(schema.decay)
        span_lo = max(start, schema.prompt_length)
        span_hi = min(start + count, schema.end - 1)
        for position in range(span_lo, span_hi):
            row = position - schema.prompt_length
            token = offset + (position - start)
            slot = len(span_members)
            span_slots[token] = slot
            span_members.append((request_id, row))
            span_lengths.append(position + 1)
            span_prefixes.append(schema.prompt_length)
            # The request's materialized width, c + t + 1: the reducer's
            # own row layout and the canonical reduction width.
            span_widths.append(schema.end)
            if row in agreement_set:
                span_agreement.append(slot)
                agreement_members.append((request_id, row))
            if row in entropy_set:
                span_entropy.append(slot)
            if row in agreement_set or row in entropy_set:
                series_slot = len(series_members)
                series_slots[token] = series_slot
                series_members.append((request_id, row))
                series_lengths.append(position + 1)
                series_widths.append(schema.end)
                if row in agreement_set:
                    series_agreement.append(series_slot)
                if row in entropy_set:
                    series_entropy.append(series_slot)
            if row in spectral_set:
                span_spectral.append(slot)
            if row in decay_set:
                span_decay.append(slot)
    return StepRowPlan(
        total_tokens=total,
        prefix_lengths=prefix,
        recency_cutoffs=recency_cutoffs(lengths),
        span_slots=span_slots,
        span_members=tuple(span_members),
        span_lengths=np.asarray(span_lengths, dtype=np.int32),
        span_agreement=np.asarray(span_agreement, dtype=np.int64),
        span_spectral=np.asarray(span_spectral, dtype=np.int64),
        span_decay=np.asarray(span_decay, dtype=np.int64),
        span_entropy=np.asarray(span_entropy, dtype=np.int64),
        span_prefixes=np.asarray(span_prefixes, dtype=np.int32),
        span_widths=np.asarray(span_widths, dtype=np.int32),
        agreement_members=tuple(agreement_members),
        series_slots=series_slots,
        series_members=tuple(series_members),
        series_lengths=np.asarray(series_lengths, dtype=np.int32),
        series_agreement=np.asarray(series_agreement, dtype=np.int64),
        series_entropy=np.asarray(series_entropy, dtype=np.int64),
        series_widths=np.asarray(series_widths, dtype=np.int32),
    )
