"""Signed rows from a decode engine, reduced into the fast lane's features.

A decode engine that signs as it decodes (or scores) hands each finished row over as a
*signed row*: the span's surfaces in the statistics contract of
:mod:`anamnesis.extraction.fast.external_stats`, plus the row's bookkeeping. The engine
names this shape and loads a reducer by dotted path; :class:`EngineReducer` is that
reducer, ``anamnesis.extraction.fast.engine_reducer:EngineReducer``. Nothing here imports
the engine: a signed row is read by its attributes.

**A signed row.** ``start`` is the row's prompt length P and ``positions`` its span length
in steps; with ``seq`` the row's full token ids (prompt and generated, ``S = len(seq)``),
``positions == S - P - 1`` (the last sampled token is never fed). ``kept`` is the number of
positions the engine held; a row with ``kept < positions`` lost its earliest positions
and is refused. ``weights`` is the engine's weights digest
(:func:`anamnesis.extraction.weights_digest.weights_digest`). ``route_sources`` gives, per
layer, the routing sources read before attention and before the MLP; ``final_sources``
the final read's. ``surfaces`` maps names to tensors over the ``kept`` positions:

* ``depth [kept, layers, hidden]``: every layer's block output;
* ``q``, ``k``, ``v``, ``gate`` ``[kept, sampled, ...]``: at the lane's sampled layers, in order;
* ``head_entropy [kept, layers, heads]``, ``mean_entropy [kept, layers]``;
* ``head_scalars [kept, sampled, 3, heads]``: prompt mass, recency mass, sink, per head;
* ``mean_rows [kept, sampled, width]``: the head-mean rows at absolute columns;
* ``logit_stats [kept, 6]``: the output reduction's per-step quantities;
* ``route [kept, layers, 2, route width]`` and ``route_final [kept, route width]``: routing
  weights, the first ``route_sources`` / ``final_sources`` entries live;
* ``committed``: a list, the embedding and then each committed block's output.

**Identity.** A reduced row's lane id is :func:`stats_lane_id` of its lane under the
``engine`` mapping the caller passes (the engine's identity and geometry), so rows from
two engines, or two pin tables, never share an id. A batched row's lane id also names the
instruction set it was reduced on (:func:`~anamnesis.extraction.fast.batched_stats.device_arch`),
so rows reduced on two GPU architectures never share one either. :attr:`EngineReducer.reducer_identity`
names this module's and the statistics reducer's source digests.

**Batched.** By default (``batched=True``) the rows of a batch are
grouped by step-count bucket (and, within it, by surface shapes and routing plan), and each
group's families, attention included, are reduced together, batchmate-invariant
(:mod:`anamnesis.extraction.fast.batched_attention`,
:mod:`anamnesis.extraction.fast.batched_stats`,
:mod:`anamnesis.extraction.fast.batched_families`); those rows carry the hybrid lane id,
never the per-row one. The batched gate top-k breaks ties at the k-th value by the lowest
column index, which the per-row reduction does not define.

**Buckets.** ``buckets`` names the bucket rule
(:data:`~anamnesis.extraction.fast.batched_stats.BUCKET_RULES`,
:func:`~anamnesis.extraction.fast.batched_stats.bucket_width`). A row of ``n`` steps is
padded with zeros to its bucket's canonical width and reduced beside rows of other step
counts in the same bucket; every reduction over its steps is masked to its own ``n``, and
every per-row position (windows, halves, trajectory points, the spectral graph's steps,
epochs) is its own, and so are its short-span placeholders. A row joins a bucket only when
its own lane's feature names are the bucket lane's, name for name; others (a one-step span)
are batched only with rows of their exact step count. The rule joins the lane id.

**Lens.** With a ``lens`` head (:class:`~anamnesis.extraction.fast.lens.LensHead`), every
row also carries the lens family (:mod:`anamnesis.extraction.fast.lens`) at the lane's
sampled layers, after the routing family. It is opt-in: the head's identity joins the engine
mapping the lane id is computed under, so a lens row never shares a lane with a row without
one, and each receipt names the lens's built-in check (``lens_final_check``: the largest gap
between the final read's entropy and the step's own). It needs the batched reduction.

**Refusals.** The first row that cannot be reduced raises :class:`ExternalCaptureError`
naming its index, so a caller can retry the others one at a time. A row is refused when its span is incomplete, its step count disagrees with its
token ids, its weights digest is not the one the reducer was built for, or a surface is
missing or mis-shaped.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from anamnesis.extraction.fast.batched_attention import attention_features
from anamnesis.extraction.fast.batched_families import (
    kvqg_features,
    outputs_features,
    routing_features,
)
from anamnesis.extraction.fast.batched_stats import (
    BUCKET_FLOOR,
    BUCKET_RULE,
    BUCKET_RULES,
    Counts,
    RowSpan,
    batched_receipt,
    block_output_features,
    bucket_width,
    finish_rows,
    take_steps,
)
from anamnesis.extraction.fast.external import ExternalCaptureError, RoutingPoint
from anamnesis.extraction.fast.external_stats import (
    STATS_SOURCE_SHA256,
    ExternalStatsSpan,
    routing_stats_from_routing,
)
from anamnesis.extraction.fast.features import GpuFeatureLane
from anamnesis.extraction.fast.lens import LensError, LensHead, lens_features

ENGINE_REDUCER_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
"""This module's source digest, part of :attr:`EngineReducer.reducer_identity`."""

SCHEMA_STEP_CAP = 8
"""Step counts at or above this share one feature schema (the fast lane's per-step cap)."""

SURFACES = ("depth", "q", "k", "v", "gate", "head_entropy", "mean_entropy", "head_scalars",
            "mean_rows", "logit_stats", "route", "route_final", "committed")
"""The surfaces a signed row must carry."""


@dataclass
class EngineReducedRow:
    """One row's signature: the feature vector (float32), the digest of its names, its
    lane id, and the reduction's receipt and bookkeeping (``meta``). Field for field the
    shape an engine's reducer protocol asks for."""

    features: np.ndarray
    names_digest: str
    lane_id: str
    meta: dict[str, Any] = field(default_factory=dict)


def names_digest(names: Sequence[str]) -> str:
    """The sha256 of a feature-name list, as every receipt records it."""
    return hashlib.sha256(json.dumps(list(names)).encode()).hexdigest()


def routing_points(row: Any) -> list[RoutingPoint]:
    """A signed row's routing in the routing family's order: per layer, the read before
    attention (when it has sources) and the read before the MLP, then the final read."""
    s = row.surfaces
    points: list[RoutingPoint] = []
    for layer, (n_attn, n_mlp) in enumerate(row.route_sources):
        if n_attn:
            points.append(("pre_attn", layer, s["route"][:, layer, 0, :n_attn]))
        points.append(("pre_mlp", layer, s["route"][:, layer, 1, :n_mlp]))
    points.append(("final", -1, s["route_final"][:, :row.final_sources]))
    return points


def _begin_signed_row(lane: GpuFeatureLane, row: Any, token_ids: Sequence[int]):
    """A signed row's checks, its span and its attention statistics reduced; returns
    ``(span, token ids)``."""
    seq = [int(t) for t in token_ids]
    start, positions = int(row.start), int(row.positions)
    if int(row.kept) != positions:
        raise ExternalCaptureError(
            f"incomplete span: {row.kept} of {positions} positions kept")
    if positions != len(seq) - start - 1:
        raise ExternalCaptureError(
            f"the row has {positions} steps but its token ids give {len(seq) - start - 1}")
    s = row.surfaces
    missing = [name for name in SURFACES if name not in s]
    if missing:
        raise ExternalCaptureError(f"the signed row lacks surfaces {missing}")
    depth = s["depth"]
    span = ExternalStatsSpan(lane, start=start, end=len(seq), num_layers=int(depth.shape[1]))
    scalars = s["head_scalars"]
    span.attention_stats(head_entropy=s["head_entropy"].float(),
                         mean_entropy=s["mean_entropy"].float(),
                         head_prompt_mass=scalars[:, :, 0].float(),
                         head_recency_mass=scalars[:, :, 1].float(),
                         head_sink=scalars[:, :, 2].float(),
                         mean_rows=s["mean_rows"].float())
    return span, seq


def _finish_arguments(lane: GpuFeatureLane, row: Any, seq: list[int],
                      engine: Mapping[str, Any]) -> dict[str, Any]:
    """The keyword arguments a signed row's span is finished with."""
    s = row.surfaces
    sampled = list(lane.config.sampled_layers)
    per_layer = {name: {layer: s[name][:, j] for j, layer in enumerate(sampled)}
                 for name in ("k", "v", "q", "gate")}
    return dict(hidden=s["depth"].permute(1, 0, 2).contiguous(), keys=per_layer["k"],
                values=per_layer["v"], queries=per_layer["q"], gates=per_layer["gate"],
                token_ids=seq, capture_identity=dict(engine),
                logit_stats=s["logit_stats"], committed=list(s["committed"]),
                routing_stats=routing_stats_from_routing(routing_points(row), sampled))


def reduce_signed_row(lane: GpuFeatureLane, row: Any, token_ids: Sequence[int],
                      engine: Mapping[str, Any]):
    """One signed row through :class:`ExternalStatsSpan`; returns its ``GpuFeatureResult``.

    Raises
    ------
    ExternalCaptureError
        When the span is incomplete, the step count disagrees with the token ids, or a
        surface is missing or mis-shaped (the span's own checks).
    """
    span, seq = _begin_signed_row(lane, row, token_ids)
    return span.finish(**_finish_arguments(lane, row, seq, engine))


_NONFINITE = "the capture holds a nonfloating or nonfinite tensor"

_FINITE_REFUSALS = dict(
    head_entropy="head_entropy holds a nonfinite value",
    mean_entropy="mean_entropy holds a nonfinite value",
    head_scalars="head_scalars holds a nonfinite value",
    mean_rows="mean_rows holds a nonfinite value",
    depth=_NONFINITE,
    k=_NONFINITE,
    v=_NONFINITE,
    q=_NONFINITE,
    gate=_NONFINITE,
    logit_stats="logit_stats must be finite float32 or float64 [{steps}, 6]",
    route="the routing weights hold a nonfinite value",
    route_final="the routing weights hold a nonfinite value",
)
"""The surfaces whose finiteness a batch checks, in the order the per-row reduction reads
them, and the refusal of a row that fails."""

_SCALARS = ("head_prompt_mass", "head_recency_mass", "head_sink")
"""The per-row statistics contract's names of ``head_scalars``' three columns."""


def _shape_key(row: Any, width: int | None = None) -> tuple:
    """What rows must share to be reduced in one batch: every stacked surface's shape (its
    steps padded to ``width``, when given) and the routing plan."""
    s = row.surfaces

    def shape(x) -> tuple:
        dims = tuple(x.shape)
        return dims if width is None or not dims else (int(width),) + dims[1:]

    return (tuple((name, shape(s[name])) for name in _FINITE_REFUSALS),
            tuple(shape(c) for c in s["committed"]),
            tuple(tuple(int(n) for n in pair) for pair in row.route_sources),
            int(row.final_sources))


def _pad_stack(tensors: Sequence[Any], width: int | None, device) -> torch.Tensor:
    """``tensors`` stacked on ``device``, each zero-padded along its steps to ``width``."""
    moved = [t.to(device) for t in tensors]
    if width is None or all(t.shape[0] == width for t in moved):
        return torch.stack(moved)
    out = moved[0].new_zeros((len(moved), int(width), *moved[0].shape[1:]))
    for b, t in enumerate(moved):
        out[b, :t.shape[0]] = t
    return out


def validate_batch(rows: Sequence[Any], device,
                   width: int | None = None) -> tuple[dict[str, Any], list[int]]:
    """The surfaces of rows that share a :func:`_shape_key`, stacked on ``device`` (each row's
    steps zero-padded to ``width``, when given), and the indices of rows holding a nonfinite
    value, found with one device-to-host copy for the whole batch."""
    stacked = {name: _pad_stack([row.surfaces[name] for row in rows], width, device)
               for name in _FINITE_REFUSALS}
    flags = torch.stack([torch.isfinite(x).reshape(len(rows), -1).all(dim=1)
                         for x in stacked.values()]).cpu()
    return stacked, [b for b in range(len(rows)) if not bool(flags[:, b].all())]


def _first_refusal(row: Any) -> str:
    """The refusal of a row :func:`validate_batch` found nonfinite, as the per-row reduction
    words it."""
    for name, message in _FINITE_REFUSALS.items():
        x = row.surfaces[name]
        if bool(torch.isfinite(x).all()):
            continue
        if name == "head_scalars":
            for i, column in enumerate(_SCALARS):
                if not bool(torch.isfinite(x[:, :, i]).all()):
                    return f"{column} holds a nonfinite value"
        return message.format(steps=int(row.positions))
    return _NONFINITE


def _batched_points(stacked: Mapping[str, Any], row: Any) -> list[RoutingPoint]:
    """:func:`routing_points` over the batch's stacked routing weights."""
    route, final = stacked["route"], stacked["route_final"]
    points: list[RoutingPoint] = []
    for layer, (n_attn, n_mlp) in enumerate(row.route_sources):
        if n_attn:
            points.append(("pre_attn", layer, route[:, :, layer, 0, :n_attn]))
        points.append(("pre_mlp", layer, route[:, :, layer, 1, :n_mlp]))
    points.append(("final", -1, final[:, :, :row.final_sources]))
    return points


def _check_signed_row(lane: GpuFeatureLane, row: Any, token_ids: Sequence[int]) -> RowSpan:
    """A signed row's checks for the batched reduction, which read shapes and settings only
    (the batch checks finiteness at once); returns its span.

    Raises
    ------
    ExternalCaptureError
        As the per-row reduction does for an incomplete span, a step count that disagrees
        with the token ids, a missing surface, or a mis-shaped one.
    """
    seq = [int(t) for t in token_ids]
    start, positions = int(row.start), int(row.positions)
    if int(row.kept) != positions:
        raise ExternalCaptureError(
            f"incomplete span: {row.kept} of {positions} positions kept")
    if positions != len(seq) - start - 1:
        raise ExternalCaptureError(
            f"the row has {positions} steps but its token ids give {len(seq) - start - 1}")
    s = row.surfaces
    missing = [name for name in SURFACES if name not in s]
    if missing:
        raise ExternalCaptureError(f"the signed row lacks surfaces {missing}")
    if start < 1 or positions < 1:
        raise ExternalCaptureError(f"invalid span start={start} end={len(seq)}")
    if not torch.are_deterministic_algorithms_enabled():
        raise ExternalCaptureError("the reduction runs with deterministic algorithms on")
    if lane.device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
        raise ExternalCaptureError("the reduction runs with TF32 off")
    depth = s["depth"]
    if not isinstance(depth, torch.Tensor) or depth.ndim != 3 or depth.shape[0] != positions:
        raise ExternalCaptureError(
            f"depth must be [{positions}, layers, width], got "
            f"{tuple(depth.shape) if isinstance(depth, torch.Tensor) else type(depth).__name__}")
    layers = int(depth.shape[1])
    sampled = list(lane.config.sampled_layers)
    if not set(sampled) <= set(range(layers)):
        raise ExternalCaptureError("a sampled layer lies outside the capture")
    he = s["head_entropy"]
    if not isinstance(he, torch.Tensor) or he.ndim != 3:
        raise ExternalCaptureError("head_entropy must be [steps, layers, heads]")
    heads = int(he.shape[2])
    expected = dict(head_entropy=(positions, layers, heads), mean_entropy=(positions, layers),
                    head_scalars=(positions, len(sampled), 3, heads))
    for name, shape in expected.items():
        if not isinstance(s[name], torch.Tensor) or tuple(s[name].shape) != shape:
            raise ExternalCaptureError(
                f"{name} must be {list(shape)}, got {list(getattr(s[name], 'shape', ()))}")
    mr = s["mean_rows"]
    if (not isinstance(mr, torch.Tensor) or mr.ndim != 3
            or tuple(mr.shape[:2]) != (positions, len(sampled)) or mr.shape[2] < len(seq) - 1):
        raise ExternalCaptureError(
            f"mean_rows must be [{positions}, {len(sampled)}, >={len(seq) - 1}], "
            f"got {tuple(getattr(mr, 'shape', ()))}")
    if (lane.pm.ndim != 3 or lane.pm.shape[0] < layers + 1
            or lane.pm.shape[1] < start + positions or lane.pm.shape[2] != depth.shape[2]):
        raise ExternalCaptureError("the lane's positional means do not cover this capture")
    for name, ndim in (("k", 4), ("v", 4), ("q", 4), ("gate", 3)):
        x = s[name]
        if (not isinstance(x, torch.Tensor) or x.ndim != ndim or x.shape[0] != positions
                or x.shape[1] != len(sampled)):
            raise ExternalCaptureError(
                f"{name} must be [{positions}, {len(sampled)}, ...], got "
                f"{tuple(getattr(x, 'shape', ()))}")
    for name in ("depth", "k", "v", "q", "gate", "head_entropy", "mean_entropy",
                 "head_scalars", "mean_rows"):
        if not s[name].is_floating_point():
            raise ExternalCaptureError(_NONFINITE)
    ls = s["logit_stats"]
    if (not isinstance(ls, torch.Tensor) or tuple(ls.shape) != (positions, 6)
            or ls.dtype not in (torch.float32, torch.float64)):
        raise ExternalCaptureError(f"logit_stats must be finite float32 or float64 [{positions}, 6]")
    return RowSpan(start=start, end=len(seq), token_ids=seq)


def _reduce_uniform(lane: GpuFeatureLane, rows: Sequence[Any], spans: Sequence[RowSpan],
                    engine: Mapping[str, Any], width: int | None = None,
                    bucket_rule: str = BUCKET_RULE, lens: LensHead | None = None) -> list[Any]:
    """Rows that share a :func:`_shape_key` (padded to ``width``), every family batched; with
    ``lens``, the lens family after them."""
    results: list[Any] = [None] * len(rows)
    stacked, bad = validate_batch(rows, lane.device, width)
    for b in bad:
        results[b] = ExternalCaptureError(_first_refusal(rows[b]))
    if bad:
        keep = [b for b in range(len(rows)) if b not in set(bad)]
        if not keep:
            return results
        index = torch.as_tensor(keep, device=lane.device)
        stacked = {name: x.index_select(0, index) for name, x in stacked.items()}
    else:
        keep = list(range(len(rows)))
    sampled = list(lane.config.sampled_layers)
    first = rows[keep[0]]
    starts = [spans[b].start for b in keep]
    steps = int(stacked["depth"].shape[1])
    padded = width is not None and bucket_rule != "exact-steps" and steps >= BUCKET_FLOOR
    counts = Counts.of([int(rows[b].positions) for b in keep], lane.device, steps, padded)
    committed = [_pad_stack([rows[b].surfaces["committed"][c] for b in keep], width, lane.device)
                 for c in range(len(first.surfaces["committed"]))]
    try:
        attention = attention_features(lane, stacked["head_entropy"], stacked["mean_entropy"],
                                       stacked["head_scalars"], stacked["mean_rows"], starts,
                                       counts)
        blocks = block_output_features(lane, stacked["depth"], starts, attention.laplacians,
                                       counts)
        kvqg = kvqg_features(lane, stacked["k"], stacked["v"], stacked["q"], stacked["gate"],
                             attention.per_head, counts)
        outputs, logprob = outputs_features(lane, stacked["logit_stats"], counts)
        routing_names, routing_values = routing_features(
            sampled, _batched_points(stacked, first), committed, counts)
        depth = stacked["depth"]
        knnlm = None
        if lane.config.enable_knnlm_baseline:
            final = depth[:, :, depth.shape[2] - 1]  # [B, steps, hidden]: the last layer
            knnlm = (final[:, steps - 1] if counts.uniform(steps)
                     else take_steps(final, (counts.dev - 1)[:, None])[:, 0])
        lensed = None
        if lens is not None:
            chosen = torch.zeros(len(keep), steps, dtype=torch.long)
            for i, b in enumerate(keep):
                n, start = int(rows[b].positions), spans[b].start
                chosen[i, :n] = torch.as_tensor(
                    [int(t) for t in spans[b].token_ids[start + 1:start + 1 + n]])
            step_entropy = stacked["logit_stats"][:, :, 0]
            lensed = lens_features(lens, depth, committed, first.route_sources, sampled,
                                   chosen.to(lane.device), step_entropy,
                                   counts.lengths(step_entropy))
            engine = dict(engine, lens=lens.identity())
        finished = finish_rows(
            lane, [spans[b] for b in keep], attention.names + blocks.names + kvqg.names
            + outputs.names,
            torch.cat([attention.values, blocks.values, kvqg.values, outputs.values], dim=1),
            logprob, knnlm, engine, num_layers=int(depth.shape[2]),
            routing_names=tuple(routing_names), routing_values=routing_values,
            retained_attention_summary_bytes=attention.retained_bytes,
            bucket_rule=bucket_rule, bucket_width=steps)
        if lensed is not None:
            finished = _with_lens(finished, lensed)
    except (ExternalCaptureError, LensError, ValueError) as err:
        err = err if isinstance(err, ExternalCaptureError) else ExternalCaptureError(str(err))
        for b in keep:
            results[b] = err
        return results
    for b, result in zip(keep, finished, strict=True):
        results[b] = result
    return results


def _with_lens(finished: Sequence[Any], lensed: Any) -> list[Any]:
    """Each finished row with the lens family's values after its own, and its receipt naming
    the lens family and the row's built-in check. A row whose lens values are nonfinite is
    refused, as any family's would be."""
    values = lensed.values.float().cpu().numpy()
    check = lensed.final_check.double().cpu().tolist()
    names: tuple[str, ...] | None = None
    out: list[Any] = []
    for b, result in enumerate(finished):
        if isinstance(result, ExternalCaptureError):
            out.append(result)
            continue
        if not np.isfinite(values[b]).all():
            out.append(ExternalCaptureError("the lens produced a nonfinite feature"))
            continue
        if names is None or names[:len(result.feature_names)] != result.feature_names:
            names = tuple(result.feature_names) + tuple(lensed.names)
        receipt = dict(result.metadata or {})
        receipt["reductions"] = dict(receipt.get("reductions", {}), lens="batched")
        receipt["feature_schema_sha256"] = hashlib.sha256(json.dumps(list(names)).encode()).hexdigest()
        receipt["lens_final_check"] = float(check[b])
        out.append(replace(result, features=np.concatenate([result.features, values[b]]),
                           feature_names=names, metadata=receipt))
    return out


def reduce_signed_rows_hybrid(lane: GpuFeatureLane, rows: Sequence[Any],
                              token_ids: Sequence[Sequence[int]],
                              engine: Mapping[str, Any],
                              bucket_rule: str = BUCKET_RULE,
                              width: int | None = None,
                              max_group_rows: int | None = None,
                              lens: LensHead | None = None) -> list[Any]:
    """Signed rows of one lane, every family batched across the rows of one step-count
    bucket (:func:`attention_features`, :func:`block_output_features`, :func:`kvqg_features`,
    :func:`outputs_features`, :func:`routing_features`) and assembled by
    :func:`finish_rows`. Rows are grouped by their bucket's canonical width under
    ``bucket_rule`` (or all at ``width``, when given) and padded to it; rows whose surfaces differ in shape (past their steps)
    or routing plan reduce in separate batches. Returns, per row in order, its
    ``GpuFeatureResult`` or the :class:`ExternalCaptureError` that refused it; a refused
    row is left out of the batch, which changes no other row's bytes.

    ``max_group_rows`` caps the rows reduced together: a larger group reduces in chunks of
    at most that many rows, in order. The reduction is batchmate-invariant, so a chunk's rows
    have the bytes they have in the whole group, and the cap is not part of the lane. It
    bounds the memory a batch takes (the stacked surfaces, padded to the bucket's width) at
    the cost of more launches. ``lens`` adds the lens family (see the module docstring)."""
    if max_group_rows is not None and int(max_group_rows) < 1:
        raise ValueError(f"max_group_rows must be at least 1, not {max_group_rows}")
    if bucket_rule not in BUCKET_RULES:
        raise ValueError(f"bucket rule must be one of {BUCKET_RULES}, not {bucket_rule!r}")
    results: list[Any] = [None] * len(rows)
    spans: dict[int, RowSpan] = {}
    for i, (row, seq) in enumerate(zip(rows, token_ids, strict=True)):
        try:
            spans[i] = _check_signed_row(lane, row, seq)
        except ExternalCaptureError as err:
            results[i] = err
    groups: dict[tuple, list[int]] = {}
    for i in sorted(spans):
        padded_to = width if width is not None else bucket_width(int(rows[i].positions), bucket_rule)
        groups.setdefault((padded_to, _shape_key(rows[i], padded_to)), []).append(i)
    for (padded_to, _), group in groups.items():
        step = len(group) if max_group_rows is None else int(max_group_rows)
        for lo in range(0, len(group), step):
            members = group[lo:lo + step]
            reduced = _reduce_uniform(lane, [rows[i] for i in members],
                                      [spans[i] for i in members], engine, padded_to, bucket_rule,
                                      lens)
            for i, result in zip(members, reduced, strict=True):
                results[i] = result
    return results


class EngineReducer:
    """Signed rows into fast-lane signatures, one lane per capped step count.

    Built from a model preset and its calibration (``preset``, ``calib_dir``,
    ``device``), the way a replay lane is; or, for tests and callers that already hold
    lanes, from ``lane_for(steps) -> GpuFeatureLane``. ``expected_weights`` is the weights
    digest every row must carry; built from a preset it defaults to the preset's
    ``checkpoint_sha256`` (none declared: rows are not checked).

    ``batched`` selects the hybrid reduction: rows of one step-count bucket reduce every
    family together (:func:`reduce_signed_rows_hybrid`), batchmate-invariant, under the
    bucket rule ``buckets`` (default
    :data:`~anamnesis.extraction.fast.batched_stats.BUCKET_RULE`). Its rows carry the hybrid
    lane id (:func:`~anamnesis.extraction.fast.batched_stats.hybrid_lane_id`) and the reducer
    identity names the batched module's digests and the bucket rule. ``max_group_rows``
    caps the rows reduced together, bounding a batch's memory; it changes no row's bytes, so
    it is not part of the lane. The default is the
    batched reduction; ``batched=False`` reduces row by row under the per-row statistics
    lane id.

    ``lens`` (a :class:`~anamnesis.extraction.fast.lens.LensHead`) or ``lens_checkpoint`` (a
    checkpoint path, read by :meth:`~anamnesis.extraction.fast.lens.LensHead.from_checkpoint`
    as the weights ``expected_weights`` names) turns on the lens family. The head must belong
    to ``expected_weights``, and the lens needs the batched reduction.
    """

    def __init__(self, *, preset: str | None = None, calib_dir: str | Path | None = None,
                 device: str = "cuda",
                 lane_for: Callable[[int], GpuFeatureLane] | None = None,
                 expected_weights: str | None = None, batched: bool = True,
                 buckets: str = BUCKET_RULE, max_group_rows: int | None = None,
                 lens: LensHead | None = None, lens_checkpoint: str | Path | None = None) -> None:
        if (lane_for is None) == (preset is None):
            raise ValueError("give a preset (with calib_dir) or lane_for, one of the two")
        if buckets not in BUCKET_RULES:
            raise ValueError(f"buckets must be one of {BUCKET_RULES}, not {buckets!r}")
        self._lanes: dict[int, GpuFeatureLane] = {}
        self.batched = bool(batched)
        self.buckets = buckets
        if max_group_rows is not None and int(max_group_rows) < 1:
            raise ValueError(f"max_group_rows must be at least 1, not {max_group_rows}")
        self.max_group_rows = None if max_group_rows is None else int(max_group_rows)
        if lane_for is not None:
            self._lane_for = lane_for
            self.expected_weights = expected_weights
            self.preset_name = None
        else:
            if calib_dir is None:
                raise ValueError("a preset needs its calib_dir")
            self._build_from_preset(preset, Path(calib_dir), device)
            self.expected_weights = expected_weights or self._preset.checkpoint_sha256
        self.lens = self._lens(lens, lens_checkpoint)
        identity = dict(engine_reducer_source_sha256=ENGINE_REDUCER_SOURCE_SHA256,
                        stats_source_sha256=STATS_SOURCE_SHA256)
        if self.lens is not None:
            identity.update(lens=self.lens.identity())
        if self.batched:
            batched_identity = batched_receipt("cpu", buckets)
            for by_device in ("row_backend", "eigvalsh", "device_arch"):  # a row's lane id names its device's
                batched_identity.pop(by_device)
            identity.update(batched=batched_identity)
        self.reducer_identity = (
            "anamnesis.extraction.fast.engine_reducer:EngineReducer@"
            + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest())

    def _lens(self, lens: LensHead | None, checkpoint: str | Path | None) -> LensHead | None:
        if lens is not None and checkpoint is not None:
            raise ValueError("give a lens head or a lens checkpoint, not both")
        if lens is None and checkpoint is None:
            return None
        if not self.batched:
            raise ValueError("the lens family needs the batched reduction")
        if not self.expected_weights:
            raise ValueError("the lens family needs expected_weights: a head reads one model")
        if checkpoint is not None:
            lens = LensHead.from_checkpoint(checkpoint, weights=self.expected_weights)
        if lens.weights != self.expected_weights:
            raise ValueError(f"the lens head belongs to weights {lens.weights}, the rows to "
                             f"{self.expected_weights}")
        return lens

    def _build_from_preset(self, preset: str, calib_dir: Path, device: str) -> None:
        from anamnesis.config import resolve_preset
        from anamnesis.extraction.calibration import load_calibration
        from anamnesis.extraction.calibration_extension import read_calibration_identity
        from anamnesis.extraction.fast.schema import resolve_gpu_schema
        from anamnesis.extraction.replay_config import native_replay_configs

        self._preset = resolve_preset(preset)
        self.preset_name = self._preset.name
        num_layers = int(self._preset.num_layers)
        extraction, families = native_replay_configs(self._preset)
        positional_means, components, mean = load_calibration(calib_dir, True)
        calibration_sha256 = read_calibration_identity(calib_dir).sha256

        def lane_for(steps: int) -> GpuFeatureLane:
            names = resolve_gpu_schema(num_layers, int(steps), extraction, families,
                                       components).feature_names
            return GpuFeatureLane(extraction, families, list(names), positional_means,
                                  components, mean, device=device,
                                  calibration_sha256=calibration_sha256, replay_path="full")

        self._lane_for = lane_for

    def _group_width(self, steps: int) -> int:
        """The canonical width a row of ``steps`` steps reduces at: its bucket's, when its own
        lane's names are the bucket lane's, name for name; else its exact step count."""
        width = bucket_width(steps, self.buckets)
        if width != steps and self.lane(steps).names != self.lane(width).names:
            return steps
        return width

    def lane(self, steps: int) -> GpuFeatureLane:
        """The lane for a span of ``steps`` steps (one per capped step count)."""
        key = min(int(steps), SCHEMA_STEP_CAP)
        if key not in self._lanes:
            self._lanes[key] = self._lane_for(int(steps))
        return self._lanes[key]

    def reduce(self, rows: Sequence[Any], *, token_ids: Sequence[Sequence[int]],
               engine: Mapping[str, Any]) -> list[EngineReducedRow]:
        """One :class:`EngineReducedRow` per signed row, in order.

        With ``batched``, the rows are grouped by step-count bucket and each group's families
        are reduced together; a row's bytes do not depend on its batchmates.

        Raises
        ------
        ExternalCaptureError
            For the first row that cannot be reduced, naming its index in the batch.
        """
        if len(rows) != len(token_ids):
            raise ExternalCaptureError(
                f"{len(rows)} rows but {len(token_ids)} token id lists")
        results: list[Any] = [None] * len(rows)
        groups: dict[int, list[int]] = {}
        for i, row in enumerate(rows):
            if self.expected_weights and getattr(row, "weights", "") != self.expected_weights:
                results[i] = ExternalCaptureError(
                    f"weights digest {getattr(row, 'weights', '') or '(none)'} is not the "
                    f"reducer's {self.expected_weights}")
                if not self.batched:
                    break
                continue
            if not self.batched:
                try:
                    results[i] = reduce_signed_row(self.lane(int(row.positions)), row,
                                                   token_ids[i], engine)
                except ExternalCaptureError as err:
                    results[i] = err
                    break
                continue
            groups.setdefault(self._group_width(int(row.positions)), []).append(i)
        for width, members in groups.items():
            reduced = reduce_signed_rows_hybrid(
                self.lane(width), [rows[i] for i in members],
                [token_ids[i] for i in members], engine, self.buckets, width,
                max_group_rows=self.max_group_rows, lens=self.lens)
            for i, result in zip(members, reduced, strict=True):
                results[i] = result
        for i, result in enumerate(results):
            if isinstance(result, ExternalCaptureError):
                raise ExternalCaptureError(f"row {i}: {result}") from result
        out = []
        digests: dict[int, tuple[Any, str]] = {}  # rows of one batch share one names tuple
        for row, result in zip(rows, results, strict=True):
            receipt = dict(result.metadata or {})
            hit = digests.get(id(result.feature_names))
            if hit is None or hit[0] is not result.feature_names:
                hit = (result.feature_names, names_digest(result.feature_names))
                digests[id(result.feature_names)] = hit
            out.append(EngineReducedRow(
                features=np.asarray(result.features, dtype=np.float32),
                names_digest=hit[1],
                lane_id=str(receipt["lane_id"]),
                meta=dict(steps=int(row.positions), feature_names=list(result.feature_names),
                          mean_logprob=float(result.mean_logprob),
                          weights=getattr(row, "weights", ""), lever_slot=int(row.lever_slot),
                          steered=bool(row.steered), reducer_identity=self.reducer_identity,
                          receipt=receipt)))
        return out
