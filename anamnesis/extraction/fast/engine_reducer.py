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
two engines, or two pin tables, never share an id. :attr:`EngineReducer.reducer_identity`
names this module's and the statistics reducer's source digests.

**Refusals.** A batch is reduced row by row; the first row that cannot be reduced raises
:class:`ExternalCaptureError` naming its index, so a caller can retry the others one at a
time. A row is refused when its span is incomplete, its step count disagrees with its
token ids, its weights digest is not the one the reducer was built for, or a surface is
missing or mis-shaped.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from torch import Tensor

from anamnesis.extraction.fast.external import ExternalCaptureError, RoutingPoint
from anamnesis.extraction.fast.external_stats import (
    STATS_SOURCE_SHA256,
    ExternalStatsSpan,
    routing_stats_from_routing,
)
from anamnesis.extraction.fast.features import GpuFeatureLane

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


def reduce_signed_row(lane: GpuFeatureLane, row: Any, token_ids: Sequence[int],
                      engine: Mapping[str, Any]):
    """One signed row through :class:`ExternalStatsSpan`; returns its ``GpuFeatureResult``.

    Raises
    ------
    ExternalCaptureError
        When the span is incomplete, the step count disagrees with the token ids, or a
        surface is missing or mis-shaped (the span's own checks).
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
    sampled = list(lane.config.sampled_layers)
    depth = s["depth"]
    span = ExternalStatsSpan(lane, start=start, end=len(seq), num_layers=int(depth.shape[1]))
    scalars = s["head_scalars"]
    span.attention_stats(head_entropy=s["head_entropy"].float(),
                         mean_entropy=s["mean_entropy"].float(),
                         head_prompt_mass=scalars[:, :, 0].float(),
                         head_recency_mass=scalars[:, :, 1].float(),
                         head_sink=scalars[:, :, 2].float(),
                         mean_rows=s["mean_rows"].float())
    per_layer = {name: {layer: s[name][:, j] for j, layer in enumerate(sampled)}
                 for name in ("k", "v", "q", "gate")}
    return span.finish(hidden=depth.permute(1, 0, 2).contiguous(), keys=per_layer["k"],
                       values=per_layer["v"], queries=per_layer["q"], gates=per_layer["gate"],
                       token_ids=seq, capture_identity=dict(engine),
                       logit_stats=s["logit_stats"], committed=list(s["committed"]),
                       routing_stats=routing_stats_from_routing(routing_points(row), sampled))


class EngineReducer:
    """Signed rows into fast-lane signatures, one lane per capped step count.

    Built from a model preset and its calibration (``preset``, ``calib_dir``,
    ``device``), the way a replay lane is; or, for tests and callers that already hold
    lanes, from ``lane_for(steps) -> GpuFeatureLane``. ``expected_weights`` is the weights
    digest every row must carry; built from a preset it defaults to the preset's
    ``checkpoint_sha256`` (none declared: rows are not checked).
    """

    def __init__(self, *, preset: str | None = None, calib_dir: str | Path | None = None,
                 device: str = "cuda",
                 lane_for: Callable[[int], GpuFeatureLane] | None = None,
                 expected_weights: str | None = None) -> None:
        if (lane_for is None) == (preset is None):
            raise ValueError("give a preset (with calib_dir) or lane_for, one of the two")
        self._lanes: dict[int, GpuFeatureLane] = {}
        if lane_for is not None:
            self._lane_for = lane_for
            self.expected_weights = expected_weights
            self.preset_name = None
        else:
            if calib_dir is None:
                raise ValueError("a preset needs its calib_dir")
            self._build_from_preset(preset, Path(calib_dir), device)
            self.expected_weights = expected_weights or self._preset.checkpoint_sha256
        self.reducer_identity = (
            "anamnesis.extraction.fast.engine_reducer:EngineReducer@"
            + hashlib.sha256(json.dumps(dict(
                engine_reducer_source_sha256=ENGINE_REDUCER_SOURCE_SHA256,
                stats_source_sha256=STATS_SOURCE_SHA256), sort_keys=True).encode()).hexdigest())

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

    def lane(self, steps: int) -> GpuFeatureLane:
        """The lane for a span of ``steps`` steps (one per capped step count)."""
        key = min(int(steps), SCHEMA_STEP_CAP)
        if key not in self._lanes:
            self._lanes[key] = self._lane_for(int(steps))
        return self._lanes[key]

    def reduce(self, rows: Sequence[Any], *, token_ids: Sequence[Sequence[int]],
               engine: Mapping[str, Any]) -> list[EngineReducedRow]:
        """One :class:`EngineReducedRow` per signed row, in order.

        Raises
        ------
        ExternalCaptureError
            For the first row that cannot be reduced, naming its index in the batch.
        """
        if len(rows) != len(token_ids):
            raise ExternalCaptureError(
                f"{len(rows)} rows but {len(token_ids)} token id lists")
        out = []
        for i, (row, seq) in enumerate(zip(rows, token_ids, strict=True)):
            try:
                if self.expected_weights and getattr(row, "weights", "") != self.expected_weights:
                    raise ExternalCaptureError(
                        f"weights digest {getattr(row, 'weights', '') or '(none)'} is not the "
                        f"reducer's {self.expected_weights}")
                result = reduce_signed_row(self.lane(int(row.positions)), row, seq, engine)
            except ExternalCaptureError as err:
                raise ExternalCaptureError(f"row {i}: {err}") from err
            receipt = dict(result.metadata or {})
            out.append(EngineReducedRow(
                features=np.asarray(result.features, dtype=np.float32),
                names_digest=names_digest(result.feature_names),
                lane_id=str(receipt["lane_id"]),
                meta=dict(steps=int(row.positions), feature_names=list(result.feature_names),
                          mean_logprob=float(result.mean_logprob),
                          weights=getattr(row, "weights", ""), lever_slot=int(row.lever_slot),
                          steered=bool(row.steered), reducer_identity=self.reducer_identity,
                          receipt=receipt)))
        return out
