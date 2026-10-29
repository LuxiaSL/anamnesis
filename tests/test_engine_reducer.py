"""Signed rows, as a decode engine hands them over, reduced by EngineReducer.

A signed row is built from a synthetic forward in the engine's layout. Its signature must
match :class:`~anamnesis.extraction.fast.external.ExternalSpan` reducing the same
forward's weights: within the statistics contract's tolerance everywhere, and byte for
byte on the output and routing families. The rest: incomplete spans, wrong step counts,
foreign weights and missing surfaces are refused naming the row; the lane id follows
the engine identity; and the reducer satisfies the engine's protocol when the engine is
importable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
import torch

from anamnesis.extraction.fast.engine_reducer import (
    EngineReducer,
    names_digest,
    routing_points,
)
from anamnesis.extraction.fast.external import ExternalCaptureError
from anamnesis.extraction.fast.external_stats import (
    LANE_PREFIX,
    logit_stats_from_logits,
    stats_from_weights,
)
from test_fast_external_stats import TOLERANCE, forward_weights, weights_reduction
from test_vllm_readout import N_LAYERS, make_example

ENGINE = {"producer": "synthetic-engine", "compile_mode": "default", "kernel_pins": "0" * 64,
          "width": 128, "weights": "w" * 64}
WEIGHTS = "w" * 64


@pytest.fixture(autouse=True)
def deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


@dataclass
class SignedRow:
    """The engine's signed row, by the attributes the reducer reads."""

    positions: int
    kept: int
    start: int
    lever_slot: int
    steered: bool
    route_sources: tuple[tuple[int, int], ...]
    final_sources: int
    weights: str = ""
    surfaces: dict[str, Any] = field(default_factory=dict)


def signed_example(steps=16, start=4, sampled=(0, 1, 2), seed=0):
    """(lane, signed row, token ids, the forward's weights, the capture) in engine layout."""
    lane, capture, kwargs, weights = make_example(torch.float32, steps=steps, start=start,
                                                  sampled_layers=sampled)
    begin, end = kwargs["start"], kwargs["end"]
    full = forward_weights(weights, begin, end, torch.float32)
    stats = stats_from_weights(full, begin, end, lane.config.sampled_layers)
    g = torch.Generator().manual_seed(seed)
    route_sources = tuple((layer, layer + 1) for layer in range(N_LAYERS))
    width = N_LAYERS + 2
    route = torch.zeros(steps, N_LAYERS, 2, width)
    for layer, (n_attn, n_mlp) in enumerate(route_sources):
        for point, n in ((0, n_attn), (1, n_mlp)):
            if n:
                w = torch.rand(steps, n, generator=g, dtype=torch.float64)
                route[:, layer, point, :n] = (w / w.sum(1, keepdim=True)).float()
    w = torch.rand(steps, width, generator=g, dtype=torch.float64)
    route_final = (w / w.sum(1, keepdim=True)).float()
    committed = [torch.randn(steps, 8, generator=g) for _ in range(4)]
    order = list(lane.config.sampled_layers)
    surfaces = dict(
        depth=capture["hidden"].permute(1, 0, 2).contiguous(),
        k=torch.stack([capture["keys"][layer] for layer in order], dim=1),
        v=torch.stack([capture["values"][layer] for layer in order], dim=1),
        q=torch.stack([capture["queries"][layer] for layer in order], dim=1),
        gate=torch.stack([capture["gates"][layer] for layer in order], dim=1),
        head_entropy=stats["head_entropy"], mean_entropy=stats["mean_entropy"],
        head_scalars=torch.stack([stats["head_prompt_mass"], stats["head_recency_mass"],
                                  stats["head_sink"]], dim=2),
        mean_rows=stats["mean_rows"],
        logit_stats=logit_stats_from_logits(capture["logits"], capture["chosen"]),
        route=route, route_final=route_final, committed=committed)
    row = SignedRow(positions=steps, kept=steps, start=begin, lever_slot=0, steered=False,
                    route_sources=route_sources, final_sources=width, weights=WEIGHTS,
                    surfaces=surfaces)
    token_ids = [0] * end
    token_ids[begin + 1:end] = capture["chosen"].tolist()
    return lane, row, token_ids, full, capture


def reducer_for(lane, **kwargs):
    return EngineReducer(lane_for=lambda steps: lane, **kwargs)


@pytest.mark.parametrize("steps,start,sampled", [(1, 4, (0, 1, 2)), (3, 4, (0, 2)),
                                                 (16, 4, (0, 2)), (40, 6, (1,))],
                         ids=["steps1", "steps3", "steps16", "steps40"])
def test_a_signed_row_reduces_to_the_weights_features(steps, start, sampled):
    lane, row, token_ids, full, capture = signed_example(steps, start, sampled)
    got = reducer_for(lane, expected_weights=WEIGHTS).reduce([row], token_ids=[token_ids],
                                                             engine=ENGINE)[0]
    reference = weights_reduction(lane, capture, full, start=row.start, end=len(token_ids),
                                  routing=routing_points(row),
                                  committed=row.surfaces["committed"])
    names = list(reference.feature_names)
    assert got.meta["feature_names"] == names
    assert got.names_digest == names_digest(names)
    np.testing.assert_allclose(got.features, reference.features, **TOLERANCE)
    exact = np.array([n.startswith(("logit_", "top1_", "top5_", "mean_chosen", "std_chosen",
                                    "mean_surprise", "std_surprise", "surprise_", "attnres_"))
                      for n in names])
    assert exact.sum() > 40
    assert got.features[exact].tobytes() == reference.features[exact].tobytes()
    assert got.meta["steps"] == steps
    assert got.meta["receipt"]["reductions"] == dict(attention="stats", logits="stats",
                                                     routing="stats")


def test_the_lane_id_follows_the_engine_identity():
    lane, row, token_ids, _, _ = signed_example()
    reducer = reducer_for(lane)
    a = reducer.reduce([row], token_ids=[token_ids], engine=ENGINE)[0]
    b = reducer.reduce([row], token_ids=[token_ids], engine={**ENGINE, "kernel_pins": "1" * 64})[0]
    assert a.lane_id.startswith(LANE_PREFIX) and b.lane_id.startswith(LANE_PREFIX)
    assert a.lane_id != b.lane_id
    assert a.features.tobytes() == b.features.tobytes()
    assert a.meta["receipt"]["capture"] == ENGINE
    assert reducer.reducer_identity.startswith(
        "anamnesis.extraction.fast.engine_reducer:EngineReducer@")


def test_a_bad_row_is_refused_naming_it():
    lane, row, token_ids, _, _ = signed_example()
    reducer = reducer_for(lane, expected_weights=WEIGHTS)

    def refused(match, bad_row, ids=token_ids):
        with pytest.raises(ExternalCaptureError, match=match):
            reducer.reduce([row, bad_row], token_ids=[token_ids, ids], engine=ENGINE)

    refused(r"row 1: incomplete span", SignedRow(**{**row.__dict__, "kept": row.positions - 1}))
    refused(r"row 1: the row has", row, ids=token_ids + [7])
    refused(r"row 1: weights digest", SignedRow(**{**row.__dict__, "weights": "x" * 64}))
    refused(r"row 1: weights digest \(none\)", SignedRow(**{**row.__dict__, "weights": ""}))
    surfaces = {k: v for k, v in row.surfaces.items() if k != "mean_rows"}
    refused(r"row 1: the signed row lacks surfaces \['mean_rows'\]",
            SignedRow(**{**row.__dict__, "surfaces": surfaces}))
    with pytest.raises(ExternalCaptureError, match="2 rows but 1 token id lists"):
        reducer.reduce([row, row], token_ids=[token_ids], engine=ENGINE)


def test_construction_needs_one_source_of_lanes():
    with pytest.raises(ValueError, match="one of the two"):
        EngineReducer()
    with pytest.raises(ValueError, match="one of the two"):
        EngineReducer(preset="8b", calib_dir=".", lane_for=lambda steps: None)
    with pytest.raises(ValueError, match="calib_dir"):
        EngineReducer(preset="8b")


def test_lanes_are_kept_per_capped_step_count():
    built = []
    lane, _, _, _, _ = signed_example()
    reducer = EngineReducer(lane_for=lambda steps: built.append(steps) or lane)
    for steps in (1, 2, 8, 9, 300, 2):
        reducer.lane(steps)
    assert built == [1, 2, 8]


def test_the_engine_protocol_accepts_it():
    signing = pytest.importorskip("kotodama.model.signing")
    protocol = getattr(signing, "SignatureReducer", None)
    if protocol is None:
        pytest.skip("this kotodama has no SignatureReducer")
    lane, _, _, _, _ = signed_example()
    assert isinstance(reducer_for(lane), protocol)
