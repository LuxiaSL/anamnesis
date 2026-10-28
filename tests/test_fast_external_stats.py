"""Attention statistics in place of weights, held to the weights' own reduction.

:class:`anamnesis.extraction.fast.external_stats.ExternalStatsSpan` reduces a span whose
attention arrives as the module's statistics contract. Fed
:func:`~anamnesis.extraction.fast.external_stats.stats_from_weights` of a forward's
weights, every coordinate must agree with
:class:`~anamnesis.extraction.fast.external.ExternalSpan` fed those weights, to a tight
tolerance (the statistics are the weights' reduction's intermediates, so most agree to
the byte), across the short-span branches and in both weight dtypes; the names and
the placeholder coordinates of short spans are the same. The rest: a malformed contract
is refused by name, and a statistics row's lane id is never an ExternalSpan id.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anamnesis.extraction.fast.external import ExternalCaptureError, external_lane_id
from anamnesis.extraction.fast.external_stats import (
    CONTRACT,
    LANE_PREFIX,
    ExternalStatsSpan,
    RoutingPointStats,
    logit_stats_from_logits,
    routing_stats_from_routing,
    stats_from_weights,
    stats_lane_id,
)
from test_fast_external import IDENTITY, external_reduction, full_weights, routing_substrate
from test_vllm_readout import N_LAYERS, make_example

TOLERANCE = dict(rtol=2e-5, atol=2e-6)


@pytest.fixture(autouse=True)
def deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def forward_weights(weights, start, end, dtype):
    return [full_weights(weights[layer], start, end)[0].to(dtype) for layer in range(N_LAYERS)]


def stats_reduction(lane, capture, full, *, start, end, routing=None, committed=None):
    span = ExternalStatsSpan(lane, start=start, end=end, num_layers=N_LAYERS)
    span.attention_stats(**stats_from_weights(full, start, end, lane.config.sampled_layers))
    token_ids = [0] * end
    token_ids[start + 1:end] = capture["chosen"].tolist()
    return span.finish(hidden=capture["hidden"], keys=capture["keys"], values=capture["values"],
                       queries=capture["queries"], gates=capture["gates"],
                       logits=capture["logits"], token_ids=token_ids,
                       capture_identity=IDENTITY, routing=routing, committed=committed)


def weights_reduction(lane, capture, full, *, start, end, routing=None, committed=None):
    """ExternalSpan over the same forward weights the statistics were taken from."""
    weights = {layer: w.unsqueeze(0)[:, :, start:end, :end].float()
               for layer, w in enumerate(full)}
    return external_reduction(lane, capture, weights, start=start, end=end,
                              routing=routing, committed=committed)


def compare(lane, steps, start, sampled, dtype, weight_dtype, routed=False):
    lane, capture, kwargs, weights = make_example(dtype, steps=steps, start=start,
                                                  sampled_layers=sampled)
    span = dict(start=kwargs["start"], end=kwargs["end"])
    full = forward_weights(weights, dtype=weight_dtype, **span)
    routing = committed = None
    if routed:
        routing, committed = routing_substrate(8, steps=steps, seed=steps)
    stats = stats_reduction(lane, capture, full, routing=routing, committed=committed, **span)
    alone = weights_reduction(lane, capture, full, routing=routing, committed=committed, **span)
    return stats, alone


CASES = [(1, 4, (0, 1, 2)), (2, 4, (0, 2)), (3, 6, (1,)), (8, 4, (0, 1, 2)), (9, 5, (0, 2)),
         (33, 4, (0, 1, 2)), (70, 6, (1,))]


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["float32", "bfloat16"])
@pytest.mark.parametrize("steps,start,sampled", CASES,
                         ids=[f"steps{s}" for s, _, _ in CASES])
def test_statistics_reduce_to_the_weights_features(steps, start, sampled, dtype, weight_dtype):
    stats, alone = compare(None, steps, start, sampled, dtype, weight_dtype, routed=steps == 33)
    assert stats.feature_names == alone.feature_names
    assert stats.mean_logprob == alone.mean_logprob
    np.testing.assert_allclose(stats.features, alone.features, **TOLERANCE,
                               err_msg=str([n for n, a, b in zip(
                                   alone.feature_names, alone.features, stats.features)
                                   if not np.isclose(a, b, **TOLERANCE)]))
    attention = ("attn_", "head_agreement", "cache_", "spectral_", "ph_")
    rest = [i for i, n in enumerate(alone.feature_names) if not n.startswith(attention)]
    assert stats.features[rest].tobytes() == alone.features[rest].tobytes()


def test_short_spans_keep_the_placeholders():
    for steps in (1, 2, 3):
        stats, alone = compare(None, steps, 4, (0, 1, 2), torch.float32, torch.float32)
        zero = alone.features == 0
        assert (stats.features[zero] == 0).all()


def test_the_lane_id_is_never_an_external_span_id():
    lane, capture, kwargs, weights = make_example(torch.float32)
    span = dict(start=kwargs["start"], end=kwargs["end"])
    stats = stats_reduction(lane, capture, forward_weights(weights, dtype=torch.float32, **span),
                            **span)
    receipt = stats.metadata
    assert receipt["lane_id"] == stats_lane_id(lane, IDENTITY)
    assert receipt["lane_id"].startswith(LANE_PREFIX)
    assert receipt["external_lane_id"] == external_lane_id(lane, IDENTITY)
    assert not external_lane_id(lane, IDENTITY).startswith(LANE_PREFIX)
    assert not external_lane_id(lane, {**IDENTITY, "capture_kind": "engine-stats"}).startswith(
        LANE_PREFIX)
    assert receipt["capture_kind"] == "engine-stats"
    assert receipt["replay_path"] == "external-stats"


def started():
    lane, capture, kwargs, weights = make_example(torch.float32)
    span = dict(start=kwargs["start"], end=kwargs["end"])
    stats = stats_from_weights(forward_weights(weights, dtype=torch.float32, **span),
                               span["start"], span["end"], lane.config.sampled_layers)
    return ExternalStatsSpan(lane, num_layers=N_LAYERS, **span), stats


@pytest.mark.parametrize("name", CONTRACT)
def test_a_malformed_contract_is_refused_by_name(name):
    span, stats = started()
    for broken in (stats[name][:-1], stats[name].double(),
                   torch.full_like(stats[name], float("nan"))):
        with pytest.raises(ExternalCaptureError, match=name):
            span.attention_stats(**{**stats, name: broken})
    assert not span.attention_reducer.seen


def test_weights_and_repeated_statistics_are_refused():
    span, stats = started()
    with pytest.raises(ExternalCaptureError, match="attention_stats"):
        span.attention(0, torch.zeros(4, 30, 30))
    span.attention_stats(**stats)
    with pytest.raises(ExternalCaptureError, match="once"):
        span.attention_stats(**stats)


def test_the_converter_matches_the_contract_shapes():
    lane, capture, kwargs, weights = make_example(torch.float32, steps=16, start=4)
    stats = stats_from_weights(forward_weights(weights, 4, 21, torch.float32), 4, 21,
                               lane.config.sampled_layers)
    n, sampled = 16, len(lane.config.sampled_layers)
    assert stats["head_entropy"].shape == (n, N_LAYERS, 4)
    assert stats["mean_entropy"].shape == (n, N_LAYERS)
    for name in ("head_prompt_mass", "head_recency_mass", "head_sink"):
        assert stats[name].shape == (n, sampled, 4)
    assert stats["mean_rows"].shape == (n, sampled, 20)
    assert all(t.dtype == torch.float32 for t in stats.values())
    total = stats["head_prompt_mass"] + 0  # masses are fractions of rows that sum to one
    assert bool(((total >= 0) & (total <= 1 + 1e-6)).all())
    # zero past each row's causal length
    for j in range(n):
        assert bool((stats["mean_rows"][j, :, 4 + j + 1:] == 0).all())


def test_the_twin_was_written_against_this_attention_reducer():
    """``StatsAttentionReducer.consume_stats`` restates ``AttentionReducer.consume``'s
    arithmetic over the mean rows. A change to the reducer's source fails here, so the
    twin is re-read with it; the agreement tests above then say whether they still agree."""
    import hashlib
    from pathlib import Path

    from anamnesis.extraction.fast import attention

    digest = hashlib.sha256(Path(attention.__file__).read_bytes()).hexdigest()
    assert digest == "57b7607c5be249080cf3fe395fc1182d5bf0b1904183f34d03e6b2d7691f6a8b"


TWINS = {
    # the output reduction after its per-step quantities
    "features.py": "a8737153447b48fe81d1aeb295afa252f751d4e1e7841489fc114ffd0a42bdf0",
    # the routing family's assembly
    "attn_res.py": "2dc0fa45320ffc8d67c5f90ffa8a872ef093c9c8fc4bbb36f3638e033865f546",
    # the block-output, key, value, query and gate reduction
    "external.py": "a707670ad44a6b51ee2344974bd5242b4a5341eaec767fb329f79f96cc9c50c0",
}


@pytest.mark.parametrize("name", list(TWINS))
def test_the_restatements_were_written_against_these_sources(name):
    """The module restates parts of these files; a change to one fails here, so its twin
    is re-read with it and the agreement tests say whether they still agree."""
    import hashlib
    from pathlib import Path

    from anamnesis.extraction.fast import external

    path = Path(external.__file__).with_name(name)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == TWINS[name]


def kotodama_routing(routing, committed, steps, seed):
    """The routing as kotodama hands it over: transposed views of ``[sources, S]``
    tensors, and committed blocks as row slices of ``[S, width]``."""
    g = torch.Generator().manual_seed(seed)
    pad = 5
    views = []
    for tag, layer, w in routing:
        full = torch.rand(w.shape[1], steps + 2 * pad, generator=g, dtype=w.dtype)
        full[:, pad:pad + steps] = w.T
        views.append((tag, layer, full[:, pad:pad + steps].T))
    blocks = []
    for c in committed:
        full = torch.randn(steps + 2 * pad, c.shape[1], generator=g, dtype=c.dtype)
        full[pad:pad + steps] = c
        blocks.append(full[pad:pad + steps])
    return views, blocks


def reduced(dtype, steps, start, sampled, *, layout):
    lane, capture, kwargs, weights = make_example(dtype, steps=steps, start=start,
                                                  sampled_layers=sampled)
    span = dict(start=kwargs["start"], end=kwargs["end"])
    full = forward_weights(weights, dtype=torch.float32, **span)
    routing, committed = routing_substrate(8, steps=steps, seed=steps + 1)
    if layout == "kotodama":
        routing, committed = kotodama_routing(routing, committed, steps, seed=steps)
    alone = weights_reduction(lane, capture, full, routing=routing, committed=committed, **span)
    s = ExternalStatsSpan(lane, num_layers=N_LAYERS, **span)
    s.attention_stats(**stats_from_weights(full, span["start"], span["end"],
                                           lane.config.sampled_layers))
    token_ids = [0] * span["end"]
    token_ids[span["start"] + 1:] = capture["chosen"].tolist()
    stats = s.finish(hidden=capture["hidden"], keys=capture["keys"], values=capture["values"],
                     queries=capture["queries"], gates=capture["gates"], logits=None,
                     logit_stats=logit_stats_from_logits(capture["logits"], capture["chosen"]),
                     token_ids=token_ids, capture_identity=IDENTITY, committed=committed,
                     routing_stats=routing_stats_from_routing(routing, lane.config.sampled_layers))
    return stats, alone


OUTPUT_STEMS = ("logit_", "top1_", "top5_", "mean_chosen", "std_chosen", "mean_surprise",
                "std_surprise", "surprise_")


@pytest.mark.parametrize("layout", ["contiguous", "kotodama"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["float32", "bfloat16"])
@pytest.mark.parametrize("steps,start,sampled", [(1, 4, (0, 1, 2)), (3, 4, (0, 2)),
                                                 (16, 4, (0, 2)), (40, 6, (1,))],
                         ids=["steps1", "steps3", "steps16", "steps40"])
def test_output_and_routing_reductions_give_their_families_byte_for_byte(
        dtype, steps, start, sampled, layout):
    stats, alone = reduced(dtype, steps, start, sampled, layout=layout)
    assert stats.feature_names == alone.feature_names
    assert stats.mean_logprob == alone.mean_logprob
    names = np.asarray(alone.feature_names)
    exact = np.array([n.startswith(OUTPUT_STEMS + ("attnres_",)) for n in names])
    assert exact.sum() > 40
    assert stats.features[exact].tobytes() == alone.features[exact].tobytes(), list(
        names[exact][stats.features[exact] != alone.features[exact]])
    np.testing.assert_allclose(stats.features, alone.features, **TOLERANCE)
    assert stats.metadata["reductions"] == dict(attention="stats", logits="stats",
                                                routing="stats")


def test_the_routing_reductions_keep_full_weights_only_where_they_are_read():
    routing, _ = routing_substrate(8, steps=12, seed=3)
    points = routing_stats_from_routing(routing, [0, 2])
    for (tag, layer, w), point in zip(routing, points, strict=True):
        full = tag == "final" or layer in (0, 2)
        assert (point.weights is not None) == full
        assert point.quantities is None or point.quantities.shape == (12, 3)


def test_malformed_reductions_are_refused():
    lane, capture, kwargs, weights = make_example(torch.float32)
    span = dict(start=kwargs["start"], end=kwargs["end"])
    full = forward_weights(weights, dtype=torch.float32, **span)
    routing, committed = routing_substrate(8, steps=16, seed=1)
    good = dict(hidden=capture["hidden"], keys=capture["keys"], values=capture["values"],
                queries=capture["queries"], gates=capture["gates"],
                token_ids=[0] * span["end"], capture_identity=IDENTITY, committed=committed)
    logit_stats = logit_stats_from_logits(capture["logits"], capture["chosen"])
    points = routing_stats_from_routing(routing, lane.config.sampled_layers)

    def attempt(match, **change):
        s = ExternalStatsSpan(lane, num_layers=N_LAYERS, **span)
        s.attention_stats(**stats_from_weights(full, span["start"], span["end"],
                                               lane.config.sampled_layers))
        args = {**good, "logit_stats": logit_stats, "routing_stats": points, **change}
        with pytest.raises(ExternalCaptureError, match=match):
            s.finish(**args)

    attempt("one of the two", logits=capture["logits"])
    attempt("logit_stats must", logit_stats=logit_stats[:, :5])
    attempt("logit_stats must", logit_stats=logit_stats.half())
    attempt("not both", routing=routing)
    sampled_point = next(p for p in points if p.weights is not None and p.tag != "final")
    reduced_wrongly = [p if p is not sampled_point else RoutingPointStats(
        p.tag, p.layer, quantities=torch.zeros(16, 3, dtype=torch.float64)) for p in points]
    attempt("needs its full weights", routing_stats=reduced_wrongly)
    attempt("neither or both", routing_stats=[RoutingPointStats("layer", 3)])
    attempt("come together", committed=None)
