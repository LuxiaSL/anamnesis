"""Graph replay of the external readout, held to ``ExternalSpan.finish`` byte for byte.

:class:`anamnesis.extraction.fast.graphed.GraphedExternalSpan` restates the
reduction ``ExternalSpan.finish`` runs so that it can be recorded as CUDA graphs.
On any device the restatement, run eagerly and assembled the way a replay is, is
held to ``ExternalSpan.finish`` on the same captures: both substrate dtypes, three
span shapes, with and without the routing family. Without CUDA a graphed span is
the eager span. With CUDA, several spans of one shape are replayed against fresh
eager reductions of the same inputs, a new shape records its own graphs, and a
reduction that reads the span's data on the host is refused and reduced eagerly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anamnesis.extraction.fast.external import ExternalSpan
from anamnesis.extraction.fast.graphed import (
    GraphCaptureRefused,
    GraphedExternalSpan,
    SpanGraphCache,
    _Capturing,
    _Record,
    _Tape,
)
from test_fast_external import IDENTITY, full_weights, routing_substrate
from test_vllm_readout import N_LAYERS, make_example

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="graph replay needs CUDA")


@pytest.fixture(autouse=True)
def deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def _to(x, device):
    if isinstance(x, dict):
        return {k: _to(v, device) for k, v in x.items()}
    return x.to(device) if isinstance(x, torch.Tensor) else x


def _example(dtype, steps, start, sampled, device="cpu", seed=411):
    lane, capture, kwargs, weights = make_example(dtype, steps=steps, start=start,
                                                  sampled_layers=sampled, seed=seed)
    if device != "cpu":
        lane = _lane_on(lane, device)
        capture = {k: _to(v, device) for k, v in capture.items()}
        weights = {k: v.to(device) for k, v in weights.items()}
    return lane, capture, kwargs, weights


def _lane_on(lane, device):
    from anamnesis.extraction.fast.features import GpuFeatureLane

    pca = lane.pca.cpu().numpy() if not isinstance(lane.pca, dict) else {
        k: v.cpu().numpy() for k, v in lane.pca.items()}
    mean = lane.pca_mean.cpu().numpy() if not isinstance(lane.pca_mean, dict) else {
        k: v.cpu().numpy() for k, v in lane.pca_mean.items()}
    return GpuFeatureLane(lane.config, lane.families, list(lane.names), lane.pm, pca, mean,
                          device=device, calibration_sha256=lane.calibration_sha256)


def _span(cls, lane, capture, weights, kwargs, **extra):
    start, end = kwargs["start"], kwargs["end"]
    span = cls(lane, start=start, end=end, num_layers=N_LAYERS, **extra)
    device = lane.device
    for layer in range(N_LAYERS):
        span.attention(layer, full_weights(weights[layer].cpu(), start, end)[0].to(device))
    return span


def _call(capture, kwargs, routing=None, committed=None):
    end, start = kwargs["end"], kwargs["start"]
    token_ids = [0] * end
    token_ids[start + 1:end] = capture["chosen"].tolist()
    return dict(hidden=capture["hidden"], keys=capture["keys"], values=capture["values"],
                queries=capture["queries"], gates=capture["gates"], logits=capture["logits"],
                token_ids=token_ids, capture_identity=IDENTITY, routing=routing, committed=committed)


def _same(a, b):
    assert a.feature_names == b.feature_names
    assert a.features.tobytes() == b.features.tobytes(), [
        name for name, x, y in zip(a.feature_names, a.features, b.features)
        if np.float32(x).tobytes() != np.float32(y).tobytes()]
    assert np.float64(a.mean_logprob).tobytes() == np.float64(b.mean_logprob).tobytes()
    assert a.metadata["lane_id"] == b.metadata["lane_id"]


SHAPES = [(16, 4, (0, 1, 2)), (16, 4, (0, 2)), (70, 6, (1,))]
SHAPE_IDS = ["all-sampled", "one-unsampled", "long-span"]


@pytest.mark.parametrize("routed", [False, True], ids=["battery", "routed"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["float32", "bfloat16"])
@pytest.mark.parametrize("steps,start,sampled", SHAPES, ids=SHAPE_IDS)
def test_the_restated_reduction_is_external_finish_byte_for_byte(dtype, steps, start, sampled, routed):
    lane, capture, kwargs, weights = _example(dtype, steps, start, sampled)
    routing, committed = routing_substrate(8, steps, seed=5) if routed else (None, None)
    call = _call(capture, kwargs, routing, committed)
    restated = _span(GraphedExternalSpan, lane, capture, weights, kwargs)._reduce_eagerly(call)
    eager = _span(ExternalSpan, lane, capture, weights, kwargs).finish(**call)
    _same(restated, eager)
    assert restated.metadata["graph_replayed"] is False


def test_without_cuda_a_graphed_span_is_the_eager_span():
    lane, capture, kwargs, weights = _example(torch.float32, 16, 4, (0, 1, 2))
    cache = SpanGraphCache()
    call = _call(capture, kwargs)
    graphed = _span(GraphedExternalSpan, lane, capture, weights, kwargs, cache=cache).finish(**call)
    eager = _span(ExternalSpan, lane, capture, weights, kwargs).finish(**call)
    _same(graphed, eager)
    assert graphed.metadata["graph_replayed"] is False
    assert cache.stats().shapes == 0


def test_a_cache_holds_at_least_one_shape():
    with pytest.raises(ValueError):
        SpanGraphCache(max_shapes=0)


@cuda
@pytest.mark.parametrize("routed", [False, True], ids=["battery", "routed"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["float32", "bfloat16"])
@pytest.mark.parametrize("steps,start,sampled", SHAPES, ids=SHAPE_IDS)
def test_replayed_spans_are_eager_spans_byte_for_byte(dtype, steps, start, sampled, routed):
    cache = SpanGraphCache()
    replayed = 0
    for seed in (411, 412, 413, 414):
        lane_seeded, capture, kwargs, weights = _example(dtype, steps, start, sampled, "cuda", seed)
        if seed == 411:
            lane = lane_seeded
        routing, committed = (routing_substrate(8, steps, seed=seed) if routed else (None, None))
        if routed:
            routing = [(t, l, w.cuda()) for t, l, w in routing]
            committed = [c.cuda() for c in committed]
        call = _call(capture, kwargs, routing, committed)
        graphed = _span(GraphedExternalSpan, lane, capture, weights, kwargs, cache=cache).finish(**call)
        eager = _span(ExternalSpan, lane, capture, weights, kwargs).finish(**call)
        _same(graphed, eager)
        replayed += graphed.metadata["graph_replayed"]
    stats = cache.stats()
    assert stats.refused == {} and stats.shapes == 1 and stats.replays == 3 and replayed == 3


@cuda
def test_each_shape_records_its_own_graphs_and_the_oldest_is_dropped():
    cache = SpanGraphCache(max_shapes=1)
    for steps in (16, 20, 16):
        lane, capture, kwargs, weights = _example(torch.float32, steps, 4, (0, 1, 2), "cuda")
        call = _call(capture, kwargs)
        graphed = _span(GraphedExternalSpan, lane, capture, weights, kwargs, cache=cache).finish(**call)
        _same(graphed, _span(ExternalSpan, lane, capture, weights, kwargs).finish(**call))
    stats = cache.stats()
    assert stats.shapes == 1 and stats.captures == 3 and stats.evictions == 2


@cuda
def test_a_host_read_of_the_spans_data_refuses_the_capture():
    data = torch.randn(8, device="cuda")
    record = _Record()
    record.host_reads.append(((), torch.bool, True))
    side = torch.cuda.Stream()
    tape = _Tape(torch.cuda.graph_pool_handle(), torch.cuda.current_stream())
    capturing = _Capturing(tape, record, {data.untyped_storage().data_ptr()}, side,
                           torch.cuda.current_stream())
    with torch.cuda.stream(side), pytest.raises(GraphCaptureRefused, match="span's data"):
        try:
            with capturing:
                capturing.begin()
                bool((data * 2).sum() > 0)
        finally:
            capturing.abandon()
    torch.cuda.synchronize()


@cuda
def test_a_host_read_of_constants_is_answered_from_the_eager_run():
    constant = torch.arange(4, device="cuda", dtype=torch.float64)
    record = _Record()
    record.host_reads.append(((), torch.bool, False))
    side = torch.cuda.Stream()
    stream = torch.cuda.current_stream()
    tape = _Tape(torch.cuda.graph_pool_handle(), stream)
    capturing = _Capturing(tape, record, set(), side, stream)
    side.wait_stream(stream)
    with torch.cuda.stream(side):
        with capturing:
            capturing.begin()
            flat = bool(constant.std(correction=0) <= 1e-12)
            capturing.end()
    stream.wait_stream(side)
    assert flat is False and capturing.reads == 1
