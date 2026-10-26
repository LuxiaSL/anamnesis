"""The fast lane's reduction of an outside capture, held to the fast lane's own.

:class:`anamnesis.extraction.fast.external.ExternalSpan` restates the per-layer loop
of ``GpuFeatureLane._reduce_capture``. The central case drives both on one synthetic
capture (the vLLM readout's, from ``test_vllm_readout``) and compares every
coordinate's bytes, in both substrate dtypes and three span shapes. The device port
of the routing family is held to the canonical numpy family on the same inputs, to
rounding. The rest are the readout's refusals and what its lane id is made of.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anamnesis.extraction.fast.attn_res import attn_res_features
from anamnesis.extraction.fast.external import (
    ExternalCaptureError,
    ExternalSpan,
    external_lane_id,
)
from anamnesis.extraction.feature_families.attn_res import extract_attn_res
from test_vllm_readout import N_LAYERS, fast_lane_reduction, make_example

IDENTITY = {"capture": "synthetic", "source_sha256": "0" * 64}


@pytest.fixture(autouse=True)
def deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def full_weights(span_weights, start, end):
    """A forward's whole ``[1, heads, end, end]`` weights holding ``span_weights`` at the span's rows."""
    heads = span_weights.shape[1]
    full = torch.rand(1, heads, end, end, generator=torch.Generator().manual_seed(7))
    full[:, :, start:end, :] = span_weights
    return full


def external_reduction(lane, capture, weights, *, start, end, routing=None, committed=None):
    span = ExternalSpan(lane, start=start, end=end, num_layers=N_LAYERS)
    for layer in range(N_LAYERS):
        span.attention(layer, full_weights(weights[layer], start, end)[0])
    token_ids = [0] * end
    token_ids[start + 1:end] = capture["chosen"].tolist()
    return span.finish(hidden=capture["hidden"], keys=capture["keys"], values=capture["values"],
                       queries=capture["queries"], gates=capture["gates"], logits=capture["logits"],
                       token_ids=token_ids, capture_identity=IDENTITY, routing=routing,
                       committed=committed)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["float32", "bfloat16"])
@pytest.mark.parametrize("steps,start,sampled", [
    (16, 4, (0, 1, 2)),
    (16, 4, (0, 2)),
    (70, 6, (1,)),
], ids=["all-sampled", "one-unsampled", "long-span"])
def test_the_external_readout_is_the_fast_lanes_reduction_byte_for_byte(dtype, steps, start, sampled):
    lane, capture, kwargs, weights = make_example(dtype, steps=steps, start=start, sampled_layers=sampled)
    actual = external_reduction(lane, capture, weights, start=kwargs["start"], end=kwargs["end"])
    expected = fast_lane_reduction(lane, capture, weights, start=kwargs["start"], end=kwargs["end"])
    assert actual.feature_names == tuple(lane.names) == tuple(expected.feature_names)
    assert actual.features.tobytes() == expected.features.tobytes(), [
        name for name, a, b in zip(actual.feature_names, actual.features, expected.features)
        if np.float32(a).tobytes() != np.float32(b).tobytes()]
    assert actual.mean_logprob == expected.mean_logprob


def routing_substrate(n_blocks, steps, seed):
    g = torch.Generator().manual_seed(seed)
    routing = []
    for layer in range(5):
        for _point in range(2):
            n_src = min(layer + 2, max(n_blocks, 2))
            w = torch.rand(steps, n_src, generator=g, dtype=torch.float64)
            routing.append(("layer", layer, (w / w.sum(1, keepdim=True)).float()))
    w = torch.rand(steps, max(n_blocks, 2), generator=g, dtype=torch.float64)
    routing.append(("final", -1, (w / w.sum(1, keepdim=True)).float()))
    committed = [torch.randn(steps, 8, generator=g) for _ in range(n_blocks)]
    return routing, committed


@pytest.mark.parametrize("n_blocks", [3, 7, 8, 12])
def test_the_device_routing_family_is_the_canonical_one(n_blocks):
    routing, committed = routing_substrate(n_blocks, steps=13, seed=n_blocks)
    sampled = [0, 2, 4]
    names, values = attn_res_features(routing, committed, sampled)
    canonical = extract_attn_res([(t, l, w.numpy()) for t, l, w in routing],
                                 committed=[c.numpy() for c in committed], sampled_layers=sampled)
    assert names == list(canonical.feature_names)
    np.testing.assert_allclose(values.numpy(), canonical.features, rtol=1e-6, atol=1e-7)


def test_the_routing_family_follows_the_battery():
    lane, capture, kwargs, weights = make_example(torch.float32)
    routing, committed = routing_substrate(8, steps=16, seed=3)
    result = external_reduction(lane, capture, weights, start=kwargs["start"], end=kwargs["end"],
                                routing=routing, committed=committed)
    battery = len(lane.names)
    assert result.feature_names[:battery] == tuple(lane.names)
    assert result.feature_names[battery:] == tuple(attn_res_features(routing, committed,
                                                                     lane.config.sampled_layers)[0])
    assert "attnres_committed_cos_b6b7_mean" in result.feature_names
    assert result.metadata["routing_family"] is True


def test_a_layer_whose_attention_was_not_reduced_is_refused():
    lane, capture, kwargs, weights = make_example(torch.float32)
    span = ExternalSpan(lane, start=kwargs["start"], end=kwargs["end"], num_layers=N_LAYERS)
    span.attention(0, full_weights(weights[0], kwargs["start"], kwargs["end"]))
    with pytest.raises(ExternalCaptureError, match="not reduced"):
        span.finish(hidden=capture["hidden"], keys=capture["keys"], values=capture["values"],
                    queries=capture["queries"], gates=capture["gates"], logits=capture["logits"],
                    token_ids=[0] * kwargs["end"], capture_identity=IDENTITY)


def test_routing_without_committed_blocks_is_refused():
    lane, capture, kwargs, weights = make_example(torch.float32)
    routing, _ = routing_substrate(8, steps=16, seed=1)
    with pytest.raises(ExternalCaptureError, match="come together"):
        external_reduction(lane, capture, weights, start=kwargs["start"], end=kwargs["end"],
                           routing=routing, committed=None)


def test_a_span_is_finished_once_and_takes_no_attention_after():
    lane, capture, kwargs, weights = make_example(torch.float32)
    external_reduction(lane, capture, weights, start=kwargs["start"], end=kwargs["end"])
    span = ExternalSpan(lane, start=kwargs["start"], end=kwargs["end"], num_layers=N_LAYERS)
    span.finished = True
    with pytest.raises(ExternalCaptureError):
        span.attention(0, full_weights(weights[0], kwargs["start"], kwargs["end"]))


def test_nondeterministic_arithmetic_is_refused():
    lane, capture, kwargs, weights = make_example(torch.float32)
    torch.use_deterministic_algorithms(False)
    with pytest.raises(ExternalCaptureError, match="deterministic|arithmetic settings changed"):
        external_reduction(lane, capture, weights, start=kwargs["start"], end=kwargs["end"])


def test_a_short_attention_matrix_is_refused():
    lane, capture, kwargs, weights = make_example(torch.float32)
    span = ExternalSpan(lane, start=kwargs["start"], end=kwargs["end"], num_layers=N_LAYERS)
    with pytest.raises(ExternalCaptureError, match="cover"):
        span.attention(0, torch.zeros(1, 4, kwargs["end"] - 1, kwargs["end"]))


def test_the_lane_id_names_the_capture():
    lane, *_ = make_example(torch.float32)
    a = external_lane_id(lane, IDENTITY)
    assert a == external_lane_id(lane, dict(IDENTITY))
    assert a != external_lane_id(lane, {**IDENTITY, "source_sha256": "1" * 64})
    assert a != lane.lane_id


def test_a_span_past_the_calibrated_positions_is_refused():
    lane, capture, kwargs, weights = make_example(torch.float32)
    covered = lane.positions_calibrated
    with pytest.raises(ExternalCaptureError, match="calibration covers"):
        ExternalSpan(lane, start=covered - 1, end=covered + 2, num_layers=N_LAYERS)


def test_arithmetic_changed_after_the_lane_was_built_is_refused():
    lane, capture, kwargs, weights = make_example(torch.float32)
    torch.use_deterministic_algorithms(not lane.identity["deterministic"])
    with pytest.raises(ExternalCaptureError, match="arithmetic settings changed"):
        ExternalSpan(lane, start=kwargs["start"], end=kwargs["end"], num_layers=N_LAYERS)
