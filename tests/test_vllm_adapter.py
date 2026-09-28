"""The vLLM lane's attention adapter, held to the fast lane's attention reducer.

:class:`anamnesis.extraction.vllm.adapter.AttentionFeatureAdapter` claims to be
:class:`anamnesis.extraction.fast.attention.AttentionReducer`'s arithmetic line
for line, over the rows the capture holds instead of the materialized weights.
Each case here builds one layer's eager-shaped attention weights, runs the real
reducer on them, and hands the adapter the capture's view of the same rows (see
``tests/vllm_observables.py``). Every coordinate the adapter emits must then be
the reducer's own fp32 scalar, bit for bit; the refusals are the malformed
observables a capture could hand it.

Everything here runs on the CPU. What needs a device is whether the instrumented
kernel and the second pass write the rows these cases compute on the host.
"""

from __future__ import annotations

import pytest
import torch

from anamnesis.extraction.fast.attention import AttentionReducer
from anamnesis.extraction.fast.ops import FeatureCollector
from anamnesis.extraction.vllm.adapter import AttentionFeatureAdapter
from vllm_observables import consume_arguments, engine_observables, span_weights


def run_pair(t: int, c: int, heads: int, seed: int, *, hidden: int = 16):
    """The reducer and the adapter over identical rows, one sampled layer."""
    weights = span_weights(t, c, heads, seed)
    g = torch.Generator().manual_seed(seed + 1)
    corrected = torch.randn(t, hidden, generator=g)
    keys = torch.randn(t, max(1, heads // 2), 8, generator=g)
    reference = FeatureCollector("cpu")
    reducer = AttentionReducer(reference, steps=t, prefix_length=c, sampled_layers=[0])
    reducer.consume(0, weights)
    reducer.smoothness(0, corrected)
    reducer.head_features(0, keys)
    observed = engine_observables(weights, c=c)
    candidate = FeatureCollector("cpu")
    adapter = AttentionFeatureAdapter(candidate, schema=observed["schema"], num_layers=1,
                                      sampled_layers=[0])
    adapter.consume(0, **consume_arguments(observed, sampled=True))
    adapter.smoothness(0, corrected)
    adapter.assert_complete()
    return reference.values, candidate.values


CASES = ((47, 13, 4, 11), (127, 9, 2, 12), (12, 5, 4, 13), (3, 4, 4, 14),
         (2, 3, 2, 15), (1, 2, 2, 16))
"""(steps, prompt length, heads, seed): spans past and under every threshold the
reducer branches on — the entropy stride at 60 steps, the decay fit at 10, the
per-head summaries at 4, the flow families at 2."""


@pytest.mark.parametrize("t,c,heads,seed", CASES)
def test_every_emitted_coordinate_is_the_reducers_bits(t, c, heads, seed):
    """On identical rows no estimator is left: every coordinate is the reducer's own."""
    reference, candidate = run_pair(t, c, heads, seed)
    mismatched = [name for name, value in candidate.items()
                  if not torch.equal(value, reference[name])]
    assert not mismatched, mismatched


def test_the_entropy_series_slice_matches_on_long_spans():
    """A span past the stride threshold exercises the scatter-then-slice."""
    reference, candidate = run_pair(127, 9, 2, 21)
    for name in ("attn_entropy_mean_L0", "attn_entropy_std_L0"):
        assert torch.equal(candidate[name], reference[name]), name


@pytest.mark.parametrize("t,c,heads,seed", CASES)
def test_the_adapter_emits_every_reducer_name_but_the_key_spread(t, c, heads, seed):
    """The key-spread pair is reduced from the captured keys by the readout, not here."""
    reference, candidate = run_pair(t, c, heads, seed)
    spread = {name for name in reference if "_kv_key_spread_" in name}
    assert len(spread) == 2 and not spread & set(candidate)
    assert set(candidate) | spread == set(reference)


def test_an_unsampled_layer_matches_the_reducer_bitwise():
    """An unsampled layer carries only the entropy and agreement series, and emits
    exactly the names the reducer emits for it."""
    weights = span_weights(127, 9, 2, 24)
    observed = engine_observables(weights, c=9)
    out = FeatureCollector("cpu")
    adapter = AttentionFeatureAdapter(out, schema=observed["schema"], num_layers=2,
                                      sampled_layers=[1])
    adapter.consume(0, **consume_arguments(observed, sampled=False))
    reference = FeatureCollector("cpu")
    reducer = AttentionReducer(reference, steps=127, prefix_length=9, sampled_layers=[1])
    reducer.consume(0, weights)
    assert set(out.values) == set(reference.values)
    for name in out.values:
        assert torch.equal(out.values[name], reference.values[name]), name


def _observed(seed: int = 18):
    weights = span_weights(12, 5, 4, seed)
    return engine_observables(weights, c=5)


def _fresh(observed, sampled=(0,), layers=1):
    return AttentionFeatureAdapter(FeatureCollector("cpu"), schema=observed["schema"],
                                   num_layers=layers, sampled_layers=sampled)


def test_a_sampled_layer_requires_every_family_operand():
    observed = _observed(23)
    full = consume_arguments(observed, sampled=True)
    with pytest.raises(ValueError, match="entropy row"):
        _fresh(observed).consume(0, **{**full, "entropy_rows": None})
    with pytest.raises(ValueError, match="span rows must be"):
        _fresh(observed).consume(0, **{**full, "span_rows": None})
    with pytest.raises(ValueError, match="head_ent"):
        _fresh(observed).consume(0, **{**full, "head_ent": None})
    with pytest.raises(ValueError, match="head_ent"):
        _fresh(observed).consume(0, **{**full, "head_ent": observed["head_ent"].float()})
    with pytest.raises(ValueError, match="head_recency"):
        _fresh(observed).consume(0, **{**full, "head_recency": observed["head_recency"][:-1]})
    with pytest.raises(ValueError, match="entropy row"):
        _fresh(observed).consume(0, **{**full, "entropy_rows": observed["entropy_rows"][:-1]})


def test_an_unsampled_layer_refuses_sampled_operands():
    observed = _observed(23)
    unsampled = consume_arguments(observed, sampled=False)
    with pytest.raises(ValueError, match="belong to sampled"):
        _fresh(observed, sampled=(1,), layers=2).consume(
            0, **unsampled, head_ent=observed["head_ent"])
    with pytest.raises(ValueError, match="belong to sampled"):
        _fresh(observed, sampled=(), layers=1).consume(
            0, **consume_arguments(observed, sampled=True))


def test_the_adapter_refuses_malformed_observables():
    observed = _observed()
    span = consume_arguments(observed, sampled=True)
    adapter = _fresh(observed)
    adapter.consume(0, **span)
    with pytest.raises(ValueError, match="twice"):
        adapter.consume(0, **span)
    with pytest.raises(ValueError, match="coverage must be"):
        _fresh(observed).consume(0, **consume_arguments(observed, sampled=False))
    with pytest.raises(ValueError, match="fp32"):
        _fresh(observed).consume(0, **{**span, "stats": observed["stats"].double()})
    with pytest.raises(ValueError, match="float64"):
        _fresh(observed).consume(0, **{**span, "h_mean": observed["h_mean"].float()})
    with pytest.raises(ValueError, match="capture width"):
        _fresh(observed).consume(
            0, **{**span, "spectral_rows": observed["spectral_rows"][:, :-1]})
    with pytest.raises(ValueError, match="capture width"):
        _fresh(observed).consume(0, **{**span, "decay_rows": observed["decay_rows"][:, :-1]})
    with pytest.raises(ValueError, match="layer outside"):
        _fresh(observed).consume(1, **span)
    bad = observed["stats"].clone()
    bad[0, 0, 2] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        _fresh(observed).consume(0, **{**span, "stats": bad})
    bad_rows = observed["span_rows"].clone()
    bad_rows[0, 0] = torch.inf
    with pytest.raises(ValueError, match="finite"):
        _fresh(observed).consume(0, **{**span, "span_rows": bad_rows})


def test_a_partially_adapted_request_is_refused():
    observed = _observed()
    span = consume_arguments(observed, sampled=True)
    with pytest.raises(ValueError, match="never adapted"):
        _fresh(observed).assert_complete()
    done = _fresh(observed)
    done.consume(0, **span)
    with pytest.raises(ValueError, match="smoothness never consumed"):
        done.assert_complete()
    with pytest.raises(ValueError, match="no retained spectral"):
        _fresh(observed).smoothness(0, torch.randn(12, 8))


def test_a_wrong_sampled_layer_set_is_a_constructor_error():
    observed = engine_observables(span_weights(4, 3, 2, 19), c=3)
    with pytest.raises(ValueError, match="layer indices"):
        _fresh(observed, sampled=[2], layers=2)
    with pytest.raises(ValueError, match="duplicate"):
        _fresh(observed, sampled=[1, 1], layers=2)
    with pytest.raises(ValueError, match="num_layers"):
        _fresh(observed, sampled=[], layers=0)
    with pytest.raises(ValueError, match="RequestRowSchema"):
        AttentionFeatureAdapter(FeatureCollector("cpu"), schema=None, num_layers=1,
                                sampled_layers=[0])
