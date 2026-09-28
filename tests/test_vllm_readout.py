"""The vLLM lane's readout, held to the fast lane's own reduction on the same capture.

:func:`anamnesis.extraction.vllm.readout.reduce_capture` restates the per-layer
loop of :meth:`anamnesis.extraction.fast.features.GpuFeatureLane._reduce_capture`
instead of calling it. The restatement is only the fast lane's arithmetic if the
two produce the same bytes, so the central case here drives both on one
synthetic capture and compares every coordinate of the full schema: the
residual, key, value, query, gate, output and PCA families through the fast
lane's own reducers, and the attention families through the adapter on one
side and :class:`anamnesis.extraction.fast.attention.AttentionReducer` on the
other. ``GpuFeatureLane`` takes ``device="cpu"``, and neither reduction launches
a kernel, so the whole comparison runs on the CPU in both substrate dtypes.

The rest are the readout's refusals: a capture that is not exactly the
substrate, a span or device that does not fit, a process that has imported the
engine or would run the readout with other arithmetic.

What needs a device: the same comparison at ``cuda:0`` under the readout
workspace, where cuBLAS rather than the CPU's BLAS carries the matrix products,
and the capture a real engine writes. The CUDA-side refusals below are exercised
by patching torch's switches, not by starting CUDA.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from anamnesis.config import ExtractionConfig, FeaturePipelineConfig
from anamnesis.extraction.fast.attention import AttentionReducer
from anamnesis.extraction.fast.families import FamilyReducer
from anamnesis.extraction.fast.features import GpuFeatureLane
from anamnesis.extraction.fast.ops import FeatureCollector
from anamnesis.extraction.fast.schema import resolve_gpu_schema
from anamnesis.extraction.vllm.envelope import READOUT_WORKSPACE
from anamnesis.extraction.vllm.readout import (
    LaneReadout,
    assert_clean_readout_process,
    reduce_capture,
)
from vllm_observables import engine_observables, span_weights

N_LAYERS, HIDDEN, VOCAB = 3, 8, 17
QUERY_HEADS, KV_HEADS, HEAD_DIM = 4, 2, 2


def lane_configuration(sampled_layers):
    """The complete declared battery at toy widths, and a PCA basis for it."""
    extraction = ExtractionConfig(
        sampled_layers=list(sampled_layers),
        pca_layers=[0, 2],
        pca_components=3,
        early_layer_cutoff=1,
        late_layer_cutoff=2,
    )
    families = FeaturePipelineConfig(
        include_core_blocks=True,
        enable_residual_trajectory=True,
        trajectory_layers=[1],
        enable_attention_flow=True,
        enable_gate_features=True,
        enable_per_head=True,
        enable_value_geometry=True,
        enable_qk_geometry=True,
        enable_kv_cka=True,
        contrastive_layers=[1],
    )
    components = np.eye(HIDDEN, dtype=np.float32)[:3]
    return extraction, families, components


def make_example(dtype, *, steps=16, start=4, sampled_layers=(0, 1, 2), seed=411):
    """A lane, a complete capture of one row, and the rows its attention came from."""
    g = torch.Generator().manual_seed(seed)
    extraction, families, components = lane_configuration(sampled_layers)
    positions = start + steps + 8
    schema = resolve_gpu_schema(N_LAYERS, steps, extraction, families, components)
    pm = (np.arange((N_LAYERS + 1) * positions * HIDDEN, dtype=np.float32)
          .reshape(N_LAYERS + 1, positions, HIDDEN) / 1000)
    lane = GpuFeatureLane(extraction, families, list(schema.feature_names), pm, components,
                          np.zeros(HIDDEN, np.float32), device="cpu",
                          calibration_sha256="a" * 64)

    def rand(*shape):
        return torch.randn(*shape, generator=g).to(dtype)

    sampled = set(sampled_layers)
    weights = {layer: span_weights(steps, start, QUERY_HEADS, 500 + layer)
               for layer in range(N_LAYERS)}
    observed = {layer: engine_observables(w, c=start) for layer, w in weights.items()}
    capture = dict(
        hidden=rand(N_LAYERS, steps, HIDDEN),
        keys={i: rand(steps, KV_HEADS, HEAD_DIM) for i in sorted(sampled)},
        values={i: rand(steps, KV_HEADS, HEAD_DIM) for i in sorted(sampled)},
        queries={i: rand(steps, QUERY_HEADS, HEAD_DIM) for i in sorted(sampled)},
        gates={i: rand(steps, 12) for i in sorted(sampled)},
        logits=rand(steps, VOCAB),
        chosen=torch.arange(steps) % VOCAB,
    )
    for key in ("stats", "h_mean", "h_heads", "entropy_rows"):
        capture[f"attn_{key}"] = {i: observed[i][key] for i in range(N_LAYERS)}
    for key in ("coverage", "spectral_rows", "decay_rows", "span_rows", "head_ent",
                "head_sink", "head_prompt", "head_recency"):
        capture[f"attn_{key}"] = {i: observed[i][key] for i in sorted(sampled)}
    kwargs = dict(start=start, end=start + steps + 1, model="synthetic")
    return lane, capture, kwargs, weights


def fast_lane_reduction(lane, capture, weights, *, start, end):
    """The fast lane's own reduction of the same capture, through ``_reduce_capture``.

    The capture is laid out the way the eager forward's hooks leave it: hidden
    states behind a distinct embedding output (so an off-by-one layer offset
    shows), per-head projections as ``[1, heads, steps, dim]``, and the chosen
    tokens inside the full token row.
    """
    n, t, d = capture["hidden"].shape
    collector = FeatureCollector("cpu")
    attention = AttentionReducer(
        collector, steps=t, prefix_length=start,
        sampled_layers=lane.config.sampled_layers,
        spectral_stride=lane.config.spectral_subsample_step,
        n_windows=lane.families.temporal_n_windows,
        include_stft=lane.families.enable_stft,
    )
    for layer in range(n):
        attention.consume(layer, weights[layer])
    state = SimpleNamespace()
    for source, target in (("keys", "pre_rope_keys"), ("values", "v_proj_values"),
                           ("queries", "queries")):
        setattr(state, target, {i: [x.permute(1, 0, 2).unsqueeze(0)]
                                for i, x in capture[source].items()})
    state.gate_activations = {i: [x.unsqueeze(0)] for i, x in capture["gates"].items()}
    loaded = SimpleNamespace(model=SimpleNamespace(layers=[None] * n), hook_state=state)
    result = SimpleNamespace(
        hidden_states=[torch.full((1, t, d), 123.0)] + [x.unsqueeze(0) for x in capture["hidden"]],
        logits=capture["logits"].unsqueeze(0),
    )
    ids = torch.zeros(1, end, dtype=torch.int64)
    ids[0, start + 1:end] = capture["chosen"]
    return lane._reduce_capture(loaded, result, ids, ids[0].tolist(), start, end, 0,
                                collector, attention,
                                FamilyReducer(collector, lane.config, lane.families))


@pytest.fixture(params=[torch.float32, torch.bfloat16], ids=["float32", "bfloat16"])
def example(monkeypatch, request):
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    return make_example(request.param)


ATTENTION_STEMS = (
    "attn_entropy_", "head_agreement_", "cache_", "spectral_", "attn_flow_", "ph_",
)
"""Name stems of the coordinates the attention adapter and the key-spread pair emit;
every other coordinate comes from the fast lane's own reducers in both paths."""


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["float32", "bfloat16"])
@pytest.mark.parametrize("steps,start,sampled", [
    (16, 4, (0, 1, 2)),
    (16, 4, (0, 2)),
    (70, 6, (1,)),
], ids=["all-sampled", "one-unsampled", "long-span"])
def test_the_readout_is_the_fast_lanes_reduction_byte_for_byte(monkeypatch, dtype, steps,
                                                               start, sampled):
    """Every coordinate of the full schema, attention included, is the fast lane's bytes.

    The non-attention coordinates are checked on their own first, so a failure
    says which half of the restatement drifted.
    """
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    lane, capture, kwargs, weights = make_example(dtype, steps=steps, start=start,
                                                  sampled_layers=sampled)
    actual = reduce_capture(lane, capture, **kwargs)
    expected = fast_lane_reduction(lane, capture, weights, start=kwargs["start"],
                                   end=kwargs["end"])
    assert actual.feature_names == tuple(lane.names) == tuple(expected.feature_names)
    substrate = [i for i, name in enumerate(actual.feature_names)
                 if not name.startswith(ATTENTION_STEMS)]
    assert substrate and len(substrate) < len(actual.feature_names)
    families = {name.split("_")[0] for name in np.asarray(actual.feature_names)[substrate]}
    assert {"activation", "delta", "res", "pca", "kv", "value", "qk", "gate", "logit",
            "surprise"} <= families
    np.testing.assert_array_equal(actual.features[substrate], expected.features[substrate])
    assert actual.features.tobytes() == expected.features.tobytes(), [
        name for name, a, b in zip(actual.feature_names, actual.features, expected.features)
        if np.float32(a).tobytes() != np.float32(b).tobytes()]


def test_the_readout_returns_the_full_schema_and_what_it_read(example):
    lane, capture, kwargs, _ = example
    result = reduce_capture(lane, capture, **kwargs)
    assert isinstance(result, LaneReadout)
    assert result.feature_names == tuple(lane.names)
    assert result.features.dtype == np.float32
    assert result.features.shape == (len(lane.names),)
    assert result.metadata["workspace"] == READOUT_WORKSPACE
    assert result.metadata["substrate_dtype"] == str(capture["hidden"].dtype)
    assert result.metadata["logits_dtype"] == str(capture["logits"].dtype)
    assert result.metadata["steps"] == kwargs["end"] - kwargs["start"] - 1
    assert "lane_id" not in result.metadata


def test_the_vocabulary_is_the_logits_full_width_and_labels_are_the_next_tokens(example):
    lane, c, kwargs, _ = example
    c["logits"].zero_()
    c["logits"][:, VOCAB - 1] = 10
    c["chosen"].fill_(VOCAB - 1)
    a = reduce_capture(lane, c, **kwargs)
    assert a.features[a.feature_names.index("mean_chosen_rank")] == 0
    c["chosen"].fill_(VOCAB - 2)
    b = reduce_capture(lane, c, **kwargs)
    assert b.features[b.feature_names.index("mean_chosen_rank")] == 1
    assert (b.features[b.feature_names.index("mean_surprise")]
            > a.features[a.feature_names.index("mean_surprise")] + 9)


def test_float32_logits_are_read_as_given(example):
    lane, c, kwargs, _ = example
    c["logits"] = c["logits"].float()
    assert reduce_capture(lane, c, **kwargs).metadata["logits_dtype"] == "torch.float32"


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda c: c["keys"].pop(1), "keys layers"),
        (lambda c: c["gates"].update({1: c["gates"][1][:, :10]}), "mixed gates"),
        (lambda c: c.update(chosen=c["chosen"][:-1]), "chosen"),
        (lambda c: c.update(chosen=c["chosen"].int()), "chosen"),
        (lambda c: c["queries"].update({0: c["queries"][0].double()}), "mixed native"),
        (lambda c: c.update(hidden=c["hidden"][:, :-1]), "span shape"),
        (lambda c: c["chosen"].fill_(VOCAB), "outside vocabulary"),
        (lambda c: c.update(logits=c["logits"][:, :4]), "vocabulary >=5"),
        (lambda c: c["hidden"].__setitem__((0, 0, 0), float("nan")), "nonfinite"),
        (lambda c: c.pop("attn_span_rows"), "missing"),
        (lambda c: c.update(logprobs=c["logits"]), "logprob"),
        (lambda c: c.update(extra=c["logits"]), "extra"),
        (lambda c: c["attn_coverage"].pop(0), "attn_coverage layers"),
        (lambda c: c["attn_stats"].pop(2), "attn_stats layers"),
        (lambda c: c["attn_h_mean"].update({0: [0.0]}), "map layers to tensors"),
    ],
)
def test_an_invalid_capture_is_refused(example, mutation, match):
    lane, c, kwargs, _ = example
    mutation(c)
    with pytest.raises(ValueError, match=match):
        reduce_capture(lane, c, **kwargs)


def test_a_malformed_span_or_identity_is_refused(example):
    lane, c, kwargs, _ = example
    with pytest.raises(ValueError, match="model identity"):
        reduce_capture(lane, c, **{**kwargs, "model": ""})
    with pytest.raises(ValueError, match="prediction span"):
        reduce_capture(lane, c, **{**kwargs, "end": kwargs["start"] + 1})
    with pytest.raises(ValueError, match="span shape"):
        reduce_capture(lane, c, **{**kwargs, "start": kwargs["start"] + 1})


def test_positional_means_that_stop_short_of_the_span_are_refused(example):
    lane, c, kwargs, _ = example
    lane.pm = np.ascontiguousarray(lane.pm[:, : kwargs["end"] - 2])
    with pytest.raises(ValueError, match="positional means"):
        reduce_capture(lane, c, **kwargs)


def test_a_capture_on_another_device_than_the_lane_is_refused(example):
    lane, c, kwargs, _ = example
    lane.device = torch.device("meta")
    with pytest.raises(ValueError, match="device"):
        reduce_capture(lane, c, **kwargs)


def test_a_lane_at_another_spectral_stride_is_refused(example):
    """The capture selected its spectral rows at the lane's fixed stride."""
    lane, c, kwargs, _ = example
    lane.config = lane.config.model_copy(update={"spectral_subsample_step": 5})
    with pytest.raises(ValueError, match="stride"):
        reduce_capture(lane, c, **kwargs)


def test_a_process_that_imported_the_engine_is_refused(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace())
    with pytest.raises(RuntimeError, match="outside"):
        assert_clean_readout_process("cpu")


def test_batch_invariant_mode_is_refused(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    with pytest.raises(RuntimeError, match="batch-invariant"):
        assert_clean_readout_process("cpu")


def test_an_installed_matrix_product_override_is_refused(monkeypatch):
    def mm(a, b):
        return a @ b

    mm.__module__ = "vllm.model_executor.layers.batch_invariant"
    monkeypatch.setattr(torch, "mm", mm)
    with pytest.raises(RuntimeError, match="override"):
        assert_clean_readout_process("cpu")


def _cuda_switches(monkeypatch, *, deterministic=True, warn_only=False, tf32=False):
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: deterministic)
    monkeypatch.setattr(torch, "is_deterministic_algorithms_warn_only_enabled",
                        lambda: warn_only)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", tf32)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)


def test_a_cuda_readout_requires_its_workspace_and_arithmetic(monkeypatch):
    """The CUDA checks read settings only, so they are exercised without starting CUDA."""
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    _cuda_switches(monkeypatch)
    with pytest.raises(RuntimeError, match=READOUT_WORKSPACE):
        assert_clean_readout_process("cuda")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", READOUT_WORKSPACE)
    assert_clean_readout_process("cuda")
    _cuda_switches(monkeypatch, deterministic=False)
    with pytest.raises(RuntimeError, match="deterministic"):
        assert_clean_readout_process("cuda")
    _cuda_switches(monkeypatch, warn_only=True)
    with pytest.raises(RuntimeError, match="deterministic"):
        assert_clean_readout_process("cuda")
    _cuda_switches(monkeypatch, tf32=True)
    with pytest.raises(RuntimeError, match="TF32"):
        assert_clean_readout_process("cuda")


def test_the_cpu_readout_does_not_read_the_cuda_settings(monkeypatch):
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    _cuda_switches(monkeypatch, deterministic=False, tf32=True)
    assert_clean_readout_process("cpu")


def test_a_cuda_lane_built_at_another_workspace_is_refused_before_reduction(example,
                                                                             monkeypatch):
    lane, c, kwargs, _ = example
    lane.device = torch.device("cuda:0")
    lane.identity["cublas_workspace_config"] = ":16:8"
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", READOUT_WORKSPACE)
    _cuda_switches(monkeypatch)
    with pytest.raises(RuntimeError, match="lane's workspace"):
        reduce_capture(lane, c, **kwargs)
