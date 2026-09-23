"""Batched replay: several spans in one pass, each row reduced as if it were alone.

Batching is a throughput mechanism, and it is the one mechanism in the lane that can
change a number without changing a formula: rows are packed to a common width, so a
row sees padding it would not see alone, and the physical cache index a row's key sits
at differs from that row's absolute position. The layout answers both — left-padded
continuations right-padded, per-row absolute positions carried in `position_ids`, and
padding sliced off before any reduction sees the tensor — and this test is what says
the answer holds.

The claim is per-row equality with the single-span reference: for each row, the anchor's
features over that span alone, computed against a freshly built prefix cache, at the
same tolerance the unbatched agreement proof uses. Ragged prefixes and ragged
continuations are both covered, in both orders, because left-padding the shorter prefix
and right-padding the shorter continuation are different arithmetic.

Two further properties, each a way batching could pass on numbers and still be wrong:

* **Identity.** A batched result carries a batch lane id distinct from the single-span
  lane id, one shared invocation id per batch, and the composition digest of the rows it
  was packed with — so a batched vector can never be read as an unbatched one, and two
  batches of the same rows are distinguishable.
* **Teardown, including on failure.** The hooks are gone and the hook state is empty
  after a normal return *and* after a reduction raises mid-batch. A leaked hook would
  corrupt whatever ran next rather than failing here.

The `independent` prefill policy stacks per-row caches built by unpadded single-row
forwards, which is why it requires equal prefix lengths and why the forward-shape
counter asserts `[1, 1, 2]` rather than `[2, 2]`.

Real tiny Llama on a CPU. Batched submission is outside what the replay CLI accepts:
it is qualified here as arithmetic, not certified as a deployment path.
"""

import numpy as np
import pytest
import torch

from test_fast_lane_equivalence import tiny_loaded
from anamnesis.config import ExtractionConfig, FeaturePipelineConfig
from anamnesis.extraction.fast.batch_layout import ReplaySpan
from anamnesis.extraction.fast.features import GpuFeatureLane
from anamnesis.extraction.fast.schema import resolve_gpu_schema
from anamnesis.extraction.replay.cached import replay_extract_cached
from anamnesis.extraction.feature_pipeline import compute_features_with_families_from_data


@pytest.mark.parametrize(
    "prefixes,continuations,prefill_policy",
    [
        ((7, 7), (17, 17), "batched"),
        ((3, 11), (17, 13), "batched"),
        ((11, 3), (13, 17), "batched"),
        ((7, 7), (17, 13), "independent"),
    ],
)
def test_batch_feature_parity_repeat_and_cleanup(
    prefixes, continuations, prefill_policy, monkeypatch
):
    loaded = tiny_loaded()
    rng = np.random.default_rng(341)
    spans = tuple(
        ReplaySpan(tuple(rng.integers(0, 64, size=p + n).tolist()), p, p + n)
        for p, n in zip(prefixes, continuations, strict=True)
    )
    pm = rng.normal(0, 0.01, size=(4, 64, 32)).astype(np.float32)
    components = rng.normal(size=(5, 32)).astype(np.float32)
    mean = np.zeros(32, dtype=np.float32)
    extraction = ExtractionConfig(
        sampled_layers=[0, 1, 2], pca_layers=[0, 1], pca_components=5,
        early_layer_cutoff=8, late_layer_cutoff=24,
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
        contrastive_layers=[8, 16, 20, 24, 28],
    )
    schema = resolve_gpu_schema(3, 16, extraction, families, components)
    lane = GpuFeatureLane(
        extraction,
        families,
        list(schema.feature_names),
        pm,
        components,
        mean,
        device="cpu",
        calibration_sha256="a" * 64,
    )
    # Transformers lazily installs its own persistent output-capture hooks.
    # Establish that baseline before checking our temporary hook teardown.
    loaded.disable_hooks()
    with torch.no_grad():
        loaded.model(
            torch.tensor([spans[0].tokens]),
            use_cache=False,
            output_hidden_states=True,
            output_attentions=True,
        )
    hooks = [tuple(m.self_attn._forward_hooks) for m in loaded.model.model.layers]
    forwards = []
    counter = loaded.model.register_forward_pre_hook(
        lambda _m, args: forwards.append(tuple(args[0].shape))
    )
    first = lane.replay_batch(loaded, spans, prefill_policy=prefill_policy)
    counter.remove()
    if prefill_policy == "batched":
        assert len(forwards) == 2 and all(shape[0] == 2 for shape in forwards)
    else:
        assert [shape[0] for shape in forwards] == [1, 1, 2]
    second = lane.replay_batch(loaded, spans, prefill_policy=prefill_policy)
    assert first[0].metadata["lane_id"] != lane.lane_id
    assert first[0].metadata["lane_id"] == first[1].metadata["lane_id"]
    assert (
        first[0].metadata["batch_invocation_id"]
        != second[0].metadata["batch_invocation_id"]
    )
    assert not loaded.hook_state.pre_rope_keys
    assert [
        tuple(m.self_attn._forward_hooks) for m in loaded.model.model.layers
    ] == hooks
    for i, span in enumerate(spans):
        assert first[i].features.tobytes() == second[i].features.tobytes()
        assert first[i].metadata["replay_id"] != second[i].metadata["replay_id"]
        loaded.disable_hooks()
        with torch.no_grad():
            prefix = loaded.model(
                torch.tensor([span.tokens[: span.start]]),
                use_cache=True,
                return_dict=True,
            )
        raw = replay_extract_cached(
            loaded,
            prefix.past_key_values,
            list(span.tokens[span.start : span.end]),
            span.start,
            pm,
        )
        reference = compute_features_with_families_from_data(
            raw, extraction, families, components, mean
        )
        failures = [
            (name, float(actual), float(expected))
            for name, actual, expected in zip(
                schema.feature_names, first[i].features, reference.features, strict=True
            )
            if not np.isclose(actual, expected, atol=5e-6, rtol=3e-5)
        ]
        assert not failures, failures
        np.testing.assert_allclose(
            first[i].knnlm_baseline, reference.knnlm_baseline, atol=1e-6, rtol=1e-5
        )
        logits = np.stack(raw.logits).astype(np.float64)
        chosen = np.asarray(raw.chosen_token_ids, dtype=np.int64)
        maxima = logits.max(axis=-1)
        log_normalizer = maxima + np.log(np.exp(logits - maxima[:, None]).sum(axis=-1))
        expected_logprob = float(
            np.mean(logits[np.arange(len(chosen)), chosen] - log_normalizer)
        )
        assert np.isclose(first[i].mean_logprob, expected_logprob, atol=1e-6, rtol=1e-6)
        assert first[i].mean_logprob == second[i].mean_logprob

    def fail(*args, **kwargs):
        raise RuntimeError("injected reduction failure")

    monkeypatch.setattr(lane, "_reduce_capture", fail)
    with pytest.raises(RuntimeError, match="injected"):
        lane.replay_batch(loaded, spans, prefill_policy=prefill_policy)
    assert not loaded.hook_state.pre_rope_keys
    assert [
        tuple(m.self_attn._forward_hooks) for m in loaded.model.model.layers
    ] == hooks
