"""The agreement proof: the fast lane against the numeric anchor, on one box.

The program has two extraction paths and one definition of a feature. The anchor
(`state_extractor` via `feature_pipeline`) says what each named number is; the fast
lane (`extraction.fast`) computes the same named vector inside the forward pass. If
the two disagree, every claim that mixes their outputs is measuring the difference
between two implementations and calling it signal. So the agreement is not an
assumption recorded in a docstring — it is this test, and it runs wherever the
suite runs.

It runs on a CPU with no checkpoint. The model is a real `LlamaForCausalLM` with
random weights, small enough to forward in milliseconds: real because the lane
reads eager attention weights out of `self_attn`'s forward hook and refuses a model
whose `model_type` is not `llama`, so the hand-built decoder in `synthetic_runtime`
cannot stand in here. Random weights are not a limitation for this claim — the
claim is about two computations over the same tensors, not about trained values.

Three things are asserted together, because each is worthless without the others:

1. **Agreement.** Every feature in the vector matches the anchor's, per name, at
   float32-reduction tolerance. A mismatch reports the feature's name, so a
   disagreement is attributable rather than a single failed comparison.
2. **Repeatability.** Two lane runs over one span are byte-identical, and each
   carries its own replay id. Agreement measured once on a nondeterministic path
   would be a coincidence.
3. **Non-interference.** After the lane runs, the hooks it installed are gone, the
   hook state is empty, and the anchor's own path reproduces its earlier vector
   byte-for-byte. A fast path that leaves the model altered has not been proven
   against anything.

The span lengths are the branch boundaries of the short-series name contract: 2 and
3 tokens hit the degenerate branches where families emit zeros under their full
names, 17 and 65 hit the windowed and spectral paths. Both replay geometries
(fresh-cache continuation and one no-cache full pass) and both PCA layouts (pooled
and per-layer) are covered, since each is a separate arithmetic path.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from anamnesis.config import ExtractionConfig, FeaturePipelineConfig, ModelConfig
from anamnesis.extraction.feature_pipeline import compute_features_with_families_from_data
from anamnesis.extraction.fast.features import GpuFeatureLane
from anamnesis.extraction.fast.schema import resolve_gpu_schema
from anamnesis.extraction.model_loader import (
    HookState,
    LoadedModel,
    _make_k_proj_hook,
    _make_v_proj_hook,
    _make_q_proj_hook,
    _make_gate_proj_hook,
)
from anamnesis.extraction.replay.cached import replay_extract_cached
from anamnesis.extraction.replay.extract import replay_extract

# The layer plan and band cutoffs the 8B preset declares. Both configuration
# classes require them explicitly, and the fast lane's schema is derived from the
# anchor's, so the two paths must be handed the same plan or they are not being
# compared.
SAMPLED_LAYERS = [0, 8, 16, 20, 24, 28, 31]
TRAJECTORY_LAYERS = [8, 16, 20, 24, 28]
CONTRASTIVE_LAYERS = [8, 16, 20, 24, 28]
EARLY_LAYER_CUTOFF = 8
LATE_LAYER_CUTOFF = 24
ANCHOR_LAYER_PLAN = ExtractionConfig(
    sampled_layers=SAMPLED_LAYERS,
    pca_layers=TRAJECTORY_LAYERS,
    early_layer_cutoff=EARLY_LAYER_CUTOFF,
    late_layer_cutoff=LATE_LAYER_CUTOFF,
)


def tiny_loaded():
    """A real Llama decoder with random weights, hooked the way `load_model` hooks one.

    Shared with the other fast-lane tests: the lane's own preconditions (dense
    Llama, eager attention, one device, hooks on the projection modules) are what
    make the fixture real rather than synthetic.
    """
    torch.manual_seed(751)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=80,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    state = HookState()
    handles = []
    for layer, module in enumerate(model.model.layers):
        for projection, fn in (
            ("k_proj", _make_k_proj_hook),
            ("v_proj", _make_v_proj_hook),
        ):
            handles.append(
                getattr(module.self_attn, projection).register_forward_hook(
                    fn(layer, state, 2, 8)
                )
            )
        handles.append(
            module.self_attn.q_proj.register_forward_hook(
                _make_q_proj_hook(layer, state, 4, 8)
            )
        )
        handles.append(
            module.mlp.gate_proj.register_forward_hook(
                _make_gate_proj_hook(layer, state)
            )
        )
    loader_config = ModelConfig(
        model_id="tiny-llama",
        torch_dtype="float32",
        num_layers=config.num_hidden_layers,
        hidden_dim=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
    )
    return LoadedModel(model, None, state, handles, loader_config)


@pytest.mark.parametrize("continuation", [2, 3, 4, 17, 65])
@pytest.mark.parametrize("replay_path", ["cached", "full"])
@pytest.mark.parametrize("per_layer_pca", [False, True])
def test_full_lane_matches_reference_and_repeats_with_fresh_cache(
    continuation, replay_path, per_layer_pca
):
    loaded = tiny_loaded()
    prefix = 11
    rng = np.random.default_rng(492)
    tokens = rng.integers(0, 64, size=prefix + continuation).tolist()
    pm = rng.normal(0, 0.01, size=(4, prefix + continuation, 32)).astype(np.float32)
    components = rng.normal(size=(5, 32)).astype(np.float32)
    mean = rng.normal(0, 0.01, size=32).astype(np.float32)
    if per_layer_pca:
        components = {0: components, 1: components * 0.3}
        mean = {0: mean, 1: mean + 0.1}
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
        contrastive_layers=CONTRASTIVE_LAYERS,
    )

    def reference_raw():
        if replay_path == "full":
            return replay_extract(loaded, tokens, prefix, pm)
        loaded.disable_hooks()
        with torch.no_grad():
            pre = loaded.model(
                torch.tensor([tokens[:prefix]]), use_cache=True, return_dict=True
            )
        return replay_extract_cached(
            loaded, pre.past_key_values, tokens[prefix:], prefix, pm
        )

    raw = reference_raw()
    ref = compute_features_with_families_from_data(raw, extraction, families, components, mean)
    schema = resolve_gpu_schema(3, continuation - 1, extraction, families, components)
    assert schema.feature_names == tuple(ref.feature_names)
    assert schema.family_slices == ref.block_slices
    original_hooks = [
        tuple(module.self_attn._forward_hooks) for module in loaded.model.model.layers
    ]
    lane = GpuFeatureLane(
        extraction,
        families,
        ref.feature_names,
        pm,
        components,
        mean,
        device="cpu",
        calibration_sha256="a" * 64,
        replay_path=replay_path,
    )
    one = lane.replay_span(loaded, tokens, prefix, len(tokens))
    two = lane.replay_span(loaded, tokens, prefix, len(tokens))
    assert one.features.tobytes() == two.features.tobytes()
    assert one.metadata["replay_id"] != two.metadata["replay_id"]
    assert one.metadata["input_tokens_sha256"] == two.metadata["input_tokens_sha256"]
    assert one.metadata["attention_layers_reduced"] == 3
    assert not one.metadata["certified"]
    assert one.metadata["lane_id"] == lane.lane_id
    assert one.metadata["replay_path"] == replay_path
    assert tuple(ref.feature_names) == one.feature_names
    failures = []
    for i, name in enumerate(ref.feature_names):
        if not np.isclose(one.features[i], ref.features[i], atol=5e-6, rtol=3e-5):
            failures.append((name, float(ref.features[i]), float(one.features[i])))
    assert not failures, failures
    np.testing.assert_array_equal(one.knnlm_baseline, ref.knnlm_baseline)
    # Hook teardown must also let the anchor's path run afterward.
    assert not loaded.hook_state.pre_rope_keys
    assert [
        tuple(module.self_attn._forward_hooks) for module in loaded.model.model.layers
    ] == original_hooks
    raw_after = reference_raw()
    ref_after = compute_features_with_families_from_data(
        raw_after, extraction, families, components, mean
    )
    assert ref_after.features.tobytes() == ref.features.tobytes()


def test_replay_path_is_part_of_lane_identity():
    families = FeaturePipelineConfig(
        include_core_blocks=True,
        enable_residual_trajectory=True,
        enable_attention_flow=True,
        enable_gate_features=True,
        enable_per_head=True,
        enable_value_geometry=True,
        enable_qk_geometry=True,
        enable_kv_cka=True,
        trajectory_layers=TRAJECTORY_LAYERS,
        contrastive_layers=CONTRASTIVE_LAYERS,
    )
    args = (
        ANCHOR_LAYER_PLAN,
        families,
        [],
        np.zeros((4, 10, 32), np.float32),
        np.zeros((5, 32), np.float32),
        np.zeros(32, np.float32),
    )
    cached = GpuFeatureLane(
        *args, device="cpu", calibration_sha256="a" * 64, replay_path="cached"
    )
    full = GpuFeatureLane(
        *args, device="cpu", calibration_sha256="a" * 64, replay_path="full"
    )
    assert cached.lane_id != full.lane_id


def test_rejects_non_eager_before_capturing():
    loaded = tiny_loaded()
    families = FeaturePipelineConfig(
        include_core_blocks=True,
        enable_residual_trajectory=True,
        enable_attention_flow=True,
        enable_gate_features=True,
        enable_per_head=True,
        enable_value_geometry=True,
        enable_qk_geometry=True,
        enable_kv_cka=True,
        trajectory_layers=TRAJECTORY_LAYERS,
        contrastive_layers=CONTRASTIVE_LAYERS,
    )
    lane = GpuFeatureLane(
        ANCHOR_LAYER_PLAN,
        families,
        [],
        np.zeros((4, 10, 32), np.float32),
        np.zeros((5, 32), np.float32),
        np.zeros(32, np.float32),
        device="cpu",
        calibration_sha256="a" * 64,
    )
    loaded.model.config._attn_implementation = "sdpa"
    with pytest.raises(ValueError, match="eager"):
        lane.replay_span(loaded, [1, 2, 3, 4], 1, 4)
