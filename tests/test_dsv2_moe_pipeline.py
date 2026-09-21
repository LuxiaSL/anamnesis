"""Offline regression: the full v2 feature pipeline runs on DeepSeek-V2-Lite (MLA+MoE) shapes.

Catches the class of dim-mismatch / None-surface crashes that the GPU onboard-smoke misses
(it exercises only the baseline extractor, not the v2 families). The original miss: the dense
L0 gate (intermediate_size 10944) vs MoE shared_experts gate (moe_intermediate*n_shared 2816)
broke gate_features' cross-layer cosine — so M6 capture gates the MoE shared branch only.

Pure-CPU (synthetic RawGenerationData); no GPU/model. Run: `python -m pytest tests/test_dsv2_moe_pipeline.py`
or `python tests/test_dsv2_moe_pipeline.py`.
"""
from __future__ import annotations

import numpy as np

from anamnesis.feature_map import FeatureMap, Method, Source
from anamnesis.config import ExtractionConfig, FeaturePipelineConfig, MODEL_PRESETS
from anamnesis.extraction.feature_families.expert_routing import N_FEATURES_PER_LAYER
from anamnesis.extraction.feature_pipeline import compute_features_v2_from_data
from anamnesis.extraction.state_extractor import ExtractionResult, RawGenerationData


def _synthetic_dsv2_raw(T: int = 24) -> tuple[RawGenerationData, list[int], list[int]]:
    rng = np.random.default_rng(0)
    p = MODEL_PRESETS["dsv2-lite"]
    sl = p.sampled_layers                      # [0,5,11,15,18,22,26]
    moe = [l for l in sl if l != 0]            # gate/router on MoE layers only
    nl, hidden, heads = 27, 2048, 16
    gate_dim, n_experts, kv_lora = 1408 * 2, 64, 512   # shared gate width; routed experts; c_KV latent

    hs = [rng.standard_normal((nl + 1, hidden)).astype(np.float32) for _ in range(T)]
    att = [rng.random((nl, heads, 6 + t)).astype(np.float32) for t in range(T)]
    keys = {l: [rng.standard_normal((1, kv_lora)).astype(np.float32) for _ in range(T)] for l in sl}
    gates = {l: [rng.standard_normal(gate_dim).astype(np.float32) for _ in range(T)] for l in moe}

    def _dist():
        x = rng.normal(0, 2, n_experts)
        e = np.exp(x - x.max())
        return (e / e.sum()).astype(np.float32)

    rdist = {l: [_dist() for _ in range(T)] for l in moe}
    # v2.1: branch norms carry a 3rd column = per-token cos(shared_out, routed_out) ∈ [-1, 1]
    rnorms = {l: [np.array([abs(rng.normal(1, .2)), abs(rng.normal(2, .3)),
                            float(np.clip(rng.normal(0, .4), -1, 1))], dtype=np.float32)
                  for _ in range(T)] for l in moe}
    # v2.1: per-token ‖router_logits‖ (pre-softmax), one scalar per generated token
    rlogit = {l: [np.float32(abs(rng.normal(5, 1))) for _ in range(T)] for l in moe}
    data = RawGenerationData(
        hidden_states=hs, attentions=att,
        logits=[rng.standard_normal(2000).astype(np.float32) for _ in range(T)],
        chosen_token_ids=np.arange(T, dtype=np.float32), pre_rope_keys=keys, prompt_length=4,
        gate_activations=gates, v_proj_values=None, queries=None,   # MLA: no v_proj/q_proj capture
        router_dist=rdist, router_branch_norms=rnorms, router_logit_norms=rlogit)
    return data, sl, moe


def _run_full_v2_pipeline() -> tuple[ExtractionResult, list[int]]:
    """The pipeline under test, and the MoE layers its arity is checked against."""
    data, sl, moe = _synthetic_dsv2_raw()
    p = MODEL_PRESETS["dsv2-lite"]
    ec = ExtractionConfig(sampled_layers=sl, pca_layers=p.pca_layers,
                          early_layer_cutoff=p.early_layer_cutoff,
                          late_layer_cutoff=p.late_layer_cutoff, enable_residual_pca=False)
    fc = FeaturePipelineConfig(
        include_core_blocks=True, enable_residual_trajectory=True, enable_attention_flow=True,
        enable_gate_features=True, enable_temporal_dynamics=False, enable_per_head=True,
        enable_stft=True, enable_contrastive_projection=False, enable_value_geometry=True,
        enable_qk_geometry=True, enable_kv_cka=True, enable_expert_routing=True,
        trajectory_layers=p.trajectory_layers, contrastive_layers=p.contrastive_layers)

    return compute_features_v2_from_data(data, ec, fc, pca_components=None, pca_mean=None), moe


def test_dsv2_full_v2_pipeline_runs_clean() -> None:
    res, moe = _run_full_v2_pipeline()

    assert len(res.features) > 0
    assert np.isfinite(res.features).all(), "non-finite features"
    # v2.1: 19 per-layer × MoE layers + (adjacent-pairs + 1 global-mean = len(moe)) cross-layer CKA.
    # For M6's 6 MoE layers → 19×6 + 6 = 120. (Older "~129" = pre-desk 15-pair CKA; ruled to 6.)
    xrt_names = [n for n in res.feature_names if n.startswith("xrt_")]
    xrt_n = len(xrt_names)
    expected = N_FEATURES_PER_LAYER * len(moe) + len(moe)
    assert N_FEATURES_PER_LAYER == 19, f"per-layer {N_FEATURES_PER_LAYER} != 19"
    assert xrt_n == expected, f"xrt {xrt_n} != {expected}"
    # cross-layer CKA block present (adjacent-5 + global mean)
    assert sum(1 for n in xrt_names if "_cka_" in n) == len(moe), "cka block wrong size"
    assert "xrt_cka_global_mean" in xrt_names
    # feature_map places everything (incl every xrt name) — the acceptance check
    fmap = FeatureMap(res.feature_names, 27)
    assert len(fmap.unclassified()) == 0
    # all four non-learned METHOD rungs are now present on the routing source (v2.1 completeness)
    xrt_methods = {t.method for t in fmap.tags if t.source == Source.expert_routing}
    for m in (Method.geometry, Method.magnitude, Method.spectral, Method.distributional):
        assert m in xrt_methods, f"xrt missing method {m}: has {xrt_methods}"


if __name__ == "__main__":
    test_dsv2_full_v2_pipeline_runs_clean()
    r, _ = _run_full_v2_pipeline()
    xn = sum(1 for n in r.feature_names if n.startswith("xrt_"))
    print(f"PASS — {len(r.features)} features, {xn} xrt (v2.1), all finite, 0 unclassified, "
          f"all 4 method rungs present")
