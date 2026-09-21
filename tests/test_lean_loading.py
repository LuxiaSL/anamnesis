"""Validate lean-mode load_raw_tensors equals full-mode for the requested surfaces.

load_raw_tensors grew opt-in `surfaces=` / `attn_layers=` parameters so callers
that read only one surface (e.g. hidden-only encoder scripts) stop paying for
full attention/gate/dense-logits reconstruction. Defaults must preserve the
exact historical behavior; lean mode must be bit-identical to full mode on
every surface the caller requested.

Covers both npz schemas:
  - v2 (save_raw_tensors): sampled hidden/attention layers, gates, per-gen pos-means
  - v3 (save_raw_tensors_v3): all-layer hidden/attention/keys/values, sampled queries/gates

Usage:
    python tests/test_lean_loading.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from anamnesis.config import ExtractionConfig, FeaturePipelineConfig
from anamnesis.extraction.feature_pipeline import compute_features_v2_from_data
from anamnesis.extraction.raw_saver import (
    VALID_SURFACES,
    load_raw_tensors,
    save_raw_tensors,
    save_raw_tensors_v3,
)
from anamnesis.extraction.state_extractor import (
    RawGenerationData,
    extract_cache_and_keys,
)

# Small synthetic model geometry (fast, CPU-only)
T = 12
NUM_LAYERS = 6  # transformer layers (hidden_states adds +1 embedding row)
HIDDEN_DIM = 16
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 8
INTERMEDIATE = 32
VOCAB = 200
PROMPT_LEN = 5
SAMPLED_LAYERS = [0, 2, 5]


def _make_synthetic_data(rng: np.random.Generator, all_layer: bool) -> RawGenerationData:
    """Build a RawGenerationData resembling a real capture (growing seq lens)."""
    kv_layers = list(range(NUM_LAYERS)) if all_layer else SAMPLED_LAYERS
    hidden_states = [
        rng.standard_normal((NUM_LAYERS + 1, HIDDEN_DIM)).astype(np.float32)
        for _ in range(T)
    ]
    attentions = []
    for t in range(T):
        seq_len = PROMPT_LEN + t + 1
        a = rng.random((NUM_LAYERS, NUM_HEADS, seq_len)).astype(np.float32)
        a /= a.sum(axis=-1, keepdims=True)  # normalized like softmax rows
        attentions.append(a)
    logits = [rng.standard_normal(VOCAB).astype(np.float32) * 3.0 for _ in range(T)]
    pre_rope_keys = {
        l: [rng.standard_normal((NUM_KV_HEADS, HEAD_DIM)).astype(np.float32) for _ in range(T)]
        for l in kv_layers
    }
    v_proj_values = {
        l: [rng.standard_normal((NUM_KV_HEADS, HEAD_DIM)).astype(np.float32) for _ in range(T)]
        for l in kv_layers
    }
    queries = {
        l: [rng.standard_normal((NUM_HEADS, HEAD_DIM)).astype(np.float32) for _ in range(T)]
        for l in SAMPLED_LAYERS
    }
    gate_activations = {
        l: [rng.standard_normal(INTERMEDIATE).astype(np.float32) for _ in range(T)]
        for l in SAMPLED_LAYERS
    }
    return RawGenerationData(
        hidden_states=hidden_states,
        attentions=attentions,
        logits=logits,
        chosen_token_ids=rng.integers(0, VOCAB, size=T).astype(np.float32),
        pre_rope_keys=pre_rope_keys,
        prompt_length=PROMPT_LEN,
        positional_means=None,
        gate_activations=gate_activations,
        v_proj_values=v_proj_values,
        queries=queries,
    )


def _layer_dicts_equal(a: dict | None, b: dict | None, name: str) -> None:
    assert (a is None) == (b is None), f"{name}: presence mismatch ({a is None} vs {b is None})"
    if a is None:
        return
    assert sorted(a.keys()) == sorted(b.keys()), f"{name}: layer keys differ"
    for l in a:
        assert len(a[l]) == len(b[l]), f"{name}[{l}]: length differs"
        for t in range(len(a[l])):
            np.testing.assert_array_equal(a[l][t], b[l][t], err_msg=f"{name}[{l}][{t}]")


def _config() -> ExtractionConfig:
    return ExtractionConfig(
        sampled_layers=SAMPLED_LAYERS,
        pca_layers=[2, 4],
        enable_residual_pca=False,          # needs a fitted PCA model
        enable_knnlm_baseline=False,  # ditto
        early_layer_cutoff=1,
        late_layer_cutoff=4,
    )


def _check_one_schema(raw_dir: Path, label: str) -> None:
    full = load_raw_tensors(0, raw_dir)

    # ── Per-surface lean loads are bit-identical to the full load ──
    lean_hidden = load_raw_tensors(0, raw_dir, surfaces=("hidden",))
    assert len(lean_hidden.hidden_states) == len(full.hidden_states)
    for t in range(len(full.hidden_states)):
        np.testing.assert_array_equal(lean_hidden.hidden_states[t], full.hidden_states[t])
    assert lean_hidden.attentions == [] and lean_hidden.logits == []
    assert lean_hidden.pre_rope_keys == {} and lean_hidden.gate_activations is None
    assert lean_hidden.prompt_length == full.prompt_length
    np.testing.assert_array_equal(lean_hidden.chosen_token_ids, full.chosen_token_ids)

    lean_kvq = load_raw_tensors(0, raw_dir, surfaces=("keys", "values", "queries"))
    _layer_dicts_equal(lean_kvq.pre_rope_keys or None, full.pre_rope_keys or None, "keys")
    _layer_dicts_equal(lean_kvq.v_proj_values, full.v_proj_values, "values")
    _layer_dicts_equal(lean_kvq.queries, full.queries, "queries")
    assert lean_kvq.hidden_states == [] and lean_kvq.attentions == []

    lean_attn = load_raw_tensors(0, raw_dir, surfaces=("attention",))
    assert len(lean_attn.attentions) == len(full.attentions)
    for t in range(len(full.attentions)):
        np.testing.assert_array_equal(lean_attn.attentions[t], full.attentions[t])

    lean_gate = load_raw_tensors(0, raw_dir, surfaces=("gate",))
    _layer_dicts_equal(lean_gate.gate_activations, full.gate_activations, "gate")

    lean_logits = load_raw_tensors(0, raw_dir, surfaces=("logits",))
    assert len(lean_logits.logits) == len(full.logits)
    for t in range(len(full.logits)):
        np.testing.assert_array_equal(lean_logits.logits[t], full.logits[t])

    # ── attn_layers: requested layers identical, others zero, shape unchanged ──
    lean_sub = load_raw_tensors(0, raw_dir, attn_layers=SAMPLED_LAYERS)
    assert len(lean_sub.attentions) == len(full.attentions)
    saved_attn = {int(l) for l in np.load(raw_dir / "gen_000.npz")["saved_layers_attn"]}
    for t in range(len(full.attentions)):
        assert lean_sub.attentions[t].shape == full.attentions[t].shape
        for l in range(NUM_LAYERS):
            if l in SAMPLED_LAYERS:
                np.testing.assert_array_equal(
                    lean_sub.attentions[t][l], full.attentions[t][l],
                    err_msg=f"attn t={t} l={l}",
                )
            elif l in saved_attn:
                assert not np.any(lean_sub.attentions[t][l]), f"attn t={t} l={l} not zeroed"

    # ── Feature-level equivalence for sampled-only readers ──
    # NOTE: attn_entropy_* and head_agreement_* deliberately read ALL attention layers
    # present, so attn_layers=sampled is NOT feature-identical for the whole set of core
    # blocks on all-layer banks. Everything that reads attention at sampled layers only
    # (cache-and-keys, attention_flow, per_head, temporal_dynamics) must be bit-identical:
    config = _config()
    ck_full, ck_names_full = extract_cache_and_keys(full, config)
    ck_lean, ck_names_lean = extract_cache_and_keys(lean_sub, config)
    assert ck_names_full == ck_names_lean
    np.testing.assert_array_equal(
        ck_full, ck_lean, err_msg="cache and keys differ under attn_layers=sampled"
    )

    fam_cfg = FeaturePipelineConfig(
        include_core_blocks=False,
        enable_attention_flow=True,
        enable_temporal_dynamics=True,
        enable_per_head=True,
        enable_gate_features=True,
        # Required per-model fields. The four families under test read `sampled_layers`
        # from the ExtractionConfig, so these two only have to be a valid layer plan
        # for this synthetic geometry; no enabled family reads them.
        trajectory_layers=SAMPLED_LAYERS,
        contrastive_layers=SAMPLED_LAYERS,
    )
    fam_full = compute_features_v2_from_data(full, config, fam_cfg)
    fam_lean = compute_features_v2_from_data(lean_sub, config, fam_cfg)
    assert fam_full.feature_names == fam_lean.feature_names
    np.testing.assert_array_equal(
        fam_full.features, fam_lean.features,
        err_msg="sampled-layer families differ under attn_layers=sampled",
    )

    # ── Error handling ──
    try:
        load_raw_tensors(0, raw_dir, surfaces=("hidden", "bogus"))
        raise AssertionError("expected ValueError for unknown surface name")
    except ValueError as e:
        assert "bogus" in str(e)

    print(f"  OK [{label}]: lean == full on every requested surface; "
          f"features identical with attn_layers={SAMPLED_LAYERS}")


def test_lean_loading_v3() -> None:
    rng = np.random.default_rng(42)
    data = _make_synthetic_data(rng, all_layer=True)
    with tempfile.TemporaryDirectory() as td:
        raw_dir = Path(td)
        save_raw_tensors_v3(data, 0, raw_dir, prompt_length=PROMPT_LEN, top_k_logits=20)
        _check_one_schema(raw_dir, "v3 all-layer")


def test_lean_loading_v2() -> None:
    rng = np.random.default_rng(43)
    data = _make_synthetic_data(rng, all_layer=False)
    data.v_proj_values = None
    data.queries = None
    with tempfile.TemporaryDirectory() as td:
        raw_dir = Path(td)
        save_raw_tensors(
            data, 0, raw_dir, config=_config(),
            prompt_length=PROMPT_LEN, top_k_logits=20,
        )
        _check_one_schema(raw_dir, "v2 sampled-layer")


def test_empty_generation() -> None:
    """T=0 files short-circuit identically in lean and full mode."""
    with tempfile.TemporaryDirectory() as td:
        raw_dir = Path(td)
        empty = RawGenerationData(
            hidden_states=[], attentions=[], logits=[],
            chosen_token_ids=np.array([], dtype=np.float32),
            pre_rope_keys={}, prompt_length=0,
        )
        save_raw_tensors_v3(empty, 0, raw_dir, prompt_length=0)
        for kwargs in ({}, {"surfaces": ("hidden",)}, {"attn_layers": [0]}):
            out = load_raw_tensors(0, raw_dir, **kwargs)
            assert out.hidden_states == [] and out.attentions == [] and out.logits == []
    print("  OK [empty]: T=0 short-circuit unchanged in lean mode")


def main() -> int:
    print(f"Valid surfaces: {sorted(VALID_SURFACES)}")
    test_lean_loading_v3()
    test_lean_loading_v2()
    test_empty_generation()
    print("ALL LEAN-LOADING TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
