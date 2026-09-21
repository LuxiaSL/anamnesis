"""The golden master: the optimized state_extractor agrees with the reference one.

`state_extractor` is the numeric anchor, and `state_extractor_reference` is the
unoptimized statement of the same math. This file is where the two meet, which is
why they port together: an optimization that changed a number would otherwise be
indistinguishable from an optimization that did not.

The synthetic data covers the shapes that break extractors: a short generation, a
four-token one, a medium one, and the empty case where every block must still emit
its fixed-length vector of zeros with the right names.

Run under pytest for the equivalence assertions, or as a script for the same
comparison plus a timing read:

    python tests/test_extraction_equivalence.py --verbose
    python tests/test_extraction_equivalence.py --benchmark
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pytest

from anamnesis.config import ExtractionConfig
from anamnesis.extraction.state_extractor import (
    ExtractionResult,
    RawGenerationData,
    extract_all_features,
    extract_norms_and_output_stats,
    extract_attention_and_deltas,
    extract_cache_and_keys,
)


def _make_synthetic_data(
    T: int = 30,
    num_layers: int = 32,
    hidden_dim: int = 4096,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    vocab_size: int = 128256,
    prompt_length: int = 50,
    sampled_layers: list[int] | None = None,
    seed: int = 42,
) -> RawGenerationData:
    rng = np.random.RandomState(seed)
    if sampled_layers is None:
        sampled_layers = [0, 8, 16, 20, 24, 28, 31]

    hidden_states = [
        rng.randn(num_layers + 1, hidden_dim).astype(np.float32)
        for _ in range(T)
    ]

    attentions = []
    for t in range(T):
        seq_len = prompt_length + t + 1
        raw_attn = rng.rand(num_layers, num_heads, seq_len).astype(np.float32)
        raw_attn /= raw_attn.sum(axis=2, keepdims=True)
        attentions.append(raw_attn)

    logits = [
        rng.randn(vocab_size).astype(np.float32)
        for _ in range(T)
    ]

    chosen_token_ids = rng.randint(0, vocab_size, size=T).astype(np.float32)

    pre_rope_keys: dict[int, list[np.ndarray]] = {}
    for l_idx in sampled_layers:
        pre_rope_keys[l_idx] = [
            rng.randn(num_kv_heads, head_dim).astype(np.float32)
            for _ in range(T)
        ]

    positional_means = rng.randn(num_layers + 1, prompt_length + T + 10, hidden_dim).astype(np.float32) * 0.01

    return RawGenerationData(
        hidden_states=hidden_states,
        attentions=attentions,
        logits=logits,
        chosen_token_ids=chosen_token_ids,
        pre_rope_keys=pre_rope_keys,
        prompt_length=prompt_length,
        positional_means=positional_means,
    )


TEST_CASES: tuple[tuple[str, int, int], ...] = (
    ("T=30 (short gen, training_numbers-like)", 30, 128256),
    ("T=4 (very short, favorite_animal-like)", 4, 128256),
    ("T=100 (medium gen)", 100, 128256),
    ("T=0 (empty, edge case)", 0, 128256),
)

DEFAULT_RTOL = 1e-4
DEFAULT_ATOL = 1e-5


def _config() -> ExtractionConfig:
    """Residual PCA is off: it needs a fitted basis, which is banked data, not synthetic."""
    return ExtractionConfig(
        sampled_layers=[0, 8, 16, 20, 24, 28, 31],
        pca_layers=[8, 16, 20, 24, 28],
        early_layer_cutoff=8,
        late_layer_cutoff=24,
        enable_norms_and_output_stats=True,
        enable_attention_and_deltas=True,
        enable_cache_and_keys=True,
        enable_residual_pca=False,
        enable_knnlm_baseline=True,
    )


def _case_data(T: int, vocab_size: int) -> RawGenerationData:
    """The synthetic capture for one case; T=0 is built empty rather than sliced."""
    if T == 0:
        return RawGenerationData(
            hidden_states=[],
            attentions=[],
            logits=[],
            chosen_token_ids=np.array([], dtype=np.float32),
            pre_rope_keys={},
            prompt_length=50,
            positional_means=None,
        )
    return _make_synthetic_data(T=T, vocab_size=vocab_size)


def _run_reference(data: RawGenerationData, config: ExtractionConfig) -> ExtractionResult:
    from anamnesis.extraction.state_extractor_reference import (
        extract_all_features as ref_extract_all,
    )
    return ref_extract_all(data, config)


def _run_optimized(data: RawGenerationData, config: ExtractionConfig) -> ExtractionResult:
    return extract_all_features(data, config)


def _compare(
    ref: ExtractionResult,
    opt: ExtractionResult,
    label: str,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    verbose: bool = False,
) -> tuple[bool, list[str]]:
    issues: list[str] = []

    if len(ref.features) != len(opt.features):
        issues.append(f"{label}: feature length mismatch: ref={len(ref.features)} opt={len(opt.features)}")
        return False, issues

    if ref.feature_names != opt.feature_names:
        diffs = [
            (i, r, o)
            for i, (r, o) in enumerate(zip(ref.feature_names, opt.feature_names))
            if r != o
        ]
        issues.append(f"{label}: {len(diffs)} name mismatches: {diffs[:5]}")

    close = np.allclose(ref.features, opt.features, rtol=rtol, atol=atol)
    if not close:
        diffs_idx = np.where(~np.isclose(ref.features, opt.features, rtol=rtol, atol=atol))[0]
        max_abs_diff = float(np.abs(ref.features - opt.features).max())
        max_rel_diff = float(
            np.abs(ref.features - opt.features)[diffs_idx].max()
            / np.maximum(np.abs(ref.features[diffs_idx]), 1e-12).max()
        ) if len(diffs_idx) > 0 else 0.0

        sample_diffs = []
        for idx in diffs_idx[:10]:
            name = ref.feature_names[idx] if idx < len(ref.feature_names) else f"[{idx}]"
            sample_diffs.append(f"  {name}: ref={ref.features[idx]:.8f} opt={opt.features[idx]:.8f}")

        issues.append(
            f"{label}: {len(diffs_idx)}/{len(ref.features)} features differ "
            f"(max_abs={max_abs_diff:.2e}, max_rel={max_rel_diff:.2e})"
        )
        if verbose:
            issues.extend(sample_diffs)

    if ref.block_slices != opt.block_slices:
        issues.append(f"{label}: block_slices differ: ref={ref.block_slices} opt={opt.block_slices}")

    return len(issues) == 0, issues


# ── the gate ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("label,T,vocab_size", TEST_CASES, ids=[c[0] for c in TEST_CASES])
def test_optimized_matches_reference(label: str, T: int, vocab_size: int) -> None:
    """Both extractors produce the same names, the same slices and the same numbers."""
    config = _config()
    data = _case_data(T, vocab_size)
    ref = _run_reference(data, config)
    opt = _run_optimized(data, config)
    passed, issues = _compare(ref, opt, label, DEFAULT_RTOL, DEFAULT_ATOL, verbose=True)
    assert passed, "\n".join(issues)


@pytest.mark.parametrize("label,T,vocab_size", TEST_CASES, ids=[c[0] for c in TEST_CASES])
def test_names_are_one_per_feature(label: str, T: int, vocab_size: int) -> None:
    """A name per dimension, and the slices partition the vector without a gap.

    A block that emits the wrong number of zeros on a short generation is the
    failure this catches: the vector still concatenates, so only the arithmetic
    between names, length and slice bounds shows it.
    """
    result = _run_optimized(_case_data(T, vocab_size), _config())
    assert len(result.feature_names) == len(result.features)
    bounds = sorted(result.block_slices.values())
    assert bounds[0][0] == 0
    for (_, end), (next_start, _) in zip(bounds, bounds[1:]):
        assert end == next_start
    assert bounds[-1][1] == len(result.features)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--benchmark", "-b", action="store_true")
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    args = parser.parse_args()

    config = _config()
    all_pass = True

    for label, T, vocab_size in TEST_CASES:
        print(f"\n{'='*60}")
        print(f"Test: {label}")
        print(f"{'='*60}")

        data = _case_data(T, vocab_size)

        try:
            ref_result = _run_reference(data, config)
        except Exception as e:
            print(f"  REFERENCE FAILED: {e}")
            all_pass = False
            continue

        try:
            opt_result = _run_optimized(data, config)
        except Exception as e:
            print(f"  OPTIMIZED FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_pass = False
            continue

        passed, issues = _compare(ref_result, opt_result, label, args.rtol, args.atol, args.verbose)

        if passed:
            print(f"  PASS: {len(ref_result.features)} features match (rtol={args.rtol}, atol={args.atol})")
        else:
            all_pass = False
            for issue in issues:
                print(f"  FAIL: {issue}")

        if args.benchmark and T > 0:
            n_runs = 5 if T <= 30 else 3
            print(f"\n  Benchmarking ({n_runs} runs)...")

            ref_times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _run_reference(data, config)
                ref_times.append(time.perf_counter() - t0)

            opt_times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _run_optimized(data, config)
                opt_times.append(time.perf_counter() - t0)

            ref_median = np.median(ref_times)
            opt_median = np.median(opt_times)
            speedup = ref_median / opt_median if opt_median > 0 else float("inf")
            print(f"  Reference: {ref_median:.3f}s (median)")
            print(f"  Optimized: {opt_median:.3f}s (median)")
            print(f"  Speedup:   {speedup:.1f}x")

    print(f"\n{'='*60}")
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
