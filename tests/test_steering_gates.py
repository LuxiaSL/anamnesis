"""The gates, each against the failure it was written to catch.

The on-policy gate needs a model on a CUDA device, so its two pure pieces — the
pilot subset and the layer arithmetic — are checked here and the agreement pass
itself belongs to the golden path. The other three gates are pure numpy and are
checked against the artifacts they were built to refuse.
"""

from __future__ import annotations

import numpy as np
import pytest

from anamnesis.steering import gates, vectors


def diagonal_spectrum(evals: list[float]) -> vectors.Spectrum:
    array = np.asarray(evals, dtype=np.float64)
    return vectors.Spectrum.from_arrays(array, np.eye(len(array)))


# ── The pilot subset ──────────────────────────────────────────────────────────
def test_pilot_generations_stride_and_drop_short_spans() -> None:
    entries = {
        str(k * 40): {"input_ids": list(range(100)), "prompt_length": 10} for k in range(5)
    }
    entries["80"] = {"input_ids": list(range(12)), "prompt_length": 10}
    pilots = gates.pilot_generations(entries, count=5, stride=40, min_generated=32)
    assert len(pilots) == 4


def test_pilot_generations_tolerates_a_sparse_bank() -> None:
    assert gates.pilot_generations({}, count=3) == []


# ── Upstream zero ─────────────────────────────────────────────────────────────
def test_deepest_layer_read_takes_the_maximum_across_a_cross_layer_name() -> None:
    """A cross-layer feature reads the deeper block too, so the max is the key."""
    assert gates.deepest_layer_read("kv_value_cka_L0_L28") == 28
    assert gates.deepest_layer_read("attn_entropy_L14") == 14
    assert gates.deepest_layer_read("logit_entropy_mean") is None


def test_delta_features_are_upstream_one_block_shallower() -> None:
    assert gates.is_upstream("attn_entropy_L13", 13, site=14) is True
    assert gates.is_upstream("attn_entropy_L14", 14, site=14) is False
    assert gates.is_upstream("delta_attn_L12", 12, site=14) is True
    assert gates.is_upstream("delta_attn_L13", 13, site=14) is False


def test_upstream_zero_passes_on_an_identical_upstream_block() -> None:
    names = ["a_L2", "b_L13", "c_L14", "d_L20"]
    unsteered = {g: (names, np.array([1.0, 2.0, 3.0, 4.0])) for g in (0, 1)}
    steered = {g: (names, np.array([1.0, 2.0, 9.0, 9.0])) for g in (0, 1)}
    report = gates.upstream_zero_check(steered, unsteered, site=14)
    assert report["PASS"] is True
    assert report["n_upstream_features"] == 2
    assert report["max_abs_upstream_delta"] == 0.0


def test_upstream_zero_fails_on_the_smallest_nonzero_difference() -> None:
    """The bar is exact zero, not a tolerance: identical tokens run identically."""
    names = ["a_L2", "b_L14"]
    unsteered = {0: (names, np.array([1.0, 3.0]))}
    steered = {0: (names, np.array([1.0 + 1e-12, 3.0]))}
    report = gates.upstream_zero_check(steered, unsteered, site=14)
    assert report["PASS"] is False
    assert report["worst_feature"] == "a_L2"
    assert report["offenders_sample"]


def test_upstream_zero_reports_a_feature_name_mismatch_rather_than_comparing() -> None:
    steered = {0: (["a_L2"], np.array([1.0]))}
    unsteered = {0: (["b_L2"], np.array([1.0]))}
    assert "mismatch" in gates.upstream_zero_check(steered, unsteered, site=14)["error"]


def test_upstream_zero_reports_an_empty_intersection() -> None:
    steered = {0: (["a_L2"], np.array([1.0]))}
    unsteered = {1: (["a_L2"], np.array([1.0]))}
    assert "no generation ids shared" in gates.upstream_zero_check(steered, unsteered, 14)["error"]


def test_signature_directory_round_trip(tmp_path) -> None:
    for gen_id in (0, 1):
        np.savez(
            tmp_path / f"gen_{gen_id:03d}.npz",
            feature_names=np.array(["a_L2", "b_L14"]),
            features=np.array([float(gen_id), 2.0], dtype=np.float32),
        )
    loaded = gates.load_signature_directory(tmp_path)
    assert sorted(loaded) == [0, 1]
    assert loaded[1][0] == ["a_L2", "b_L14"]
    assert loaded[1][1][0] == pytest.approx(1.0)


# ── A direction's own matched null ────────────────────────────────────────────
def test_the_mean_difference_null_is_the_top_k_trace_share() -> None:
    """The derivation that makes this null free — and it needs no data."""
    spectrum = diagonal_spectrum([10.0, 5.0, 4.0, 1.0])
    expected = (10.0 + 5.0) / 20.0
    assert gates.analytic_null_topk(spectrum, "diff_of_means", 2) == pytest.approx(expected)


def test_the_isotropic_null_is_k_over_d() -> None:
    spectrum = diagonal_spectrum([9.0, 4.0, 1.0, 0.5])
    assert gates.analytic_null_topk(spectrum, "isotropic_random", 2) == pytest.approx(0.5)


def test_the_gate_refuses_a_gradient_rather_than_answering() -> None:
    """A number that renders as a verdict is worse than a crash."""
    spectrum = diagonal_spectrum([9.0, 4.0, 1.0])
    with pytest.raises(NotImplementedError, match="NO KNOWN NULL"):
        gates.assert_against_own_null(np.array([1.0, 0, 0]), "gradient", spectrum, k=1, n_draws=10)


def test_the_gate_refuses_a_whitened_direction_without_shuffled_labels() -> None:
    spectrum = diagonal_spectrum([9.0, 4.0, 1.0])
    with pytest.raises(NotImplementedError, match="shuffled-label"):
        gates.assert_against_own_null(np.array([1.0, 0, 0]), "lda_whitened", spectrum, k=1, n_draws=10)


def test_the_gate_refuses_an_undeclared_construction() -> None:
    spectrum = diagonal_spectrum([2.0, 1.0])
    with pytest.raises(ValueError, match="declare one of"):
        gates.assert_against_own_null(np.array([1.0, 0.0]), "vibes", spectrum)  # type: ignore[arg-type]


def test_the_same_vector_reads_differently_against_two_nulls() -> None:
    """The failure the gate exists for: the null decides the verdict, not the vector."""
    spectrum = diagonal_spectrum([4.0, 4.0] + [1.0] * 18)
    # Top-2 energy set to exactly Σ's top-2 trace share: what a label-free mean
    # difference is expected to look like, and four times the isotropic expectation.
    share = float(spectrum.evals[:2].sum() / spectrum.evals.sum())
    top_heavy = np.concatenate([
        np.full(2, np.sqrt(share / 2)), np.full(18, np.sqrt((1.0 - share) / 18))
    ])
    against_isotropic = gates.assert_against_own_null(
        top_heavy, "isotropic_random", spectrum, k=2, n_draws=400
    )
    against_own = gates.assert_against_own_null(
        top_heavy, "diff_of_means", spectrum, k=2, n_draws=400
    )
    assert against_isotropic.percentile > 95
    assert "ABOVE its own null" in against_isotropic.verdict
    assert "INSIDE its own null" in against_own.verdict
    assert against_own.observed_topk == pytest.approx(against_isotropic.observed_topk)


def test_a_verdict_inside_the_null_says_the_position_is_the_construction() -> None:
    spectrum = diagonal_spectrum([4.0] * 8)
    rng = np.random.default_rng(2)
    verdict = gates.assert_against_own_null(
        rng.standard_normal(8), "isotropic_random", spectrum, k=4, n_draws=400
    )
    assert "INSIDE its own null" in verdict.verdict
    assert "isotropic_random" in str(verdict)


def test_empirical_null_is_reproducible_under_a_seed() -> None:
    spectrum = diagonal_spectrum([5.0, 3.0, 1.0, 0.5])
    first = gates.empirical_null_topk(spectrum, "diff_of_means", 2, n_draws=50, seed=7)
    again = gates.empirical_null_topk(spectrum, "diff_of_means", 2, n_draws=50, seed=7)
    assert np.allclose(first, again)


# ── Shape ─────────────────────────────────────────────────────────────────────
def test_shape_audit_catches_a_degenerate_row_axis_that_covariates_call_clean() -> None:
    """The artifact that passes location and scale: a handful of rows holding an axis."""
    rng = np.random.default_rng(13)
    score = rng.normal(0.0, 1e-3, size=2000)
    score[:6] = 50.0
    covariate = rng.normal(size=2000)
    report = gates.audit_axis(score, {"generation_length": covariate})
    assert report["covariates"]["generation_length"]["location"] == pytest.approx(0.0, abs=0.1)
    assert report["kurtosis"] > gates.KURTOSIS_FLAG
    assert report["verdict"].startswith("ARTIFACT-SHAPED")
    assert any(flag.startswith("KURTOSIS") for flag in report["flags"])


def test_shape_audit_passes_a_healthy_axis() -> None:
    rng = np.random.default_rng(14)
    report = gates.audit_axis(rng.normal(size=4000), {"length": rng.normal(size=4000)})
    assert report["flags"] == []
    assert report["verdict"] == "shape OK"


def test_shape_audit_flags_a_scale_coupled_covariate() -> None:
    """Session one's leg: a score whose magnitude tracks a nuisance variable."""
    rng = np.random.default_rng(15)
    covariate = rng.uniform(0.0, 1.0, size=3000)
    score = rng.normal(size=3000) * (0.05 + covariate)
    report = gates.audit_axis(score, {"cap": covariate})
    assert abs(report["covariates"]["cap"]["scale"]) > 0.3
    assert any(flag.startswith("SCALE vs cap") for flag in report["flags"])


def test_minority_cluster_peels_a_shard_and_splits_a_real_axis() -> None:
    shard = np.concatenate([np.zeros(500), np.full(3, 40.0)])
    assert gates.minority_cluster(shard) == 3
    rng = np.random.default_rng(16)
    bimodal = np.concatenate([rng.normal(-4, 0.5, 250), rng.normal(4, 0.5, 250)])
    assert gates.minority_cluster(bimodal) > 200


def test_a_controlled_covariate_is_named_rather_than_correlated() -> None:
    rng = np.random.default_rng(17)
    report = gates.audit_axis(rng.normal(size=500), {"fixed": np.ones(500)})
    assert report["covariates"]["fixed"]["note"].startswith("no variance")
    assert report["covariates"]["fixed"]["location"] is None


def test_audit_axes_carries_the_variance_share_and_the_thresholds() -> None:
    """The parameterized form: scores in, verdicts out, no corpus loader involved."""
    rng = np.random.default_rng(18)
    scores = {"PC1": rng.normal(size=1500), "PC2": np.concatenate([np.zeros(1495), np.full(5, 30.0)])}
    report = gates.audit_axes(scores, {"length": rng.normal(size=1500)}, {"PC1": 0.4, "PC2": 0.18})
    assert report["axes"]["PC1"]["verdict"] == "shape OK"
    assert report["axes"]["PC2"]["verdict"].startswith("ARTIFACT-SHAPED")
    assert report["axes"]["PC2"]["var_ratio"] == pytest.approx(0.18)
    assert report["thresholds"]["kurtosis"] == gates.KURTOSIS_FLAG
