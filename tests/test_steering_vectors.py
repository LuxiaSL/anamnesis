"""The construction conventions, each against the property it exists for.

Every check here runs on synthetic arrays small enough to reason about, which is
what makes them checks rather than regression snapshots: a failure names a broken
convention, not a changed number.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from anamnesis.steering import vectors


def spectrum_with_eigenvalues(evals: list[float], seed: int = 0) -> vectors.Spectrum:
    """A spectrum on a random orthonormal basis with the given eigenvalues."""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((len(evals), len(evals))))
    return vectors.Spectrum.from_arrays(np.asarray(evals, dtype=np.float64), basis)


# ── Elementary geometry ───────────────────────────────────────────────────────
def test_unit_refuses_a_zero_vector() -> None:
    with pytest.raises(ValueError, match="zero vector"):
        vectors.unit(np.zeros(4))


def test_cosine_is_scale_invariant() -> None:
    a, b = np.array([1.0, 2.0, 3.0]), np.array([3.0, 1.0, 0.5])
    assert vectors.cosine(a, b) == pytest.approx(vectors.cosine(5.0 * a, 0.1 * b))


# ── The spectrum's descending invariant ───────────────────────────────────────
def test_spectrum_sorts_ascending_input_descending() -> None:
    """An eigendecomposition stored ascending is the standing trap; the sort is the fix."""
    evals = np.array([0.1, 1.0, 10.0])
    basis = np.eye(3)
    spectrum = vectors.Spectrum.from_arrays(evals, basis)
    assert list(spectrum.evals) == [10.0, 1.0, 0.1]
    # The top eigenvector must now be the column that carried the largest eigenvalue.
    assert np.allclose(spectrum.evecs[:, 0], basis[:, 2])


def test_spectrum_from_npz_sorts_and_reads_the_ridge(tmp_path) -> None:
    path = tmp_path / "sigma.npz"
    np.savez(path, evals=np.array([0.5, 4.0]), evecs=np.eye(2), ridge=np.float64(0.25))
    spectrum = vectors.Spectrum.from_npz(path)
    assert list(spectrum.evals) == [4.0, 0.5]
    assert spectrum.ridge == 0.25


def test_energy_profile_sums_to_one_and_mass_profile_partitions_it() -> None:
    spectrum = spectrum_with_eigenvalues([9.0, 4.0, 1.0, 0.5, 0.1, 0.01], seed=3)
    rng = np.random.default_rng(7)
    v = rng.standard_normal(6)
    assert spectrum.energy_profile(v).sum() == pytest.approx(1.0)
    profile = spectrum.mass_profile(v, band=(1, 4))
    assert sum(profile.values()) == pytest.approx(1.0)


def test_mahalanobis_is_larger_in_the_tail_than_at_the_top() -> None:
    """The whole screen rests on this ordering: low variance is expensive."""
    spectrum = vectors.Spectrum.from_arrays(np.array([100.0, 1.0, 0.01]), np.eye(3))
    top = spectrum.mahalanobis(np.array([1.0, 0.0, 0.0]))
    tail = spectrum.mahalanobis(np.array([0.0, 0.0, 1.0]))
    assert tail > top


def test_band_out_of_range_is_refused() -> None:
    spectrum = spectrum_with_eigenvalues([3.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="lo < hi"):
        spectrum.band_basis((2, 1))
    with pytest.raises(ValueError, match="past the spectrum"):
        spectrum.band_basis((5, 9))


# ── Mean difference: the two laws ─────────────────────────────────────────────
def test_paired_and_unpaired_agree_on_a_balanced_pairing() -> None:
    rng = np.random.default_rng(11)
    pos, neg = rng.standard_normal((6, 4)), rng.standard_normal((6, 4))
    unpaired = vectors.mean_difference(pos, neg)
    paired = vectors.paired_mean_difference(list(zip(pos, neg)))
    assert np.allclose(unpaired, paired)


def test_paired_and_unpaired_disagree_on_an_unbalanced_one() -> None:
    """The reason both laws exist: with different counts per condition they differ."""
    pos = np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]])
    neg = np.array([[0.0, 0.0], [2.0, 0.0]])
    unpaired = vectors.mean_difference(pos, neg)
    paired = vectors.paired_mean_difference([(pos[0], neg[0]), (pos[1], neg[1])])
    assert not np.allclose(unpaired, paired)


def test_mean_difference_keeps_the_raw_norm() -> None:
    pos = np.array([[2.0, 0.0]])
    neg = np.array([[0.0, 0.0]])
    assert np.linalg.norm(vectors.mean_difference(pos, neg)) == pytest.approx(2.0)


def test_per_prompt_average_collapses_repeats() -> None:
    rows = np.array([[0.0, 0.0], [2.0, 4.0], [1.0, 1.0]])
    grouped = vectors.per_prompt_average(rows, ["p1", "p1", "p2"])
    assert set(grouped) == {"p1", "p2"}
    assert np.allclose(grouped["p1"], [1.0, 2.0])


def test_pair_on_prompts_refuses_a_thin_pairing() -> None:
    a = {f"p{i}": np.zeros(2) for i in range(3)}
    b = {f"p{i}": np.ones(2) for i in range(3)}
    with pytest.raises(ValueError, match="shared prompts"):
        vectors.pair_on_prompts(a, b)
    A, B, shared = vectors.pair_on_prompts(a, b, min_shared=3)
    assert shared == ["p0", "p1", "p2"] and A.shape == B.shape == (3, 2)


# ── Whitening ─────────────────────────────────────────────────────────────────
def test_whitened_direction_recovers_the_discriminant_under_anisotropy() -> None:
    """The finding this construction answers: Σ⁻¹Δ need not point along Δ."""
    sigma = np.diag([100.0, 1.0])
    delta = np.array([1.0, 1.0])
    w = vectors.whitened_direction(delta, sigma)
    assert abs(w[1]) > abs(w[0]) * 50
    assert abs(vectors.cosine(delta, w)) < 0.75


def test_pooled_within_class_rows_removes_the_between_class_shift() -> None:
    pos = np.array([[10.0, 0.0], [12.0, 0.0]])
    neg = np.array([[-10.0, 0.0], [-8.0, 0.0]])
    pooled = vectors.pooled_within_class_rows(pos, neg)
    assert np.allclose(pooled.mean(axis=0), 0.0)
    assert pooled.shape == (4, 2)


def test_shrink_scale_moves_the_shrinkage_and_clips_to_one() -> None:
    rng = np.random.default_rng(5)
    rows = rng.standard_normal((12, 20))
    _sigma, auto = vectors.ledoit_wolf_covariance(rows)
    _scaled_sigma, halved = vectors.ledoit_wolf_covariance(rows, shrink_scale=0.5)
    assert halved == pytest.approx(auto * 0.5)
    _clipped_sigma, clipped = vectors.ledoit_wolf_covariance(rows, shrink_scale=1e6)
    assert clipped == 1.0


def test_whitening_diagnostics_report_the_alignment_and_the_counts() -> None:
    rng = np.random.default_rng(9)
    pos = rng.standard_normal((8, 5)) + np.array([2.0, 0, 0, 0, 0])
    neg = rng.standard_normal((7, 5))
    sigma, shrinkage = vectors.pooled_within_class_covariance(pos, neg)
    report = vectors.whitening_diagnostics(pos, neg, sigma, shrinkage)
    assert 0.0 <= report["cos_delta_whitened"] <= 1.0
    assert report["mahalanobis_d"] > 0
    assert report["n_pos"] == 8 and report["n_neg"] == 7


# ── Band projection and orthogonalization ─────────────────────────────────────
def test_band_pass_lands_inside_the_band_and_is_unit() -> None:
    spectrum = vectors.Spectrum.from_arrays(np.array([9.0, 4.0, 1.0, 0.1]), np.eye(4))
    v = np.array([1.0, 1.0, 1.0, 1.0])
    member = vectors.band_pass(v, spectrum, band=(1, 3))
    assert np.linalg.norm(member) == pytest.approx(1.0)
    assert member[0] == pytest.approx(0.0) and member[3] == pytest.approx(0.0)


def test_band_pass_refuses_a_vector_with_no_band_component() -> None:
    spectrum = vectors.Spectrum.from_arrays(np.array([9.0, 4.0, 1.0]), np.eye(3))
    with pytest.raises(ValueError, match="no component inside the band"):
        vectors.band_pass(np.array([1.0, 0.0, 0.0]), spectrum, band=(1, 3))


def test_band_pass_anatomy_reports_the_surviving_fraction() -> None:
    spectrum = vectors.Spectrum.from_arrays(np.array([9.0, 4.0, 1.0, 0.1]), np.eye(4))
    mostly_top = np.array([10.0, 1.0, 0.0, 0.0])
    anatomy = vectors.band_pass_anatomy(mostly_top, spectrum, band=(1, 3))
    assert anatomy["band_mass_of_raw"] < 0.2
    mostly_band = np.array([0.1, 10.0, 1.0, 0.0])
    assert vectors.band_pass_anatomy(mostly_band, spectrum, band=(1, 3))["band_mass_of_raw"] > 0.9


def test_orthogonalize_produces_an_orthogonal_unit_vector() -> None:
    reference = vectors.unit(np.array([1.0, 1.0, 0.0]))
    perp = vectors.orthogonalize(np.array([1.0, 0.0, 0.0]), reference)
    assert vectors.cosine(perp, reference) == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(perp) == pytest.approx(1.0)


def test_orthogonalize_refuses_a_parallel_vector() -> None:
    reference = np.array([0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="parallel"):
        vectors.orthogonalize(reference * 3.0, reference)


def test_orthogonalization_anatomy_hard_checks_the_cosine() -> None:
    reference = vectors.unit(np.array([1.0, 2.0, 3.0]))
    perp, anatomy = vectors.orthogonalization_anatomy(np.array([3.0, 1.0, 0.0]), reference)
    assert abs(anatomy["cos_perp_reference"]) <= 1e-8
    assert 0.0 < anatomy["residual_norm_fraction"] <= 1.0
    assert vectors.cosine(perp, reference) == pytest.approx(0.0, abs=1e-12)


# ── Nulls and dose ────────────────────────────────────────────────────────────
def test_random_unit_vectors_are_seeded_and_unit() -> None:
    first = vectors.random_unit_vectors(16, count=3, seed=1234)
    again = vectors.random_unit_vectors(16, count=3, seed=1234)
    assert set(first) == {"R1", "R2", "R3"}
    for key, value in first.items():
        assert np.linalg.norm(value) == pytest.approx(1.0, abs=1e-6)
        assert np.allclose(value, again[key])
    assert not np.allclose(first["R1"], vectors.random_unit_vectors(16, seed=99)["R1"])


def test_random_band_vector_is_confined_to_its_band() -> None:
    """The matched-null discipline: a band member's null is drawn in the band."""
    spectrum = vectors.Spectrum.from_arrays(np.array([9.0, 4.0, 1.0, 0.1]), np.eye(4))
    null = vectors.random_band_vector(spectrum, band=(1, 3), rng=np.random.default_rng(3))
    assert np.linalg.norm(null) == pytest.approx(1.0)
    assert null[0] == pytest.approx(0.0) and null[3] == pytest.approx(0.0)


def test_dose_alpha_scales_by_the_median_norm_and_refuses_a_nonpositive_one() -> None:
    assert vectors.dose_alpha(0.1, 40.0) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="must be positive"):
        vectors.dose_alpha(0.1, 0.0)


def test_median_row_norm_pools_positions_rather_than_generations() -> None:
    """One long generation of small norms must outweigh one short row of a large one."""
    rows = np.vstack([np.full((9, 2), [1.0, 0.0]), np.array([[100.0, 0.0]])])
    assert vectors.median_row_norm(rows) == pytest.approx(1.0)


# ── The sweep law ─────────────────────────────────────────────────────────────
def test_heldout_d_is_smaller_than_the_in_sample_value_on_noise() -> None:
    """Why the split exists: fitting and reading the same rows invents separation."""
    rng = np.random.default_rng(21)
    a, b = rng.standard_normal((40, 60)), rng.standard_normal((40, 60))
    in_sample = vectors.heldout_cohens_d(a, a, b, b, min_direction_norm=1e-8, sd_floor=1e-8)
    held_out = vectors.heldout_cohens_d(
        a[:20], a[20:], b[:20], b[20:], min_direction_norm=1e-8, sd_floor=1e-8
    )
    assert in_sample > 1.0
    assert abs(held_out) < in_sample


def test_the_two_degenerate_conventions_differ_only_where_they_degenerate() -> None:
    """Both donor conventions are kept because they disagree at zero variance."""
    a_fit = np.array([[1.0], [1.0]])
    b_fit = np.array([[0.0], [0.0]])
    a_eval = np.array([[1.0], [1.0]])
    b_eval = np.array([[0.0], [0.0]])
    floored = vectors.heldout_cohens_d(
        a_fit, a_eval, b_fit, b_eval, min_direction_norm=1e-8, sd_floor=1e-8
    )
    zeroed = vectors.heldout_cohens_d(
        a_fit, a_eval, b_fit, b_eval, min_direction_norm=1e-8, sd_floor=None
    )
    assert floored > 1e6
    assert zeroed == 0.0


def test_absent_direction_returns_zero() -> None:
    same = np.ones((4, 3))
    assert vectors.heldout_cohens_d(
        same, same, same, same, min_direction_norm=1e-8, sd_floor=1e-8
    ) == 0.0


def test_half_split_sweep_peaks_at_the_planted_layer() -> None:
    rng = np.random.default_rng(31)
    n, layers, dim = 24, 5, 8
    a = rng.standard_normal((n, layers, dim))
    b = rng.standard_normal((n, layers, dim))
    a[:, 3, 0] += 6.0
    d_mean, d_sd = vectors.half_split_sweep(a, b, k_splits=12, rng=np.random.default_rng(1))
    assert int(np.argmax(np.abs(d_mean))) == 3
    assert d_sd.shape == (layers,)


def test_half_split_sweep_refuses_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match=r"\[n, L, d\]"):
        vectors.half_split_sweep(np.zeros((4, 3, 2)), np.zeros((4, 2, 2)))


# ── Banking ───────────────────────────────────────────────────────────────────
def test_bank_round_trips_under_the_frozen_names(tmp_path) -> None:
    bank = {"V3_L14": np.ones(4, dtype=np.float32) / 2.0, "R1": np.eye(4, dtype=np.float32)[0]}
    stamps = {"model": "3b", "sites": [14], "median_resid_norms": {"L14": 40.0}}
    npz_path, stamps_path = vectors.save_vector_bank(tmp_path, bank, stamps)
    assert npz_path.name == "a5_vectors.npz" and stamps_path.name == "a5_vectors_stamps.json"
    loaded, loaded_stamps = vectors.load_vector_bank(tmp_path)
    assert set(loaded) == set(bank)
    assert loaded_stamps == stamps
    assert json.loads(stamps_path.read_text())["median_resid_norms"]["L14"] == 40.0


def test_load_vector_bank_refuses_a_missing_bank(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="no vector bank"):
        vectors.load_vector_bank(tmp_path)


def test_load_vector_warns_on_a_non_unit_vector(caplog) -> None:
    bank = {"V3_L14": np.full(4, 5.0, dtype=np.float32)}
    with caplog.at_level("WARNING"):
        vectors.load_vector(bank, "V3_L14")
    assert "norm" in caplog.text
    with pytest.raises(KeyError, match="not in"):
        vectors.load_vector(bank, "absent")


def test_site_of_key_reads_the_suffix_and_falls_back_for_site_independent_keys() -> None:
    assert vectors.site_of_key("V3_L22", 14) == 22
    assert vectors.site_of_key("R1", 14) == 14
    assert vectors.site_of_key("V3sel_bare", 16) == 16


def test_install_lever_takes_sigma_and_dose_from_the_target(monkeypatch) -> None:
    """The install convention: Σ and the dose currency come from the injection target."""
    rng = np.random.default_rng(41)
    donor = {12: rng.standard_normal((10, 6)) + 3.0}
    base = {12: rng.standard_normal((10, 6))}
    target_tokens = {12: rng.standard_normal((80, 6)) * np.array([5, 1, 1, 1, 1, 1])}
    bank, diagnostics = vectors.build_install_vectors(
        donor, base, target_tokens, band=(1, 4)
    )
    assert set(bank) == {"Install_L12", "InstallW_L12", "Rband_L12", "median_norm_L12"}
    assert np.linalg.norm(bank["Install_L12"]) == pytest.approx(1.0)
    assert diagnostics["L12"]["n_target_tokens"] == 80
    assert diagnostics["L12"]["target_median_token_norm"] == pytest.approx(
        vectors.median_row_norm(target_tokens[12])
    )
