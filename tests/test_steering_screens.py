"""The screens, on synthetic covariances where the answer is known in advance.

The captures need a model on a CUDA device and are exercised by the golden path
rather than here; everything the screens actually decide with is pure numpy and
is checked against a planted structure.
"""

from __future__ import annotations

import numpy as np
import pytest

from anamnesis.steering import screens, vectors


def diagonal_spectrum(evals: list[float], ridge_rel: float = 0.0) -> vectors.Spectrum:
    array = np.asarray(evals, dtype=np.float64)
    return vectors.Spectrum.from_arrays(array, np.eye(len(array)), ridge=ridge_rel * array.mean())


# ── The covariance screen ─────────────────────────────────────────────────────
def test_tail_direction_costs_more_than_a_top_direction() -> None:
    """The screen's whole claim: the same unit magnitude does different damage."""
    spectrum = diagonal_spectrum([100.0, 10.0, 1.0, 0.01], ridge_rel=1e-3)
    top = screens.screen_vector(np.array([1.0, 0, 0, 0]), spectrum, tail_fraction=0.25)
    tail = screens.screen_vector(np.array([0, 0, 0, 1.0]), spectrum, tail_fraction=0.25)
    assert tail["mahalanobis"] > top["mahalanobis"] * 100
    assert tail["tail_over_top"] > 1.0 > top["tail_over_top"]


def test_eigenmass_entries_are_fractions_of_a_unit_vector() -> None:
    spectrum = diagonal_spectrum([4.0, 3.0, 2.0, 1.0])
    row = screens.screen_vector(np.array([1.0, 1.0, 0.0, 0.0]), spectrum, tail_fraction=0.5)
    assert row["top_2_eigenmass"] == pytest.approx(1.0)
    assert row["bottom_2_eigenmass"] == pytest.approx(0.0)


def test_screen_vector_refuses_a_width_mismatch() -> None:
    with pytest.raises(ValueError, match="dimensional"):
        screens.screen_vector(np.ones(3), diagonal_spectrum([1.0, 1.0]))


def test_screen_bank_selects_by_site_and_skips_other_models_widths() -> None:
    spectrum = diagonal_spectrum([4.0, 2.0, 1.0])
    bank = {
        "V3_L14": np.array([1.0, 0, 0]),
        "V3_L22": np.array([0, 1.0, 0]),
        "R1": np.array([0, 0, 1.0]),
        "V3_L14_other_model": np.ones(7),
        "median_norm_L14": np.asarray(40.0),
    }
    rows = screens.screen_bank(bank, spectrum, site=14)
    assert set(rows) == {"V3_L14", "R1"}


def test_band_mass_agrees_with_the_square_root_of_the_band_fraction() -> None:
    spectrum = diagonal_spectrum([9.0, 4.0, 1.0, 0.1])
    report = screens.band_mass(np.array([1.0, 2.0, 2.0, 1.0]), spectrum, band=(1, 3))
    assert report["band_mass"] == pytest.approx(report["sqrt_band_massfrac_check"])
    assert sum(report["mass_fractions"].values()) == pytest.approx(1.0)


def test_band_mass_is_low_for_an_amplified_residue() -> None:
    """A member built from a source with no band content is a renormalized remainder."""
    spectrum = diagonal_spectrum([9.0, 4.0, 1.0, 0.1])
    residue = screens.band_mass(np.array([100.0, 0.5, 0.5, 0.0]), spectrum, band=(1, 3))
    genuine = screens.band_mass(np.array([0.1, 5.0, 5.0, 0.0]), spectrum, band=(1, 3))
    assert residue["band_mass"] < 0.05 < 0.9 < genuine["band_mass"]


def test_band_mass_refuses_a_zero_vector_and_a_width_mismatch() -> None:
    spectrum = diagonal_spectrum([2.0, 1.0])
    with pytest.raises(ValueError, match="zero vector"):
        screens.band_mass(np.zeros(2), spectrum, band=(0, 2))
    with pytest.raises(ValueError, match="dimensional"):
        screens.band_mass(np.ones(5), spectrum, band=(0, 2))


# ── The deformation curve ─────────────────────────────────────────────────────
def test_linear_response_holds_the_ratio_to_alpha_squared_flat() -> None:
    rng = np.random.default_rng(4)
    baseline = rng.standard_normal((400, 6))
    direction = np.array([1.0, 0, 0, 0, 0, 0])
    steered = {alpha: baseline + alpha * direction for alpha in (0.1, 0.2, 0.4)}
    curve = screens.deformation_curve(baseline, steered, median_site_input_norm=2.0)
    ratios = [curve["by_dose"][str(a)]["maha_over_alpha2"] for a in (0.1, 0.2, 0.4)]
    assert max(ratios) / min(ratios) < 1.05


def test_superlinear_response_makes_the_ratio_rise() -> None:
    """The graded-Goodhart readout: growth faster than α² is off-manifold."""
    rng = np.random.default_rng(5)
    baseline = rng.standard_normal((400, 6))
    direction = np.array([0, 0, 0, 0, 0, 1.0])
    steered = {alpha: baseline + (alpha**3) * direction for alpha in (0.1, 0.4, 0.8)}
    curve = screens.deformation_curve(baseline, steered, median_site_input_norm=2.0)
    ratios = [curve["by_dose"][str(a)]["maha_over_alpha2"] for a in (0.1, 0.4, 0.8)]
    assert ratios[0] < ratios[1] < ratios[2]


def test_deformation_curve_reports_no_ratio_at_zero_dose() -> None:
    baseline = np.random.default_rng(6).standard_normal((50, 4))
    curve = screens.deformation_curve(baseline, {0.0: baseline}, median_site_input_norm=1.0)
    assert curve["by_dose"]["0.0"]["maha_over_alpha2"] is None


# ── Layer separation ──────────────────────────────────────────────────────────
def test_two_fold_d_separates_a_planted_shift_and_not_noise() -> None:
    rng = np.random.default_rng(8)
    a, b = rng.standard_normal((24, 5)), rng.standard_normal((24, 5))
    assert abs(screens.two_fold_heldout_d(a, b)) < 1.0
    assert screens.two_fold_heldout_d(a + np.array([5.0, 0, 0, 0, 0]), b) > 2.0


def test_two_fold_d_returns_zero_when_no_fold_can_carry_a_variance() -> None:
    assert screens.two_fold_heldout_d(np.ones((2, 3)), np.zeros((2, 3))) == 0.0


def test_centroid_ratio_is_scale_free() -> None:
    rng = np.random.default_rng(10)
    a = rng.standard_normal((20, 4)) + np.array([3.0, 0, 0, 0])
    b = rng.standard_normal((20, 4))
    assert screens.centroid_ratio(a, b) == pytest.approx(screens.centroid_ratio(7.0 * a, 7.0 * b))


def test_centroid_ratio_is_zero_without_within_class_spread() -> None:
    assert screens.centroid_ratio(np.ones((3, 2)), np.zeros((3, 2))) == 0.0


def test_axis_separation_rows_name_their_layer_and_depth() -> None:
    """A sparse routing substrate and a dense residual one must both say which layer."""
    rng = np.random.default_rng(12)
    pos = rng.standard_normal((16, 3, 5))
    neg = rng.standard_normal((16, 3, 5))
    pos[:, 1, :] += 4.0
    rows = screens.axis_separation_rows(pos, neg, layer_indices=[5, 11, 18], n_layers=28)
    assert [row["layer"] for row in rows] == [5, 11, 18]
    assert rows[1]["depth_pct"] == pytest.approx(round(100 * 11 / 28, 1))
    assert screens.peak_layer(rows)["layer"] == 11


def test_axis_separation_drops_non_finite_rows_per_layer() -> None:
    pos = np.zeros((6, 2, 3))
    neg = np.ones((6, 2, 3))
    pos[0, 0, :] = np.nan
    rows = screens.axis_separation_rows(pos, neg, layer_indices=[1, 2], n_layers=4)
    assert rows[0]["n_pos"] == 5
    assert rows[1]["n_pos"] == 6


def test_axis_separation_refuses_an_index_count_mismatch() -> None:
    with pytest.raises(ValueError, match="measured layers"):
        screens.axis_separation_rows(np.zeros((4, 3, 2)), np.zeros((4, 3, 2)), [1, 2], 10)


def test_peak_layer_refuses_an_empty_table() -> None:
    with pytest.raises(ValueError, match="no separation rows"):
        screens.peak_layer([])
