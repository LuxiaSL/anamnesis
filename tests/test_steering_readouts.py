"""The readouts: cell parsing, the target/off-target split, matched-support nulls.

The lever and the contrast frames run over banked signature directories, so the
directory-level readouts are exercised through synthetic banks written to a
temporary path; the arithmetic is checked directly where it can be.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from anamnesis.steering import readouts


def write_signature_bank(directory, rows, feature_names=("f0", "f1", "f2", "f3")) -> None:
    """A signatures directory of ``gen_*.npz`` files, one per row."""
    directory.mkdir(parents=True, exist_ok=True)
    for gen_id, row in enumerate(rows):
        np.savez(
            directory / f"gen_{gen_id:03d}.npz",
            feature_names=np.array(list(feature_names)),
            features=np.asarray(row, dtype=np.float32),
        )


def write_metadata(path, texts) -> None:
    path.write_text(json.dumps({
        "generations": [
            {"generated_text": text, "num_generated_tokens": len(text.split())} for text in texts
        ]
    }))


# ── Cell names ────────────────────────────────────────────────────────────────
def test_signed_dose_spellings_parse_to_the_same_number() -> None:
    assert readouts.parse_cell_name("V3_L14_a0.3", 14)["alpha_frac"] == pytest.approx(0.3)
    assert readouts.parse_cell_name("V3_L14_an0.3", 14)["alpha_frac"] == pytest.approx(-0.3)
    assert readouts.parse_cell_name("V7_L14_p03", 14)["alpha_frac"] == pytest.approx(0.3)
    assert readouts.parse_cell_name("V7_L14_m01", 14)["alpha_frac"] == pytest.approx(-0.1)


def test_a_bare_baseline_cell_parses_at_zero_dose() -> None:
    parsed = readouts.parse_cell_name("baseline", 16)
    assert parsed == {"vector": "baseline", "site": 16, "alpha_frac": 0.0}


def test_the_longest_vector_name_binds_before_its_prefix() -> None:
    assert readouts.parse_cell_name("V3sel_bare_L14_a0.1", 14)["vector"] == "V3sel_bare"
    assert readouts.parse_cell_name("Rband2_L14_a0.1", 14)["vector"] == "Rband2"


def test_the_other_spelling_handles_riders_and_cross_site_injection() -> None:
    doubled = readouts.parse_cell_name("V1_L14_L14_a0.0", 20)
    assert doubled == {"vector": "V1", "site": 14, "alpha_frac": 0.0}
    cross = readouts.parse_cell_name("V4_L14_at7_a0.3", 20)
    assert cross["site"] == 7


def test_a_non_cell_directory_is_skipped_rather_than_defaulted() -> None:
    assert readouts.parse_cell_name("signatures_v3", 14) is None
    assert readouts.parse_cell_name("judge_results", 14) is None


# ── The decomposition ─────────────────────────────────────────────────────────
def test_pole_axis_puts_the_threshold_between_the_two_corpora() -> None:
    a = np.array([[2.0, 0.0], [4.0, 0.0]])
    b = np.array([[-4.0, 0.0], [-2.0, 0.0]])
    axis, projection_a, projection_b, threshold = readouts.pole_axis(a, b)
    assert np.allclose(axis, [1.0, 0.0])
    assert projection_a == pytest.approx(3.0) and projection_b == pytest.approx(-3.0)
    assert threshold == pytest.approx(0.0)


def test_decompose_shift_splits_on_and_off_axis_movement() -> None:
    axis = np.array([1.0, 0.0, 0.0])
    decomposed = readouts.decompose_shift(np.array([3.0, 4.0, 0.0]), axis)
    assert decomposed["target_shift"] == pytest.approx(3.0)
    assert decomposed["off_target"] == pytest.approx(4.0)
    assert decomposed["effect_per_offtarget"] == pytest.approx(0.75)


def test_effect_per_offtarget_is_dose_invariant() -> None:
    """Both parts scale with the injection, so the ratio is what survives dose."""
    axis = np.array([1.0, 0.0])
    small = readouts.decompose_shift(np.array([1.0, 2.0]), axis)
    large = readouts.decompose_shift(np.array([10.0, 20.0]), axis)
    assert small["effect_per_offtarget"] == pytest.approx(large["effect_per_offtarget"])


# ── The lever, end to end over banked directories ─────────────────────────────
def test_lever_readout_separates_a_real_vector_from_its_random_controls(tmp_path) -> None:
    rng = np.random.default_rng(3)
    floor = rng.normal(0.0, 1.0, size=(40, 4))
    write_signature_bank(tmp_path / "floor", floor)
    write_signature_bank(tmp_path / "pole_a", rng.normal(0.0, 1.0, (20, 4)) + [6.0, 0, 0, 0])
    write_signature_bank(tmp_path / "pole_b", rng.normal(0.0, 1.0, (20, 4)) - [6.0, 0, 0, 0])

    run = tmp_path / "run"
    base = rng.normal(0.0, 1.0, (20, 4))
    write_signature_bank(run / "V3_L14_a0.0", base)
    write_signature_bank(run / "V3_L14_a0.1", base + [4.0, 0, 0, 0])
    write_signature_bank(run / "R1_L14_a0.1", base + [0.0, 4.0, 0, 0])
    write_signature_bank(run / "R2_L14_a0.1", base + [0.0, 0.0, 4.0, 0])

    report = readouts.lever_readout(
        run, tmp_path / "pole_a", tmp_path / "pole_b", tmp_path / "floor",
        map_site=14, sig_subdir="", baseline_cell="V3_L14_a0.0",
    )
    lever = report["lever_ratio_by_dose"]["L14_a0.1"]
    assert lever["lever_ratio"] > 5.0
    steered = next(r for r in report["per_cell"] if r["cell"] == "V3_L14_a0.1")
    control = next(r for r in report["per_cell"] if r["cell"] == "R1_L14_a0.1")
    assert steered["target_shift"] > abs(control["target_shift"])
    assert control["off_target"] > control["target_shift"]
    assert steered["frac_pole_a_vs_baseline"] > 0.0


def test_lever_readout_refuses_a_missing_baseline(tmp_path) -> None:
    rng = np.random.default_rng(4)
    write_signature_bank(tmp_path / "floor", rng.normal(size=(20, 4)))
    write_signature_bank(tmp_path / "pole_a", rng.normal(size=(10, 4)) + 3.0)
    write_signature_bank(tmp_path / "pole_b", rng.normal(size=(10, 4)) - 3.0)
    run = tmp_path / "run"
    write_signature_bank(run / "V3_L14_a0.1", rng.normal(size=(10, 4)))
    with pytest.raises(ValueError, match="baseline cell"):
        readouts.lever_readout(
            run, tmp_path / "pole_a", tmp_path / "pole_b", tmp_path / "floor",
            map_site=14, sig_subdir="",
        )


# ── Matched-support efficiency ────────────────────────────────────────────────
def test_support_is_read_from_the_name() -> None:
    assert readouts.support_of("V3top") == "top"
    assert readouts.support_of("V3tail") == "tail"
    assert readouts.support_of("Rband2") == "band"
    assert readouts.support_of("V7") == "band"
    assert readouts.support_of("V3") == "full"


def test_nulls_are_identified_by_their_prefix() -> None:
    assert readouts.is_null("R1") and readouts.is_null("Rtail")
    assert not readouts.is_null("V3top")


def test_nulls_are_never_pooled_across_supports() -> None:
    """The constraint: a tail null must not stand in for a band null, or vice versa."""
    axis = np.array([1.0, 0.0, 0.0])
    centroid = np.zeros(3)
    cells = {
        "V3tail_a0.1": {"z": np.array([[2.0, 1.0, 0.0]]), "vector": "V3tail", "site": 14,
                        "alpha_frac": 0.1},
        "Rtail_a0.1": {"z": np.array([[1.0, 1.0, 0.0]]), "vector": "Rtail", "site": 14,
                       "alpha_frac": 0.1},
        "V7_a0.1": {"z": np.array([[4.0, 1.0, 0.0]]), "vector": "V7", "site": 14,
                    "alpha_frac": 0.1},
    }
    rows = {row["cell"]: row for row in
            readouts.matched_support_efficiency(cells, centroid, axis)}
    tail = rows["V3tail_a0.1"]
    band = rows["V7_a0.1"]
    assert tail["support"] == "tail" and band["support"] == "band"
    assert tail["efficiency_over_matched_null"] is not None
    # No band null was supplied, so the band target reports no ratio rather than
    # borrowing the tail's.
    assert band["efficiency_over_matched_null"] is None


def test_the_promotion_flag_reads_the_matched_null_ratio() -> None:
    axis = np.array([1.0, 0.0])
    cells = {
        "V3_a0.1": {"z": np.array([[3.0, 0.0]]), "vector": "V3", "site": 14, "alpha_frac": 0.1},
        "R1_a0.1": {"z": np.array([[0.5, 2.0]]), "vector": "R1", "site": 14, "alpha_frac": 0.1},
    }
    rows = {r["cell"]: r for r in readouts.matched_support_efficiency(cells, np.zeros(2), axis)}
    assert rows["V3_a0.1"]["efficiency_over_matched_null"] > 1.5
    assert rows["V3_a0.1"]["clears_1.5x_matched_null"] is True
    assert "clears_1.5x_matched_null" not in rows["R1_a0.1"]


def test_coherence_statistics_ride_alongside_when_metadata_is_given(tmp_path) -> None:
    """The C4 consumer: a cell whose text collapsed has not shown what it looks like."""
    healthy = tmp_path / "healthy.json"
    degenerate = tmp_path / "degenerate.json"
    write_metadata(healthy, ["the state of the art moves quickly and unevenly across tasks"])
    write_metadata(degenerate, ["loop loop loop " * 20])
    axis = np.array([1.0, 0.0])
    cells = {
        "V3_a0.1": {"z": np.array([[2.0, 0.0]]), "vector": "V3", "site": 14, "alpha_frac": 0.1,
                    "metadata": healthy},
        "V4_a0.1": {"z": np.array([[2.0, 0.0]]), "vector": "V4", "site": 14, "alpha_frac": 0.1,
                    "metadata": degenerate},
    }
    rows = {r["cell"]: r for r in readouts.matched_support_efficiency(cells, np.zeros(2), axis)}
    assert rows["V3_a0.1"]["coherence"]["mean_trigram_rep"] < 0.1
    assert rows["V4_a0.1"]["coherence"]["mean_trigram_rep"] > 0.8
    assert rows["V3_a0.1"]["coherence"]["mean_ttr"] > rows["V4_a0.1"]["coherence"]["mean_ttr"]


def test_construction_mahalanobis_orders_top_below_tail() -> None:
    evals = np.array([100.0, 1.0, 0.01])
    evecs = np.eye(3)
    top = readouts.construction_mahalanobis(np.array([1.0, 0, 0]), evals, evecs)
    tail = readouts.construction_mahalanobis(np.array([0, 0, 1.0]), evals, evecs)
    assert tail > top * 1000


# ── Checkpoint series ─────────────────────────────────────────────────────────
def test_seed_floor_is_the_within_class_distance() -> None:
    rng = np.random.default_rng(6)
    base = rng.normal(0.0, 1.0, size=(30, 5))
    floor = readouts.seed_floor(base, list(range(30)), group_size=10, per_group=4)
    assert floor > 0.0


def test_seed_floor_refuses_a_bank_with_no_within_class_pairs() -> None:
    with pytest.raises(ValueError, match="no within-class pairs"):
        readouts.seed_floor(np.zeros((1, 3)), [0], group_size=10, per_group=1)


def test_sign_flip_p_is_small_for_an_agreeing_direction_and_large_for_noise() -> None:
    agreeing = np.full(40, 1.0)
    assert readouts.sign_flip_p(agreeing, n_perm=2000) < 0.001
    rng = np.random.default_rng(7)
    assert readouts.sign_flip_p(rng.normal(size=40), n_perm=2000) > 0.05


def test_directional_series_finds_the_onset_before_the_magnitude_bar(tmp_path) -> None:
    rng = np.random.default_rng(8)
    direction = np.zeros(6)
    direction[0] = 1.0
    fields = {}
    for step, scale in (("0001", 0.0), ("0002", 0.02), ("0003", 0.4)):
        fields[step] = rng.normal(0.0, 0.01, size=(30, 6)) + scale * direction
    report = readouts.directional_series(fields, ["0001", "0002", "0003"], floor=0.5, n_perm=2000)
    assert report["directional_onset_step"] == "0002"
    assert report["per_checkpoint"][1]["above_visibility_bar"] is False
    assert report["per_checkpoint"][2]["above_visibility_bar"] is True
    assert report["install_axis"].startswith("step-0003")


def test_an_external_axis_is_named_in_the_artifact() -> None:
    fields = {"0001": np.ones((10, 3))}
    report = readouts.directional_series(
        fields, ["0001"], floor=0.5, axis=np.array([1.0, 0.0, 0.0]), n_perm=200
    )
    assert report["install_axis"] == "external"


def test_directional_series_refuses_an_empty_step_list() -> None:
    with pytest.raises(ValueError, match="at least one step"):
        readouts.directional_series({}, [], floor=0.1)


def test_the_matched_control_frame_cancels_the_drift_the_vs_base_frame_keeps() -> None:
    """The stamp matters because the two frames measure different things."""
    drift = np.array([5.0, 0.0])
    trait = np.array([0.0, 1.0])
    base = (np.zeros((4, 2)), [0, 1, 2, 3])
    arm = {"0001": (np.tile(drift + trait, (4, 1)), [0, 1, 2, 3])}
    control = {"0001": (np.tile(drift, (4, 1)), [0, 1, 2, 3])}

    contrasted, frame = readouts.contrast_fields(arm, control, base, ["0001"])
    assert frame == "vs-matched-control"
    assert np.allclose(contrasted["0001"].mean(axis=0), trait)

    total, frame = readouts.contrast_fields(arm, None, base, ["0001"])
    assert frame == "vs-base"
    assert np.allclose(total["0001"].mean(axis=0), drift + trait)


def test_contrast_fields_refuses_an_unmatched_probe_set() -> None:
    base = (np.zeros((2, 2)), [0, 1])
    arm = {"0001": (np.ones((2, 2)), [0, 1])}
    control = {"0001": (np.ones((2, 2)), [7, 8])}
    with pytest.raises(ValueError, match="no probes shared"):
        readouts.contrast_fields(arm, control, base, ["0001"])


# ── The identity sidecar ──────────────────────────────────────────────────────
def test_expert_usage_histogram_carries_its_warning_and_finds_unused_experts(tmp_path) -> None:
    rng = np.random.default_rng(9)
    n_positions, n_layers, n_experts = 12, 2, 8
    router = rng.dirichlet(np.ones(n_experts), size=(n_positions, n_layers))
    router[..., 7] = 0.0
    router = router / router.sum(axis=-1, keepdims=True)
    path = tmp_path / "raw.npz"
    np.savez(path, router_dist=router, router_layer_indices=np.array([5, 11]))
    report = readouts.expert_usage_histogram([path], top_k=2)
    assert "IDENTITY channel" in report["WARNING"]
    assert "never enter a signature" in report["WARNING"]
    assert report["lineage"]["router_layer_indices"] == [5, 11]
    assert report["hard_usage_pooled"]["n_experts_unused"] >= 1
    assert 0.0 < report["hard_usage_pooled"]["entropy_frac_of_uniform"] <= 1.0
    assert len(report["per_layer_hard_entropy_frac"]) == n_layers


def test_expert_usage_histogram_refuses_a_bank_with_no_routing(tmp_path) -> None:
    path = tmp_path / "raw.npz"
    np.savez(path, something_else=np.zeros(3))
    with pytest.raises(ValueError, match="no 'router_dist' array"):
        readouts.expert_usage_histogram([path])


def test_write_json_creates_its_directory(tmp_path) -> None:
    out = readouts.write_json(tmp_path / "nested" / "readout.json", {"a": 1})
    assert json.loads(out.read_text()) == {"a": 1}
