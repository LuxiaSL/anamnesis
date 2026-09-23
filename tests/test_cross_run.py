"""Cross-run transfer: the scoring, the wildcard, and the linear leg beside it.

The transfer itself trains a small network several times, so the tests split into the parts
that can be pinned by value and one end-to-end pass at small parameters:

  * **scoring** against the map a mode mapping declares: what counts as correct, what the wildcard
    does (reported, never scored), and that chance is one over the training vocabulary;
  * **the assignment**, which counts test generations by nearest training centroid, and the
    similarity matrix, which is cosine and therefore scale-free;
  * **the layer filter**, which reads a feature's layer out of its name in both spellings
    and can keep or drop the unlayered global statistics;
  * **loading**, which refuses a missing key by name rather than returning a short vector,
    and which can concatenate several keys in the order asked for;
  * a **width mismatch between the two runs is refused**: transfer means one projection
    reading both banks, so two widths are two instruments;
  * the **LDA leg** on a planted corpus: a run separable in its own discriminant space and
    another that is not, which is the dissociation the leg exists to make visible;
  * the headline carries the 3B comparison beside every number rather than after it.

CPU only; the banks are synthetic and the seeds are few.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anamnesis.analysis.cross_run import (
    build_layer_indices,
    compute_centroids,
    cross_run_transfer,
    headline,
    lda_direction_test,
    load_feature_names,
    load_run_features,
    nearest_centroid_assignment,
    score_mapping,
    similarity_matrix,
)
from anamnesis.modes import DEFAULT_MODE_MAPPING, mapping_wildcards, mode_mapping

TRAIN_MODES = ["dialectical", "linear", "socratic"]
TEST_MODES = ["deliberative", "pedagogical", "structured"]


def write_run(
    outputs: Path,
    run: str,
    modes: list[str],
    *,
    width: int = 4,
    n_per_mode: int = 6,
    seed: int = 0,
    extra_key: str | None = None,
) -> Path:
    """A run's signatures: one cluster per mode, separated on its own column."""
    sig_dir = outputs / "runs" / run / "signatures"
    sig_dir.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    gid = 0
    for index, mode in enumerate(modes):
        for _ in range(n_per_mode):
            vector = 0.2 * rng.standard_normal(width).astype(np.float32)
            vector[index % width] += 5.0
            arrays = {
                "features": vector,
                "feature_names": np.array(
                    [f"attn_prompt_mass_L{8 * i}" for i in range(width - 1)] + ["logit_entropy"]
                ),
            }
            if extra_key is not None:
                arrays[extra_key] = np.ones(2, dtype=np.float32)
            np.savez(sig_dir / f"gen_{gid:03d}.npz", **arrays)
            (sig_dir / f"gen_{gid:03d}.json").write_text(
                json.dumps(
                    {
                        "generation_id": gid,
                        "mode": mode,
                        "mode_idx": index,
                        "topic": f"topic-{gid % 3}",
                        "topic_idx": gid % 3,
                        "num_generated_tokens": 70,
                    }
                )
            )
            gid += 1
    return sig_dir


def test_a_map_is_scored_only_over_the_pairs_it_predicted() -> None:
    assignment = np.array([[8, 1, 1], [0, 4, 6], [2, 2, 6], [3, 4, 3]])
    modes_test = ["deliberative", "pedagogical", "structured", "compressed"]
    predicted = {"deliberative": "dialectical", "pedagogical": "socratic", "structured": "linear"}
    scored = score_mapping(assignment, modes_test, ["dialectical", "socratic", "linear"], predicted)
    assert scored.modes_total == 3, "the wildcard is not one of the predictions"
    assert scored.total_mapped == 30, "and its generations are not in the denominator"
    assert scored.correct_mapped == 8 + 4 + 6, "one row per prediction, at its own column"
    assert scored.modes_correct == 2, "pedagogical's nearest centroid is not the predicted one"
    assert scored.chance_accuracy == pytest.approx(1 / 3)
    assert scored.per_mode["compressed"]["predicted"] is None
    assert scored.per_mode["compressed"]["samples"] == 10
    assert scored.per_mode["deliberative"]["match"] is True
    assert scored.per_mode["pedagogical"]["match"] is False, (
        "its nearest centroid is not the one predicted"
    )


def test_a_mode_with_no_generations_scores_zero_rather_than_dividing_by_zero() -> None:
    assignment = np.zeros((2, 2), dtype=np.int64)
    scored = score_mapping(assignment, ["a", "b"], ["x", "y"], {"a": "x"})
    assert scored.overall_accuracy == 0.0
    assert scored.per_mode["a"]["accuracy"] == 0.0
    assert scored.per_mode["a"]["actual_nearest"] is None


def test_the_assignment_counts_by_nearest_centroid() -> None:
    centroids = {"dialectical": np.array([1.0, 0.0]), "socratic": np.array([0.0, 1.0])}
    embedded = np.array([[1.0, 0.1], [0.9, 0.2], [0.1, 1.0], [0.0, 0.8]], dtype=np.float32)
    y = np.array([0, 0, 1, 1])
    counts = nearest_centroid_assignment(
        embedded, y, ["deliberative", "pedagogical"], centroids, ["dialectical", "socratic"]
    )
    assert counts.tolist() == [[2, 0], [0, 2]]


def test_the_similarity_matrix_is_cosine_and_therefore_scale_free() -> None:
    rows = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 2.0])}
    columns = {"x": np.array([3.0, 0.0]), "y": np.array([0.0, 0.5])}
    matrix = similarity_matrix(rows, columns, ["a", "b"], ["x", "y"])
    assert matrix[0, 0] == pytest.approx(1.0)
    assert matrix[0, 1] == pytest.approx(0.0)
    assert matrix[1, 1] == pytest.approx(1.0)


def test_centroids_are_the_mode_means_of_the_embedding() -> None:
    embedded = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 0.0]], dtype=np.float32)
    centroids = compute_centroids(embedded, np.array([0, 0, 1]), ["a", "b"])
    assert centroids["a"].tolist() == [1.0, 1.0]
    assert centroids["b"].tolist() == [4.0, 0.0]


def test_the_layer_filter_reads_both_spellings_and_can_keep_the_global_statistics() -> None:
    names = ["attn_mass_L8_mean", "resid_velocity_L31", "logit_entropy", "gate_L16_sparsity"]
    assert build_layer_indices(names) == [0, 1, 3]
    assert build_layer_indices(names, layers={8}) == [0]
    assert build_layer_indices(names, layers={8}, include_unlayered=True) == [0, 2]
    assert build_layer_indices(names, include_unlayered=True) == [0, 1, 2, 3]


def test_loading_names_the_key_it_cannot_find(tmp_path: Path) -> None:
    write_run(tmp_path, "run_a", TRAIN_MODES, seed=1)
    names = load_feature_names("run_a", tmp_path)
    assert len(names) == 4
    with pytest.raises(KeyError, match="features_missing"):
        load_run_features("run_a", tmp_path, feature_key="features_missing")
    with pytest.raises(FileNotFoundError, match="signatures dir not found"):
        load_run_features("absent", tmp_path)


def test_loading_concatenates_several_keys_in_the_order_asked_for(tmp_path: Path) -> None:
    write_run(tmp_path, "run_a", TRAIN_MODES, seed=1, extra_key="features_addon")
    X, y, modes, labels = load_run_features(
        "run_a", tmp_path, feature_key=["features", "features_addon"]
    )
    assert X.shape == (18, 6), "four baseline columns plus the two-wide addon"
    assert modes == sorted(TRAIN_MODES), "the label encoder orders the modes"
    assert len(y) == len(labels) == 18


def test_a_column_selection_applies_after_the_concatenation(tmp_path: Path) -> None:
    write_run(tmp_path, "run_a", TRAIN_MODES, seed=1, extra_key="features_addon")
    X, _y, _modes, _labels = load_run_features(
        "run_a", tmp_path, feature_key=["features", "features_addon"], feature_indices=[4, 5]
    )
    assert X.shape == (18, 2)
    with pytest.raises(ValueError, match="feature_indices is empty"):
        load_run_features("run_a", tmp_path, feature_indices=[])


def test_two_runs_of_different_widths_are_refused(tmp_path: Path) -> None:
    write_run(tmp_path, "run_a", TRAIN_MODES, width=4, seed=1)
    write_run(tmp_path, "run_b", TEST_MODES, width=5, seed=2)
    with pytest.raises(RuntimeError, match="feature widths must match"):
        cross_run_transfer(
            train_run="run_a", test_run="run_b", n_seeds=1, n_epochs=5, outputs_base=tmp_path
        )


def test_the_lda_leg_separates_a_shared_space_from_an_unshared_one() -> None:
    rng = np.random.default_rng(0)
    n, d = 30, 5
    X_a = rng.standard_normal((n, d))
    y_a = np.array([i % 3 for i in range(n)])
    for cls in range(3):
        X_a[y_a == cls, cls] += 6.0

    shared = X_a + 0.1 * rng.standard_normal((n, d))
    y_b = y_a.copy()
    result = lda_direction_test(X_a, y_a, ["a", "b", "c"], shared, y_b, ["x", "y", "z"])
    assert result["r2_sil_own_lda"] > 0.2, "the fitted run separates in its own space"
    assert result["r3_sil_in_r2_lda"] > 0.0, "a run sharing the directions projects into them"
    assert result["projection_dim"] == 2, "three classes give two discriminants"
    assert set(result["r3_per_mode_sil"]) == {"x", "y", "z"}
    assert np.asarray(result["cross_prediction_matrix"]).shape == (3, 3)
    assert "directions-versus-manifolds" in result["interpretation"]

    unshared = rng.standard_normal((n, d))
    scrambled = lda_direction_test(X_a, y_a, ["a", "b", "c"], unshared, y_b, ["x", "y", "z"])
    assert scrambled["r3_sil_in_r2_lda"] < result["r3_sil_in_r2_lda"], (
        "a run with no structure in those directions reads lower"
    )


def test_one_transfer_pass_runs_end_to_end_and_carries_its_comparison(tmp_path: Path) -> None:
    write_run(tmp_path, "run_train", TRAIN_MODES, seed=1)
    write_run(tmp_path, "run_test", TEST_MODES, seed=2)
    results = cross_run_transfer(
        train_run="run_train",
        test_run="run_test",
        n_seeds=2,
        n_epochs=10,
        bottleneck_dim=4,
        outputs_base=tmp_path,
    )
    assert results["train_modes"] == sorted(TRAIN_MODES)
    assert results["test_modes"] == sorted(TEST_MODES)
    forward = results["transfer_forward"]
    assert forward["n_seeds"] == 2
    assert len(forward["mapping_accuracy_per_seed"]) == 2
    assert forward["chance_accuracy"] == pytest.approx(1 / 3)
    mapping = mode_mapping(DEFAULT_MODE_MAPPING)
    forward_wildcard, _ = mapping_wildcards(DEFAULT_MODE_MAPPING)
    assert set(forward["per_mode_accuracy"]) == set(mapping.pairs)
    assert forward_wildcard not in results["test_modes"], (
        "this corpus has no wildcard mode, so no wildcard row is reported"
    )
    assert "wildcard_assignments" not in forward
    assert results["mapping"] == DEFAULT_MODE_MAPPING
    assert results["forward_mapping"] == dict(mapping.pairs)
    assert results["reverse_mapping"] == mapping.reverse_pairs()
    assert mapping.reference is not None
    assert results["reference"] == mapping.reference.model_dump()
    assert np.asarray(forward["median_similarity_matrix"]).shape == (3, 3)

    lines = headline(results)
    assert len(lines) == 4
    model = mapping.reference.model
    assert all(model in line for line in lines), "every number travels with its comparison"
