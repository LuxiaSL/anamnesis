"""Reading banked gauntlet results across runs: the joins, and what they refuse.

Every reading here is over files another pass produced, so the tests build those files and
check the readings by value:

  * a run that is absent or unreadable is **named and skipped**, not fatal, because a report
    over the runs that exist is the normal case;
  * a section that does not validate leaves the rest of the file readable;
  * pair difficulty is read off the pair's two modes, and the three buckets are kept apart
    because averaging them hides the hard pairs;
  * the complementarity matrix is over **hard pairs only**, a tier at ceiling on all of them
    is dropped by name rather than entering as a flat profile, and the correlation is sorted
    so the most complementary pair reads first;
  * feature importance is grouped by family and by sub-family, including the baseline blocks
    that carry no family prefix;
  * the hardest confusion is read off the matrix that carries its own labels — the reading
    the frozen record could not produce, because it looked for a field that does not exist;
  * value-add is a delta against the baseline composites, and a missing baseline is reported
    as ``None`` rather than as a zero delta.

CPU only; no classifier runs here at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anamnesis.analysis.complementarity import (
    COMPLEMENTARY_BAR,
    DIVERGENCE_BAR,
    analyze_complementarity,
    analyze_confusion,
    analyze_consistency,
    analyze_resolution,
    analyze_subfamily_importance,
    analyze_tier_ordering,
    analyze_value_add,
    complementarity_report,
    feature_family,
    feature_subfamily,
    load_report_inputs,
    load_results,
    pair_difficulty,
    pair_name,
)

LABELS = ["analogical", "contrastive", "dialectical", "linear", "socratic"]


def tier_block(
    accuracy: float,
    *,
    pairwise: dict[str, float] | None = None,
    confusion: list[list[int]] | None = None,
) -> dict[str, object]:
    """One tier's section of a banked classification result, in its own schema.

    Every required field is present, because the reader validates the section and a
    fixture that skipped one would be testing the fallback rather than the reading.
    """
    return {
        "rf_5way": {
            "accuracy": accuracy,
            "fold_accuracies": [accuracy] * 5,
            "confusion_matrix": confusion,
            "labels": LABELS if confusion else None,
        },
        "topic_heldout": {
            "accuracy": accuracy,
            "fold_accuracies": [accuracy] * 5,
            "n_groups": 20,
        },
        "linear_probe": {"accuracy": accuracy, "fold_accuracies": [accuracy] * 5},
        "pairwise_binary": {
            pair: {"accuracy": value, "fold_accuracies": [value]}
            for pair, value in (pairwise or {}).items()
        },
        "rf_4way_no_analogical": {
            "accuracy": accuracy,
            "fold_accuracies": [accuracy] * 5,
            "confusion_matrix": None,
            "labels": None,
        },
    }


def hard_pairs(values: dict[tuple[str, str], float]) -> dict[str, float]:
    return {pair_name(a, b): value for (a, b), value in values.items()}


def write_results(
    root: Path,
    run: str,
    *,
    by_tier: dict[str, dict[str, object]],
    n_samples: int = 100,
    top_features: list[tuple[str, float]] | None = None,
) -> Path:
    directory = root / run
    directory.mkdir(parents=True, exist_ok=True)
    document: dict[str, object] = {
        "run_name": run,
        "n_samples": n_samples,
        "classification": {"by_tier": by_tier},
    }
    if top_features is not None:
        ranked = [
            {"name": name, "importance": importance} for name, importance in top_features
        ]
        document["tier_ablation"] = {
            "per_tier_accuracy": {"T2": 0.5},
            "pairwise_tier_combinations": {},
            "leave_one_tier_out": {},
            "tier_ranking": [],
            "tier_inversion_t25_gt_t2_gt_t1": False,
            "top_features_rf": ranked,
            "top_features_rf_t2t25": ranked,
            "top_features_lr_t2t25": ranked,
            "tier_contribution_ratio": {},
            "std_vs_mean": {},
            "cohens_d_per_topic": {
                "per_topic": {},
                "mean_d": None,
                "median_d": None,
                "std_d": None,
                "min_d": None,
                "max_d": None,
                "all_positive": None,
                "n_topics": 0,
            },
        }
    (directory / "results.json").write_text(json.dumps(document))
    return directory / "results.json"


def test_a_pairs_difficulty_is_read_off_its_two_modes() -> None:
    assert pair_name("socratic", "linear") == "linear_vs_socratic", "the key is alphabetical"
    assert pair_difficulty("linear_vs_socratic") == "hard"
    assert pair_difficulty("compressed_vs_linear") == "cross"
    assert pair_difficulty("associative_vs_compressed") == "easy-easy"
    assert pair_difficulty("linear") == "unknown"


def test_an_absent_or_broken_run_is_named_and_skipped(tmp_path: Path) -> None:
    assert load_results(tmp_path / "nothing" / "results.json") is None
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "results.json").write_text("{not json")
    assert load_results(broken / "results.json") is None

    write_results(tmp_path, "8b_v2", by_tier={"T1": tier_block(0.4)})
    loaded = load_report_inputs(tmp_path)
    assert set(loaded) == {"8b_v2"}, "only the run that exists is loaded"


def test_a_section_that_does_not_validate_leaves_the_file_readable(tmp_path: Path) -> None:
    directory = tmp_path / "8b_v2"
    directory.mkdir()
    (directory / "results.json").write_text(
        json.dumps({"n_samples": 40, "classification": {"by_tier": {"T1": {"nonsense": 1}}}})
    )
    results = load_results(directory / "results.json")
    assert results is not None
    assert results["n_samples"] == 40, "the rest of the file is still readable"
    assert isinstance(results["classification"], dict), (
        "an unvalidatable section stays raw rather than taking the file with it"
    )
    assert analyze_resolution({"8b_v2": results}) == {}, (
        "a reading over a section it cannot type reports nothing for that run"
    )


def test_consistency_flags_the_tiers_that_moved(tmp_path: Path) -> None:
    write_results(tmp_path, "8b_baseline", by_tier={"T1": tier_block(0.40), "T2": tier_block(0.60)})
    write_results(
        tmp_path, "8b_v2_5way", by_tier={"T1": tier_block(0.41), "T2": tier_block(0.80)}
    )
    results = load_report_inputs(tmp_path)
    comparisons = analyze_consistency(results)["comparisons"]
    assert len(comparisons) == 1, "the 3B pair is absent and is skipped"
    tiers = comparisons[0]["tiers"]
    assert tiers["T1"]["divergent"] is False
    assert tiers["T2"]["divergent"] is True
    assert tiers["T2"]["diff"] == pytest.approx(0.20)
    assert abs(tiers["T1"]["diff"]) < DIVERGENCE_BAR


def test_resolution_keeps_the_difficulty_buckets_apart(tmp_path: Path) -> None:
    pairwise = hard_pairs(
        {
            ("linear", "socratic"): 0.60,
            ("linear", "dialectical"): 0.70,
            ("compressed", "linear"): 0.95,
            ("associative", "compressed"): 0.99,
        }
    )
    write_results(tmp_path, "8b_v2", by_tier={"T2": tier_block(0.5, pairwise=pairwise)})
    resolution = analyze_resolution(load_report_inputs(tmp_path))["8b_v2"]["T2"]
    assert resolution["hard"]["n_pairs"] == 2
    assert resolution["hard"]["mean"] == pytest.approx(0.65)
    assert resolution["hard"]["min"] == pytest.approx(0.60)
    assert resolution["cross"]["n_pairs"] == 1
    assert resolution["easy-easy"]["mean"] == pytest.approx(0.99)


def test_the_complementarity_matrix_drops_a_tier_with_no_profile(tmp_path: Path) -> None:
    hard = {
        ("linear", "socratic"): 0.6,
        ("linear", "dialectical"): 0.9,
        ("socratic", "dialectical"): 0.7,
    }
    inverse = {key: 1.5 - value for key, value in hard.items()}
    flat = {key: 1.0 for key in hard}
    write_results(
        tmp_path,
        "8b_v2",
        by_tier={
            "T2": tier_block(0.5, pairwise=hard_pairs(hard)),
            "T2.5": tier_block(0.5, pairwise=hard_pairs(inverse)),
            "T3": tier_block(0.5, pairwise=hard_pairs(flat)),
        },
    )
    out = analyze_complementarity(load_report_inputs(tmp_path))["8b_v2"]
    assert len(out["hard_pairs"]) == 3
    pairs = out["pairs_by_correlation"]
    assert [row["r"] for row in pairs] == sorted(row["r"] for row in pairs)
    assert {row["tier_a"] for row in pairs} | {row["tier_b"] for row in pairs} == {"T2", "T2.5"}, (
        "the tier at ceiling on every hard pair has no profile to correlate"
    )
    assert pairs[0]["r"] < COMPLEMENTARY_BAR and pairs[0]["reading"] == "COMPLEMENTARY"


def test_feature_names_are_grouped_by_family_and_sub_family() -> None:
    assert feature_family("cp_L16_t3_d07") == "contrastive_projection"
    assert feature_family("af_L8_recency_bias") == "attention_flow"
    assert feature_family("key_drift_L16") == "T2.5"
    assert feature_family("attn_entropy_L8") == "T2"
    assert feature_family("pca_resid_L16") == "T3"
    assert feature_family("logit_entropy") == "T1"
    assert feature_family("mystery").startswith("unknown(")

    assert feature_subfamily("cp_L16_t3_d07") == "cp_t3"
    assert feature_subfamily("td_L16_key_drift_w0") == "td_key_drift"
    assert feature_subfamily("af_L8_sysprompt_decay_rate") == "af_sysprompt_decay"
    assert feature_subfamily("af_L8_region_early_gen") == "af_region_early"
    assert feature_subfamily("gf_L16_sparsity_mean") == "gf_sparsity"
    assert feature_subfamily("rt_L24_velocity_norm") == "rt_velocity"
    assert feature_subfamily("something_else").startswith("other(")


def test_importance_is_summed_per_family_and_per_sub_family(tmp_path: Path) -> None:
    write_results(
        tmp_path,
        "8b_v2",
        by_tier={"T2": tier_block(0.5)},
        top_features=[
            ("af_L8_recency_bias_mean", 0.10),
            ("af_L16_recency_bias_mean", 0.05),
            ("td_L16_key_drift_w0", 0.20),
        ],
    )
    out = analyze_subfamily_importance(load_report_inputs(tmp_path))["8b_v2"]
    assert out["family_importance"]["attention_flow"] == pytest.approx(0.15)
    assert out["family_importance"]["temporal_dynamics"] == pytest.approx(0.20)
    assert out["subfam_importance"]["af_recency_bias"] == pytest.approx(0.15)
    assert out["n_features_ranked"] == 3
    assert out["subfam_importance_t2t25"]["td_key_drift"] == pytest.approx(0.20)


def test_the_hardest_confusion_is_read_off_the_matrix_that_carries_its_labels(
    tmp_path: Path,
) -> None:
    confusion = [
        [10, 0, 0, 0, 0],
        [0, 10, 0, 0, 0],
        [0, 0, 10, 0, 0],
        [0, 0, 0, 5, 5],
        [0, 0, 0, 5, 5],
    ]
    write_results(tmp_path, "8b_v2", by_tier={"T2": tier_block(0.8, confusion=confusion)})
    out = analyze_confusion(load_report_inputs(tmp_path))["8b_v2"]
    assert out["tiers_analyzed"] == ["T2"]
    hardest = out["hardest_confusion"]["T2"]
    assert hardest["pair"] == pair_name("linear", "socratic"), (
        "the two classes the matrix recovers half the time are the hardest pair"
    )
    assert hardest["mean_diagonal"] == pytest.approx(0.5)


def test_a_confusion_matrix_without_labels_is_skipped(tmp_path: Path) -> None:
    write_results(tmp_path, "8b_v2", by_tier={"T2": tier_block(0.8)})
    out = analyze_confusion(load_report_inputs(tmp_path))["8b_v2"]
    assert out["tiers_analyzed"] == []


def test_the_ordering_check_reports_the_three_accuracies_and_its_verdict(tmp_path: Path) -> None:
    write_results(
        tmp_path,
        "8b_v2",
        n_samples=100,
        by_tier={
            "T1": tier_block(0.30),
            "T2": tier_block(0.50),
            "T2.5": tier_block(0.62),
            "T3": tier_block(0.40),
        },
    )
    out = analyze_tier_ordering(load_report_inputs(tmp_path))["8b_v2"]
    assert out["inversion"] is True
    assert out["n_modes"] == 5, "the mode count comes from the samples and the topics per mode"
    assert out["T3"] == pytest.approx(0.40)


def test_value_add_is_a_delta_and_a_missing_baseline_is_not_a_zero(tmp_path: Path) -> None:
    write_results(
        tmp_path, "8b_baseline", by_tier={"T2+T2.5": tier_block(0.60), "combined": tier_block(0.70)}
    )
    write_results(
        tmp_path,
        "8b_v2_5way",
        by_tier={
            "attention_flow": tier_block(0.66),
            "engineered": tier_block(0.75),
        },
    )
    out = analyze_value_add(load_report_inputs(tmp_path))["8B"]
    assert out["families"]["attention_flow"]["delta_vs_baseline_t2t25"] == pytest.approx(0.06)
    assert out["composites"]["engineered"]["delta_vs_baseline_combined"] == pytest.approx(0.05)

    write_results(tmp_path, "3b_run4", by_tier={"T1": tier_block(0.3)})
    write_results(tmp_path, "3b_v2_5way", by_tier={"attention_flow": tier_block(0.5)})
    out_3b = analyze_value_add(load_report_inputs(tmp_path))["3B"]
    assert out_3b["baseline_t2t25"] is None
    assert out_3b["families"]["attention_flow"]["delta_vs_baseline_t2t25"] is None


def test_the_report_carries_all_seven_readings(tmp_path: Path) -> None:
    write_results(tmp_path, "8b_v2", by_tier={"T1": tier_block(0.4), "T2": tier_block(0.6)})
    report = complementarity_report(load_report_inputs(tmp_path))
    assert set(report) == {
        "runs",
        "consistency",
        "resolution",
        "complementarity",
        "subfamily_importance",
        "confusion",
        "tier_ordering",
        "value_add",
    }
    assert report["runs"] == ["8b_v2"]
