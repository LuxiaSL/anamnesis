"""Reading banked gauntlet results across runs: the joins, and what they refuse.

Every reading here is over files another pass produced, so the tests build those files and
check the readings by value:

  * a run that is absent or unreadable is **named and skipped**, not fatal, because a report
    over the runs that exist is the normal case;
  * a section that does not validate leaves the rest of the file readable;
  * pair difficulty is read off the pair's two modes, and the three buckets are kept apart
    because averaging them hides the hard pairs;
  * the complementarity matrix is over **hard pairs only**, a block at ceiling on all of them
    is dropped by name rather than entering as a flat profile, and the correlation is sorted
    so the most complementary pair reads first;
  * feature importance is grouped by family and by sub-family, including the core blocks
    that carry no family prefix;
  * the hardest confusion is read off the matrix that carries its own labels — the reading
    the frozen record could not produce, because it looked for a field that does not exist;
  * value-add is a delta against the baseline composites, and a missing baseline is reported
    as ``None`` rather than as a zero delta;
  * a banked union is reported under its legacy label, never under a current one.

CPU only; no classifier runs here at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anamnesis.analysis.complementarity import (
    BLOCK_BY_FAMILY,
    COMPLEMENTARY_BAR,
    DIVERGENCE_BAR,
    analyze_complementarity,
    analyze_confusion,
    analyze_consistency,
    analyze_resolution,
    analyze_subfamily_importance,
    analyze_block_ordering,
    analyze_value_add,
    complementarity_report,
    feature_family,
    feature_subfamily,
    load_report_inputs,
    load_results,
    pair_difficulty,
    pair_name,
)
from anamnesis.analysis.gauntlet.schemas.compat import (
    LEGACY_EVERYTHING,
    LEGACY_FAMILY_UNION,
    LEGACY_UNION_LABELS,
    READ_ONLY_LABELS,
)
from anamnesis.analysis.subfamily import classify_signal
from anamnesis.analysis.gauntlet.signature_io import (
    ALL_CORE,
    ALL_FAMILIES,
    ATTENTION_AND_CACHE,
    ATTENTION_AND_DELTAS,
    ATTENTION_FLOW,
    CACHE_AND_KEYS,
    GATE_FEATURES,
    NORMS_AND_OUTPUT_STATS,
    RESIDUAL_PCA,
    RESIDUAL_TRAJECTORY,
)
from anamnesis.feature_map import (
    FAMILY_ATTENTION_FLOW,
    FAMILY_GATE,
    FAMILY_LABELS,
    FAMILY_RESIDUAL_TRAJECTORY,
    named_family,
)

LABELS = ["analogical", "contrastive", "dialectical", "linear", "socratic"]


def banked_spelling(legacy: str) -> str:
    """The spelling a banked results file carries for a legacy union.

    Taken from the compat table rather than written here, so the banked vocabulary has
    one home and a fixture spelled with it is spelled the way the reader expects.
    """
    return next(banked for banked, label in LEGACY_UNION_LABELS.items() if label == legacy)


def block_block(
    accuracy: float,
    *,
    pairwise: dict[str, float] | None = None,
    confusion: list[list[int]] | None = None,
) -> dict[str, object]:
    """One block's section of a banked classification result, in its own schema.

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
    by_block: dict[str, dict[str, object]],
    n_samples: int = 100,
    top_features: list[tuple[str, float]] | None = None,
) -> Path:
    directory = root / run
    directory.mkdir(parents=True, exist_ok=True)
    document: dict[str, object] = {
        "run_name": run,
        "n_samples": n_samples,
        "classification": {"by_block": by_block},
    }
    if top_features is not None:
        ranked = [
            {"name": name, "importance": importance} for name, importance in top_features
        ]
        document["legacy_bin_readout"] = {
            "per_block_accuracy": {ATTENTION_AND_DELTAS: 0.5},
            "pairwise_block_combinations": {},
            "leave_one_block_out": {},
            "block_ranking": [],
            "cache_beats_attention_beats_norms": False,
            "top_features_rf": ranked,
            "top_features_rf_attention_and_cache": ranked,
            "top_features_lr_attention_and_cache": ranked,
            "block_contribution_ratio": {},
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

    write_results(tmp_path, "8b_v2", by_block={NORMS_AND_OUTPUT_STATS: block_block(0.4)})
    loaded = load_report_inputs(tmp_path)
    assert set(loaded) == {"8b_v2"}, "only the run that exists is loaded"


def test_a_section_that_does_not_validate_leaves_the_file_readable(tmp_path: Path) -> None:
    directory = tmp_path / "8b_v2"
    directory.mkdir()
    (directory / "results.json").write_text(
        json.dumps({"n_samples": 40, "classification": {"by_block": {NORMS_AND_OUTPUT_STATS: {"nonsense": 1}}}})
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


def test_consistency_flags_the_blocks_that_moved(tmp_path: Path) -> None:
    write_results(tmp_path, "8b_baseline", by_block={NORMS_AND_OUTPUT_STATS: block_block(0.40), ATTENTION_AND_DELTAS: block_block(0.60)})
    write_results(
        tmp_path, "8b_v2_5way", by_block={NORMS_AND_OUTPUT_STATS: block_block(0.41), ATTENTION_AND_DELTAS: block_block(0.80)}
    )
    results = load_report_inputs(tmp_path)
    comparisons = analyze_consistency(results)["comparisons"]
    assert len(comparisons) == 1, "the 3B pair is absent and is skipped"
    blocks = comparisons[0]["blocks"]
    assert blocks[NORMS_AND_OUTPUT_STATS]["divergent"] is False
    assert blocks[ATTENTION_AND_DELTAS]["divergent"] is True
    assert blocks[ATTENTION_AND_DELTAS]["diff"] == pytest.approx(0.20)
    assert abs(blocks[NORMS_AND_OUTPUT_STATS]["diff"]) < DIVERGENCE_BAR


def test_resolution_keeps_the_difficulty_buckets_apart(tmp_path: Path) -> None:
    pairwise = hard_pairs(
        {
            ("linear", "socratic"): 0.60,
            ("linear", "dialectical"): 0.70,
            ("compressed", "linear"): 0.95,
            ("associative", "compressed"): 0.99,
        }
    )
    write_results(tmp_path, "8b_v2", by_block={ATTENTION_AND_DELTAS: block_block(0.5, pairwise=pairwise)})
    resolution = analyze_resolution(load_report_inputs(tmp_path))["8b_v2"][ATTENTION_AND_DELTAS]
    assert resolution["hard"]["n_pairs"] == 2
    assert resolution["hard"]["mean"] == pytest.approx(0.65)
    assert resolution["hard"]["min"] == pytest.approx(0.60)
    assert resolution["cross"]["n_pairs"] == 1
    assert resolution["easy-easy"]["mean"] == pytest.approx(0.99)


def test_the_complementarity_matrix_drops_a_block_with_no_profile(tmp_path: Path) -> None:
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
        by_block={
            ATTENTION_AND_DELTAS: block_block(0.5, pairwise=hard_pairs(hard)),
            CACHE_AND_KEYS: block_block(0.5, pairwise=hard_pairs(inverse)),
            RESIDUAL_PCA: block_block(0.5, pairwise=hard_pairs(flat)),
        },
    )
    out = analyze_complementarity(load_report_inputs(tmp_path))["8b_v2"]
    assert len(out["hard_pairs"]) == 3
    pairs = out["pairs_by_correlation"]
    assert [row["r"] for row in pairs] == sorted(row["r"] for row in pairs)
    assert {row["block_a"] for row in pairs} | {row["block_b"] for row in pairs} == {ATTENTION_AND_DELTAS, CACHE_AND_KEYS}, (
        "the block at ceiling on every hard pair has no profile to correlate"
    )
    assert pairs[0]["r"] < COMPLEMENTARY_BAR and pairs[0]["reading"] == "COMPLEMENTARY"


def test_feature_names_are_grouped_by_family_and_sub_family() -> None:
    """Every spelling here is one a banked corpus carries.

    Key geometry is written ``kv_key_drift_L16`` and a residual norm
    ``activation_norm_mean_L0``; a rule keyed on ``key_drift`` or ``act_norm`` matches
    no column in any bank and credits nothing.
    """
    assert feature_family("cp_L16_t3_d07") == "contrastive_projection"
    assert feature_family("af_L8_recency_bias") == ATTENTION_FLOW
    assert feature_family("attn_flow_recency_bias_L16") == ATTENTION_FLOW, (
        "one family, two spellings — the corpora that named it either way"
    )
    assert feature_family("kv_key_drift_L16") == CACHE_AND_KEYS
    assert feature_family("cache_recency_bias_L8") == CACHE_AND_KEYS
    assert feature_family("attn_entropy_L8") == ATTENTION_AND_DELTAS
    assert feature_family("spectral_fiedler_L28") == ATTENTION_AND_DELTAS, (
        "the similarity graph is built from attention distributions"
    )
    assert feature_family("delta_norm_mean_L0") == ATTENTION_AND_DELTAS, (
        "a cross-layer residual delta is addressed in the attention block, not the norms one"
    )
    assert feature_family("pca_resid_L16") == RESIDUAL_PCA
    assert feature_family("logit_entropy") == NORMS_AND_OUTPUT_STATS
    assert feature_family("activation_norm_mean_L0") == NORMS_AND_OUTPUT_STATS
    assert feature_family("mystery").startswith("unknown(")

    assert feature_subfamily("cp_L16_t3_d07") == "cp_t3"
    assert feature_subfamily("td_L16_key_drift_w0") == "td_key_drift"
    assert feature_subfamily("af_L8_sysprompt_decay_rate") == "af_sysprompt_decay"
    assert feature_subfamily("af_L8_region_early_gen") == "af_region_early_gen"
    assert feature_subfamily("gf_L16_sparsity_mean") == "gf_sparsity"
    assert feature_subfamily("rt_L24_velocity_norm") == "rt_velocity"
    assert feature_subfamily("something_else").startswith("other(")


BLOCKLESS_FAMILIES = frozenset({
    "value_geometry", "qk_geometry", "kv_cka", "per_head", "attn_res", "expert_routing",
    "path_signature", "path_signature_output", "path_signature_attention",
})
"""Families extracted after these corpora were banked, so no block of a banked result
holds them. A new family lands in one of the three sets by decision, not by default: this
one, the translation table, or the read-only labels — whose family label and banked block
label are one string, which is why the fallback translates them."""

CLASSIFIED_SPELLINGS = (
    "cp_L16_t3_d07", "af_L8_recency_bias", "attn_flow_recency_bias_L16", "gf_L16_sparsity_mean",
    "gate_sparsity_mean_L16", "rt_L24_velocity_norm", "res_traj_velocity_L24",
    "td_L16_key_drift_w0", "kv_key_drift_L16", "cache_recency_bias_L8", "epoch_n_transitions_mean",
    "attn_entropy_mean_L8", "head_agreement_mean_L8", "delta_norm_mean_L0",
    "spectral_fiedler_L28", "pca_resid_L16", "logit_entropy_mean", "activation_norm_mean_L0",
    "top1_prob_mean", "surprise_traj0", "mean_chosen_rank", "std_surprise",
    "value_key_corr_L8", "qk_align_L8", "kv_value_cka_L8", "ph_head_entropy_L8",
    "res_sig_lvl2_L16", "out_sig_lvl1_entropy", "attn_sig_lvl2_recent",
    "attnres_committed_cos_L8", "xrt_switch_rate",
)
"""One spelling per family, from the corpora each family's extraction produced."""


def test_one_classifier_answers_the_family_question() -> None:
    """This module classifies no feature name itself: it translates.

    :func:`anamnesis.feature_map.named_family` is the classifier, and the translation
    table is keyed by the family constants that module defines, so the two cannot drift
    onto two spellings of one family. Every family the classifier can return is either
    translated to a block here or declared blockless.
    """
    assert set(BLOCK_BY_FAMILY) <= FAMILY_LABELS, (
        "a label is translated that the classifier never returns"
    )
    assert FAMILY_LABELS - set(BLOCK_BY_FAMILY) == BLOCKLESS_FAMILIES | READ_ONLY_LABELS, (
        "a family was added to the taxonomy without saying which block, if any, holds it"
    )
    for label, block in (
        (FAMILY_ATTENTION_FLOW, ATTENTION_FLOW),
        (FAMILY_GATE, GATE_FEATURES),
        (FAMILY_RESIDUAL_TRAJECTORY, RESIDUAL_TRAJECTORY),
    ):
        assert BLOCK_BY_FAMILY[label] == block

    for name in CLASSIFIED_SPELLINGS:
        family = named_family(name)
        assert family is not None, f"{name} is a banked spelling and has to classify"
        assert feature_family(name) == BLOCK_BY_FAMILY.get(family, family)
    assert {named_family(name) for name in CLASSIFIED_SPELLINGS} == set(FAMILY_LABELS), (
        "every family needs a spelling here, or the translation is untested for it"
    )


def test_one_classifier_answers_the_sub_family_question_too() -> None:
    """This module holds no naming rules of its own on either question.

    The sub-family table is printed beside the family one, so a second reading of the
    same conventions would key two tables two ways. It is also how a classifier goes
    stale unnoticed: the one nobody calls from a decomposition keeps its old spellings,
    misses the whole ranked list it is pointed at, and reports a bucket per feature.
    """
    assert feature_subfamily is not classify_signal, "the delegation is a named local step"
    for name in CLASSIFIED_SPELLINGS:
        assert feature_subfamily(name) == classify_signal(name)


def test_the_top_of_a_ranked_list_groups_into_signals_not_into_names() -> None:
    """The call site's population: bare core-block names, which carry no family prefix.

    Every one of them is a miss for a classifier that knows only the engineered prefixes,
    and the table it feeds then has one row per feature.
    """
    ranked = (
        "activation_norm_mean_L0",
        "activation_norm_std_L16",
        "attn_entropy_mean_L8",
        "head_agreement_mean_L8",
        "cache_recency_bias_L8",
        "cache_recency_traj0_L8",
        "kv_key_novelty_mean_L16",
        "kv_key_novelty_std_L16",
        "pca_L8_t0_c0",
        "pca_L8_t0_c1",
    )
    grouped = {feature_subfamily(name) for name in ranked}
    assert not any(signal.startswith("other(") for signal in grouped)
    assert grouped == {
        "activation_norm",
        "attn_entropy",
        "head_agreement",
        "cache_recency",
        "kv_key_novelty",
        "pca_t0",
    }


def test_importance_is_summed_per_family_and_per_sub_family(tmp_path: Path) -> None:
    write_results(
        tmp_path,
        "8b_v2",
        by_block={ATTENTION_AND_DELTAS: block_block(0.5)},
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
    assert out["subfam_importance_attention_and_cache"]["td_key_drift"] == pytest.approx(0.20)


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
    write_results(tmp_path, "8b_v2", by_block={ATTENTION_AND_DELTAS: block_block(0.8, confusion=confusion)})
    out = analyze_confusion(load_report_inputs(tmp_path))["8b_v2"]
    assert out["blocks_analyzed"] == [ATTENTION_AND_DELTAS]
    hardest = out["hardest_confusion"][ATTENTION_AND_DELTAS]
    assert hardest["pair"] == pair_name("linear", "socratic"), (
        "the two classes the matrix recovers half the time are the hardest pair"
    )
    assert hardest["mean_diagonal"] == pytest.approx(0.5)


def test_a_confusion_matrix_without_labels_is_skipped(tmp_path: Path) -> None:
    write_results(tmp_path, "8b_v2", by_block={ATTENTION_AND_DELTAS: block_block(0.8)})
    out = analyze_confusion(load_report_inputs(tmp_path))["8b_v2"]
    assert out["blocks_analyzed"] == []


def test_the_ordering_check_reports_the_three_accuracies_and_its_verdict(tmp_path: Path) -> None:
    write_results(
        tmp_path,
        "8b_v2",
        n_samples=100,
        by_block={
            NORMS_AND_OUTPUT_STATS: block_block(0.30),
            ATTENTION_AND_DELTAS: block_block(0.50),
            CACHE_AND_KEYS: block_block(0.62),
            RESIDUAL_PCA: block_block(0.40),
        },
    )
    out = analyze_block_ordering(load_report_inputs(tmp_path))["8b_v2"]
    assert out["inversion"] is True
    assert out["n_modes"] == 5, "the mode count comes from the samples and the topics per mode"
    assert out[RESIDUAL_PCA] == pytest.approx(0.40)


def test_value_add_is_a_delta_and_a_missing_baseline_is_not_a_zero(tmp_path: Path) -> None:
    write_results(
        tmp_path, "8b_baseline", by_block={ATTENTION_AND_CACHE: block_block(0.60), ALL_CORE: block_block(0.70)}
    )
    write_results(
        tmp_path,
        "8b_v2_5way",
        by_block={
            "attention_flow": block_block(0.66),
            ALL_FAMILIES: block_block(0.75),
        },
    )
    out = analyze_value_add(load_report_inputs(tmp_path))["8B"]
    assert out["families"]["attention_flow"]["delta_vs_baseline_attention_and_cache"] == pytest.approx(0.06)
    assert out["composites"][ALL_FAMILIES]["delta_vs_baseline_combined"] == pytest.approx(0.05)

    write_results(tmp_path, "3b_run4", by_block={NORMS_AND_OUTPUT_STATS: block_block(0.3)})
    write_results(tmp_path, "3b_v2_5way", by_block={"attention_flow": block_block(0.5)})
    out_3b = analyze_value_add(load_report_inputs(tmp_path))["3B"]
    assert out_3b["baseline_attention_and_cache"] is None
    assert out_3b["families"]["attention_flow"]["delta_vs_baseline_attention_and_cache"] is None


def test_value_add_reports_a_banked_union_under_its_legacy_label(tmp_path: Path) -> None:
    """The banked spellings arrive as the legacy unions, never as the current ones."""
    write_results(tmp_path, "8b_baseline", by_block={ALL_CORE: block_block(0.70)})
    write_results(
        tmp_path,
        "8b_v2_5way",
        by_block={
            banked_spelling(LEGACY_FAMILY_UNION): block_block(0.75),
            banked_spelling(LEGACY_EVERYTHING): block_block(0.80),
        },
    )
    composites = analyze_value_add(load_report_inputs(tmp_path))["8B"]["composites"]
    assert set(composites) == {LEGACY_FAMILY_UNION, LEGACY_EVERYTHING}
    assert composites[LEGACY_FAMILY_UNION]["delta_vs_baseline_combined"] == pytest.approx(0.05)


def test_the_complementarity_matrix_leaves_out_the_whole_vector_under_either_membership(
    tmp_path: Path,
) -> None:
    """The every-block union correlates with its members by construction, banked or not."""
    pairwise_a = hard_pairs({("linear", "socratic"): 0.6, ("linear", "dialectical"): 0.8})
    pairwise_b = hard_pairs({("linear", "socratic"): 0.9, ("linear", "dialectical"): 0.5})
    write_results(
        tmp_path,
        "8b_v2",
        by_block={
            ATTENTION_AND_DELTAS: block_block(0.5, pairwise=pairwise_a),
            CACHE_AND_KEYS: block_block(0.5, pairwise=pairwise_b),
            banked_spelling(LEGACY_EVERYTHING): block_block(0.9, pairwise=pairwise_a),
        },
    )
    out = analyze_complementarity(load_report_inputs(tmp_path))["8b_v2"]
    assert set(out["blocks"]) == {ATTENTION_AND_DELTAS, CACHE_AND_KEYS}


def test_the_report_carries_all_seven_readings(tmp_path: Path) -> None:
    write_results(tmp_path, "8b_v2", by_block={NORMS_AND_OUTPUT_STATS: block_block(0.4), ATTENTION_AND_DELTAS: block_block(0.6)})
    report = complementarity_report(load_report_inputs(tmp_path))
    assert set(report) == {
        "runs",
        "consistency",
        "resolution",
        "complementarity",
        "subfamily_importance",
        "confusion",
        "block_ordering",
        "value_add",
    }
    assert report["runs"] == ["8b_v2"]
