"""Section 10: Prediction scorecard — evaluate pre-registered 8B predictions."""

from __future__ import annotations

from typing import Any

from .schemas import (
    CCGPResult,
    ClassificationResult,
    IntrinsicDimensionResult,
    ScorecardPrediction,
    ScorecardResult,
    ScorecardSummary,
    TierAblationResult,
    TopologyResult,
)


def run_scorecard(all_results: dict[str, Any]) -> ScorecardResult:
    """Evaluate 9 pre-registered predictions against computed results.

    Predictions from research/notes/8b-predictions-preregistration.md
    """
    predictions: list[ScorecardPrediction] = []

    # ── Prediction 1: CCGP = 1.0 ──
    ccgp_result = all_results.get("ccgp")
    ccgp_scores: list[float] = []
    if isinstance(ccgp_result, CCGPResult):
        ccgp_scores = [v.ccgp_score for v in ccgp_result.variants.values()]

    min_ccgp = min(ccgp_scores) if ccgp_scores else None
    p1_outcome = "CONFIRMED" if min_ccgp is not None and min_ccgp >= 1.0 else (
        "PARTIAL" if min_ccgp is not None and min_ccgp >= 0.89 else "WRONG"
    )
    predictions.append(ScorecardPrediction(
        prediction="1. CCGP = 1.0",
        confidence="95%",
        importance="HIGH",
        outcome=p1_outcome,
        metric=f"min CCGP across variants = {min_ccgp}",
        surprise_threshold="CCGP < 0.89",
    ))

    # ── Prediction 2: Centroid topology preserved ──
    topo_result = all_results.get("topology")
    nearest = ""
    outgroup_ratio = 0.0
    if isinstance(topo_result, TopologyResult):
        euc = topo_result.topology_summary.get("euclidean")
        if euc is not None:
            nearest = euc.nearest_pair
            outgroup_ratio = euc.analogical_outgroup_ratio

    expected_pairs = {"contrastive-dialectical", "dialectical-contrastive",
                      "linear-socratic", "socratic-linear"}
    has_expected_near = nearest in expected_pairs
    has_outgroup = outgroup_ratio > 1.2

    p2_outcome = "CONFIRMED" if has_expected_near and has_outgroup else (
        "PARTIAL" if has_expected_near or has_outgroup else "WRONG"
    )
    predictions.append(ScorecardPrediction(
        prediction="2. Centroid topology preserved",
        confidence="85%",
        importance="HIGH",
        outcome=p2_outcome,
        metric=f"nearest={nearest}, outgroup_ratio={outgroup_ratio:.2f}",
        detail=f"expected_near={has_expected_near}, outgroup={has_outgroup}",
    ))

    # ── Prediction 3: T1/T2/T2.5 ID convergence ──
    id_result = all_results.get("intrinsic_dimension")
    convergence_values: dict[str, float] = {}
    max_diff: float | None = None
    if isinstance(id_result, IntrinsicDimensionResult) and id_result.tier_convergence is not None:
        max_diff = id_result.tier_convergence.max_pairwise_diff
        convergence_values = id_result.tier_convergence.values

    p3_outcome = "CONFIRMED" if max_diff is not None and max_diff < 4.0 else (
        "PARTIAL" if max_diff is not None and max_diff < 6.0 else "WRONG"
    )
    predictions.append(ScorecardPrediction(
        prediction="3. T1/T2/T2.5 ID convergence (within ±2)",
        confidence="75%",
        importance="HIGH",
        outcome=p3_outcome,
        metric=f"max pairwise diff = {max_diff}",
        values=convergence_values,
    ))

    # ── Prediction 4: Per-mode ID ordering preserved ──
    mode_ids: dict[str, float] = {}
    if isinstance(id_result, IntrinsicDimensionResult) and id_result.per_mode is not None:
        for mode, mdata in id_result.per_mode.items():
            if isinstance(mdata.dadapy_id, (int, float)):
                mode_ids[mode] = float(mdata.dadapy_id)

    expected_order = ["linear", "contrastive", "dialectical", "socratic", "analogical"]
    actual_order: list[str]
    if len(mode_ids) >= 5:
        actual_order = sorted(mode_ids.keys(), key=lambda m: mode_ids[m])
        n_in_order = sum(
            1 for i, m in enumerate(expected_order)
            if m in actual_order and abs(actual_order.index(m) - i) <= 1
        )
        p4_outcome = "CONFIRMED" if n_in_order >= 4 else (
            "PARTIAL" if n_in_order >= 3 else "WRONG"
        )
    else:
        p4_outcome = "INSUFFICIENT_DATA"
        actual_order = []

    predictions.append(ScorecardPrediction(
        prediction="4. Per-mode ID ordering: linear < contrastive < dialectical < socratic ≈ analogical",
        confidence="70%",
        importance="MEDIUM",
        outcome=p4_outcome,
        expected_order=expected_order,
        actual_order=actual_order,
        mode_ids=mode_ids,
    ))

    # ── Prediction 5: T3 remains elevated ──
    t3_id: float | None = None
    t1_id: float | None = None
    t2_id: float | None = None
    if isinstance(id_result, IntrinsicDimensionResult) and id_result.global_ is not None:
        for tier_key, target in (("T3", "t3"), ("T1", "t1"), ("T2", "t2")):
            entry = id_result.global_.get(tier_key)
            if entry is not None and isinstance(entry.dadapy_id, (int, float)):
                if target == "t3":
                    t3_id = float(entry.dadapy_id)
                elif target == "t1":
                    t1_id = float(entry.dadapy_id)
                else:
                    t2_id = float(entry.dadapy_id)

    mean_other: float | None = None
    if t3_id is not None and t1_id is not None and t2_id is not None:
        mean_other = (t1_id + t2_id) / 2
        elevated = t3_id > mean_other + 5
        p5_outcome = "CONFIRMED" if elevated else (
            "PARTIAL" if t3_id > mean_other + 2 else "WRONG"
        )
    else:
        p5_outcome = "INSUFFICIENT_DATA"

    predictions.append(ScorecardPrediction(
        prediction="5. T3 remains elevated relative to T1/T2/T2.5",
        confidence="80%",
        importance="MEDIUM",
        outcome=p5_outcome,
        t3_id=t3_id,
        mean_t1_t2=mean_other,
    ))

    # ── Prediction 6: T2.5 load-bearing (tier inversion) ──
    ablation = all_results.get("tier_ablation")
    tier_inversion: bool | None = None
    per_tier: dict[str, float] = {}
    removal_cost: dict[str, float | None] = {"T1": None, "T2": None, "T2.5": None}
    if isinstance(ablation, TierAblationResult):
        tier_inversion = ablation.tier_inversion_t25_gt_t2_gt_t1
        per_tier = ablation.per_tier_accuracy
        for tier_key in ("T1", "T2", "T2.5"):
            entry = ablation.leave_one_tier_out.get(tier_key)
            removal_cost[tier_key] = entry.cost_of_removal if entry is not None else None

    p6_outcome = "CONFIRMED" if tier_inversion else (
        "PARTIAL" if per_tier.get("T2.5", 0) >= per_tier.get("T1", 0) else "WRONG"
    )
    predictions.append(ScorecardPrediction(
        prediction="6. T2.5 load-bearing (T2.5 > T2 > T1 accuracy)",
        confidence="70%",
        importance="MEDIUM",
        outcome=p6_outcome,
        tier_inversion_holds=tier_inversion,
        per_tier_accuracy=per_tier,
        removal_costs=removal_cost,
    ))

    # ── Prediction 7: 5-way accuracy ~67-73% ──
    clf_result = all_results.get("classification")
    t2t25_acc: float | None = None
    combined_acc: float | None = None
    if isinstance(clf_result, ClassificationResult):
        t2t25 = clf_result.by_tier.get("T2+T2.5")
        combined = clf_result.by_tier.get("combined")
        t2t25_acc = t2t25.rf_5way.accuracy if t2t25 is not None else None
        combined_acc = combined.rf_5way.accuracy if combined is not None else None

    best_acc = max(t2t25_acc or 0.0, combined_acc or 0.0)
    p7_outcome = "CONFIRMED" if 0.57 <= best_acc <= 0.83 else (
        "PARTIAL" if 0.50 <= best_acc <= 0.90 else "WRONG"
    )
    predictions.append(ScorecardPrediction(
        prediction="7. 5-way accuracy ~67-73%",
        confidence="50%",
        importance="LOW",
        outcome=p7_outcome,
        t2t25_accuracy=t2t25_acc,
        combined_accuracy=combined_acc,
    ))

    # ── Prediction 8: Hard pairs improve more ──
    pairwise: dict[str, float] = {}
    if isinstance(clf_result, ClassificationResult):
        t2t25 = clf_result.by_tier.get("T2+T2.5")
        if t2t25 is not None:
            pairwise = {p: entry.accuracy for p, entry in t2t25.pairwise_binary.items()}

    hard_pairs = ["linear_vs_socratic", "linear_vs_contrastive", "contrastive_vs_socratic"]
    easy_pairs = ["analogical_vs_contrastive", "analogical_vs_dialectical",
                  "analogical_vs_linear", "analogical_vs_socratic"]

    hard_accs = [pairwise[p] for p in hard_pairs if p in pairwise]
    easy_accs = [pairwise[p] for p in easy_pairs if p in pairwise]

    mean_hard: float | None
    mean_easy: float | None
    if hard_accs and easy_accs:
        mean_hard = float(sum(hard_accs) / len(hard_accs))
        mean_easy = float(sum(easy_accs) / len(easy_accs))
        p8_outcome = "NOTED"
    else:
        mean_hard = mean_easy = None
        p8_outcome = "INSUFFICIENT_DATA"

    predictions.append(ScorecardPrediction(
        prediction="8. Hard pairs improve more than easy pairs",
        confidence="55%",
        importance="LOW",
        outcome=p8_outcome,
        mean_hard_pair_accuracy=mean_hard,
        mean_easy_pair_accuracy=mean_easy,
        all_pairwise=pairwise if pairwise else {},
    ))

    # ── Prediction 9: Delta-hyperbolicity slight decrease ──
    delta_rel: float | None = None
    if isinstance(topo_result, TopologyResult):
        delta_rel = topo_result.gromov_delta_euclidean.delta_relative

    if delta_rel is not None:
        p9_outcome = "CONFIRMED" if delta_rel < 0.154 else (
            "PARTIAL" if delta_rel < 0.20 else "WRONG"
        )
    else:
        p9_outcome = "INSUFFICIENT_DATA"

    predictions.append(ScorecardPrediction(
        prediction="9. Delta-hyperbolicity slight decrease (more tree-like)",
        confidence="50%",
        importance="LOW",
        outcome=p9_outcome,
        delta_rel_8b=delta_rel,
        delta_rel_3b=0.154,
    ))

    # Summary
    outcomes = [p.outcome for p in predictions]
    summary = ScorecardSummary(
        confirmed=outcomes.count("CONFIRMED"),
        partial=outcomes.count("PARTIAL"),
        wrong=outcomes.count("WRONG"),
        noted=outcomes.count("NOTED"),
        insufficient=outcomes.count("INSUFFICIENT_DATA"),
        total=len(predictions),
    )

    return ScorecardResult(predictions=predictions, summary=summary)
