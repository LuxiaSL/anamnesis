"""Section 10: Prediction scorecard — evaluate pre-registered 8B predictions.

This section reads other sections rather than the data, which makes the error
stub a possible value of every one of its inputs: a section that could not run
returns its reason in place of its numbers. So each prediction is scored only
from evidence that is actually there, and a prediction whose evidence is absent
reads INSUFFICIENT_DATA with the upstream's own reason recorded beside it.

That distinction is the point of the section. A prediction scored WRONG because
its input was missing is a finding the run did not make — the same overstatement
as a union reported at a width it does not have — so the scoring branches are
reached only once their inputs exist, and `anamnesis/analysis/gauntlet/utils.py`
· `section_reading` is what decides that.
"""

from __future__ import annotations

from typing import Any

from .signature_io import (
    ALL_CORE,
    ATTENTION_AND_CACHE,
    ATTENTION_AND_DELTAS,
    CACHE_AND_KEYS,
    NORMS_AND_OUTPUT_STATS,
    RESIDUAL_PCA,
)
from .schemas import (
    CCGPResult,
    ClassificationResult,
    IntrinsicDimensionResult,
    ScorecardPrediction,
    ScorecardResult,
    ScorecardSummary,
    LegacyBinReadoutResult,
    TopologyResult,
)
from .utils import section_reading


def run_scorecard(all_results: dict[str, Any]) -> ScorecardResult:
    """Evaluate 9 pre-registered predictions against computed results.

    Each prediction was registered with its confidence and its importance before
    the 8B results existed, and is restated at the branch that scores it: the
    text, the threshold and the registered confidence travel together in the
    `ScorecardPrediction` row, so the scorecard is readable without a second
    document beside it.

    Every row comes back. A row whose upstream section did not run carries
    ``outcome="INSUFFICIENT_DATA"`` and that section's reason in
    ``unscorable_because``; if no row could be scored, the result also carries
    ``error``, so a pass over a corpus that supported none of the nine is short by
    this section rather than reporting nine verdicts it did not earn.
    """
    predictions: list[ScorecardPrediction] = []

    ccgp_result, ccgp_gap = section_reading(all_results, "ccgp", CCGPResult)
    topo_result, topo_gap = section_reading(all_results, "topology", TopologyResult)
    id_result, id_gap = section_reading(
        all_results, "intrinsic_dimension", IntrinsicDimensionResult,
    )
    ablation, ablation_gap = section_reading(
        all_results, "legacy_bin_readout", LegacyBinReadoutResult,
    )
    clf_result, clf_gap = section_reading(
        all_results, "classification", ClassificationResult,
    )

    # ── Prediction 1: CCGP = 1.0 ──
    ccgp_scores: list[float] = []
    if ccgp_result is not None and ccgp_result.variants is not None:
        ccgp_scores = [v.ccgp_score for v in ccgp_result.variants.values()]
    p1_gap = ccgp_gap if ccgp_result is None else (
        "section 'ccgp' scored no variant on this corpus" if not ccgp_scores else None
    )

    min_ccgp = min(ccgp_scores) if ccgp_scores else None
    if min_ccgp is None:
        p1_outcome = "INSUFFICIENT_DATA"
    else:
        p1_outcome = "CONFIRMED" if min_ccgp >= 1.0 else (
            "PARTIAL" if min_ccgp >= 0.89 else "WRONG"
        )
    predictions.append(ScorecardPrediction(
        prediction="1. CCGP = 1.0",
        confidence="95%",
        importance="HIGH",
        outcome=p1_outcome,
        unscorable_because=p1_gap,
        metric=f"min CCGP across variants = {min_ccgp}",
        surprise_threshold="CCGP < 0.89",
    ))

    # ── Prediction 2: Centroid topology preserved ──
    euc = None
    if topo_result is not None:
        euc = (topo_result.topology_summary or {}).get("euclidean")
    p2_gap = topo_gap if topo_result is None else (
        "section 'topology' carries no euclidean centroid summary" if euc is None else None
    )

    if euc is None:
        p2_outcome = "INSUFFICIENT_DATA"
        nearest, outgroup_ratio = "", 0.0
        has_expected_near = has_outgroup = False
    else:
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
        unscorable_because=p2_gap,
        metric=f"nearest={nearest}, outgroup_ratio={outgroup_ratio:.2f}",
        detail=f"expected_near={has_expected_near}, outgroup={has_outgroup}",
    ))

    # ── Prediction 3: intrinsic dimension converges across the core blocks ──
    convergence_values: dict[str, float] = {}
    max_diff: float | None = None
    if id_result is not None and id_result.block_convergence is not None:
        max_diff = id_result.block_convergence.max_pairwise_diff
        convergence_values = id_result.block_convergence.values
    p3_gap = id_gap if id_result is None else (
        "the three core blocks' intrinsic dimensions were not all estimated"
        if max_diff is None else None
    )

    if max_diff is None:
        p3_outcome = "INSUFFICIENT_DATA"
    else:
        p3_outcome = "CONFIRMED" if max_diff < 4.0 else (
            "PARTIAL" if max_diff < 6.0 else "WRONG"
        )
    predictions.append(ScorecardPrediction(
        prediction=(
            "3. Intrinsic dimension converges across the first three core blocks "
            "(within ±2)"
        ),
        confidence="75%",
        importance="HIGH",
        outcome=p3_outcome,
        unscorable_because=p3_gap,
        metric=f"max pairwise diff = {max_diff}",
        values=convergence_values,
    ))

    # ── Prediction 4: Per-mode ID ordering preserved ──
    mode_ids: dict[str, float] = {}
    if id_result is not None and id_result.per_mode is not None:
        for mode, mdata in id_result.per_mode.items():
            if isinstance(mdata.dadapy_id, (int, float)):
                mode_ids[mode] = float(mdata.dadapy_id)

    expected_order = ["linear", "contrastive", "dialectical", "socratic", "analogical"]
    actual_order: list[str]
    p4_gap: str | None = None
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
        p4_gap = id_gap or (
            f"per-mode intrinsic dimension read for {len(mode_ids)} modes, "
            "and the ordering needs five"
        )

    predictions.append(ScorecardPrediction(
        prediction="4. Per-mode ID ordering: linear < contrastive < dialectical < socratic ≈ analogical",
        confidence="70%",
        importance="MEDIUM",
        outcome=p4_outcome,
        unscorable_because=p4_gap,
        expected_order=expected_order,
        actual_order=actual_order,
        mode_ids=mode_ids,
    ))

    # ── Prediction 5: residual PCA stays elevated ──
    def block_id(label: str) -> float | None:
        if id_result is None or id_result.global_ is None:
            return None
        entry = id_result.global_.get(label)
        if entry is None or not isinstance(entry.dadapy_id, (int, float)):
            return None
        return float(entry.dadapy_id)

    residual_pca_id = block_id(RESIDUAL_PCA)
    norms_id = block_id(NORMS_AND_OUTPUT_STATS)
    attention_id = block_id(ATTENTION_AND_DELTAS)

    mean_other: float | None = None
    p5_gap: str | None = None
    if residual_pca_id is not None and norms_id is not None and attention_id is not None:
        mean_other = (norms_id + attention_id) / 2
        elevated = residual_pca_id > mean_other + 5
        p5_outcome = "CONFIRMED" if elevated else (
            "PARTIAL" if residual_pca_id > mean_other + 2 else "WRONG"
        )
    else:
        p5_outcome = "INSUFFICIENT_DATA"
        p5_gap = id_gap or (
            "the residual-PCA, norms and attention blocks were not all estimated"
        )

    predictions.append(ScorecardPrediction(
        prediction=(
            "5. The residual-PCA block's intrinsic dimension stays elevated "
            "relative to the other core blocks"
        ),
        confidence="80%",
        importance="MEDIUM",
        outcome=p5_outcome,
        unscorable_because=p5_gap,
        residual_pca_id=residual_pca_id,
        mean_norms_and_attention_id=mean_other,
    ))

    # ── Prediction 6: the cache-and-keys block is load-bearing ──
    block_inversion: bool | None = None
    per_block: dict[str, float] = {}
    removal_cost: dict[str, float | None] = {NORMS_AND_OUTPUT_STATS: None, ATTENTION_AND_DELTAS: None, CACHE_AND_KEYS: None}
    if ablation is not None:
        block_inversion = ablation.cache_beats_attention_beats_norms
        per_block = ablation.per_block_accuracy
        for block_key in (NORMS_AND_OUTPUT_STATS, ATTENTION_AND_DELTAS, CACHE_AND_KEYS):
            entry = ablation.leave_one_block_out.get(block_key)
            removal_cost[block_key] = entry.cost_of_removal if entry is not None else None

    # The three blocks the prediction orders have to be measured for the ordering
    # to mean anything; a corpus without them scores nothing rather than WRONG.
    ordered_blocks = (CACHE_AND_KEYS, ATTENTION_AND_DELTAS, NORMS_AND_OUTPUT_STATS)
    p6_gap: str | None = None
    if ablation is None:
        p6_outcome = "INSUFFICIENT_DATA"
        p6_gap = ablation_gap
    elif not all(block in per_block for block in ordered_blocks):
        p6_outcome = "INSUFFICIENT_DATA"
        missing = [block for block in ordered_blocks if block not in per_block]
        p6_gap = f"no accuracy for {', '.join(missing)} in this corpus"
    else:
        p6_outcome = "CONFIRMED" if block_inversion else (
            "PARTIAL" if per_block[CACHE_AND_KEYS] >= per_block[NORMS_AND_OUTPUT_STATS]
            else "WRONG"
        )
    predictions.append(ScorecardPrediction(
        prediction=(
            "6. Cache reads and key geometry are load-bearing: that block beats "
            "attention-and-deltas, which beats norms-and-output-stats"
        ),
        confidence="70%",
        importance="MEDIUM",
        outcome=p6_outcome,
        unscorable_because=p6_gap,
        cache_beats_attention_beats_norms=block_inversion,
        per_block_accuracy=per_block,
        removal_costs=removal_cost,
    ))

    # ── Prediction 7: 5-way accuracy ~67-73% ──
    attention_and_cache_acc: float | None = None
    combined_acc: float | None = None
    if clf_result is not None:
        attention_and_cache = clf_result.by_block.get(ATTENTION_AND_CACHE)
        combined = clf_result.by_block.get(ALL_CORE)
        attention_and_cache_acc = attention_and_cache.rf_5way.accuracy if attention_and_cache is not None else None
        combined_acc = combined.rf_5way.accuracy if combined is not None else None

    scored = [acc for acc in (attention_and_cache_acc, combined_acc) if acc is not None]
    p7_gap: str | None = None
    if not scored:
        p7_outcome = "INSUFFICIENT_DATA"
        p7_gap = clf_gap or (
            f"neither {ATTENTION_AND_CACHE} nor {ALL_CORE} was classified in this corpus"
        )
        best_acc = None
    else:
        best_acc = max(scored)
        p7_outcome = "CONFIRMED" if 0.57 <= best_acc <= 0.83 else (
            "PARTIAL" if 0.50 <= best_acc <= 0.90 else "WRONG"
        )
    predictions.append(ScorecardPrediction(
        prediction="7. 5-way accuracy ~67-73%",
        confidence="50%",
        importance="LOW",
        outcome=p7_outcome,
        unscorable_because=p7_gap,
        attention_and_cache_accuracy=attention_and_cache_acc,
        combined_accuracy=combined_acc,
    ))

    # ── Prediction 8: Hard pairs improve more ──
    pairwise: dict[str, float] = {}
    if clf_result is not None:
        attention_and_cache = clf_result.by_block.get(ATTENTION_AND_CACHE)
        if attention_and_cache is not None:
            pairwise = {p: entry.accuracy for p, entry in attention_and_cache.pairwise_binary.items()}

    hard_pairs = ["linear_vs_socratic", "linear_vs_contrastive", "contrastive_vs_socratic"]
    easy_pairs = ["analogical_vs_contrastive", "analogical_vs_dialectical",
                  "analogical_vs_linear", "analogical_vs_socratic"]

    hard_accs = [pairwise[p] for p in hard_pairs if p in pairwise]
    easy_accs = [pairwise[p] for p in easy_pairs if p in pairwise]

    mean_hard: float | None
    mean_easy: float | None
    p8_gap: str | None = None
    if hard_accs and easy_accs:
        mean_hard = float(sum(hard_accs) / len(hard_accs))
        mean_easy = float(sum(easy_accs) / len(easy_accs))
        p8_outcome = "NOTED"
    else:
        mean_hard = mean_easy = None
        p8_outcome = "INSUFFICIENT_DATA"
        p8_gap = clf_gap or (
            f"no pairwise accuracies for {ATTENTION_AND_CACHE} covering both a hard "
            "and an easy pair"
        )

    predictions.append(ScorecardPrediction(
        prediction="8. Hard pairs improve more than easy pairs",
        confidence="55%",
        importance="LOW",
        outcome=p8_outcome,
        unscorable_because=p8_gap,
        mean_hard_pair_accuracy=mean_hard,
        mean_easy_pair_accuracy=mean_easy,
        all_pairwise=pairwise if pairwise else {},
    ))

    # ── Prediction 9: Delta-hyperbolicity slight decrease ──
    delta_rel: float | None = None
    if topo_result is not None and topo_result.gromov_delta_euclidean is not None:
        delta_rel = topo_result.gromov_delta_euclidean.delta_relative

    p9_gap: str | None = None
    if delta_rel is not None:
        p9_outcome = "CONFIRMED" if delta_rel < 0.154 else (
            "PARTIAL" if delta_rel < 0.20 else "WRONG"
        )
    else:
        p9_outcome = "INSUFFICIENT_DATA"
        p9_gap = topo_gap or "section 'topology' carries no delta-hyperbolicity"

    predictions.append(ScorecardPrediction(
        prediction="9. Delta-hyperbolicity slight decrease (more tree-like)",
        confidence="50%",
        importance="LOW",
        outcome=p9_outcome,
        unscorable_because=p9_gap,
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

    # A scorecard that scored nothing is a section the pass is short by, so it
    # carries the reason where `section_shortfall` reads it. One that scored some
    # rows landed: its unscored rows say what they were waiting on.
    scored_nothing = summary.insufficient == summary.total
    gaps = [p.unscorable_because for p in predictions if p.unscorable_because]
    error = (
        f"no prediction could be scored: {'; '.join(dict.fromkeys(gaps))}"
        if scored_nothing else None
    )

    return ScorecardResult(predictions=predictions, summary=summary, error=error)
