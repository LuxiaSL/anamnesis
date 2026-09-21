"""Complementarity: which families see different things, read across banked runs.

Every other analysis here reads signatures. This one reads *results* — the
gauntlet's own banked JSON, several runs of it at once — and asks the questions
that only exist between runs and between families:

1. **Consistency.** The same tier, the same modes, two corpora. A tier whose
   accuracy moves by more than five points between them is flagged, because a
   tier that is not stable across corpora is not a property of the model.
2. **Resolution.** Pairwise accuracy split by how hard the pair is. The
   format-controlled five are the hard pairs; anything involving a
   format-free mode is easier and averaging the two hides it.
3. **Complementarity.** Two tiers' accuracy *profiles* over the hard pairs,
   correlated. A low correlation means they fail on different pairs, which is what
   makes them worth combining; a high one means one of them is redundant. Easy
   pairs are excluded because their shared ceiling dominates the correlation, and a
   tier at ceiling on every hard pair has no profile at all and is dropped by name.
4. **Sub-family importance.** The banked feature importances, grouped by family and
   by sub-family, so importance is read at the resolution the names support.
5. **Confusion.** Which pair each tier finds hardest, off its own confusion matrix.
6. **Ordering.** Whether the historical tier ordering holds in each run.
7. **Value-add.** The engineered families and composites against the baseline
   composites, on the mode subset the two corpora share.

Nothing here computes a classification; everything here is a reading of one. That
is the point of the separation: the gauntlet is expensive and its results are
banked, so the cross-run questions are answered by re-reading rather than by
re-running.

The tier and family names in these tables are the record's own, because they are
the keys in banked result files. The taxonomy sweep renames the code, never the
keys on disk.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from anamnesis.analysis.gauntlet.schemas import ClassificationResult, TierAblationResult

logger = logging.getLogger(__name__)

HARD_MODES: frozenset[str] = frozenset(
    {"linear", "socratic", "contrastive", "dialectical", "analogical"}
)
"""The format-controlled five: a pair drawn from these is a hard pair."""

EASY_MODES: frozenset[str] = frozenset({"compressed", "structured", "associative"})
"""Format-free modes, which carry a format tell and are therefore easier."""

CORE_RUNS: tuple[str, ...] = ("8b_baseline", "3b_run4", "8b_v2", "3b_v2")
"""Runs the report reads by default: two baselines and two engineered corpora."""

SUBSET_RUNS: tuple[str, ...] = ("8b_v2_5way", "3b_v2_5way")
"""The mode-subset passes, read when present — they are what make an engineered
corpus comparable to a five-mode baseline."""

BASELINE_TIER_ORDER: tuple[str, ...] = ("T1", "T2", "T2.5", "T3", "T2+T2.5", "combined")
"""Reading order for the consistency table. The names are banked keys."""

CONSISTENCY_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("8b_baseline", "8b_v2_5way", "8B: baseline vs engineered (five modes)"),
    ("3b_run4", "3b_v2_5way", "3B: baseline vs engineered (five modes)"),
)

DIVERGENCE_BAR = 0.05
"""Accuracy difference at which two corpora are called divergent for a tier."""

COMPLEMENTARY_BAR = 0.30
REDUNDANT_BAR = 0.60
"""Profile-correlation bars: below the first two tiers are complementary, above the
second they are redundant, between them moderate."""

ZERO_VARIANCE = 1e-10

COMPOSITE_TIERS: frozenset[str] = frozenset(
    {"T2+T2.5", "combined", "engineered", "combined_v2", "T2+T2.5+engineered"}
)
"""Tiers that are unions of others. They belong in the correlation matrix but not in
the per-pair table, where they would double-count their members."""

NEW_FAMILIES: tuple[str, ...] = (
    "residual_trajectory",
    "attention_flow",
    "gate_features",
    "temporal_dynamics",
    "contrastive_projection",
)
V2_COMPOSITES: tuple[str, ...] = ("engineered", "T2+T2.5+engineered", "combined_v2")

FAMILY_BY_PREFIX: dict[str, str] = {
    "cp": "contrastive_projection",
    "af": "attention_flow",
    "td": "temporal_dynamics",
    "rt": "residual_trajectory",
    "gf": "gate_features",
}
"""Engineered families carry a two-letter prefix. The baseline blocks do not, which
is why the fallback below reads their signal names instead."""

BASELINE_BLOCK_BY_SIGNAL: tuple[tuple[str, str], ...] = (
    ("lookback", "T2.5"),
    ("key_drift", "T2.5"),
    ("key_novelty", "T2.5"),
    ("epoch", "T2.5"),
    ("attn_entropy", "T2"),
    ("head_agree", "T2"),
    ("residual", "T2"),
    ("pca_", "T3"),
    ("act_norm", "T1"),
    ("logit", "T1"),
    ("token", "T1"),
    ("delta", "T1"),
)
"""Which baseline block a feature name belongs to, by the signal it names. Order
matters: the more specific signals are matched first."""


def pair_name(mode_a: str, mode_b: str) -> str:
    """A pair's canonical key, alphabetical so both orders agree."""
    return f"{min(mode_a, mode_b)}_vs_{max(mode_a, mode_b)}"


def pair_difficulty(pair: str) -> str:
    """``hard``, ``cross``, ``easy-easy`` or ``unknown`` for a pair key."""
    modes = pair.split("_vs_")
    if len(modes) != 2:
        return "unknown"
    first, second = modes
    if first in HARD_MODES and second in HARD_MODES:
        return "hard"
    if first in EASY_MODES or second in EASY_MODES:
        return "easy-easy" if first in EASY_MODES and second in EASY_MODES else "cross"
    return "unknown"


def load_results(path: Path) -> dict[str, Any] | None:
    """One banked ``results.json``, with its typed sections validated on read.

    Returns None where the file is absent or unreadable, because a report over the
    runs that exist is the normal case and a missing run is named rather than
    fatal. A section that fails validation is left as the raw mapping and warned
    about: the rest of the file is still readable, and a reader who needs that
    section gets nothing rather than something mis-shaped.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"{path}: unreadable ({exc})")
        return None
    for key, model in (
        ("classification", ClassificationResult),
        ("tier_ablation", TierAblationResult),
    ):
        if isinstance(raw.get(key), dict):
            try:
                raw[key] = model.model_validate(raw[key])
            except ValueError as exc:
                logger.warning(f"{path}: {key} does not validate ({exc})")
    return raw


def load_report_inputs(
    analysis_dir: Path,
    *,
    runs: Sequence[str] = CORE_RUNS,
    subset_runs: Sequence[str] = SUBSET_RUNS,
    include_subsets: bool = True,
) -> dict[str, dict[str, Any]]:
    """Every named run's results that are present under an analysis directory."""
    wanted = list(runs) + (list(subset_runs) if include_subsets else [])
    loaded: dict[str, dict[str, Any]] = {}
    for run in wanted:
        results = load_results(Path(analysis_dir) / run / "results.json")
        if results is None:
            logger.info(f"  missing: {run}")
            continue
        loaded[run] = results
        logger.info(f"  loaded: {run}")
    return loaded


def _by_tier(results: Mapping[str, Any] | None) -> dict[str, Any]:
    """A run's per-tier classification, or empty where the section is untyped."""
    if not results:
        return {}
    section = results.get("classification")
    return section.by_tier if isinstance(section, ClassificationResult) else {}


def _accuracy(by_tier: Mapping[str, Any], tier: str) -> float | None:
    entry = by_tier.get(tier)
    return entry.rf_5way.accuracy if entry is not None else None


def analyze_consistency(
    results: Mapping[str, Mapping[str, Any]],
    *,
    pairs: Sequence[tuple[str, str, str]] = CONSISTENCY_PAIRS,
    tiers: Sequence[str] = BASELINE_TIER_ORDER,
) -> dict[str, Any]:
    """The same tier across two corpora, with the divergent ones flagged."""
    comparisons: list[dict[str, Any]] = []
    for run_a, run_b, label in pairs:
        left, right = results.get(run_a), results.get(run_b)
        if not left or not right:
            logger.info(f"  {label}: skipped (missing {run_a if not left else run_b})")
            continue
        a_tiers, b_tiers = _by_tier(left), _by_tier(right)
        entry: dict[str, Any] = {"label": label, "tiers": {}}
        for tier in tiers:
            acc_a, acc_b = _accuracy(a_tiers, tier), _accuracy(b_tiers, tier)
            if acc_a is None or acc_b is None:
                continue
            difference = acc_b - acc_a
            entry["tiers"][tier] = {
                "run_a": acc_a,
                "run_b": acc_b,
                "diff": difference,
                "divergent": abs(difference) > DIVERGENCE_BAR,
            }
        comparisons.append(entry)
    return {"comparisons": comparisons}


def analyze_resolution(
    results: Mapping[str, Mapping[str, Any]], *, runs: Sequence[str] = ("8b_v2", "3b_v2")
) -> dict[str, Any]:
    """Pairwise accuracy per tier, bucketed by pair difficulty."""
    out: dict[str, Any] = {}
    for run in runs:
        by_tier = _by_tier(results.get(run))
        if not by_tier:
            continue
        per_tier: dict[str, dict[str, Any]] = {}
        for tier in sorted(t for t, v in by_tier.items() if v.pairwise_binary):
            buckets: dict[str, list[float]] = {"hard": [], "cross": [], "easy-easy": []}
            for pair, entry in by_tier[tier].pairwise_binary.items():
                bucket = pair_difficulty(pair)
                if bucket in buckets:
                    buckets[bucket].append(entry.accuracy)
            per_tier[tier] = {
                bucket: {
                    "mean": float(np.mean(values)) if values else None,
                    "min": float(np.min(values)) if values else None,
                    "max": float(np.max(values)) if values else None,
                    "n_pairs": len(values),
                }
                for bucket, values in buckets.items()
            }
        out[run] = per_tier
    return out


def analyze_complementarity(
    results: Mapping[str, Mapping[str, Any]], *, runs: Sequence[str] = ("8b_v2",)
) -> dict[str, Any]:
    """Correlate tiers' hard-pair accuracy profiles; lower means complementary.

    A tier absent from a pair is read at chance (0.5) rather than dropped, so every
    tier's profile spans the same pairs and the correlation is over one index.
    """
    out: dict[str, Any] = {}
    for run in runs:
        by_tier = _by_tier(results.get(run))
        if not by_tier:
            continue
        tiers = [t for t, v in by_tier.items() if v.pairwise_binary and t != "combined_v2"]
        pairs = sorted({p for t in tiers for p in by_tier[t].pairwise_binary})
        profile = np.full((len(tiers), len(pairs)), 0.5, dtype=np.float64)
        for i, tier in enumerate(tiers):
            for j, pair in enumerate(pairs):
                entry = by_tier[tier].pairwise_binary.get(pair)
                if entry is not None:
                    profile[i, j] = entry.accuracy

        hard_columns = [j for j, pair in enumerate(pairs) if pair_difficulty(pair) == "hard"]
        if len(tiers) <= 1 or not hard_columns:
            continue
        hard = profile[:, hard_columns]
        keep = [i for i in range(len(tiers)) if np.std(hard[i]) > ZERO_VARIANCE]
        for i in range(len(tiers)):
            if i not in keep:
                logger.info(f"  ({tiers[i]} skipped — no variance over the hard pairs)")

        ranked: list[dict[str, Any]] = []
        if len(keep) > 1:
            correlation = np.corrcoef(hard[keep])
            for a in range(len(keep)):
                for b in range(a + 1, len(keep)):
                    value = correlation[a, b]
                    if np.isnan(value):
                        continue
                    ranked.append(
                        {
                            "tier_a": tiers[keep[a]],
                            "tier_b": tiers[keep[b]],
                            "r": float(value),
                            "reading": (
                                "COMPLEMENTARY" if value < COMPLEMENTARY_BAR
                                else "MODERATE" if value < REDUNDANT_BAR
                                else "REDUNDANT"
                            ),
                        }
                    )
            ranked.sort(key=lambda row: row["r"])

        out[run] = {
            "tiers": tiers,
            "hard_pairs": [pairs[j] for j in hard_columns],
            "individual_tiers": [t for t in tiers if t not in COMPOSITE_TIERS],
            "pairs_by_correlation": ranked,
        }
    return out


def feature_family(name: str) -> str:
    """Which family a banked feature name belongs to."""
    prefix = name.split("_")[0] if "_" in name else name
    if prefix in FAMILY_BY_PREFIX:
        return FAMILY_BY_PREFIX[prefix]
    for signal, block in BASELINE_BLOCK_BY_SIGNAL:
        if name.startswith(signal):
            return block
    return f"unknown({name[:20]})"


def feature_subfamily(name: str) -> str:
    """Which sub-family a banked feature name belongs to, at importance resolution.

    This reads the same naming conventions
    :mod:`anamnesis.analysis.subfamily` cuts a family by, at the coarser grain a
    ranked importance list supports: a layer is dropped, because importance summed
    over one layer of one signal is a number over two or three features.
    """
    parts = name.split("_")
    head = parts[0]
    if head == "cp":
        return f"cp_{parts[2]}" if len(parts) >= 3 else "cp_unknown"
    if head == "td":
        if len(parts) < 3:
            return "td_unknown"
        signal = parts[2]
        if signal == "attn":
            return "td_attn_entropy"
        if signal == "head":
            return "td_head_agreement"
        if signal == "key":
            return f"td_key_{parts[3]}" if len(parts) >= 4 else "td_key"
        if signal == "lookback":
            return "td_lookback_ratio"
        return "td_unknown"
    if head == "af":
        if len(parts) < 3:
            return "af_unknown"
        signal = parts[2]
        if signal == "sysprompt":
            return (
                "af_sysprompt_decay"
                if len(parts) >= 4 and parts[3] == "decay"
                else "af_sysprompt_mass"
            )
        if signal == "recency":
            return "af_recency_bias"
        if signal == "region":
            return f"af_region_{parts[3]}" if len(parts) >= 4 else "af_region"
        if signal == "head":
            return "af_head_diversity"
        return "af_unknown"
    if head == "rt":
        return f"rt_{parts[2]}" if len(parts) >= 3 else "rt_unknown"
    if head == "gf":
        if len(parts) < 2:
            return "gf_unknown"
        if parts[1].startswith("L"):
            return f"gf_{parts[2]}" if len(parts) >= 3 else "gf_unknown"
        return f"gf_{parts[1]}"
    return f"other({name[:20]})"


def _sum_importance(features: Sequence[Any], key: Any) -> dict[str, float]:
    totals: dict[str, float] = {}
    for feature in features:
        totals[key(feature.name)] = totals.get(key(feature.name), 0.0) + feature.importance
    return totals


def analyze_subfamily_importance(
    results: Mapping[str, Mapping[str, Any]], *, runs: Sequence[str] = ("8b_v2",), top_n: int = 20
) -> dict[str, Any]:
    """Banked feature importance, summed by family and by sub-family."""
    out: dict[str, Any] = {}
    for run in runs:
        entry = results.get(run)
        if not entry:
            continue
        ablation = entry.get("tier_ablation")
        if not isinstance(ablation, TierAblationResult) or not ablation.top_features_rf:
            logger.info(f"  {run}: no banked feature importance")
            continue
        row: dict[str, Any] = {
            "family_importance": _sum_importance(ablation.top_features_rf, feature_family),
            "subfam_importance": _sum_importance(ablation.top_features_rf, feature_subfamily),
            "n_features_ranked": len(ablation.top_features_rf),
        }
        if ablation.top_features_rf_t2t25:
            row["subfam_importance_t2t25"] = _sum_importance(
                ablation.top_features_rf_t2t25[:top_n], feature_subfamily
            )
        out[run] = row
    return out


def analyze_confusion(
    results: Mapping[str, Mapping[str, Any]], *, runs: Sequence[str] = ("8b_v2",)
) -> dict[str, Any]:
    """Each tier's hardest confusion, off its own banked confusion matrix.

    The labels come from the matrix that carries them, which is where a confusion
    matrix's row and column order is stated. A pair's difficulty is read as the mean
    of the two modes' diagonal rates: a pair both of whose classes are recovered
    poorly is the pair a tier cannot separate.
    """
    out: dict[str, Any] = {}
    for run in runs:
        by_tier = _by_tier(results.get(run))
        if not by_tier:
            continue
        hardest: dict[str, dict[str, Any]] = {}
        for tier, entry in sorted(by_tier.items()):
            matrix = entry.rf_5way.confusion_matrix
            labels = entry.rf_5way.labels
            if not matrix or not labels:
                continue
            counts = np.asarray(matrix, dtype=np.float64)
            if counts.shape[0] < 3 or len(labels) != counts.shape[0]:
                continue
            normalized = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
            candidates = [
                ((normalized[i, i] + normalized[j, j]) / 2, labels[i], labels[j])
                for i in range(counts.shape[0])
                for j in range(counts.shape[0])
                if i != j
            ]
            rate, first, second = min(candidates)
            hardest[tier] = {"pair": pair_name(first, second), "mean_diagonal": float(rate)}
            logger.info(f"    {tier:<25} hardest: {first}-{second} ({rate:.1%} diagonal)")
        out[run] = {"tiers_analyzed": sorted(hardest), "hardest_confusion": hardest}
    return out


def analyze_tier_ordering(
    results: Mapping[str, Mapping[str, Any]],
    *,
    runs: Sequence[str] = CORE_RUNS + SUBSET_RUNS,
    topics_per_mode: int = 20,
) -> dict[str, Any]:
    """Whether the historical ordering of the baseline blocks holds, per run.

    The mode count is derived from the sample count and the corpus's topics per
    mode, which is what makes the row readable beside runs of different widths.
    """
    out: dict[str, Any] = {}
    for run in runs:
        entry = results.get(run)
        by_tier = _by_tier(entry)
        if not by_tier:
            continue
        accuracies = {tier: _accuracy(by_tier, tier) for tier in ("T1", "T2", "T2.5", "T3")}
        if any(accuracies[tier] is None for tier in ("T1", "T2", "T2.5")):
            continue
        ordered = accuracies["T2.5"] > accuracies["T2"] > accuracies["T1"]
        n_modes = int(entry.get("n_samples", 0)) // topics_per_mode if entry else 0
        out[run] = {**accuracies, "inversion": bool(ordered), "n_modes": n_modes}
        logger.info(
            f"  {run:<18} T1={accuracies['T1']:.1%}  T2={accuracies['T2']:.1%}  "
            f"T2.5={accuracies['T2.5']:.1%}  ordered={ordered}  ({n_modes} modes)"
        )
    return out


def analyze_value_add(
    results: Mapping[str, Mapping[str, Any]],
    *,
    pairs: Sequence[tuple[str, str, str]] = (
        ("8b_baseline", "8b_v2_5way", "8B"),
        ("3b_run4", "3b_v2_5way", "3B"),
    ),
) -> dict[str, Any]:
    """Engineered families and composites against the baseline composites."""
    out: dict[str, Any] = {}
    for baseline_run, engineered_run, model in pairs:
        baseline, engineered = results.get(baseline_run), results.get(engineered_run)
        if not baseline or not engineered:
            logger.info(f"  {model}: skipped (missing data)")
            continue
        base_tiers, new_tiers = _by_tier(baseline), _by_tier(engineered)
        base_t2t25 = _accuracy(base_tiers, "T2+T2.5")
        base_combined = _accuracy(base_tiers, "combined")
        out[model] = {
            "baseline_t2t25": base_t2t25,
            "baseline_combined": base_combined,
            "families": {
                family: {
                    "accuracy": _accuracy(new_tiers, family),
                    "delta_vs_baseline_t2t25": (
                        None if _accuracy(new_tiers, family) is None or base_t2t25 is None
                        else _accuracy(new_tiers, family) - base_t2t25
                    ),
                }
                for family in NEW_FAMILIES
                if _accuracy(new_tiers, family) is not None
            },
            "composites": {
                composite: {
                    "accuracy": _accuracy(new_tiers, composite),
                    "delta_vs_baseline_combined": (
                        None if _accuracy(new_tiers, composite) is None or base_combined is None
                        else _accuracy(new_tiers, composite) - base_combined
                    ),
                }
                for composite in V2_COMPOSITES
                if _accuracy(new_tiers, composite) is not None
            },
        }
    return out


def complementarity_report(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """All seven readings over the runs that were loaded."""
    return {
        "runs": sorted(results),
        "consistency": analyze_consistency(results),
        "resolution": analyze_resolution(results),
        "complementarity": analyze_complementarity(results),
        "subfamily_importance": analyze_subfamily_importance(results),
        "confusion": analyze_confusion(results),
        "tier_ordering": analyze_tier_ordering(results),
        "value_add": analyze_value_add(results),
    }


__all__ = [
    "BASELINE_TIER_ORDER",
    "COMPLEMENTARY_BAR",
    "CORE_RUNS",
    "DIVERGENCE_BAR",
    "EASY_MODES",
    "HARD_MODES",
    "NEW_FAMILIES",
    "REDUNDANT_BAR",
    "SUBSET_RUNS",
    "V2_COMPOSITES",
    "analyze_complementarity",
    "analyze_confusion",
    "analyze_consistency",
    "analyze_resolution",
    "analyze_subfamily_importance",
    "analyze_tier_ordering",
    "analyze_value_add",
    "complementarity_report",
    "feature_family",
    "feature_subfamily",
    "load_report_inputs",
    "load_results",
    "pair_difficulty",
    "pair_name",
]
