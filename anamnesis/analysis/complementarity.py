"""Complementarity: which families see different things, read across banked runs.

Every other analysis here reads signatures. This one reads *results* — the
gauntlet's own banked JSON, several runs of it at once — and asks the questions
that only exist between runs and between families:

1. **Consistency.** The same block, the same modes, two corpora. A block whose
   accuracy moves by more than five points between them is flagged, because a
   block that is not stable across corpora is not a property of the model.
2. **Resolution.** Pairwise accuracy split by how hard the pair is. The
   format-controlled five are the hard pairs; anything involving a
   format-free mode is easier and averaging the two hides it.
3. **Complementarity.** Two blocks' accuracy *profiles* over the hard pairs,
   correlated. A low correlation means they fail on different pairs, which is what
   makes them worth combining; a high one means one of them is redundant. Easy
   pairs are excluded because their shared ceiling dominates the correlation, and a
   block at ceiling on every hard pair has no profile at all and is dropped by name.
4. **Sub-family importance.** The banked feature importances, grouped by family and
   by sub-family, so importance is read at the resolution the names support.
5. **Confusion.** Which pair each block finds hardest, off its own confusion matrix.
6. **Ordering.** Whether the registered accuracy ordering of the core blocks holds
   in each run: cache-and-keys above attention-and-deltas above norms-and-output-stats.
7. **Value-add.** The engineered families and composites against the baseline
   composites, on the mode subset the two corpora share.

Nothing here computes a classification; everything here is a reading of one. That
is the point of the separation: the gauntlet is expensive and its results are
banked, so the cross-run questions are answered by re-reading rather than by
re-running.

The block and family labels in these tables are the keys banked result files use. They
are addresses into those files, so a label is read as "this column range" and never as a
claim about what the columns measure.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from anamnesis.analysis.gauntlet.schemas import (
    ClassificationResult,
    LegacyBinReadoutResult,
    migrate_banked_results,
)
from anamnesis.analysis.gauntlet.signature_io import (
    ALL_CORE,
    ALL_FAMILIES,
    ATTENTION_AND_CACHE,
    ATTENTION_AND_CACHE_WITH_FAMILIES,
    ATTENTION_AND_DELTAS,
    ATTENTION_FLOW,
    CACHE_AND_KEYS,
    CONTRASTIVE_PROJECTION,
    EVERYTHING,
    GATE_FEATURES,
    NORMS_AND_OUTPUT_STATS,
    RESIDUAL_PCA,
    RESIDUAL_TRAJECTORY,
    TEMPORAL_DYNAMICS,
)

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

CORE_BLOCK_ORDER: tuple[str, ...] = (
    NORMS_AND_OUTPUT_STATS, ATTENTION_AND_DELTAS, CACHE_AND_KEYS, RESIDUAL_PCA,
    ATTENTION_AND_CACHE, ALL_CORE,
)
"""Reading order for the consistency table: the four blocks the numeric anchor
builds, then the two unions over them."""

CONSISTENCY_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("8b_baseline", "8b_v2_5way", "8B: baseline vs engineered (five modes)"),
    ("3b_run4", "3b_v2_5way", "3B: baseline vs engineered (five modes)"),
)

DIVERGENCE_BAR = 0.05
"""Accuracy difference at which two corpora are called divergent for a block."""

COMPLEMENTARY_BAR = 0.30
REDUNDANT_BAR = 0.60
"""Profile-correlation bars: below the first two blocks are complementary, above the
second they are redundant, between them moderate."""

ZERO_VARIANCE = 1e-10

BLOCK_UNION_LABELS: frozenset[str] = frozenset({
    ATTENTION_AND_CACHE, ALL_CORE, ALL_FAMILIES, EVERYTHING,
    ATTENTION_AND_CACHE_WITH_FAMILIES,
})
"""Blocks that are unions of others. They belong in the correlation matrix but not in
the per-pair table, where they would double-count their members."""

NEW_FAMILIES: tuple[str, ...] = (
    RESIDUAL_TRAJECTORY,
    ATTENTION_FLOW,
    GATE_FEATURES,
    TEMPORAL_DYNAMICS,
    CONTRASTIVE_PROJECTION,
)
V2_COMPOSITES: tuple[str, ...] = (
    ALL_FAMILIES, ATTENTION_AND_CACHE_WITH_FAMILIES, EVERYTHING,
)

FAMILY_BY_PREFIX: dict[str, str] = {
    "cp": CONTRASTIVE_PROJECTION,
    "af": ATTENTION_FLOW,
    "td": TEMPORAL_DYNAMICS,
    "rt": RESIDUAL_TRAJECTORY,
    "gf": GATE_FEATURES,
}
"""Engineered families carry a two-letter prefix. The core blocks do not, which is
why the fallback below reads their signal names instead."""

CORE_BLOCK_BY_SIGNAL: tuple[tuple[str, str], ...] = (
    ("lookback", CACHE_AND_KEYS),
    ("key_drift", CACHE_AND_KEYS),
    ("key_novelty", CACHE_AND_KEYS),
    ("epoch", CACHE_AND_KEYS),
    ("attn_entropy", ATTENTION_AND_DELTAS),
    ("head_agree", ATTENTION_AND_DELTAS),
    ("residual", ATTENTION_AND_DELTAS),
    ("pca_", RESIDUAL_PCA),
    ("act_norm", NORMS_AND_OUTPUT_STATS),
    ("logit", NORMS_AND_OUTPUT_STATS),
    ("token", NORMS_AND_OUTPUT_STATS),
    ("delta", NORMS_AND_OUTPUT_STATS),
)
"""Which core block a feature name belongs to, by the signal it names. Order
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
        raw = migrate_banked_results(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"{path}: unreadable ({exc})")
        return None
    for key, model in (
        ("classification", ClassificationResult),
        ("legacy_bin_readout", LegacyBinReadoutResult),
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


def _by_block(results: Mapping[str, Any] | None) -> dict[str, Any]:
    """A run's per-block classification, or empty where the section is untyped."""
    if not results:
        return {}
    section = results.get("classification")
    return section.by_block if isinstance(section, ClassificationResult) else {}


def _accuracy(by_block: Mapping[str, Any], block: str) -> float | None:
    entry = by_block.get(block)
    return entry.rf_5way.accuracy if entry is not None else None


def analyze_consistency(
    results: Mapping[str, Mapping[str, Any]],
    *,
    pairs: Sequence[tuple[str, str, str]] = CONSISTENCY_PAIRS,
    blocks: Sequence[str] = CORE_BLOCK_ORDER,
) -> dict[str, Any]:
    """The same block across two corpora, with the divergent ones flagged."""
    comparisons: list[dict[str, Any]] = []
    for run_a, run_b, label in pairs:
        left, right = results.get(run_a), results.get(run_b)
        if not left or not right:
            logger.info(f"  {label}: skipped (missing {run_a if not left else run_b})")
            continue
        a_blocks, b_blocks = _by_block(left), _by_block(right)
        entry: dict[str, Any] = {"label": label, "blocks": {}}
        for block in blocks:
            acc_a, acc_b = _accuracy(a_blocks, block), _accuracy(b_blocks, block)
            if acc_a is None or acc_b is None:
                continue
            difference = acc_b - acc_a
            entry["blocks"][block] = {
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
    """Pairwise accuracy per block, bucketed by pair difficulty."""
    out: dict[str, Any] = {}
    for run in runs:
        by_block = _by_block(results.get(run))
        if not by_block:
            continue
        per_block: dict[str, dict[str, Any]] = {}
        for block in sorted(t for t, v in by_block.items() if v.pairwise_binary):
            buckets: dict[str, list[float]] = {"hard": [], "cross": [], "easy-easy": []}
            for pair, entry in by_block[block].pairwise_binary.items():
                bucket = pair_difficulty(pair)
                if bucket in buckets:
                    buckets[bucket].append(entry.accuracy)
            per_block[block] = {
                bucket: {
                    "mean": float(np.mean(values)) if values else None,
                    "min": float(np.min(values)) if values else None,
                    "max": float(np.max(values)) if values else None,
                    "n_pairs": len(values),
                }
                for bucket, values in buckets.items()
            }
        out[run] = per_block
    return out


def analyze_complementarity(
    results: Mapping[str, Mapping[str, Any]], *, runs: Sequence[str] = ("8b_v2",)
) -> dict[str, Any]:
    """Correlate blocks' hard-pair accuracy profiles; lower means complementary.

    A block absent from a pair is read at chance (0.5) rather than dropped, so every
    block's profile spans the same pairs and the correlation is over one index.
    """
    out: dict[str, Any] = {}
    for run in runs:
        by_block = _by_block(results.get(run))
        if not by_block:
            continue
        blocks = [t for t, v in by_block.items() if v.pairwise_binary and t != EVERYTHING]
        pairs = sorted({p for t in blocks for p in by_block[t].pairwise_binary})
        profile = np.full((len(blocks), len(pairs)), 0.5, dtype=np.float64)
        for i, block in enumerate(blocks):
            for j, pair in enumerate(pairs):
                entry = by_block[block].pairwise_binary.get(pair)
                if entry is not None:
                    profile[i, j] = entry.accuracy

        hard_columns = [j for j, pair in enumerate(pairs) if pair_difficulty(pair) == "hard"]
        if len(blocks) <= 1 or not hard_columns:
            continue
        hard = profile[:, hard_columns]
        keep = [i for i in range(len(blocks)) if np.std(hard[i]) > ZERO_VARIANCE]
        for i in range(len(blocks)):
            if i not in keep:
                logger.info(f"  ({blocks[i]} skipped — no variance over the hard pairs)")

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
                            "block_a": blocks[keep[a]],
                            "block_b": blocks[keep[b]],
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
            "blocks": blocks,
            "hard_pairs": [pairs[j] for j in hard_columns],
            "individual_blocks": [t for t in blocks if t not in BLOCK_UNION_LABELS],
            "pairs_by_correlation": ranked,
        }
    return out


def feature_family(name: str) -> str:
    """Which family a banked feature name belongs to."""
    prefix = name.split("_")[0] if "_" in name else name
    if prefix in FAMILY_BY_PREFIX:
        return FAMILY_BY_PREFIX[prefix]
    for signal, block in CORE_BLOCK_BY_SIGNAL:
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
        ablation = entry.get("legacy_bin_readout")
        if not isinstance(ablation, LegacyBinReadoutResult) or not ablation.top_features_rf:
            logger.info(f"  {run}: no banked feature importance")
            continue
        row: dict[str, Any] = {
            "family_importance": _sum_importance(ablation.top_features_rf, feature_family),
            "subfam_importance": _sum_importance(ablation.top_features_rf, feature_subfamily),
            "n_features_ranked": len(ablation.top_features_rf),
        }
        if ablation.top_features_rf_attention_and_cache:
            row["subfam_importance_attention_and_cache"] = _sum_importance(
                ablation.top_features_rf_attention_and_cache[:top_n], feature_subfamily
            )
        out[run] = row
    return out


def analyze_confusion(
    results: Mapping[str, Mapping[str, Any]], *, runs: Sequence[str] = ("8b_v2",)
) -> dict[str, Any]:
    """Each block's hardest confusion, off its own banked confusion matrix.

    The labels come from the matrix that carries them, which is where a confusion
    matrix's row and column order is stated. A pair's difficulty is read as the mean
    of the two modes' diagonal rates: a pair both of whose classes are recovered
    poorly is the pair a block cannot separate.
    """
    out: dict[str, Any] = {}
    for run in runs:
        by_block = _by_block(results.get(run))
        if not by_block:
            continue
        hardest: dict[str, dict[str, Any]] = {}
        for block, entry in sorted(by_block.items()):
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
            hardest[block] = {"pair": pair_name(first, second), "mean_diagonal": float(rate)}
            logger.info(f"    {block:<25} hardest: {first}-{second} ({rate:.1%} diagonal)")
        out[run] = {"blocks_analyzed": sorted(hardest), "hardest_confusion": hardest}
    return out


def analyze_block_ordering(
    results: Mapping[str, Mapping[str, Any]],
    *,
    runs: Sequence[str] = CORE_RUNS + SUBSET_RUNS,
    topics_per_mode: int = 20,
) -> dict[str, Any]:
    """Whether the registered accuracy ordering of the core blocks holds, per run.

    The mode count is derived from the sample count and the corpus's topics per
    mode, which is what makes the row readable beside runs of different widths.
    """
    out: dict[str, Any] = {}
    for run in runs:
        entry = results.get(run)
        by_block = _by_block(entry)
        if not by_block:
            continue
        accuracies = {
            block: _accuracy(by_block, block)
            for block in (
                NORMS_AND_OUTPUT_STATS, ATTENTION_AND_DELTAS, CACHE_AND_KEYS, RESIDUAL_PCA,
            )
        }
        ranked = (CACHE_AND_KEYS, ATTENTION_AND_DELTAS, NORMS_AND_OUTPUT_STATS)
        if any(accuracies[block] is None for block in ranked):
            continue
        cache, attention, norms = (accuracies[block] for block in ranked)
        ordered = cache > attention > norms
        n_modes = int(entry.get("n_samples", 0)) // topics_per_mode if entry else 0
        out[run] = {**accuracies, "inversion": bool(ordered), "n_modes": n_modes}
        logger.info(
            f"  {run:<18} norms={norms:.1%}  attention={attention:.1%}  "
            f"cache={cache:.1%}  ordered={ordered}  ({n_modes} modes)"
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
    """Engineered families and composites against the unions of the core blocks."""
    out: dict[str, Any] = {}
    for baseline_run, engineered_run, model in pairs:
        baseline, engineered = results.get(baseline_run), results.get(engineered_run)
        if not baseline or not engineered:
            logger.info(f"  {model}: skipped (missing data)")
            continue
        base_blocks, new_blocks = _by_block(baseline), _by_block(engineered)
        base_attention_and_cache = _accuracy(base_blocks, ATTENTION_AND_CACHE)
        base_combined = _accuracy(base_blocks, ALL_CORE)
        out[model] = {
            "baseline_attention_and_cache": base_attention_and_cache,
            "baseline_combined": base_combined,
            "families": {
                family: {
                    "accuracy": _accuracy(new_blocks, family),
                    "delta_vs_baseline_attention_and_cache": (
                        None if _accuracy(new_blocks, family) is None or base_attention_and_cache is None
                        else _accuracy(new_blocks, family) - base_attention_and_cache
                    ),
                }
                for family in NEW_FAMILIES
                if _accuracy(new_blocks, family) is not None
            },
            "composites": {
                composite: {
                    "accuracy": _accuracy(new_blocks, composite),
                    "delta_vs_baseline_combined": (
                        None if _accuracy(new_blocks, composite) is None or base_combined is None
                        else _accuracy(new_blocks, composite) - base_combined
                    ),
                }
                for composite in V2_COMPOSITES
                if _accuracy(new_blocks, composite) is not None
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
        "block_ordering": analyze_block_ordering(results),
        "value_add": analyze_value_add(results),
    }


__all__ = [
    "CORE_BLOCK_ORDER",
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
    "analyze_block_ordering",
    "analyze_value_add",
    "complementarity_report",
    "feature_family",
    "feature_subfamily",
    "load_report_inputs",
    "load_results",
    "pair_difficulty",
    "pair_name",
]
