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

A union label names a membership, and a banked run may report a union whose members
differ from any union a current run builds. Reading maps such a union onto a legacy
label (:data:`anamnesis.analysis.gauntlet.schemas.compat.LEGACY_UNION_LABELS`), so a
reading that sets two runs side by side meets two labels where the memberships differ.
Where it asks for one and a run carries only the other, it reports the pair as not
comparable and leaves the numbers apart.
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
from anamnesis.analysis.gauntlet.schemas.compat import (
    LEGACY_ATTENTION_AND_CACHE_WITH_FAMILIES,
    LEGACY_EVERYTHING,
    LEGACY_FAMILY_UNION,
    union_counterpart,
)
from anamnesis.analysis.gauntlet.signature_io import (
    ALL_CORE,
    ALL_FAMILIES,
    ATTENTION_AND_CACHE,
    ATTENTION_AND_CACHE_WITH_FAMILIES,
    ATTENTION_AND_DELTAS,
    ATTENTION_FLOW,
    CACHE_AND_KEYS,
    EVERYTHING,
    GATE_FEATURES,
    NORMS_AND_OUTPUT_STATS,
    RESIDUAL_PCA,
    RESIDUAL_TRAJECTORY,
)
from anamnesis.analysis.gauntlet.utils import error_stub_reason, is_error_stub
from anamnesis.analysis.subfamily import classify_signal
from anamnesis.feature_map import (
    FAMILY_ATTENTION_FLOW,
    FAMILY_GATE,
    FAMILY_RESIDUAL_TRAJECTORY,
    STORED_FAMILY_ATTENTION_OTHER,
    STORED_FAMILY_ATTENTION_SPECTRAL,
    STORED_FAMILY_CACHE_AND_KEYS,
    STORED_FAMILY_NORMS_AND_OUTPUT_STATS,
    STORED_FAMILY_RESIDUAL_PCA,
    named_family,
)
from anamnesis.modes import easy_modes, hard_modes

logger = logging.getLogger(__name__)

HARD_MODES: frozenset[str] = hard_modes()
"""The format-controlled five: a pair drawn from these is a hard pair."""

EASY_MODES: frozenset[str] = easy_modes()
"""The three modes the eight-mode set adds to the five. They carry the same format
constraint, so what makes a pair drawn from them easier is that the ways of working
they ask for are computationally more distinctive."""

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
    LEGACY_FAMILY_UNION, LEGACY_ATTENTION_AND_CACHE_WITH_FAMILIES, LEGACY_EVERYTHING,
})
"""Blocks that are unions of others, current and legacy. They belong in the correlation
matrix but not in the per-pair table, where they would double-count their members."""

WHOLE_VECTOR_UNIONS: frozenset[str] = frozenset({EVERYTHING, LEGACY_EVERYTHING})
"""The union of every block a run holds, under either membership. It is left out of the
correlation matrix, where it would correlate with each of its members by construction."""

NEW_FAMILIES: tuple[str, ...] = (
    RESIDUAL_TRAJECTORY,
    ATTENTION_FLOW,
    GATE_FEATURES,
)
VALUE_ADD_COMPOSITES: tuple[str, ...] = (
    ALL_FAMILIES, ATTENTION_AND_CACHE_WITH_FAMILIES, EVERYTHING,
    LEGACY_FAMILY_UNION, LEGACY_ATTENTION_AND_CACHE_WITH_FAMILIES, LEGACY_EVERYTHING,
)
"""The composites value-add reports, each under the label its run carries: a banked run
reports the legacy unions and a current run the current ones, so a table across both
holds them in separate rows."""

BLOCK_BY_FAMILY: dict[str, str] = {
    STORED_FAMILY_NORMS_AND_OUTPUT_STATS: NORMS_AND_OUTPUT_STATS,
    STORED_FAMILY_ATTENTION_OTHER: ATTENTION_AND_DELTAS,
    STORED_FAMILY_ATTENTION_SPECTRAL: ATTENTION_AND_DELTAS,
    STORED_FAMILY_CACHE_AND_KEYS: CACHE_AND_KEYS,
    STORED_FAMILY_RESIDUAL_PCA: RESIDUAL_PCA,
    FAMILY_RESIDUAL_TRAJECTORY: RESIDUAL_TRAJECTORY,
    FAMILY_ATTENTION_FLOW: ATTENTION_FLOW,
    FAMILY_GATE: GATE_FEATURES,
}
"""The block a family's features are addressed in, for the families that have one.

:func:`anamnesis.feature_map.named_family` is the one classifier of feature names, and
it answers in families. A family is the finer cut — attention-and-deltas is two of them,
one reading attention distributions and one reading their graph spectrum — so a family
maps onto a block and not the other way round. A family in
:data:`anamnesis.feature_map.FAMILY_LABELS` and absent here is reported under its family
label. For the two whose columns only a banked corpus carries, that label and the block
label a banked result addresses them under are one string, so the fallback is the
translation."""


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

    An error stub is not such a failure. A section that could not run banks its
    reason in place of its numbers, and that is a value the section is allowed to
    have, so it is reported at info and left as the mapping it is — a warning there
    would say a file is malformed when it is only incomplete, and the readings
    below already skip an untyped section.
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
        if not isinstance(raw.get(key), dict):
            continue
        if is_error_stub(raw[key]):
            logger.info(f"{path}: {key} did not run ({error_stub_reason(raw[key])})")
            continue
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
    """The same block across two corpora, with the divergent ones flagged.

    A union asked for by a label one run carries while the other run carries only its
    counterpart (:func:`anamnesis.analysis.gauntlet.schemas.compat.union_counterpart`)
    is listed under ``not_comparable`` with the label each side holds, and no
    difference is taken: the two numbers are accuracies over different blocks.
    """
    comparisons: list[dict[str, Any]] = []
    for run_a, run_b, label in pairs:
        left, right = results.get(run_a), results.get(run_b)
        if not left or not right:
            logger.info(f"  {label}: skipped (missing {run_a if not left else run_b})")
            continue
        a_blocks, b_blocks = _by_block(left), _by_block(right)
        entry: dict[str, Any] = {"label": label, "blocks": {}, "not_comparable": {}}
        for block in blocks:
            acc_a, acc_b = _accuracy(a_blocks, block), _accuracy(b_blocks, block)
            if acc_a is None or acc_b is None:
                mismatch = _membership_mismatch(a_blocks, b_blocks, block)
                if mismatch is not None:
                    entry["not_comparable"][block] = mismatch
                    logger.warning(
                        f"  {label}: {block} not compared — {run_a} carries "
                        f"{mismatch['run_a_label']}, {run_b} carries "
                        f"{mismatch['run_b_label']}, and the two unions hold "
                        "different blocks"
                    )
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


def _membership_mismatch(
    a_blocks: Mapping[str, Any], b_blocks: Mapping[str, Any], block: str
) -> dict[str, str] | None:
    """The labels two runs hold for ``block`` when one has it and the other its counterpart.

    None when there is no counterpart, when both runs hold ``block``, or when a side holds
    neither — an absent union is absence, not a mismatch.
    """
    counterpart = union_counterpart(block)
    if counterpart is None:
        return None
    a_label = block if block in a_blocks else counterpart if counterpart in a_blocks else None
    b_label = block if block in b_blocks else counterpart if counterpart in b_blocks else None
    if a_label is None or b_label is None or a_label == b_label:
        return None
    return {"run_a_label": a_label, "run_b_label": b_label}


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
        blocks = [
            t for t, v in by_block.items()
            if v.pairwise_binary and t not in WHOLE_VECTOR_UNIONS
        ]
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
    """Which block a banked feature name's importance is credited to.

    The classification is :func:`anamnesis.feature_map.named_family`'s, translated
    through :data:`BLOCK_BY_FAMILY` so that a summed importance is keyed the way the
    accuracy tables beside it are. A name that classifier cannot place is reported as
    unknown rather than credited to a block, because an importance attributed to a block
    that never held the feature reads as a finding about that block.
    """
    family = named_family(name)
    if family is None:
        return f"unknown({name[:20]})"
    return BLOCK_BY_FAMILY.get(family, family)


def feature_subfamily(name: str) -> str:
    """Which sub-family a banked feature name belongs to, at importance resolution.

    The classification is :func:`anamnesis.analysis.subfamily.classify_signal`'s. This
    module classifies no feature name itself, on this question or the family one: a
    second reading of the same naming conventions is a second vocabulary, and the two
    tables are printed beside each other.
    """
    return classify_signal(name)


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
    """Engineered families and composites against the unions of the core blocks.

    Each delta is taken within one pair of runs, against the baseline's core unions,
    whose membership is fixed. A composite is reported under the label its run carries,
    so a banked run's rows are the legacy unions and cannot be read as the current ones.
    """
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
                for composite in VALUE_ADD_COMPOSITES
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
    "BLOCK_BY_FAMILY",
    "CORE_BLOCK_ORDER",
    "COMPLEMENTARY_BAR",
    "CORE_RUNS",
    "DIVERGENCE_BAR",
    "EASY_MODES",
    "HARD_MODES",
    "NEW_FAMILIES",
    "REDUNDANT_BAR",
    "SUBSET_RUNS",
    "VALUE_ADD_COMPOSITES",
    "WHOLE_VECTOR_UNIONS",
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
