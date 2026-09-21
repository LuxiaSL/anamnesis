"""Section 3: a readout over the stored feature blocks, and feature importance.

This section measures each stored block on its own, the blocks in pairs and triples,
and the cost of leaving each one out. Those blocks — the bins banked results are
expressed in — are the whole reason the section exists: a number banked per bin can
only be compared per bin, so the readout keeps computing exactly what it computed.

It is a compatibility readout, not the decomposition of record. Three of the four
blocks the numeric anchor builds span more than one substrate, so a block's accuracy
localizes nothing. The decomposition of record cuts by family and sub-family —
`anamnesis/scripts/run_subfamily_decomp.py`, over `anamnesis/analysis/subfamily.py` —
and the taxonomy it reads is `anamnesis/feature_map.py`.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .signature_io import (
    ATTENTION_AND_CACHE,
    ATTENTION_AND_CACHE_WITH_FAMILIES,
    ATTENTION_AND_DELTAS,
    CACHE_AND_KEYS,
    CORE_BLOCKS,
    EVERYTHING,
    FAMILY_BLOCKS,
    NORMS_AND_OUTPUT_STATS,
    ALL_CORE,
    AnalysisData,
)
from .schemas import (
    CohensDPerTopicResult,
    CrossGroupAblation,
    FeatureImportanceEntry,
    LeaveOneOutEntry,
    PairwiseBlockCombo,
    PerTopicEffectSize,
    StdVsMeanResult,
    LegacyBinReadoutResult,
    BlockRankingEntry,
    TripleBlockCombo,
)

_RF_KWARGS = dict(n_estimators=100, n_jobs=1)


def _rf_accuracy(X: NDArray, y: NDArray, seed: int = 42) -> float:
    """Quick RF 5-fold CV accuracy."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs: list[float] = []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = RandomForestClassifier(**_RF_KWARGS, random_state=seed)
        clf.fit(X_tr, y[train_idx])
        accs.append(float(accuracy_score(y[test_idx], clf.predict(X_te))))
    return float(np.mean(accs))


def _get_feature_importance(
    X: NDArray, y: NDArray, feature_names: list[str], seed: int = 42,
) -> list[FeatureImportanceEntry]:
    """Train RF on full data, return sorted feature importances."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    clf.fit(X_s, y)
    importances = clf.feature_importances_

    ranked = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )
    return [FeatureImportanceEntry(name=n, importance=float(v)) for n, v in ranked[:30]]


def _get_lr_importance(
    X: NDArray, y: NDArray, feature_names: list[str], seed: int = 42,
) -> list[FeatureImportanceEntry]:
    """LogReg coefficient magnitudes (mean across classes)."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    # Multinomial (softmax over all classes at once, not one-vs-rest) is what
    # this coefficient ranking is read from, and with the default lbfgs solver it
    # is the only multiclass fit scikit-learn performs — so it is the behaviour
    # here, stated in prose because scikit-learn does not take it as an argument.
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_s, y)

    mean_abs_coef = np.mean(np.abs(clf.coef_), axis=0)
    ranked = sorted(
        zip(feature_names, mean_abs_coef),
        key=lambda x: x[1],
        reverse=True,
    )
    return [FeatureImportanceEntry(name=n, importance=float(v)) for n, v in ranked[:30]]


def run_legacy_bin_readout(data: AnalysisData) -> LegacyBinReadoutResult:
    """Measure every stored block, its unions, and the cost of dropping each one.

    Returns the accuracies, the pairwise and triple combinations, the leave-one-out
    costs, the ranking, and the feature importances on the widest union present.
    """
    y = data.modes

    present_core = [t for t in CORE_BLOCKS if t in data.run4.block_features]
    present_families = [t for t in FAMILY_BLOCKS if t in data.run4.block_features]
    all_individual = present_core + present_families

    print(f"  Core blocks: {present_core}")
    if present_families:
        print(f"  Family blocks: {present_families}")

    # ── Per-block accuracy (each block alone) ──
    per_block_accuracy: dict[str, float] = {}
    for block in all_individual:
        X = data.get_block(block)
        per_block_accuracy[block] = _rf_accuracy(X, y)
        print(f"    {block}: {per_block_accuracy[block]:.3f} ({X.shape[1]} features)")

    for group_name in data.run4.group_features:
        X = data.get_block(group_name)
        per_block_accuracy[group_name] = _rf_accuracy(X, y)
        print(f"    {group_name}: {per_block_accuracy[group_name]:.3f} ({X.shape[1]} features)")

    # ── Pairwise within the core blocks ──
    print("  Pairwise combinations within the core blocks...")
    pairwise_blocks: dict[str, PairwiseBlockCombo] = {}
    for left, right in combinations(present_core, 2):
        key = f"{left}+{right}"
        X_pair = np.concatenate([data.get_block(left), data.get_block(right)], axis=1)
        acc = _rf_accuracy(X_pair, y)
        expected = max(per_block_accuracy[left], per_block_accuracy[right])
        pairwise_blocks[key] = PairwiseBlockCombo(
            accuracy=acc,
            n_features=int(X_pair.shape[1]),
            individual_max=expected,
            gain_over_best_individual=acc - expected,
        )

    # ── Triple within the core blocks ──
    triple_blocks: dict[str, TripleBlockCombo] | None = None
    if len(present_core) >= 3:
        print("  Triple combinations within the core blocks...")
        triple_blocks = {}
        for combo in combinations(present_core, 3):
            key = "+".join(combo)
            X_triple = np.concatenate([data.get_block(t) for t in combo], axis=1)
            acc = _rf_accuracy(X_triple, y)
            best_pair_acc = max(
                (pairwise_blocks[f"{a}+{b}"].accuracy for a, b in combinations(combo, 2)),
                default=0.0,
            )
            triple_blocks[key] = TripleBlockCombo(
                accuracy=acc,
                n_features=int(X_triple.shape[1]),
                best_pairwise_subset=best_pair_acc,
                gain_over_best_pair=acc - best_pair_acc,
            )

    # ── Cross-group: the baseline composite + each engineered family ──
    cross_group: dict[str, CrossGroupAblation] | None = None
    cross_group_baseline: str | None = None
    if present_families and len(present_core) >= 2:
        print("  Cross-group ablation (baseline + each engineered)...")
        baseline_key = ATTENTION_AND_CACHE if ATTENTION_AND_CACHE in data.run4.group_features else None
        if baseline_key is None and len(present_core) >= 2:
            baseline_key = "+".join(present_core)
        if baseline_key and baseline_key in per_block_accuracy:
            cross_group = {}
            cross_group_baseline = baseline_key
            baseline_acc = per_block_accuracy[baseline_key]
            X_base = data.get_block(baseline_key)
            for eng_block in present_families:
                X_eng = data.get_block(eng_block)
                X_combined = np.concatenate([X_base, X_eng], axis=1)
                acc = _rf_accuracy(X_combined, y)
                cross_group[f"{baseline_key}+{eng_block}"] = CrossGroupAblation(
                    accuracy=acc,
                    n_features=int(X_combined.shape[1]),
                    baseline_accuracy=baseline_acc,
                    engineered_alone=per_block_accuracy[eng_block],
                    gain_over_baseline=acc - baseline_acc,
                )

    # ── Leave-one-block-out (from all individual blocks) ──
    leave_one_out: dict[str, LeaveOneOutEntry] = {}
    leave_one_out_baseline: float | None = None
    if len(all_individual) >= 2:
        all_concat = np.concatenate([data.get_block(t) for t in all_individual], axis=1)
        all_acc = _rf_accuracy(all_concat, y)
        leave_one_out_baseline = all_acc
        for block in all_individual:
            remaining = [t for t in all_individual if t != block]
            X_without = np.concatenate([data.get_block(t) for t in remaining], axis=1)
            acc_without = _rf_accuracy(X_without, y)
            leave_one_out[block] = LeaveOneOutEntry(
                accuracy_without=acc_without,
                cost_of_removal=all_acc - acc_without,
            )

    # ── Block ranking ──
    ranking_list = sorted(
        [(t, per_block_accuracy[t]) for t in all_individual],
        key=lambda x: x[1],
        reverse=True,
    )
    block_ranking = [BlockRankingEntry(block=t, accuracy=a) for t, a in ranking_list]
    block_inversion = False
    if all(t in per_block_accuracy for t in [CACHE_AND_KEYS, ATTENTION_AND_DELTAS, NORMS_AND_OUTPUT_STATS]):
        block_inversion = (
            per_block_accuracy[CACHE_AND_KEYS] > per_block_accuracy[ATTENTION_AND_DELTAS]
            > per_block_accuracy[NORMS_AND_OUTPUT_STATS]
        )

    # ── Feature importance on best available composite ──
    top_features_rf: list[FeatureImportanceEntry] | None = None
    top_features_lr: list[FeatureImportanceEntry] | None = None
    feature_importance_composite: str | None = None
    best_composite = None
    for candidate in [EVERYTHING, ALL_CORE, ATTENTION_AND_CACHE_WITH_FAMILIES, ATTENTION_AND_CACHE]:
        if candidate in data.run4.group_features:
            best_composite = candidate
            break

    if best_composite:
        print(f"  Feature importance ({best_composite})...")
        X_key = data.get_block(best_composite)
        key_names: list[str] = []
        from .signature_io import BLOCK_UNIONS
        composite_members = []
        if best_composite in BLOCK_UNIONS:
            composite_members = [t for t in BLOCK_UNIONS[best_composite]
                                 if t in data.run4.block_features]
        for block in composite_members:
            block_names = data.run4.block_feature_names.get(block, np.array([]))
            key_names.extend(list(block_names))
        if len(key_names) != X_key.shape[1]:
            key_names = [f"feat_{i}" for i in range(X_key.shape[1])]

        top_features_rf = _get_feature_importance(X_key, y, key_names)
        top_features_lr = _get_lr_importance(X_key, y, key_names)
        feature_importance_composite = best_composite

    # The attention-and-cache union is reported separately: banked results carry it
    top_features_rf_attention_and_cache: list[FeatureImportanceEntry] = []
    top_features_lr_attention_and_cache: list[FeatureImportanceEntry] = []
    if ATTENTION_AND_CACHE in data.run4.group_features:
        print("  Feature importance (attention and cache)...")
        X_attention_and_cache = data.get_block(ATTENTION_AND_CACHE)
        attention_and_cache_names = list(data.run4.block_feature_names.get(ATTENTION_AND_DELTAS, [])) + \
                      list(data.run4.block_feature_names.get(CACHE_AND_KEYS, []))
        if len(attention_and_cache_names) != X_attention_and_cache.shape[1]:
            attention_and_cache_names = [f"feat_{i}" for i in range(X_attention_and_cache.shape[1])]
        top_features_rf_attention_and_cache = _get_feature_importance(X_attention_and_cache, y, attention_and_cache_names)
        top_features_lr_attention_and_cache = _get_lr_importance(X_attention_and_cache, y, attention_and_cache_names)

    # ── Block contribution ratio ──
    block_contribution: dict[str, float] = {}
    if len(all_individual) >= 2:
        X_all = np.concatenate([data.get_block(t) for t in all_individual], axis=1)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_all)
        clf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=1)
        clf.fit(X_s, y)
        importances = clf.feature_importances_

        offset = 0
        for block in all_individual:
            dim = data.get_block(block).shape[1]
            block_contribution[block] = float(np.sum(importances[offset:offset + dim]))
            offset += dim
        total = sum(block_contribution.values())
        if total > 0:
            block_contribution = {k: v / total for k, v in block_contribution.items()}

    # std vs mean features
    print("  std vs mean feature split...")
    std_vs_mean = _std_vs_mean_split(data, y)

    # Cohen's d per topic
    print("  Cohen's d per topic...")
    cohens_d = _topic_controlled_effect_sizes(data, y)

    return LegacyBinReadoutResult(
        per_block_accuracy=per_block_accuracy,
        pairwise_block_combinations=pairwise_blocks,
        triple_block_combinations=triple_blocks,
        cross_group_ablation=cross_group,
        cross_group_baseline=cross_group_baseline,
        leave_one_block_out=leave_one_out,
        leave_one_out_baseline_accuracy=leave_one_out_baseline,
        block_ranking=block_ranking,
        cache_beats_attention_beats_norms=block_inversion,
        top_features_rf=top_features_rf,
        top_features_lr=top_features_lr,
        feature_importance_composite=feature_importance_composite,
        top_features_rf_attention_and_cache=top_features_rf_attention_and_cache,
        top_features_lr_attention_and_cache=top_features_lr_attention_and_cache,
        top_features_rf_combined=None,
        block_contribution_ratio=block_contribution,
        std_vs_mean=std_vs_mean,
        cohens_d_per_topic=cohens_d,
    )


def _std_vs_mean_split(data: AnalysisData, y: NDArray) -> StdVsMeanResult:
    """Compare RF accuracy on *_std features vs *_mean features."""
    X = data.get_block(ATTENTION_AND_CACHE)
    names = list(data.run4.block_feature_names.get(ATTENTION_AND_DELTAS, [])) + \
            list(data.run4.block_feature_names.get(CACHE_AND_KEYS, []))

    if len(names) != X.shape[1]:
        return StdVsMeanResult(
            error="feature name mismatch",
            n_names=len(names),
            n_features=int(X.shape[1]),
        )

    std_mask = np.array(["_std" in n for n in names])
    mean_mask = np.array(["_mean" in n for n in names])

    n_std_features = int(np.sum(std_mask))
    n_mean_features = int(np.sum(mean_mask))

    std_accuracy: float | None = None
    mean_accuracy: float | None = None
    if n_std_features > 0:
        std_accuracy = _rf_accuracy(X[:, std_mask], y)
    if n_mean_features > 0:
        mean_accuracy = _rf_accuracy(X[:, mean_mask], y)
    std_beats_mean: bool | None = None
    if std_accuracy is not None and mean_accuracy is not None:
        std_beats_mean = std_accuracy > mean_accuracy

    return StdVsMeanResult(
        n_std_features=n_std_features,
        n_mean_features=n_mean_features,
        std_accuracy=std_accuracy,
        mean_accuracy=mean_accuracy,
        std_beats_mean=std_beats_mean,
    )


def _topic_controlled_effect_sizes(data: AnalysisData, y: NDArray) -> CohensDPerTopicResult:
    """Cohen's d per topic: within-mode vs between-mode distances on attention and cache."""
    from scipy.spatial.distance import pdist

    X = data.get_block(ATTENTION_AND_CACHE)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    topics = data.topics

    unique_topics = sorted(set(topics))
    per_topic: dict[str, PerTopicEffectSize] = {}

    for topic in unique_topics:
        topic_mask = topics == topic
        X_topic = X_std[topic_mask]
        y_topic = y[topic_mask]

        if len(X_topic) < 3:
            per_topic[topic] = PerTopicEffectSize(
                error="too few samples", n=int(len(X_topic)),
            )
            continue

        within_dists: list[float] = []
        between_dists: list[float] = []

        modes_in_topic = sorted(set(y_topic))
        for mode in modes_in_topic:
            mode_mask = y_topic == mode
            if np.sum(mode_mask) >= 2:
                dists = pdist(X_topic[mode_mask])
                within_dists.extend(dists.tolist())

        for i in range(len(X_topic)):
            for j in range(i + 1, len(X_topic)):
                if y_topic[i] != y_topic[j]:
                    d = float(np.linalg.norm(X_topic[i] - X_topic[j]))
                    between_dists.append(d)

        if not within_dists or not between_dists:
            per_topic[topic] = PerTopicEffectSize(
                error="insufficient pairs", n=int(len(X_topic)),
            )
            continue

        within_arr = np.array(within_dists)
        between_arr = np.array(between_dists)

        mean_w = float(np.mean(within_arr))
        mean_b = float(np.mean(between_arr))
        pooled_std = float(np.sqrt(
            (np.var(within_arr) * len(within_arr) + np.var(between_arr) * len(between_arr))
            / (len(within_arr) + len(between_arr))
        ))

        d = (mean_b - mean_w) / max(pooled_std, 1e-10)

        per_topic[topic] = PerTopicEffectSize(
            cohens_d=float(d),
            mean_within=mean_w,
            mean_between=mean_b,
            n_within_pairs=len(within_dists),
            n_between_pairs=len(between_dists),
            n_samples=int(np.sum(topic_mask)),
        )

    d_values = [v.cohens_d for v in per_topic.values() if v.cohens_d is not None]

    return CohensDPerTopicResult(
        per_topic=per_topic,
        mean_d=float(np.mean(d_values)) if d_values else None,
        median_d=float(np.median(d_values)) if d_values else None,
        std_d=float(np.std(d_values)) if d_values else None,
        min_d=float(np.min(d_values)) if d_values else None,
        max_d=float(np.max(d_values)) if d_values else None,
        all_positive=all(d > 0 for d in d_values) if d_values else None,
        n_topics=len(d_values),
    )
