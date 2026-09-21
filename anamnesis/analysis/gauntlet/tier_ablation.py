"""Section 3: Tier ablation and feature importance."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .signature_io import AnalysisData, BASELINE_TIERS, ENGINEERED_TIERS
from .schemas import (
    CohensDPerTopicResult,
    CrossGroupAblation,
    FeatureImportanceEntry,
    LeaveOneOutEntry,
    PairwiseTierCombo,
    PerTopicEffectSize,
    StdVsMeanResult,
    TierAblationResult,
    TierRankingEntry,
    TripleTierCombo,
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


def run_tier_ablation(data: AnalysisData) -> TierAblationResult:
    """Run tier ablation and feature importance analyses."""
    y = data.modes

    present_baseline = [t for t in BASELINE_TIERS if t in data.run4.tier_features]
    present_engineered = [t for t in ENGINEERED_TIERS if t in data.run4.tier_features]
    all_individual = present_baseline + present_engineered

    print(f"  Baseline tiers: {present_baseline}")
    if present_engineered:
        print(f"  Engineered tiers: {present_engineered}")

    # ── Per-tier accuracy (each tier alone) ──
    per_tier_accuracy: dict[str, float] = {}
    for tier in all_individual:
        X = data.get_tier(tier)
        per_tier_accuracy[tier] = _rf_accuracy(X, y)
        print(f"    {tier}: {per_tier_accuracy[tier]:.3f} ({X.shape[1]} features)")

    for group_name in data.run4.group_features:
        X = data.get_tier(group_name)
        per_tier_accuracy[group_name] = _rf_accuracy(X, y)
        print(f"    {group_name}: {per_tier_accuracy[group_name]:.3f} ({X.shape[1]} features)")

    # ── Pairwise within baseline ──
    print("  Pairwise baseline tier combinations...")
    pairwise_tiers: dict[str, PairwiseTierCombo] = {}
    for t1, t2 in combinations(present_baseline, 2):
        key = f"{t1}+{t2}"
        X_pair = np.concatenate([data.get_tier(t1), data.get_tier(t2)], axis=1)
        acc = _rf_accuracy(X_pair, y)
        expected = max(per_tier_accuracy[t1], per_tier_accuracy[t2])
        pairwise_tiers[key] = PairwiseTierCombo(
            accuracy=acc,
            n_features=int(X_pair.shape[1]),
            individual_max=expected,
            gain_over_best_individual=acc - expected,
        )

    # ── Triple within baseline ──
    triple_tiers: dict[str, TripleTierCombo] | None = None
    if len(present_baseline) >= 3:
        print("  Triple baseline tier combinations...")
        triple_tiers = {}
        for combo in combinations(present_baseline, 3):
            key = "+".join(combo)
            X_triple = np.concatenate([data.get_tier(t) for t in combo], axis=1)
            acc = _rf_accuracy(X_triple, y)
            best_pair_acc = max(
                (pairwise_tiers[f"{a}+{b}"].accuracy for a, b in combinations(combo, 2)),
                default=0.0,
            )
            triple_tiers[key] = TripleTierCombo(
                accuracy=acc,
                n_features=int(X_triple.shape[1]),
                best_pairwise_subset=best_pair_acc,
                gain_over_best_pair=acc - best_pair_acc,
            )

    # ── Cross-group: baseline composite + each engineered tier ──
    cross_group: dict[str, CrossGroupAblation] | None = None
    cross_group_baseline: str | None = None
    if present_engineered and len(present_baseline) >= 2:
        print("  Cross-group ablation (baseline + each engineered)...")
        baseline_key = "T2+T2.5" if "T2+T2.5" in data.run4.group_features else None
        if baseline_key is None and len(present_baseline) >= 2:
            baseline_key = "+".join(present_baseline)
        if baseline_key and baseline_key in per_tier_accuracy:
            cross_group = {}
            cross_group_baseline = baseline_key
            baseline_acc = per_tier_accuracy[baseline_key]
            X_base = data.get_tier(baseline_key)
            for eng_tier in present_engineered:
                X_eng = data.get_tier(eng_tier)
                X_combined = np.concatenate([X_base, X_eng], axis=1)
                acc = _rf_accuracy(X_combined, y)
                cross_group[f"{baseline_key}+{eng_tier}"] = CrossGroupAblation(
                    accuracy=acc,
                    n_features=int(X_combined.shape[1]),
                    baseline_accuracy=baseline_acc,
                    engineered_alone=per_tier_accuracy[eng_tier],
                    gain_over_baseline=acc - baseline_acc,
                )

    # ── Leave-one-tier-out (from all individual tiers) ──
    leave_one_out: dict[str, LeaveOneOutEntry] = {}
    leave_one_out_baseline: float | None = None
    if len(all_individual) >= 2:
        all_concat = np.concatenate([data.get_tier(t) for t in all_individual], axis=1)
        all_acc = _rf_accuracy(all_concat, y)
        leave_one_out_baseline = all_acc
        for tier in all_individual:
            remaining = [t for t in all_individual if t != tier]
            X_without = np.concatenate([data.get_tier(t) for t in remaining], axis=1)
            acc_without = _rf_accuracy(X_without, y)
            leave_one_out[tier] = LeaveOneOutEntry(
                accuracy_without=acc_without,
                cost_of_removal=all_acc - acc_without,
            )

    # ── Tier ranking ──
    ranking_list = sorted(
        [(t, per_tier_accuracy[t]) for t in all_individual],
        key=lambda x: x[1],
        reverse=True,
    )
    tier_ranking = [TierRankingEntry(tier=t, accuracy=a) for t, a in ranking_list]
    tier_inversion = False
    if all(t in per_tier_accuracy for t in ["T2.5", "T2", "T1"]):
        tier_inversion = (
            per_tier_accuracy["T2.5"] > per_tier_accuracy["T2"]
            > per_tier_accuracy["T1"]
        )

    # ── Feature importance on best available composite ──
    top_features_rf: list[FeatureImportanceEntry] | None = None
    top_features_lr: list[FeatureImportanceEntry] | None = None
    feature_importance_composite: str | None = None
    best_composite = None
    for candidate in ["combined_v2", "combined", "T2+T2.5+engineered", "T2+T2.5"]:
        if candidate in data.run4.group_features:
            best_composite = candidate
            break

    if best_composite:
        print(f"  Feature importance ({best_composite})...")
        X_key = data.get_tier(best_composite)
        key_names: list[str] = []
        from .signature_io import TIER_GROUPS
        composite_members = []
        if best_composite in TIER_GROUPS:
            composite_members = [t for t in TIER_GROUPS[best_composite]
                                 if t in data.run4.tier_features]
        for tier in composite_members:
            tier_names = data.run4.tier_feature_names.get(tier, np.array([]))
            key_names.extend(list(tier_names))
        if len(key_names) != X_key.shape[1]:
            key_names = [f"feat_{i}" for i in range(X_key.shape[1])]

        top_features_rf = _get_feature_importance(X_key, y, key_names)
        top_features_lr = _get_lr_importance(X_key, y, key_names)
        feature_importance_composite = best_composite

    # Also do T2+T2.5 importance for backward compatibility
    top_features_rf_t2t25: list[FeatureImportanceEntry] = []
    top_features_lr_t2t25: list[FeatureImportanceEntry] = []
    if "T2+T2.5" in data.run4.group_features:
        print("  Feature importance (T2+T2.5)...")
        X_t2t25 = data.get_tier("T2+T2.5")
        t2t25_names = list(data.run4.tier_feature_names.get("T2", [])) + \
                      list(data.run4.tier_feature_names.get("T2.5", []))
        if len(t2t25_names) != X_t2t25.shape[1]:
            t2t25_names = [f"feat_{i}" for i in range(X_t2t25.shape[1])]
        top_features_rf_t2t25 = _get_feature_importance(X_t2t25, y, t2t25_names)
        top_features_lr_t2t25 = _get_lr_importance(X_t2t25, y, t2t25_names)

    # ── Tier contribution ratio ──
    tier_contribution: dict[str, float] = {}
    if len(all_individual) >= 2:
        X_all = np.concatenate([data.get_tier(t) for t in all_individual], axis=1)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_all)
        clf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=1)
        clf.fit(X_s, y)
        importances = clf.feature_importances_

        offset = 0
        for tier in all_individual:
            dim = data.get_tier(tier).shape[1]
            tier_contribution[tier] = float(np.sum(importances[offset:offset + dim]))
            offset += dim
        total = sum(tier_contribution.values())
        if total > 0:
            tier_contribution = {k: v / total for k, v in tier_contribution.items()}

    # std vs mean features
    print("  std vs mean feature split...")
    std_vs_mean = _std_vs_mean_split(data, y)

    # Cohen's d per topic
    print("  Cohen's d per topic...")
    cohens_d = _topic_controlled_effect_sizes(data, y)

    return TierAblationResult(
        per_tier_accuracy=per_tier_accuracy,
        pairwise_tier_combinations=pairwise_tiers,
        triple_tier_combinations=triple_tiers,
        cross_group_ablation=cross_group,
        cross_group_baseline=cross_group_baseline,
        leave_one_tier_out=leave_one_out,
        leave_one_out_baseline_accuracy=leave_one_out_baseline,
        tier_ranking=tier_ranking,
        tier_inversion_t25_gt_t2_gt_t1=tier_inversion,
        top_features_rf=top_features_rf,
        top_features_lr=top_features_lr,
        feature_importance_composite=feature_importance_composite,
        top_features_rf_t2t25=top_features_rf_t2t25,
        top_features_lr_t2t25=top_features_lr_t2t25,
        top_features_rf_combined=None,
        tier_contribution_ratio=tier_contribution,
        std_vs_mean=std_vs_mean,
        cohens_d_per_topic=cohens_d,
    )


def _std_vs_mean_split(data: AnalysisData, y: NDArray) -> StdVsMeanResult:
    """Compare RF accuracy on *_std features vs *_mean features."""
    X = data.get_tier("T2+T2.5")
    names = list(data.run4.tier_feature_names.get("T2", [])) + \
            list(data.run4.tier_feature_names.get("T2.5", []))

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
    """Cohen's d per topic: within-mode vs between-mode distances on T2+T2.5."""
    from scipy.spatial.distance import pdist

    X = data.get_tier("T2+T2.5")
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
