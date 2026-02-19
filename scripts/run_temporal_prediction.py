#!/usr/bin/env python3
"""Experiment 3: Temporal prediction — do early compute features predict late dynamics?

Tests whether computational state features from the first half of a generation
predict computational dynamics in the second half, and whether this prediction
is better than what semantic content alone provides.

Design:
  Uses the 5-point trajectory features already embedded in the feature vectors
  (sampled at positions [0, T//4, T//2, 3T//4, T-1]). Splits into:
    - "early" = traj{0,1} (positions 0 and T//4)
    - "late"  = traj{3,4} (positions 3T//4 and T-1)

  Tasks:
    Diagnostic: Can trajectory features classify mode at all? (prerequisite)
    Task 1: compute→compute — Ridge(early_compute → late_compute), by tier
    Task 2: semantic→compute — Ridge(early_semantic → late_compute), by tier
    Task 3: semantic→semantic — Ridge(early_semantic → late_semantic), calibration
    Task 4: Cross-generation control — mode-centroid early predicts late?
    Task 5: Within-mode residual — after subtracting mode centroids
    Task 6: Cross-half mode consistency — classify each half independently

  All regressions use GroupKFold by topic (5 folds, 4 topics per fold).
  Permutation test on Task 1 vs Task 2 delta (200 permutations).

Key constraint: T2+T2.5 summary stats (the strongest mode features) are NOT
temporally decomposable. Trajectory features come mainly from T1 (weak for mode)
and T3 (captures format, not mode). T2.5 has 63 trajectory features (cache_recency
+ key_novelty) — these are the mechanistically interesting subset.

Pre-registered predictions:
  - Task 1 median R²: 0.05-0.20 overall; T2.5 key novelty should show strongest
    self-prediction; T3 may show high R² for format-consistency reasons
  - Task 2 median R²: -0.05 to +0.05 (near zero)
  - Delta (Task 1 - Task 2): 0.05-0.15, positive and significant
  - Diagnostic: 35-50% 5-way from trajectory features (above 20% chance)
  - Cross-generation ≈ within-generation → mode fingerprinting
  - Cross-generation << within-generation → genuine temporal coherence

Usage:
    python scripts/run_temporal_prediction.py
    python scripts/run_temporal_prediction.py --n-permutations 200
    python scripts/run_temporal_prediction.py --skip-permutation

Expected runtime: ~10-30 minutes on CPU (depends on permutation count).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, silhouette_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase05" / "temporal_prediction"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature index extraction — parse trajectory structure from feature names
# ---------------------------------------------------------------------------

# Tier boundaries in the 1837-dim feature vector
TIER_SLICES = {
    "T1": (0, 221),
    "T2": (221, 442),
    "T2.5": (442, 587),
    "T3": (587, 1837),
}


def build_trajectory_indices(
    feature_names: np.ndarray,
) -> dict[str, dict[str, list[int]]]:
    """Parse feature names to identify trajectory-indexed features by tier.

    Returns:
        {tier: {f"traj{t}": [global_indices...]}} for each tier and timepoint.
        Also includes a special "matched" key mapping early features to late features.
    """
    result: dict[str, dict[str, list[int]]] = {}

    for tier_name, (start, end) in TIER_SLICES.items():
        tier_traj: dict[str, list[int]] = {}
        for i in range(start, end):
            name = str(feature_names[i])
            # Match traj{N} or _t{N}_ patterns
            m = re.search(r"traj(\d+)", name)
            if not m:
                m = re.search(r"_t(\d+)_", name)
            if m:
                t = int(m.group(1))
                key = f"traj{t}"
                tier_traj.setdefault(key, []).append(i)
        result[tier_name] = tier_traj

    return result


def _feature_base_name(name: str) -> str:
    """Strip the trajectory index from a feature name to get its base identity.

    Examples:
        'activation_norm_traj0_L5' → 'activation_norm_L5'
        'kv_key_novelty_traj2_L14' → 'kv_key_novelty_L14'
        'pca_L7_t0_c15' → 'pca_L7_c15'
    """
    # Remove traj{N} or _t{N}_ patterns
    base = re.sub(r"_?traj\d+_?", "_", name).strip("_")
    base = re.sub(r"_t\d+_", "_", base)
    return base


def extract_early_late_features(
    X: np.ndarray,
    feature_names: np.ndarray,
    traj_indices: dict[str, dict[str, list[int]]],
    tiers: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]], list[str]]:
    """Extract matched early (traj0,1) and late (traj3,4) feature matrices.

    Matching is done by base feature name (stripping timepoint index), NOT by
    position. This correctly handles tiers where some features exist at fewer
    timepoints (e.g., cache_recency has traj0-3 but not traj4).

    Pairing: traj0↔traj3, traj1↔traj4. Features that exist in the early
    timepoint but not the late timepoint are dropped from the matched set.

    Returns:
        (X_early, X_late, per_tier_dict, matched_feature_names)
        where per_tier_dict maps tier_name -> (X_early_tier, X_late_tier)
    """
    if tiers is None:
        tiers = [t for t in TIER_SLICES if t != "T2"]  # T2 has no trajectory features

    all_early_idx: list[int] = []
    all_late_idx: list[int] = []
    per_tier: dict[str, tuple[list[int], list[int]]] = {}
    matched_names: list[str] = []

    for tier_name in tiers:
        tier_data = traj_indices.get(tier_name, {})
        if not tier_data:
            continue

        # Determine available timepoints
        available_traj = sorted(tier_data.keys())
        max_traj = max(int(k.replace("traj", "")) for k in available_traj)

        # Define early↔late pairs
        if max_traj >= 4:
            pairs = [("traj0", "traj3"), ("traj1", "traj4")]
        elif max_traj >= 3:
            pairs = [("traj0", "traj2"), ("traj1", "traj3")]
        else:
            logger.warning(f"  {tier_name}: insufficient trajectory points ({available_traj}), skipping")
            continue

        # Build base_name → index lookup for each timepoint
        tier_early: list[int] = []
        tier_late: list[int] = []

        for early_key, late_key in pairs:
            e_indices = tier_data.get(early_key, [])
            l_indices = tier_data.get(late_key, [])

            # Build base_name → global_index maps
            e_by_base: dict[str, int] = {}
            for idx in e_indices:
                base = _feature_base_name(str(feature_names[idx]))
                e_by_base[base] = idx

            l_by_base: dict[str, int] = {}
            for idx in l_indices:
                base = _feature_base_name(str(feature_names[idx]))
                l_by_base[base] = idx

            # Match by base name — only include features present in BOTH
            common_bases = sorted(set(e_by_base) & set(l_by_base))

            for base in common_bases:
                tier_early.append(e_by_base[base])
                tier_late.append(l_by_base[base])
                matched_names.append(f"{tier_name}:{base}")

        per_tier[tier_name] = (tier_early, tier_late)
        all_early_idx.extend(tier_early)
        all_late_idx.extend(tier_late)

    X_early = X[:, all_early_idx]
    X_late = X[:, all_late_idx]

    per_tier_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tier_name, (e_idx, l_idx) in per_tier.items():
        per_tier_arrays[tier_name] = (X[:, e_idx], X[:, l_idx])

    return X_early, X_late, per_tier_arrays, matched_names


# ---------------------------------------------------------------------------
# Semantic embedding — split text at midpoint, embed each half
# ---------------------------------------------------------------------------

def compute_semantic_embeddings(
    metadata_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Split generated text at midpoint, embed each half with sentence-transformers.

    Returns:
        (sem_early, sem_late) — (N, embed_dim) arrays
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence-transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load metadata for generated text
    with open(metadata_path) as f:
        raw = json.load(f)
    all_meta = raw["generations"] if isinstance(raw, dict) and "generations" in raw else raw
    ab_meta = [m for m in all_meta if m["prompt_set"] in ("A", "B")]
    ab_meta.sort(key=lambda m: m["generation_id"])

    early_texts: list[str] = []
    late_texts: list[str] = []

    for m in ab_meta:
        text = m["generated_text"]
        mid = len(text) // 2
        # Find nearest sentence boundary near midpoint
        for offset in range(0, min(200, len(text) // 4)):
            if mid + offset < len(text) and text[mid + offset] in ".!?\n":
                mid = mid + offset + 1
                break
            if mid - offset >= 0 and text[mid - offset] in ".!?\n":
                mid = mid - offset + 1
                break

        early_texts.append(text[:mid].strip())
        late_texts.append(text[mid:].strip())

    logger.info(f"  Embedding {len(early_texts)} text halves...")
    sem_early = model.encode(early_texts, show_progress_bar=False, batch_size=32)
    sem_late = model.encode(late_texts, show_progress_bar=False, batch_size=32)

    return np.array(sem_early), np.array(sem_late)


# ---------------------------------------------------------------------------
# Ridge regression prediction — core analysis
# ---------------------------------------------------------------------------

def ridge_predict_cv(
    X_pred: np.ndarray,
    Y_target: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> dict[str, Any]:
    """Ridge regression with GroupKFold: predict each target feature from predictors.

    Returns per-feature R² (test), aggregated statistics.
    """
    gkf = GroupKFold(n_splits=n_splits)
    n_targets = Y_target.shape[1]

    # Collect per-fold, per-feature R²
    fold_r2s: list[np.ndarray] = []

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X_pred, groups=groups)):
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        X_train = scaler_x.fit_transform(X_pred[train_idx])
        X_test = scaler_x.transform(X_pred[test_idx])
        Y_train = scaler_y.fit_transform(Y_target[train_idx])
        Y_test = scaler_y.transform(Y_target[test_idx])

        # Fit separate Ridge for each target feature
        feature_r2s = np.full(n_targets, np.nan)
        for j in range(n_targets):
            y_train_j = Y_train[:, j]
            y_test_j = Y_test[:, j]

            # Skip constant targets
            if np.std(y_train_j) < 1e-10:
                feature_r2s[j] = 0.0
                continue

            try:
                ridge = RidgeCV(alphas=alphas)
                ridge.fit(X_train, y_train_j)
                y_pred = ridge.predict(X_test)
                feature_r2s[j] = r2_score(y_test_j, y_pred)
            except Exception:
                feature_r2s[j] = 0.0

        fold_r2s.append(feature_r2s)

    # Average R² across folds for each feature
    r2_matrix = np.stack(fold_r2s)  # (n_folds, n_targets)
    mean_r2_per_feature = np.nanmean(r2_matrix, axis=0)

    return {
        "median_r2": float(np.nanmedian(mean_r2_per_feature)),
        "mean_r2": float(np.nanmean(mean_r2_per_feature)),
        "std_r2": float(np.nanstd(mean_r2_per_feature)),
        "q25_r2": float(np.nanpercentile(mean_r2_per_feature, 25)),
        "q75_r2": float(np.nanpercentile(mean_r2_per_feature, 75)),
        "n_positive": int(np.sum(mean_r2_per_feature > 0)),
        "n_features": int(n_targets),
        "frac_positive": float(np.mean(mean_r2_per_feature > 0)),
        "per_feature_r2": mean_r2_per_feature,  # keep for tier decomposition
    }


# ---------------------------------------------------------------------------
# Diagnostic: mode classification from trajectory features
# ---------------------------------------------------------------------------

def run_diagnostic_classification(
    X_early: np.ndarray,
    X_late: np.ndarray,
    X_full_traj: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    mode_names: list[str],
    per_tier_early: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Prerequisite check: can trajectory features classify mode at all?

    Tests: early-only, late-only, full-trajectory, and per-tier early.
    Uses kNN with GroupKFold to match existing validation strategy.
    """
    logger.info("=== Diagnostic: mode classification from trajectory features ===")

    results: dict[str, Any] = {}

    conditions = {
        "early_all": X_early,
        "late_all": X_late,
        "full_traj_all": X_full_traj,
    }

    # Add per-tier early
    for tier_name, (tier_early, _) in per_tier_early.items():
        conditions[f"early_{tier_name}"] = tier_early

    gkf = GroupKFold(n_splits=5)

    for label, X_cond in conditions.items():
        fold_accs: list[float] = []

        for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X_cond, groups=groups)):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_cond[train_idx])
            X_test = scaler.transform(X_cond[test_idx])
            y_train, y_test = y[train_idx], y[test_idx]

            knn = KNeighborsClassifier(n_neighbors=3, metric="cosine")
            knn.fit(X_train, y_train)
            fold_accs.append(float(knn.score(X_test, y_test)))

        mean_acc = float(np.mean(fold_accs))
        std_acc = float(np.std(fold_accs))
        results[label] = {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "fold_accs": fold_accs,
            "n_features": int(X_cond.shape[1]),
        }
        above_chance = "YES" if mean_acc > 0.25 else "marginal" if mean_acc > 0.22 else "NO"
        logger.info(
            f"  {label:20s}: {mean_acc:.1%} +/- {std_acc:.1%} "
            f"({X_cond.shape[1]} features) [{above_chance}]"
        )

    return results


# ---------------------------------------------------------------------------
# Cross-generation control
# ---------------------------------------------------------------------------

def run_cross_generation_control(
    X_early: np.ndarray,
    X_late: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    mode_names: list[str],
) -> dict[str, Any]:
    """Cross-generation control: does mode-centroid early predict individual late?

    For each sample i of mode m, replace its early features with the
    leave-one-out mean of all same-mode samples' early features (excluding i).
    If cross-gen R² ≈ within-gen R², prediction is mode fingerprinting.
    If cross-gen << within-gen, genuine temporal coherence.
    """
    logger.info("=== Cross-generation control ===")

    n_samples = X_early.shape[0]
    n_features = X_late.shape[1]
    unique_modes = np.unique(y)

    # Build leave-one-out mode centroid for early features
    X_early_centroid = np.zeros_like(X_early)
    for mode_idx in unique_modes:
        mode_mask = y == mode_idx
        mode_indices = np.where(mode_mask)[0]
        mode_early = X_early[mode_mask]

        for local_i, global_i in enumerate(mode_indices):
            # LOO mean: exclude this sample
            loo_mask = np.ones(len(mode_early), dtype=bool)
            loo_mask[local_i] = False
            X_early_centroid[global_i] = mode_early[loo_mask].mean(axis=0)

    # Run same ridge prediction with centroid early features
    cross_gen = ridge_predict_cv(X_early_centroid, X_late, groups)
    within_gen = ridge_predict_cv(X_early, X_late, groups)

    delta = within_gen["median_r2"] - cross_gen["median_r2"]

    logger.info(f"  Within-generation median R²: {within_gen['median_r2']:.4f}")
    logger.info(f"  Cross-generation median R²:  {cross_gen['median_r2']:.4f}")
    logger.info(f"  Delta (within - cross):       {delta:.4f}")

    if abs(delta) < 0.02:
        interpretation = "MODE_FINGERPRINTING — prediction is driven by mode label persistence"
    elif delta > 0.05:
        interpretation = "TEMPORAL_COHERENCE — genuine within-generation temporal structure beyond mode"
    else:
        interpretation = "MARGINAL — small advantage for within-generation, not conclusive"

    logger.info(f"  Interpretation: {interpretation}")

    return {
        "within_gen_median_r2": within_gen["median_r2"],
        "within_gen_mean_r2": within_gen["mean_r2"],
        "cross_gen_median_r2": cross_gen["median_r2"],
        "cross_gen_mean_r2": cross_gen["mean_r2"],
        "delta_median": float(delta),
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Within-mode residual prediction
# ---------------------------------------------------------------------------

def run_within_mode_residual(
    X_early: np.ndarray,
    X_late: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    """After subtracting mode centroids, does early→late prediction still work?

    Tests within-mode temporal structure beyond mode label.
    """
    logger.info("=== Within-mode residual prediction ===")

    # Subtract per-mode mean from both halves
    X_early_resid = X_early.copy()
    X_late_resid = X_late.copy()

    for mode_idx in np.unique(y):
        mask = y == mode_idx
        X_early_resid[mask] -= X_early[mask].mean(axis=0)
        X_late_resid[mask] -= X_late[mask].mean(axis=0)

    result = ridge_predict_cv(X_early_resid, X_late_resid, groups)

    logger.info(f"  Residual median R²: {result['median_r2']:.4f}")
    logger.info(f"  Residual frac positive: {result['frac_positive']:.1%}")

    return {
        "median_r2": result["median_r2"],
        "mean_r2": result["mean_r2"],
        "frac_positive": result["frac_positive"],
        "n_positive": result["n_positive"],
        "n_features": result["n_features"],
    }


# ---------------------------------------------------------------------------
# Cross-half mode consistency
# ---------------------------------------------------------------------------

def run_cross_half_consistency(
    X_early: np.ndarray,
    X_late: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    mode_names: list[str],
) -> dict[str, Any]:
    """Classify mode independently from each half, check agreement rate."""
    logger.info("=== Cross-half mode consistency ===")

    gkf = GroupKFold(n_splits=5)

    early_preds = np.full(len(y), -1)
    late_preds = np.full(len(y), -1)

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X_early, groups=groups)):
        # Early classifier
        scaler_e = StandardScaler()
        X_e_train = scaler_e.fit_transform(X_early[train_idx])
        X_e_test = scaler_e.transform(X_early[test_idx])

        knn_e = KNeighborsClassifier(n_neighbors=3, metric="cosine")
        knn_e.fit(X_e_train, y[train_idx])
        early_preds[test_idx] = knn_e.predict(X_e_test)

        # Late classifier
        scaler_l = StandardScaler()
        X_l_train = scaler_l.fit_transform(X_late[train_idx])
        X_l_test = scaler_l.transform(X_late[test_idx])

        knn_l = KNeighborsClassifier(n_neighbors=3, metric="cosine")
        knn_l.fit(X_l_train, y[train_idx])
        late_preds[test_idx] = knn_l.predict(X_l_test)

    # Agreement: both halves predict the same mode
    agreement = float(np.mean(early_preds == late_preds))
    both_correct = float(np.mean((early_preds == y) & (late_preds == y)))
    early_acc = float(np.mean(early_preds == y))
    late_acc = float(np.mean(late_preds == y))

    # Per-mode agreement
    per_mode: dict[str, dict[str, float]] = {}
    for i, name in enumerate(mode_names):
        mask = y == i
        if mask.sum() == 0:
            continue
        per_mode[name] = {
            "agreement": float(np.mean(early_preds[mask] == late_preds[mask])),
            "early_acc": float(np.mean(early_preds[mask] == y[mask])),
            "late_acc": float(np.mean(late_preds[mask] == y[mask])),
            "both_correct": float(np.mean(
                (early_preds[mask] == y[mask]) & (late_preds[mask] == y[mask])
            )),
        }

    logger.info(f"  Early accuracy:  {early_acc:.1%}")
    logger.info(f"  Late accuracy:   {late_acc:.1%}")
    logger.info(f"  Agreement rate:  {agreement:.1%}")
    logger.info(f"  Both correct:    {both_correct:.1%}")
    logger.info(f"  Per-mode agreement:")
    for name, info in sorted(per_mode.items(), key=lambda x: x[1]["agreement"], reverse=True):
        logger.info(f"    {name:15s}: agree={info['agreement']:.0%}, "
                     f"early={info['early_acc']:.0%}, late={info['late_acc']:.0%}")

    return {
        "early_accuracy": early_acc,
        "late_accuracy": late_acc,
        "agreement_rate": agreement,
        "both_correct_rate": both_correct,
        "per_mode": per_mode,
    }


# ---------------------------------------------------------------------------
# Permutation test on Task 1 vs Task 2 delta
# ---------------------------------------------------------------------------

def run_permutation_test(
    X_early_compute: np.ndarray,
    X_early_semantic: np.ndarray,
    X_late_compute: np.ndarray,
    groups: np.ndarray,
    y: np.ndarray,
    observed_delta: float,
    n_permutations: int = 200,
) -> dict[str, Any]:
    """Permutation test: shuffle mode labels, recompute Task 1 - Task 2 delta.

    Under null, mode labels don't structure the data, so compute→compute and
    semantic→compute should have similar (poor) R². The delta should be ~0.
    """
    logger.info(f"=== Permutation test ({n_permutations} shuffles) ===")

    rng = np.random.RandomState(42)
    null_deltas: list[float] = []

    for perm_i in range(n_permutations):
        # Shuffle mode labels but keep topic structure intact for GroupKFold
        y_shuffled = rng.permutation(y)

        # Recompute both tasks with shuffled labels
        # The labels affect nothing in Ridge regression (unsupervised prediction)
        # but we shuffle to break any mode-structured correlation
        # Actually — the right permutation here is to shuffle the sample ORDER
        # (break the early↔late pairing within each sample)
        shuffle_idx = rng.permutation(len(y))
        X_late_shuffled = X_late_compute[shuffle_idx]

        r2_compute = ridge_predict_cv(X_early_compute, X_late_shuffled, groups)
        r2_semantic = ridge_predict_cv(X_early_semantic, X_late_shuffled, groups)
        null_delta = r2_compute["median_r2"] - r2_semantic["median_r2"]
        null_deltas.append(null_delta)

        if (perm_i + 1) % 25 == 0:
            n_above = sum(1 for d in null_deltas if d >= observed_delta)
            logger.info(
                f"  Permutation {perm_i + 1}/{n_permutations}: "
                f"null_delta={null_delta:.4f}, "
                f"running p={n_above}/{len(null_deltas)}"
            )

    null_arr = np.array(null_deltas)
    n_above = int(np.sum(null_arr >= observed_delta))
    p_value = (n_above + 1) / (n_permutations + 1)

    logger.info(f"  Observed delta: {observed_delta:.4f}")
    logger.info(f"  Null mean: {np.mean(null_arr):.4f}, std: {np.std(null_arr):.4f}")
    logger.info(f"  Null max: {np.max(null_arr):.4f}")
    logger.info(f"  p-value: {p_value:.4f} ({n_above}/{n_permutations} above observed)")

    return {
        "observed_delta": float(observed_delta),
        "null_mean": float(np.mean(null_arr)),
        "null_std": float(np.std(null_arr)),
        "null_max": float(np.max(null_arr)),
        "n_above": n_above,
        "p_value": float(p_value),
        "n_permutations": n_permutations,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal prediction experiment")
    parser.add_argument("--n-permutations", type=int, default=200,
                        help="Permutation count for delta test (default: 200)")
    parser.add_argument("--skip-permutation", action="store_true",
                        help="Skip permutation test (faster for debugging)")
    parser.add_argument("--skip-semantic", action="store_true",
                        help="Skip semantic embedding (requires sentence-transformers)")
    args = parser.parse_args()

    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------

    logger.info("Loading data...")
    data = np.load(PROJECT_ROOT / "review_pack" / "features.npz", allow_pickle=True)
    X_full = data["features"].astype(np.float64)
    X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
    feature_names = data["feature_names"]
    labels_str = data["labels"]
    topics = data["topics"]

    le = LabelEncoder()
    y = le.fit_transform(labels_str)
    mode_names = le.classes_.tolist()

    # Topic groups for GroupKFold
    unique_topics = np.unique(topics)
    topic_to_id = {t: i for i, t in enumerate(unique_topics)}
    groups = np.array([topic_to_id[t] for t in topics])

    logger.info(f"  {X_full.shape[0]} samples, {X_full.shape[1]} features, "
                f"{len(mode_names)} modes, {len(unique_topics)} topics")

    # -----------------------------------------------------------------------
    # Parse trajectory structure
    # -----------------------------------------------------------------------

    logger.info("Parsing trajectory feature structure...")
    traj_indices = build_trajectory_indices(feature_names)

    for tier_name, tier_data in traj_indices.items():
        n_traj = sum(len(v) for v in tier_data.values())
        tp_counts = {k: len(v) for k, v in tier_data.items()}
        logger.info(f"  {tier_name}: {n_traj} trajectory features across {len(tier_data)} timepoints: {tp_counts}")

    # -----------------------------------------------------------------------
    # Extract early/late features
    # -----------------------------------------------------------------------

    logger.info("Extracting early/late feature splits...")
    X_early, X_late, per_tier, matched_names = extract_early_late_features(
        X_full, feature_names, traj_indices,
    )
    logger.info(f"  Early: {X_early.shape}, Late: {X_late.shape}")
    for tier_name, (te, tl) in per_tier.items():
        logger.info(f"    {tier_name}: early={te.shape[1]}, late={tl.shape[1]}")

    # Also build full trajectory features (all timepoints) for diagnostic
    all_traj_idx = []
    for tier_name, tier_data in traj_indices.items():
        for tp_indices in tier_data.values():
            all_traj_idx.extend(tp_indices)
    all_traj_idx = sorted(set(all_traj_idx))
    X_full_traj = X_full[:, all_traj_idx]
    logger.info(f"  Full trajectory features: {X_full_traj.shape[1]}")

    all_results: dict[str, Any] = {
        "experiment": "temporal_prediction",
        "n_samples": int(X_full.shape[0]),
        "n_features_total": int(X_full.shape[1]),
        "n_features_early": int(X_early.shape[1]),
        "n_features_late": int(X_late.shape[1]),
        "n_features_full_traj": int(X_full_traj.shape[1]),
        "per_tier_counts": {
            tier: {"early": te.shape[1], "late": tl.shape[1]}
            for tier, (te, tl) in per_tier.items()
        },
        "modes": mode_names,
        "n_topics": int(len(unique_topics)),
    }

    # Save partial results after each section
    def save_partial(label: str = "") -> None:
        partial_path = OUTPUT_DIR / "temporal_prediction_partial.json"
        serializable = {}
        for k, v in all_results.items():
            if isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            elif isinstance(v, dict):
                serializable[k] = _make_serializable(v)
            else:
                serializable[k] = v
        with open(partial_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        if label:
            logger.info(f"  [partial save after {label}]")

    # -----------------------------------------------------------------------
    # DIAGNOSTIC: Mode classification from trajectory features
    # -----------------------------------------------------------------------

    diag_results = run_diagnostic_classification(
        X_early, X_late, X_full_traj, y, groups, mode_names, per_tier,
    )
    all_results["diagnostic_classification"] = diag_results
    save_partial("diagnostic")

    # Check prerequisite: do early trajectory features carry ANY mode info?
    early_acc = diag_results.get("early_all", {}).get("mean_accuracy", 0.0)
    if early_acc < 0.22:
        logger.warning(
            "!!! PREREQUISITE FAILED: early trajectory features cannot classify mode. "
            "Prediction results will be uninterpretable for mode dynamics. "
            "Continuing anyway to characterize the features."
        )

    # -----------------------------------------------------------------------
    # TASK 1: compute→compute (per-tier decomposition)
    # -----------------------------------------------------------------------

    logger.info("\n" + "=" * 70)
    logger.info("TASK 1: compute→compute (early trajectory → late trajectory)")
    logger.info("=" * 70)

    task1_overall = ridge_predict_cv(X_early, X_late, groups)
    # Strip non-serializable per-feature array for JSON, keep for analysis
    task1_per_feature = task1_overall.pop("per_feature_r2")

    logger.info(f"  Overall median R²: {task1_overall['median_r2']:.4f}")
    logger.info(f"  Overall frac positive: {task1_overall['frac_positive']:.1%}")

    task1_by_tier: dict[str, dict[str, Any]] = {}
    for tier_name, (tier_early, tier_late) in per_tier.items():
        tier_result = ridge_predict_cv(tier_early, tier_late, groups)
        tier_pf = tier_result.pop("per_feature_r2")
        task1_by_tier[tier_name] = tier_result
        logger.info(
            f"  {tier_name:5s}: median R²={tier_result['median_r2']:.4f}, "
            f"frac_positive={tier_result['frac_positive']:.1%} "
            f"({tier_result['n_features']} features)"
        )

    all_results["task1_compute_to_compute"] = {
        "overall": task1_overall,
        "by_tier": task1_by_tier,
    }
    save_partial("task1")

    # -----------------------------------------------------------------------
    # TASK 2 & 3: semantic→compute and semantic→semantic
    # -----------------------------------------------------------------------

    if not args.skip_semantic:
        logger.info("\n" + "=" * 70)
        logger.info("TASKS 2-3: semantic predictions")
        logger.info("=" * 70)

        metadata_path = (
            PROJECT_ROOT / "outputs" / "runs" / "run4_format_controlled" / "metadata.json"
        )
        sem_early, sem_late = compute_semantic_embeddings(metadata_path)
        logger.info(f"  Semantic embeddings: early={sem_early.shape}, late={sem_late.shape}")

        # Task 2: semantic→compute (overall and per-tier)
        logger.info("\nTask 2: semantic→compute")
        task2_overall = ridge_predict_cv(sem_early, X_late, groups)
        task2_per_feature = task2_overall.pop("per_feature_r2")
        logger.info(f"  Overall median R²: {task2_overall['median_r2']:.4f}")

        task2_by_tier: dict[str, dict[str, Any]] = {}
        for tier_name, (_, tier_late) in per_tier.items():
            tier_result = ridge_predict_cv(sem_early, tier_late, groups)
            tier_result.pop("per_feature_r2")
            task2_by_tier[tier_name] = tier_result
            logger.info(
                f"  {tier_name:5s}: median R²={tier_result['median_r2']:.4f}, "
                f"frac_positive={tier_result['frac_positive']:.1%}"
            )

        all_results["task2_semantic_to_compute"] = {
            "overall": task2_overall,
            "by_tier": task2_by_tier,
        }

        # Task 3: semantic→semantic (calibration)
        logger.info("\nTask 3: semantic→semantic (calibration)")
        task3_result = ridge_predict_cv(sem_early, sem_late, groups)
        task3_result.pop("per_feature_r2")
        logger.info(f"  Median R²: {task3_result['median_r2']:.4f}")
        all_results["task3_semantic_to_semantic"] = task3_result

        # Delta: Task 1 - Task 2
        delta = task1_overall["median_r2"] - task2_overall["median_r2"]
        logger.info(f"\n  DELTA (Task 1 - Task 2): {delta:.4f}")

        delta_by_tier: dict[str, float] = {}
        for tier_name in task1_by_tier:
            if tier_name in task2_by_tier:
                td = task1_by_tier[tier_name]["median_r2"] - task2_by_tier[tier_name]["median_r2"]
                delta_by_tier[tier_name] = float(td)
                logger.info(f"    {tier_name}: delta={td:.4f}")

        all_results["delta_task1_minus_task2"] = {
            "overall": float(delta),
            "by_tier": delta_by_tier,
        }
        save_partial("tasks2-3")
    else:
        logger.info("\nSkipping semantic tasks (--skip-semantic)")
        delta = None

    # -----------------------------------------------------------------------
    # Cross-generation control
    # -----------------------------------------------------------------------

    logger.info("\n" + "=" * 70)
    logger.info("TASK 4: Cross-generation control")
    logger.info("=" * 70)

    cross_gen = run_cross_generation_control(X_early, X_late, y, groups, mode_names)
    all_results["cross_generation_control"] = cross_gen
    save_partial("cross-gen")

    # -----------------------------------------------------------------------
    # Within-mode residual prediction
    # -----------------------------------------------------------------------

    logger.info("\n" + "=" * 70)
    logger.info("TASK 5: Within-mode residual prediction")
    logger.info("=" * 70)

    residual = run_within_mode_residual(X_early, X_late, y, groups)
    all_results["within_mode_residual"] = residual
    save_partial("residual")

    # -----------------------------------------------------------------------
    # Cross-half mode consistency
    # -----------------------------------------------------------------------

    logger.info("\n" + "=" * 70)
    logger.info("TASK 6: Cross-half mode consistency")
    logger.info("=" * 70)

    consistency = run_cross_half_consistency(X_early, X_late, y, groups, mode_names)
    all_results["cross_half_consistency"] = consistency
    save_partial("consistency")

    # -----------------------------------------------------------------------
    # Permutation test on Task 1 vs Task 2 delta
    # -----------------------------------------------------------------------

    if delta is not None and not args.skip_permutation:
        logger.info("\n" + "=" * 70)
        logger.info("PERMUTATION TEST: Task 1 vs Task 2 delta")
        logger.info("=" * 70)

        perm_result = run_permutation_test(
            X_early, sem_early, X_late, groups, y,
            observed_delta=delta,
            n_permutations=args.n_permutations,
        )
        all_results["permutation_test"] = perm_result
        save_partial("permutation")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    elapsed = time.time() - t_start
    all_results["elapsed_seconds"] = float(elapsed)

    logger.info(f"\n{'=' * 70}")
    logger.info("EXPERIMENT 3 SUMMARY")
    logger.info(f"{'=' * 70}")

    # Diagnostic
    logger.info("\nDiagnostic (mode classification from trajectory features):")
    for label, info in diag_results.items():
        logger.info(f"  {label:20s}: {info['mean_accuracy']:.1%} ({info['n_features']} features)")

    # Task 1 by tier
    logger.info("\nTask 1 (compute→compute) by tier:")
    logger.info(f"  {'Overall':10s}: median R²={task1_overall['median_r2']:.4f}")
    for tier, info in sorted(task1_by_tier.items()):
        logger.info(f"  {tier:10s}: median R²={info['median_r2']:.4f}")

    # Tasks 2-3 comparison
    if delta is not None:
        logger.info(f"\nTask 2 (semantic→compute): median R²={task2_overall['median_r2']:.4f}")
        logger.info(f"Task 3 (semantic→semantic): median R²={task3_result['median_r2']:.4f}")
        logger.info(f"Delta (Task 1 - Task 2): {delta:.4f}")
        if "permutation_test" in all_results:
            pt = all_results["permutation_test"]
            logger.info(f"  Permutation p-value: {pt['p_value']:.4f}")

    # Cross-generation
    logger.info(f"\nCross-generation control:")
    logger.info(f"  Within-gen R²: {cross_gen['within_gen_median_r2']:.4f}")
    logger.info(f"  Cross-gen R²:  {cross_gen['cross_gen_median_r2']:.4f}")
    logger.info(f"  Interpretation: {cross_gen['interpretation']}")

    # Residual
    logger.info(f"\nWithin-mode residual: median R²={residual['median_r2']:.4f}")

    # Consistency
    logger.info(f"\nCross-half consistency:")
    logger.info(f"  Early acc: {consistency['early_accuracy']:.1%}, "
                f"Late acc: {consistency['late_accuracy']:.1%}, "
                f"Agreement: {consistency['agreement_rate']:.1%}")

    # Verdict
    logger.info(f"\nElapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # -----------------------------------------------------------------------
    # Save final results
    # -----------------------------------------------------------------------

    final_results = _make_serializable(all_results)
    out_path = OUTPUT_DIR / "temporal_prediction.json"
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    return obj


if __name__ == "__main__":
    main()
