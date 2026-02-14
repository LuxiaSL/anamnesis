#!/usr/bin/env python3
"""Phase 1, Experiment 1.75: Tier ablation on contrastive MLP.

Tests whether the Phase 0 tier inversion (T2.5 > T2 > T1 under format control)
is a property of the *signal* or the *classifier*. If the contrastive MLP
reproduces the same tier ranking, the inversion is intrinsic to the data geometry.

Three analyses:
  1. Per-tier MLP ablation — train contrastive MLP on each tier's features alone
  2. Cross-tier interaction — pairwise tier combinations to detect super-additivity
  3. MLP weight importance — first-layer weight norms vs RF feature importance

Both Run 3 (format-free ceiling) and Run 4 (format-controlled floor) are analyzed.

Usage:
    python scripts/run_tier_ablation.py
    python scripts/run_tier_ablation.py --skip-permutation --skip-importance
    python scripts/run_tier_ablation.py --run 4
    python scripts/run_tier_ablation.py --n-permutations 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase1" / "tier_ablation"
FIGURES_DIR = OUTPUT_DIR / "figures"

# ---------------------------------------------------------------------------
# Torch imports
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    logger.error("PyTorch required for this experiment")
    sys.exit(1)

from scripts.run_contrastive_projection import (
    _train_projection_mlp,
    run_mlp_permutation_test,
)
from scripts.run_cross_run_transfer import train_full_data_mlp


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Feature tier index ranges in the 1837-dim vector
TIER_RANGES: dict[str, tuple[int, int]] = {
    "T1": (0, 221),
    "T2": (221, 442),
    "T2.5": (442, 587),
    "T3": (587, 1837),
}

TIER_NAMES_ORDERED = ["T1", "T2", "T2.5", "T3"]

# Phase 0 RF tier ablation baselines (Run 4)
RF_TIER_ABLATION: dict[str, dict[str, float]] = {
    "T1": {"rf_accuracy": 0.54, "rf_silhouette": 0.002},
    "T2": {"rf_accuracy": 0.59, "rf_silhouette": 0.031},
    "T2.5": {"rf_accuracy": 0.64, "rf_silhouette": 0.035},
}

# Pairwise and combo tier combinations to test
TIER_COMBINATIONS: list[tuple[str, ...]] = [
    ("T1", "T2"),
    ("T1", "T2.5"),
    ("T2", "T2.5"),
    ("T1", "T2", "T2.5"),  # engineered-only
]

# Pre-registered predictions
PRE_REGISTERED = {
    "per_tier_run4": {
        "T1": {"knn_predicted": "35-45%"},
        "T2": {"knn_predicted": "45-55%"},
        "T2.5": {"knn_predicted": "55-65%"},
        "T3": {"knn_predicted": "40-50%"},
    },
    "per_tier_run3": {
        "T1": {"knn_predicted": "70-80%"},
        "T2": {"knn_predicted": "75-85%"},
        "T2.5": {"knn_predicted": "70-80%"},
        "T3": {"knn_predicted": "55-70%"},
    },
    "key_predictions": [
        "Run 4 tier ranking: T2.5 > T2 > T1 persists under MLP",
        "Run 3 tier ranking: flatter — modes produce signal across all tiers",
        "T3 comes alive under MLP (vs dead under RF)",
        "Run 4: T2+T2.5 > either alone substantially",
        "Run 4: Adding T1 to T2+T2.5 adds ~nothing",
    ],
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RunData:
    """All data for a single run, including per-tier feature matrices."""

    X_full: np.ndarray           # (N, 1837)
    tiers: dict[str, np.ndarray]  # {"T1": (N,221), ...}
    y: np.ndarray                 # (N,) integer labels
    mode_names: list[str]
    labels_str: np.ndarray
    feature_names: np.ndarray     # (1837,) for weight importance
    n_samples: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_samples = self.X_full.shape[0]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run4_data() -> RunData:
    """Load Run 4 features from review_pack/features.npz with per-tier arrays."""
    review_pack = PROJECT_ROOT / "review_pack" / "features.npz"
    data = np.load(review_pack, allow_pickle=True)

    X = data["features"].astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    labels_str = data["labels"]
    feature_names = data["feature_names"]

    tiers = {
        "T1": data["tier1"].astype(np.float64),
        "T2": data["tier2"].astype(np.float64),
        "T2.5": data["tier2_5"].astype(np.float64),
        "T3": data["tier3"].astype(np.float64),
    }
    for tier_name, tier_arr in tiers.items():
        tiers[tier_name] = np.nan_to_num(tier_arr, nan=0.0, posinf=0.0, neginf=0.0)

    le = LabelEncoder()
    y = le.fit_transform(labels_str)
    mode_names = le.classes_.tolist()

    logger.info(f"Run 4: {X.shape[0]} samples, {X.shape[1]} features, modes: {mode_names}")
    for tn, ta in tiers.items():
        logger.info(f"  {tn}: {ta.shape}")

    return RunData(
        X_full=X, tiers=tiers, y=y, mode_names=mode_names,
        labels_str=labels_str, feature_names=feature_names,
    )


def load_run3_data() -> RunData:
    """Load Run 3 Set A+B features from per-signature .npz files with per-tier arrays."""
    run_dir = PROJECT_ROOT / "outputs" / "runs" / "run3_process_modes"
    with open(run_dir / "metadata.json") as f:
        raw = json.load(f)
    all_meta = raw["generations"] if isinstance(raw, dict) and "generations" in raw else raw
    ab_meta = [m for m in all_meta if m["prompt_set"] in ("A", "B")]

    features_list: list[np.ndarray] = []
    tier_lists: dict[str, list[np.ndarray]] = {
        "T1": [], "T2": [], "T2.5": [], "T3": [],
    }
    mode_labels: list[str] = []
    feature_names_arr: np.ndarray | None = None

    sig_dir = run_dir / "signatures"
    for m in ab_meta:
        gid = m["generation_id"]
        sig_path = sig_dir / f"gen_{gid:03d}.npz"
        if not sig_path.exists():
            logger.warning(f"Missing signature: {sig_path}")
            continue
        sig = np.load(sig_path, allow_pickle=True)
        features_list.append(sig["features"])
        tier_lists["T1"].append(sig["features_tier1"])
        tier_lists["T2"].append(sig["features_tier2"])
        tier_lists["T2.5"].append(sig["features_tier2_5"])
        tier_lists["T3"].append(sig["features_tier3"])
        mode_labels.append(m["mode"])
        if feature_names_arr is None:
            feature_names_arr = sig["feature_names"]

    X = np.stack(features_list).astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    labels_str = np.array(mode_labels)

    tiers: dict[str, np.ndarray] = {}
    for tier_name, tier_list in tier_lists.items():
        arr = np.stack(tier_list).astype(np.float64)
        tiers[tier_name] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    le = LabelEncoder()
    y = le.fit_transform(labels_str)
    mode_names = le.classes_.tolist()

    if feature_names_arr is None:
        feature_names_arr = np.array([f"f{i}" for i in range(X.shape[1])])

    logger.info(f"Run 3: {X.shape[0]} samples, {X.shape[1]} features, modes: {mode_names}")
    for tn, ta in tiers.items():
        logger.info(f"  {tn}: {ta.shape}")

    return RunData(
        X_full=X, tiers=tiers, y=y, mode_names=mode_names,
        labels_str=labels_str, feature_names=feature_names_arr,
    )


# ---------------------------------------------------------------------------
# Analysis 1: Per-tier MLP ablation
# ---------------------------------------------------------------------------

def run_per_tier_mlp(
    X: np.ndarray,
    y: np.ndarray,
    mode_names: list[str],
    tier_label: str,
    run_label: str,
    bottleneck_dim: int = 32,
    n_epochs: int = 200,
) -> dict[str, Any]:
    """Run contrastive MLP CV on a single tier's features with per-mode test silhouette."""
    label = f"{run_label} {tier_label}"
    logger.info(f"=== {label}: {X.shape[1]} features ===")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()

    fold_sils: list[float] = []
    fold_knns: list[float] = []
    all_test_y: list[np.ndarray] = []
    all_test_sil_samples: list[np.ndarray] = []
    all_y_test: list[int] = []
    all_y_pred: list[int] = []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        emb_train, emb_test, final_loss = _train_projection_mlp(
            X_train, y_train, X_test,
            bottleneck_dim=bottleneck_dim, n_epochs=n_epochs,
            torch_seed=42 + fold_i,
        )

        sil_test = float("nan")
        if len(np.unique(y_test)) > 1:
            sil_test = silhouette_score(emb_test, y_test, metric="cosine")
            fold_sils.append(sil_test)
            sil_samples = silhouette_samples(emb_test, y_test, metric="cosine")
            all_test_y.append(y_test)
            all_test_sil_samples.append(sil_samples)

        knn = KNeighborsClassifier(n_neighbors=3, metric="cosine")
        knn.fit(emb_train, y_train)
        y_pred = knn.predict(emb_test)
        knn_acc = float(np.mean(y_pred == y_test))
        fold_knns.append(knn_acc)
        all_y_test.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        logger.info(f"  Fold {fold_i}: sil_test={sil_test:.4f}, kNN={knn_acc:.2%}")

    # Per-mode test silhouette aggregated across folds
    per_mode_test_sil: dict[str, float] = {}
    if all_test_y:
        all_y_arr = np.concatenate(all_test_y)
        all_sil_arr = np.concatenate(all_test_sil_samples)
        for i, name in enumerate(mode_names):
            mask = all_y_arr == i
            if mask.sum() > 0:
                per_mode_test_sil[name] = float(np.mean(all_sil_arr[mask]))

    # Per-mode recall
    y_test_arr = np.array(all_y_test)
    y_pred_arr = np.array(all_y_pred)
    per_mode_recall: dict[str, float] = {}
    for i, name in enumerate(mode_names):
        mask = y_test_arr == i
        if mask.sum() > 0:
            per_mode_recall[name] = float(np.mean(y_pred_arr[mask] == y_test_arr[mask]))

    results: dict[str, Any] = {
        "tier": tier_label,
        "n_features": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "silhouette_test_mean": float(np.mean(fold_sils)) if fold_sils else 0.0,
        "silhouette_test_std": float(np.std(fold_sils)) if fold_sils else 0.0,
        "silhouette_test_per_fold": [float(s) for s in fold_sils],
        "knn_accuracy_mean": float(np.mean(fold_knns)),
        "knn_accuracy_std": float(np.std(fold_knns)),
        "knn_accuracy_per_fold": [float(k) for k in fold_knns],
        "per_mode_test_silhouette": per_mode_test_sil,
        "per_mode_recall": per_mode_recall,
    }

    logger.info(f"  {label}: sil={results['silhouette_test_mean']:.4f} "
                f"+/- {results['silhouette_test_std']:.4f}, "
                f"kNN={results['knn_accuracy_mean']:.2%}")
    if per_mode_test_sil:
        logger.info(f"  Per-mode test sil:")
        for name, sil in sorted(per_mode_test_sil.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"    {name}: {sil:.4f}")

    return results


def run_per_tier_analysis(
    run_data: RunData,
    run_label: str,
    skip_permutation: bool = False,
    n_permutations: int = 50,
) -> dict[str, Any]:
    """Run per-tier MLP ablation for all tiers + combined on a single run."""
    results: dict[str, Any] = {}

    # Each tier
    for tier_name in TIER_NAMES_ORDERED:
        X_tier = run_data.tiers[tier_name]
        tier_results = run_per_tier_mlp(
            X_tier, run_data.y, run_data.mode_names,
            tier_label=tier_name, run_label=run_label,
        )

        # Permutation test for tiers with signal above chance
        if not skip_permutation:
            observed_sil = tier_results["silhouette_test_mean"]
            if observed_sil > 0.01:  # Only test tiers with non-trivial signal
                logger.info(f"  Running permutation test for {tier_name} "
                            f"(observed sil={observed_sil:.4f})...")
                perm_results = run_mlp_permutation_test(
                    X_tier, run_data.y, observed_sil,
                    n_permutations=n_permutations,
                )
                tier_results["permutation"] = perm_results
            else:
                logger.info(f"  Skipping permutation for {tier_name} "
                            f"(sil={observed_sil:.4f} too low)")

        results[tier_name] = tier_results

    # Combined (reference)
    combined_results = run_per_tier_mlp(
        run_data.X_full, run_data.y, run_data.mode_names,
        tier_label="combined", run_label=run_label,
    )
    results["combined"] = combined_results

    return results


# ---------------------------------------------------------------------------
# Analysis 2: Cross-tier interaction
# ---------------------------------------------------------------------------

def run_cross_tier_analysis(
    run_data: RunData,
    run_label: str,
) -> dict[str, Any]:
    """Run MLP on pairwise and combo tier combinations to detect interactions."""
    results: dict[str, Any] = {}

    for combo in TIER_COMBINATIONS:
        combo_label = "+".join(combo)
        X_combo = np.hstack([run_data.tiers[t] for t in combo])

        combo_results = run_per_tier_mlp(
            X_combo, run_data.y, run_data.mode_names,
            tier_label=combo_label, run_label=run_label,
        )
        results[combo_label] = combo_results

    return results


# ---------------------------------------------------------------------------
# Analysis 3: MLP weight importance
# ---------------------------------------------------------------------------

def run_weight_importance(
    run_data: RunData,
    rf_top_20: list[dict[str, Any]],
) -> dict[str, Any]:
    """Train full-data MLP and extract first-layer weight norms for importance."""
    logger.info("=== MLP weight importance analysis ===")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(run_data.X_full)

    model, final_loss = train_full_data_mlp(X_scaled, run_data.y)
    logger.info(f"  Full-data MLP trained, final_loss={final_loss:.4f}")

    # Extract first-layer weight norms
    W = model.net[0].weight.data.numpy()  # shape (256, 1837)
    importance = np.linalg.norm(W, axis=0)  # L2 norm per input feature → (1837,)

    # Map each feature to its tier
    feature_tiers: list[str] = []
    for i in range(len(importance)):
        assigned = False
        for tier_name, (start, end) in TIER_RANGES.items():
            if start <= i < end:
                feature_tiers.append(tier_name)
                assigned = True
                break
        if not assigned:
            feature_tiers.append("unknown")

    # Per-tier importance statistics
    per_tier_stats: dict[str, dict[str, float]] = {}
    for tier_name in TIER_NAMES_ORDERED:
        tier_mask = np.array([ft == tier_name for ft in feature_tiers])
        tier_imp = importance[tier_mask]
        if len(tier_imp) > 0:
            per_tier_stats[tier_name] = {
                "mean": float(np.mean(tier_imp)),
                "median": float(np.median(tier_imp)),
                "max": float(np.max(tier_imp)),
                "std": float(np.std(tier_imp)),
                "sum": float(np.sum(tier_imp)),
                "n_features": int(len(tier_imp)),
            }

    # Top-20 features by MLP weight importance
    top_indices = np.argsort(importance)[::-1][:20]
    top_20_mlp = []
    for idx in top_indices:
        top_20_mlp.append({
            "name": str(run_data.feature_names[idx]),
            "importance": float(importance[idx]),
            "index": int(idx),
            "tier": feature_tiers[idx],
        })

    # Compare to RF top-20
    rf_names = {item["name"] for item in rf_top_20}
    mlp_names = {item["name"] for item in top_20_mlp}
    overlap = rf_names & mlp_names
    overlap_count = len(overlap)

    # Spearman correlation between MLP and RF importance across all features
    # Build RF importance vector (features not in RF top-20 get 0)
    rf_importance_vec = np.zeros(len(importance))
    for item in rf_top_20:
        idx = item["index"]
        if 0 <= idx < len(importance):
            rf_importance_vec[idx] = item["importance"]

    # Only compute correlation where at least one has nonzero importance
    # (otherwise we'd be correlating zeros which is meaningless)
    nonzero_mask = (importance > 0) | (rf_importance_vec > 0)
    if nonzero_mask.sum() > 2:
        spearman_r, spearman_p = stats.spearmanr(
            importance[nonzero_mask], rf_importance_vec[nonzero_mask]
        )
    else:
        spearman_r, spearman_p = 0.0, 1.0

    # Also compute rank correlation using just top-100 from each
    top100_mlp_idx = set(np.argsort(importance)[::-1][:100].tolist())
    top100_rf_idx = set(np.argsort(rf_importance_vec)[::-1][:100].tolist())
    union_idx = sorted(top100_mlp_idx | top100_rf_idx)
    if len(union_idx) > 2:
        spearman_top100_r, spearman_top100_p = stats.spearmanr(
            importance[union_idx], rf_importance_vec[union_idx]
        )
    else:
        spearman_top100_r, spearman_top100_p = 0.0, 1.0

    results: dict[str, Any] = {
        "per_tier_stats": per_tier_stats,
        "top_20_mlp": top_20_mlp,
        "top_20_rf": rf_top_20,
        "overlap_count": overlap_count,
        "overlap_features": sorted(overlap),
        "spearman_correlation": float(spearman_r),
        "spearman_p_value": float(spearman_p),
        "spearman_top100_correlation": float(spearman_top100_r),
        "spearman_top100_p_value": float(spearman_top100_p),
        "final_loss": float(final_loss),
        "importance_vector_stored": False,  # too large for JSON
    }

    logger.info(f"  Per-tier importance stats:")
    for tn, ts in per_tier_stats.items():
        logger.info(f"    {tn}: mean={ts['mean']:.4f}, median={ts['median']:.4f}, "
                    f"max={ts['max']:.4f}, sum={ts['sum']:.2f}")
    logger.info(f"  Top-5 MLP features:")
    for item in top_20_mlp[:5]:
        logger.info(f"    {item['name']} ({item['tier']}): {item['importance']:.4f}")
    logger.info(f"  RF vs MLP top-20 overlap: {overlap_count}/20 features")
    if overlap:
        logger.info(f"    Shared: {sorted(overlap)}")
    logger.info(f"  Spearman correlation (all nonzero): r={spearman_r:.4f}, p={spearman_p:.4f}")
    logger.info(f"  Spearman correlation (top-100 union): "
                f"r={spearman_top100_r:.4f}, p={spearman_top100_p:.4f}")

    return results, importance, feature_tiers


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_tier_comparison(
    run4_per_tier: dict[str, Any] | None,
    run3_per_tier: dict[str, Any] | None,
    save_path: Path,
) -> None:
    """Grouped bar chart: T1/T2/T2.5/T3/Combined x Run3/Run4, kNN accuracy."""
    tier_labels = TIER_NAMES_ORDERED + ["combined"]

    has_r4 = run4_per_tier is not None
    has_r3 = run3_per_tier is not None
    n_bars = int(has_r4) + int(has_r3)
    if n_bars == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(tier_labels))
    width = 0.35 if n_bars == 2 else 0.5

    # kNN accuracy
    bar_offset = 0
    if has_r4:
        r4_knn = [run4_per_tier.get(t, {}).get("knn_accuracy_mean", 0.0) for t in tier_labels]
        offset = -width / 2 if n_bars == 2 else 0
        bars = ax1.bar(x + offset, r4_knn, width, label="Run 4 (format-ctrl)", color="steelblue", alpha=0.8)
        for bar, val in zip(bars, r4_knn):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.0%}", ha="center", va="bottom", fontsize=7)

    if has_r3:
        r3_knn = [run3_per_tier.get(t, {}).get("knn_accuracy_mean", 0.0) for t in tier_labels]
        offset = width / 2 if n_bars == 2 else 0
        bars = ax1.bar(x + offset, r3_knn, width, label="Run 3 (format-free)", color="coral", alpha=0.8)
        for bar, val in zip(bars, r3_knn):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.0%}", ha="center", va="bottom", fontsize=7)

    ax1.axhline(0.20, color="gray", linestyle="--", alpha=0.5, label="chance (20%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tier_labels)
    ax1.set_ylabel("kNN Accuracy (5-way)")
    ax1.set_title("Per-Tier kNN Accuracy")
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 1.05)

    # Silhouette
    if has_r4:
        r4_sil = [run4_per_tier.get(t, {}).get("silhouette_test_mean", 0.0) for t in tier_labels]
        offset = -width / 2 if n_bars == 2 else 0
        bars = ax2.bar(x + offset, r4_sil, width, label="Run 4 (format-ctrl)", color="steelblue", alpha=0.8)
        for bar, val in zip(bars, r4_sil):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    if has_r3:
        r3_sil = [run3_per_tier.get(t, {}).get("silhouette_test_mean", 0.0) for t in tier_labels]
        offset = width / 2 if n_bars == 2 else 0
        bars = ax2.bar(x + offset, r3_sil, width, label="Run 3 (format-free)", color="coral", alpha=0.8)
        for bar, val in zip(bars, r3_sil):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax2.set_xticks(x)
    ax2.set_xticklabels(tier_labels)
    ax2.set_ylabel("Silhouette (cosine, test)")
    ax2.set_title("Per-Tier Test Silhouette")
    ax2.legend(fontsize=9)

    fig.suptitle("Experiment 1.75: Tier Ablation — Contrastive MLP", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {save_path}")


def plot_interaction_heatmap(
    per_tier: dict[str, Any],
    cross_tier: dict[str, Any],
    run_label: str,
    save_path: Path,
) -> None:
    """Matrix showing combination performance vs single-tier, with interaction coloring."""
    # Build matrix: rows = combinations, cols = metrics
    combo_labels = list(cross_tier.keys())
    single_tiers = TIER_NAMES_ORDERED

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # For each combination, compute: combo kNN, best single constituent kNN, difference
    rows = []
    for combo_label in combo_labels:
        combo_knn = cross_tier[combo_label]["knn_accuracy_mean"]
        combo_sil = cross_tier[combo_label]["silhouette_test_mean"]
        constituents = combo_label.split("+")
        best_single_knn = max(
            per_tier.get(t, {}).get("knn_accuracy_mean", 0.0) for t in constituents
        )
        best_single_sil = max(
            per_tier.get(t, {}).get("silhouette_test_mean", 0.0) for t in constituents
        )
        sum_marginal_knn = sum(
            per_tier.get(t, {}).get("knn_accuracy_mean", 0.0) - 0.20  # subtract chance
            for t in constituents
        ) + 0.20  # add chance back
        rows.append({
            "combo": combo_label,
            "combo_knn": combo_knn,
            "combo_sil": combo_sil,
            "best_single_knn": best_single_knn,
            "best_single_sil": best_single_sil,
            "knn_vs_best": combo_knn - best_single_knn,
            "sil_vs_best": combo_sil - best_single_sil,
            "sum_marginal_knn": sum_marginal_knn,
            "knn_vs_sum": combo_knn - sum_marginal_knn,
        })

    # Also add combined (all tiers) for reference
    if "combined" in per_tier:
        combined_knn = per_tier["combined"]["knn_accuracy_mean"]
        combined_sil = per_tier["combined"]["silhouette_test_mean"]
        best_eng_knn = cross_tier.get("T1+T2+T2.5", {}).get("knn_accuracy_mean", 0.0)
        rows.append({
            "combo": "T1+T2+T2.5+T3 (all)",
            "combo_knn": combined_knn,
            "combo_sil": combined_sil,
            "best_single_knn": max(per_tier.get(t, {}).get("knn_accuracy_mean", 0.0)
                                   for t in single_tiers),
            "best_single_sil": max(per_tier.get(t, {}).get("silhouette_test_mean", 0.0)
                                   for t in single_tiers),
            "knn_vs_best": combined_knn - max(per_tier.get(t, {}).get("knn_accuracy_mean", 0.0)
                                               for t in single_tiers),
            "sil_vs_best": combined_sil - max(per_tier.get(t, {}).get("silhouette_test_mean", 0.0)
                                               for t in single_tiers),
            "sum_marginal_knn": 0.0,
            "knn_vs_sum": 0.0,
        })

    # Build heatmap data
    combo_names = [r["combo"] for r in rows]
    metric_names = ["kNN", "sil", "kNN-best", "sil-best"]
    data = np.array([
        [r["combo_knn"], r["combo_sil"], r["knn_vs_best"], r["sil_vs_best"]]
        for r in rows
    ])

    im = ax.imshow(data, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_yticks(range(len(combo_names)))
    ax.set_yticklabels(combo_names, fontsize=9)

    for i in range(len(combo_names)):
        for j in range(len(metric_names)):
            val = data[i, j]
            if j in (2, 3):  # delta metrics (signed)
                fmt = f"{val:+.2%}" if j == 2 else f"{val:+.3f}"
            else:  # absolute metrics
                fmt = f"{val:.2%}" if j == 0 else f"{val:.3f}"
            color = "white" if abs(val) > 0.3 else "black"
            ax.text(j, i, fmt, ha="center", va="center", fontsize=9, color=color)

    ax.set_title(f"{run_label}: Cross-Tier Interaction")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {save_path}")


def plot_weight_importance(
    importance: np.ndarray,
    feature_tiers: list[str],
    rf_top_20: list[dict[str, Any]],
    save_path: Path,
) -> None:
    """Violin/box plots of MLP weight norms grouped by tier."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: violin/box plots by tier
    tier_data = []
    tier_labels_plot = []
    for tier_name in TIER_NAMES_ORDERED:
        tier_mask = np.array([ft == tier_name for ft in feature_tiers])
        tier_data.append(importance[tier_mask])
        tier_labels_plot.append(f"{tier_name}\n(n={tier_mask.sum()})")

    bp = axes[0].boxplot(tier_data, labels=tier_labels_plot, patch_artist=True,
                         showfliers=False, widths=0.6)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay swarm-like scatter
    for i, (td, color) in enumerate(zip(tier_data, colors)):
        jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(td))
        axes[0].scatter(np.full(len(td), i + 1) + jitter, td,
                        alpha=0.15, s=5, c=color, zorder=2)

    axes[0].set_ylabel("L2 weight norm (first layer)")
    axes[0].set_title("MLP Weight Importance by Tier")

    # Right: top-20 comparison (MLP vs RF)
    # Build RF importance for comparison
    rf_importance_full = np.zeros(len(importance))
    for item in rf_top_20:
        idx = item["index"]
        if 0 <= idx < len(importance):
            rf_importance_full[idx] = item["importance"]

    # Normalize both to [0, 1] for comparison
    mlp_norm = importance / (importance.max() + 1e-10)
    rf_norm = rf_importance_full / (rf_importance_full.max() + 1e-10)

    # Scatter: each point is a feature, x=RF importance, y=MLP importance
    # Color by tier
    tier_color_map = {"T1": "#4C72B0", "T2": "#55A868", "T2.5": "#C44E52", "T3": "#8172B2"}
    for tier_name in TIER_NAMES_ORDERED:
        tier_mask = np.array([ft == tier_name for ft in feature_tiers])
        # Only plot features where at least one has nonzero importance
        relevant = tier_mask & ((mlp_norm > 0.1) | (rf_norm > 0.1))
        if relevant.sum() > 0:
            axes[1].scatter(
                rf_norm[relevant], mlp_norm[relevant],
                c=tier_color_map[tier_name], label=tier_name,
                alpha=0.6, s=20, edgecolors="k", linewidth=0.3,
            )

    axes[1].set_xlabel("RF importance (normalized)")
    axes[1].set_ylabel("MLP weight norm (normalized)")
    axes[1].set_title("MLP vs RF Feature Importance")
    axes[1].legend(fontsize=9)
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x")

    fig.suptitle("Experiment 1.75: MLP Weight Importance Analysis", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {save_path}")


def plot_top20_comparison(
    top_20_mlp: list[dict[str, Any]],
    top_20_rf: list[dict[str, Any]],
    save_path: Path,
) -> None:
    """Side-by-side ranked list of MLP vs RF top features with tier coloring."""
    tier_color_map = {"T1": "#4C72B0", "T2": "#55A868", "T2.5": "#C44E52", "T3": "#8172B2"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10))

    # MLP top-20
    mlp_names = [f["name"] for f in top_20_mlp]
    mlp_vals = [f["importance"] for f in top_20_mlp]
    mlp_colors = [tier_color_map.get(f.get("tier", "T1"), "gray") for f in top_20_mlp]
    rf_name_set = {f["name"] for f in top_20_rf}

    y_pos = np.arange(len(mlp_names))
    bars = ax1.barh(y_pos, mlp_vals, color=mlp_colors, alpha=0.8)
    ax1.set_yticks(y_pos)
    labels = []
    for name in mlp_names:
        marker = " *" if name in rf_name_set else ""
        labels.append(f"{name}{marker}")
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("L2 weight norm")
    ax1.set_title("MLP Top-20 Features")

    # RF top-20
    rf_names = [f["name"] for f in top_20_rf]
    rf_vals = [f["importance"] for f in top_20_rf]
    # Determine tier for RF features by index
    rf_colors = []
    for f in top_20_rf:
        idx = f["index"]
        tier = "unknown"
        for tn, (start, end) in TIER_RANGES.items():
            if start <= idx < end:
                tier = tn
                break
        rf_colors.append(tier_color_map.get(tier, "gray"))
    mlp_name_set = {f["name"] for f in top_20_mlp}

    y_pos2 = np.arange(len(rf_names))
    bars2 = ax2.barh(y_pos2, rf_vals, color=rf_colors, alpha=0.8)
    ax2.set_yticks(y_pos2)
    labels2 = []
    for name in rf_names:
        marker = " *" if name in mlp_name_set else ""
        labels2.append(f"{name}{marker}")
    ax2.set_yticklabels(labels2, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("RF importance")
    ax2.set_title("RF Top-20 Features (Phase 0)")

    # Legend for tier colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=t) for t, c in tier_color_map.items()]
    legend_elements.append(Patch(facecolor="white", edgecolor="black", label="* = shared"))
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=9)

    fig.suptitle("Top-20 Feature Comparison: MLP vs RF", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Tier ablation experiment")
    parser.add_argument("--skip-permutation", action="store_true",
                        help="Skip permutation tests (faster iteration)")
    parser.add_argument("--skip-interaction", action="store_true",
                        help="Skip cross-tier interaction analysis")
    parser.add_argument("--skip-importance", action="store_true",
                        help="Skip MLP weight importance analysis")
    parser.add_argument("--n-permutations", type=int, default=50,
                        help="Number of permutations per tier (default: 50)")
    parser.add_argument("--run", type=str, default="both", choices=["3", "4", "both"],
                        help="Which run to analyze (default: both)")
    args = parser.parse_args()

    t_start = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {
        "experiment": "tier_ablation",
        "pre_registered_predictions": PRE_REGISTERED,
    }

    # --- Load Phase 0 RF comparison data ---
    rf_results_path = PROJECT_ROOT / "outputs" / "runs" / "run4_format_controlled" / "results.json"
    rf_top_20: list[dict[str, Any]] = []
    rf_tier_ablation: dict[str, Any] = {}
    try:
        with open(rf_results_path) as f:
            rf_data = json.load(f)
        fi = rf_data.get("feature_importance", {})
        rf_top_20 = fi.get("top_20_features", [])
        rf_tier_ablation = fi.get("tier_ablation", {})
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load RF results: {e}")

    # --- Run 4 ---
    run4_per_tier: dict[str, Any] | None = None
    run4_cross_tier: dict[str, Any] | None = None

    if args.run in ("4", "both"):
        logger.info(f"\n{'='*60}")
        logger.info("RUN 4 (format-controlled)")
        logger.info(f"{'='*60}")

        run4 = load_run4_data()

        run4_per_tier = run_per_tier_analysis(
            run4, "Run 4",
            skip_permutation=args.skip_permutation,
            n_permutations=args.n_permutations,
        )

        run4_results: dict[str, Any] = {
            "n_samples": run4.n_samples,
            "modes": run4.mode_names,
            "per_tier": run4_per_tier,
        }

        # Cross-tier interaction
        if not args.skip_interaction:
            run4_cross_tier = run_cross_tier_analysis(run4, "Run 4")
            run4_results["cross_tier"] = run4_cross_tier

        # Weight importance
        if not args.skip_importance and rf_top_20:
            importance_results, importance_vec, feature_tiers = run_weight_importance(
                run4, rf_top_20,
            )
            run4_results["weight_importance"] = importance_results

            # Visualizations for weight importance
            plot_weight_importance(
                importance_vec, feature_tiers, rf_top_20,
                FIGURES_DIR / "weight_importance_by_tier.png",
            )
            plot_top20_comparison(
                importance_results["top_20_mlp"], rf_top_20,
                FIGURES_DIR / "top20_comparison.png",
            )

        # Phase 0 RF comparison
        if rf_tier_ablation:
            phase0_comparison: dict[str, Any] = {}
            for tier_name in ["T1", "T2", "T2.5"]:
                rf_key = tier_name.replace(".", "_")
                rf_info = rf_tier_ablation.get(rf_key, {})
                mlp_info = run4_per_tier.get(tier_name, {})
                phase0_comparison[tier_name] = {
                    "rf_accuracy": rf_info.get("accuracy_alone", RF_TIER_ABLATION.get(tier_name, {}).get("rf_accuracy")),
                    "rf_silhouette": rf_info.get("silhouette_alone", RF_TIER_ABLATION.get(tier_name, {}).get("rf_silhouette")),
                    "mlp_knn": mlp_info.get("knn_accuracy_mean"),
                    "mlp_silhouette": mlp_info.get("silhouette_test_mean"),
                }
            run4_results["phase0_rf_comparison"] = phase0_comparison

        all_results["run4"] = run4_results

    # --- Run 3 ---
    run3_per_tier: dict[str, Any] | None = None
    run3_cross_tier: dict[str, Any] | None = None

    if args.run in ("3", "both"):
        logger.info(f"\n{'='*60}")
        logger.info("RUN 3 (process-prescriptive, format-free)")
        logger.info(f"{'='*60}")

        run3 = load_run3_data()

        run3_per_tier = run_per_tier_analysis(
            run3, "Run 3",
            skip_permutation=args.skip_permutation,
            n_permutations=args.n_permutations,
        )

        run3_results: dict[str, Any] = {
            "n_samples": run3.n_samples,
            "modes": run3.mode_names,
            "per_tier": run3_per_tier,
        }

        if not args.skip_interaction:
            run3_cross_tier = run_cross_tier_analysis(run3, "Run 3")
            run3_results["cross_tier"] = run3_cross_tier

        all_results["run3"] = run3_results

    # --- Visualization ---
    plot_tier_comparison(
        run4_per_tier, run3_per_tier,
        FIGURES_DIR / "tier_comparison.png",
    )

    if run4_per_tier and run4_cross_tier:
        plot_interaction_heatmap(
            run4_per_tier, run4_cross_tier,
            "Run 4", FIGURES_DIR / "interaction_heatmap_run4.png",
        )

    if run3_per_tier and run3_cross_tier:
        plot_interaction_heatmap(
            run3_per_tier, run3_cross_tier,
            "Run 3", FIGURES_DIR / "interaction_heatmap_run3.png",
        )

    # --- Summary ---
    elapsed = time.time() - t_start
    all_results["elapsed_seconds"] = float(elapsed)

    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT 1.75 SUMMARY")
    logger.info(f"{'='*60}")

    for run_label, per_tier_data in [("Run 4", run4_per_tier), ("Run 3", run3_per_tier)]:
        if per_tier_data is None:
            continue
        logger.info(f"\n{run_label} Per-Tier Results:")
        logger.info(f"  {'Tier':<12} {'kNN':>8} {'Sil':>8}")
        logger.info(f"  {'-'*28}")
        for tier_name in TIER_NAMES_ORDERED + ["combined"]:
            info = per_tier_data.get(tier_name, {})
            knn = info.get("knn_accuracy_mean", 0.0)
            sil = info.get("silhouette_test_mean", 0.0)
            perm_p = ""
            if "permutation" in info:
                perm_p = f"  p={info['permutation']['p_value']:.4f}"
            logger.info(f"  {tier_name:<12} {knn:>7.2%} {sil:>8.4f}{perm_p}")

    if run4_per_tier:
        # Check tier ranking prediction
        tier_knns = {t: run4_per_tier.get(t, {}).get("knn_accuracy_mean", 0.0)
                     for t in TIER_NAMES_ORDERED}
        ranking = sorted(tier_knns.items(), key=lambda x: x[1], reverse=True)
        ranking_str = " > ".join(f"{t}({v:.0%})" for t, v in ranking)
        logger.info(f"\nRun 4 tier ranking: {ranking_str}")
        predicted_holds = (
            tier_knns.get("T2.5", 0) > tier_knns.get("T2", 0) >
            tier_knns.get("T1", 0)
        )
        logger.info(f"Predicted T2.5 > T2 > T1: {'CONFIRMED' if predicted_holds else 'NOT CONFIRMED'}")

    if run4_cross_tier:
        logger.info(f"\nRun 4 Cross-Tier Interaction:")
        for combo, info in run4_cross_tier.items():
            constituents = combo.split("+")
            best_single = max(
                run4_per_tier.get(t, {}).get("knn_accuracy_mean", 0.0)
                for t in constituents
            )
            combo_knn = info["knn_accuracy_mean"]
            diff = combo_knn - best_single
            logger.info(f"  {combo:<15} kNN={combo_knn:.2%} "
                        f"(vs best single {best_single:.2%}, diff={diff:+.2%})")

    if "weight_importance" in all_results.get("run4", {}):
        wi = all_results["run4"]["weight_importance"]
        logger.info(f"\nWeight Importance:")
        logger.info(f"  RF vs MLP top-20 overlap: {wi['overlap_count']}/20")
        logger.info(f"  Spearman (top-100 union): r={wi['spearman_top100_correlation']:.4f}")

    logger.info(f"\nElapsed: {elapsed:.1f}s")

    # --- Save ---
    out_path = OUTPUT_DIR / "tier_ablation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
