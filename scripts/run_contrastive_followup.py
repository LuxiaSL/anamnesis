#!/usr/bin/env python3
"""Follow-up analyses for Experiment 1 contrastive projection.

Computes two things the main script didn't:
1. 4-way (no analogical) MLP metrics — silhouette and kNN on the hard subset
2. Per-mode silhouette from CV test folds (not the inflated full-data version)

Also runs a small permutation test on the 4-way subset.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE1_DIR = PROJECT_ROOT / "outputs" / "phase1" / "contrastive_projection"

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.error("PyTorch required for this script")
    sys.exit(1)

# Import shared components from main script
from scripts.run_contrastive_projection import (
    load_features,
    _train_projection_mlp,
)


def run_mlp_with_per_mode_test_sil(
    X: np.ndarray,
    y: np.ndarray,
    mode_names: list[str],
    bottleneck_dim: int = 32,
    n_splits: int = 5,
    n_epochs: int = 200,
    label: str = "5-way",
) -> dict[str, Any]:
    """Run contrastive MLP CV and collect per-mode silhouette on TEST folds."""
    logger.info(f"=== {label} MLP with per-mode test silhouette ===")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = StandardScaler()

    # Collect all test embeddings and labels across folds
    all_test_emb: list[np.ndarray] = []
    all_test_y: list[np.ndarray] = []
    all_test_sil_samples: list[np.ndarray] = []

    fold_sils: list[float] = []
    fold_knns: list[float] = []
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

            # Per-sample silhouette on this fold's test set
            sil_samples = silhouette_samples(emb_test, y_test, metric="cosine")
            all_test_emb.append(emb_test)
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

    # Aggregate per-mode test silhouette across all folds
    all_y_arr = np.concatenate(all_test_y)
    all_sil_arr = np.concatenate(all_test_sil_samples)

    per_mode_test_sil = {}
    for i, name in enumerate(mode_names):
        mask = all_y_arr == i
        if mask.sum() > 0:
            per_mode_test_sil[name] = float(np.mean(all_sil_arr[mask]))
        else:
            per_mode_test_sil[name] = float("nan")

    # Per-mode recall
    y_test_arr = np.array(all_y_test)
    y_pred_arr = np.array(all_y_pred)
    per_mode_recall = {}
    for i, name in enumerate(mode_names):
        mask = y_test_arr == i
        if mask.sum() > 0:
            per_mode_recall[name] = float(np.mean(y_pred_arr[mask] == y_test_arr[mask]))

    results = {
        "label": label,
        "n_samples": int(X.shape[0]),
        "n_modes": len(mode_names),
        "modes": mode_names,
        "silhouette_test_mean": float(np.mean(fold_sils)) if fold_sils else 0.0,
        "silhouette_test_std": float(np.std(fold_sils)) if fold_sils else 0.0,
        "silhouette_test_per_fold": [float(s) for s in fold_sils],
        "knn_accuracy_mean": float(np.mean(fold_knns)),
        "knn_accuracy_std": float(np.std(fold_knns)),
        "knn_accuracy_per_fold": [float(k) for k in fold_knns],
        "per_mode_test_silhouette": per_mode_test_sil,
        "per_mode_recall": per_mode_recall,
    }

    logger.info(f"  {label} summary: sil_test={results['silhouette_test_mean']:.4f} "
                f"+/- {results['silhouette_test_std']:.4f}, "
                f"kNN={results['knn_accuracy_mean']:.2%}")
    logger.info(f"  Per-mode test silhouette:")
    for name, sil in sorted(per_mode_test_sil.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"    {name}: {sil:.4f}")
    logger.info(f"  Per-mode recall:")
    for name, recall in sorted(per_mode_recall.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"    {name}: {recall:.2%}")

    return results


def run_4way_permutation(
    X: np.ndarray,
    y: np.ndarray,
    observed_sil: float,
    n_permutations: int = 50,
    bottleneck_dim: int = 32,
    n_splits: int = 5,
    n_epochs: int = 200,
) -> dict[str, Any]:
    """Permutation test on the 4-way (no analogical) subset."""
    logger.info(f"=== 4-way permutation test ({n_permutations} shuffles) ===")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    precomputed_folds = list(cv.split(X, y))

    null_silhouettes: list[float] = []
    null_knn_accs: list[float] = []
    rng = np.random.RandomState(0)

    for perm_i in range(n_permutations):
        y_shuffled = rng.permutation(y)
        scaler = StandardScaler()
        fold_sils: list[float] = []
        fold_knns: list[float] = []

        for fold_i, (train_idx, test_idx) in enumerate(precomputed_folds):
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            y_train, y_test = y_shuffled[train_idx], y_shuffled[test_idx]

            emb_train, emb_test, _ = _train_projection_mlp(
                X_train, y_train, X_test,
                bottleneck_dim=bottleneck_dim, n_epochs=n_epochs,
                torch_seed=42 + fold_i,
            )

            if len(np.unique(y_test)) > 1:
                sil = silhouette_score(emb_test, y_test, metric="cosine")
                fold_sils.append(sil)

            knn = KNeighborsClassifier(n_neighbors=3, metric="cosine")
            knn.fit(emb_train, y_train)
            fold_knns.append(knn.score(emb_test, y_test))

        null_silhouettes.append(float(np.mean(fold_sils)) if fold_sils else 0.0)
        null_knn_accs.append(float(np.mean(fold_knns)) if fold_knns else 0.0)

        if (perm_i + 1) % 10 == 0:
            n_above = sum(1 for s in null_silhouettes if s >= observed_sil)
            logger.info(f"  Permutation {perm_i + 1}/{n_permutations}: "
                        f"null_sil={null_silhouettes[-1]:.4f}, "
                        f"running p={n_above}/{len(null_silhouettes)}")

    n_above = sum(1 for s in null_silhouettes if s >= observed_sil)
    p_value = (n_above + 1) / (n_permutations + 1)

    results = {
        "n_permutations": n_permutations,
        "observed_silhouette": float(observed_sil),
        "null_silhouette_mean": float(np.mean(null_silhouettes)),
        "null_silhouette_std": float(np.std(null_silhouettes)),
        "null_silhouette_max": float(np.max(null_silhouettes)),
        "null_knn_mean": float(np.mean(null_knn_accs)),
        "null_knn_std": float(np.std(null_knn_accs)),
        "n_above_observed": int(n_above),
        "p_value": float(p_value),
        "null_distribution": [float(s) for s in null_silhouettes],
    }

    logger.info(f"  4-way permutation: p={p_value:.4f} "
                f"({n_above}/{n_permutations} above observed {observed_sil:.4f})")
    logger.info(f"  Null: mean={results['null_silhouette_mean']:.4f}, "
                f"max={results['null_silhouette_max']:.4f}")

    return results


def main() -> None:
    t_start = time.time()

    X, y, mode_names, labels_str = load_features()

    all_results: dict[str, Any] = {}

    # --- 5-way with per-mode test silhouette ---
    results_5way = run_mlp_with_per_mode_test_sil(
        X, y, mode_names, label="5-way"
    )
    all_results["5way_per_mode_test_sil"] = results_5way

    # --- 4-way (no analogical) ---
    analogical_idx = mode_names.index("analogical")
    mask_no_ana = y != analogical_idx
    X_4way = X[mask_no_ana]
    labels_4way = labels_str[mask_no_ana]

    le_4way = LabelEncoder()
    y_4way = le_4way.fit_transform(labels_4way)
    mode_names_4way = le_4way.classes_.tolist()

    logger.info(f"\n4-way subset: {X_4way.shape[0]} samples, "
                f"modes: {mode_names_4way}")

    results_4way = run_mlp_with_per_mode_test_sil(
        X_4way, y_4way, mode_names_4way, label="4-way (no analogical)"
    )
    all_results["4way_no_analogical"] = results_4way

    # --- 4-way permutation test (50 shuffles — ~20 min on CPU) ---
    observed_4way_sil = results_4way["silhouette_test_mean"]
    perm_4way = run_4way_permutation(
        X_4way, y_4way, observed_4way_sil, n_permutations=50,
    )
    all_results["4way_permutation"] = perm_4way

    elapsed = time.time() - t_start
    all_results["elapsed_seconds"] = float(elapsed)

    logger.info(f"\n{'='*60}")
    logger.info("FOLLOW-UP SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"5-way per-mode test silhouette: {results_5way['per_mode_test_silhouette']}")
    logger.info(f"4-way sil_test: {results_4way['silhouette_test_mean']:.4f}")
    logger.info(f"4-way kNN: {results_4way['knn_accuracy_mean']:.2%}")
    logger.info(f"4-way permutation p: {perm_4way['p_value']:.4f}")
    logger.info(f"Elapsed: {elapsed:.1f}s")

    out_path = PHASE1_DIR / "contrastive_followup.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
