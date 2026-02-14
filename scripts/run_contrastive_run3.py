#!/usr/bin/env python3
"""Contrastive projection on Run 3 (process-prescriptive modes) features.

Contextualizes Run 4 results by testing the same MLP architecture on modes
with genuine format variation and distinctive processing strategies.

Run 3 modes: structured, associative, deliberative, compressed, pedagogical
Run 3 4-way (no compressed): 91.7% RF accuracy

If Run 3 4-way MLP silhouette >> Run 4's 0.047, that empirically demonstrates
Run 4 is the floor (hardest case), not the ceiling (general capability).

Usage:
    python scripts/run_contrastive_run3.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase1" / "contrastive_projection"

try:
    import torch
except ImportError:
    logger.error("PyTorch required")
    sys.exit(1)

from scripts.run_contrastive_projection import (
    _train_projection_mlp,
    run_lda_projection,
    per_mode_silhouette,
    plot_projections,
    plot_comparison,
)


def load_run3_features() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Load Run 3 Set A+B features from signatures."""
    run_dir = PROJECT_ROOT / "outputs" / "runs" / "run3_process_modes"
    with open(run_dir / "metadata.json") as f:
        raw = json.load(f)
    all_meta = raw["generations"] if isinstance(raw, dict) and "generations" in raw else raw
    ab_meta = [m for m in all_meta if m["prompt_set"] in ("A", "B")]

    features_list, mode_labels = [], []
    sig_dir = run_dir / "signatures"
    for m in ab_meta:
        gid = m["generation_id"]
        sig_path = sig_dir / f"gen_{gid:03d}.npz"
        if not sig_path.exists():
            logger.warning(f"Missing signature: {sig_path}")
            continue
        sig = np.load(sig_path, allow_pickle=True)
        features_list.append(sig["features"])
        mode_labels.append(m["mode"])

    X = np.stack(features_list).astype(np.float64)
    labels_str = np.array(mode_labels)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    le = LabelEncoder()
    y = le.fit_transform(labels_str)
    mode_names = le.classes_.tolist()

    logger.info(f"Loaded {X.shape[0]} samples, {X.shape[1]} features, modes: {mode_names}")
    return X, y, mode_names, labels_str


def run_mlp_full(
    X: np.ndarray,
    y: np.ndarray,
    mode_names: list[str],
    label: str,
    bottleneck_dim: int = 32,
    n_splits: int = 5,
    n_epochs: int = 200,
) -> dict[str, Any]:
    """Run contrastive MLP with full diagnostics: per-mode test silhouette,
    confusion matrix, per-mode recall."""
    logger.info(f"=== {label}: Contrastive MLP ({bottleneck_dim}d) ===")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
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

    # Per-mode test silhouette
    all_y_arr = np.concatenate(all_test_y)
    all_sil_arr = np.concatenate(all_test_sil_samples)
    per_mode_test_sil = {}
    for i, name in enumerate(mode_names):
        mask = all_y_arr == i
        if mask.sum() > 0:
            per_mode_test_sil[name] = float(np.mean(all_sil_arr[mask]))

    # Confusion matrix and recall
    y_test_arr = np.array(all_y_test)
    y_pred_arr = np.array(all_y_pred)
    n_classes = len(mode_names)
    cm = confusion_matrix(y_test_arr, y_pred_arr, labels=list(range(n_classes)))
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
        "confusion_matrix": cm.tolist(),
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


def run_permutation(
    X: np.ndarray,
    y: np.ndarray,
    observed_sil: float,
    label: str,
    n_permutations: int = 50,
    bottleneck_dim: int = 32,
    n_splits: int = 5,
    n_epochs: int = 200,
) -> dict[str, Any]:
    """Permutation test."""
    logger.info(f"=== {label}: Permutation test ({n_permutations} shuffles) ===")

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
        "label": label,
        "n_permutations": n_permutations,
        "observed_silhouette": float(observed_sil),
        "null_silhouette_mean": float(np.mean(null_silhouettes)),
        "null_silhouette_std": float(np.std(null_silhouettes)),
        "null_silhouette_max": float(np.max(null_silhouettes)),
        "null_knn_mean": float(np.mean(null_knn_accs)),
        "n_above_observed": int(n_above),
        "p_value": float(p_value),
        "null_distribution": [float(s) for s in null_silhouettes],
    }

    logger.info(f"  {label} permutation: p={p_value:.4f} "
                f"({n_above}/{n_permutations} above observed {observed_sil:.4f})")

    return results


def run_cv_stability(
    X: np.ndarray,
    y: np.ndarray,
    label: str,
    n_seeds: int = 30,
    bottleneck_dim: int = 32,
    n_splits: int = 5,
    n_epochs: int = 200,
) -> dict[str, Any]:
    """CV stability analysis across seeds."""
    logger.info(f"=== {label}: CV stability ({n_seeds} seeds) ===")

    seed_silhouettes: list[float] = []
    seed_knn_accs: list[float] = []

    for seed_i in range(n_seeds):
        cv_seed = seed_i * 7 + 13
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)
        scaler = StandardScaler()
        fold_sils: list[float] = []
        fold_knns: list[float] = []

        for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            y_train, y_test = y[train_idx], y[test_idx]

            emb_train, emb_test, _ = _train_projection_mlp(
                X_train, y_train, X_test,
                bottleneck_dim=bottleneck_dim, n_epochs=n_epochs,
                torch_seed=cv_seed + fold_i,
            )

            if len(np.unique(y_test)) > 1:
                sil = silhouette_score(emb_test, y_test, metric="cosine")
                fold_sils.append(sil)

            knn = KNeighborsClassifier(n_neighbors=3, metric="cosine")
            knn.fit(emb_train, y_train)
            fold_knns.append(knn.score(emb_test, y_test))

        seed_silhouettes.append(float(np.mean(fold_sils)) if fold_sils else 0.0)
        seed_knn_accs.append(float(np.mean(fold_knns)) if fold_knns else 0.0)

        if (seed_i + 1) % 10 == 0:
            logger.info(f"  Seed {seed_i + 1}/{n_seeds}: "
                        f"running_median_sil={np.median(seed_silhouettes):.4f}, "
                        f"running_median_knn={np.median(seed_knn_accs):.2%}")

    sil_arr = np.array(seed_silhouettes)
    knn_arr = np.array(seed_knn_accs)

    results = {
        "label": label,
        "n_seeds": n_seeds,
        "silhouette_mean": float(np.mean(sil_arr)),
        "silhouette_median": float(np.median(sil_arr)),
        "silhouette_std": float(np.std(sil_arr)),
        "silhouette_ci95": [float(np.percentile(sil_arr, 2.5)),
                            float(np.percentile(sil_arr, 97.5))],
        "knn_mean": float(np.mean(knn_arr)),
        "knn_median": float(np.median(knn_arr)),
        "knn_std": float(np.std(knn_arr)),
        "knn_ci95": [float(np.percentile(knn_arr, 2.5)),
                     float(np.percentile(knn_arr, 97.5))],
    }

    logger.info(f"  {label} CV stability: sil median={results['silhouette_median']:.4f}, "
                f"95% CI=[{results['silhouette_ci95'][0]:.4f}, "
                f"{results['silhouette_ci95'][1]:.4f}]")
    logger.info(f"  {label} CV stability: kNN median={results['knn_median']:.2%}, "
                f"95% CI=[{results['knn_ci95'][0]:.2%}, {results['knn_ci95'][1]:.2%}]")

    return results


def main() -> None:
    t_start = time.time()

    X, y, mode_names, labels_str = load_run3_features()
    all_results: dict[str, Any] = {
        "experiment": "contrastive_projection_run3",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "modes": mode_names,
        "context": "Run 3 (process-prescriptive, format-free) vs Run 4 (format-controlled). "
                   "Tests whether contrastive projection's capability scales with mode distinctiveness.",
    }

    # --- Baseline ---
    X_scaled = StandardScaler().fit_transform(X)
    sil_raw = silhouette_score(X_scaled, y, metric="cosine")
    logger.info(f"Baseline silhouette (raw, cosine): {sil_raw:.4f}")
    all_results["baseline_silhouette"] = float(sil_raw)

    # --- LDA for comparison ---
    lda_results, X_proj_lda = run_lda_projection(X, y)
    lda_results["per_mode"] = per_mode_silhouette(X_proj_lda, y, mode_names)
    all_results["lda"] = lda_results

    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_projections(X_proj_lda, y, mode_names,
                     f"Run 3 LDA Projection (sil={lda_results['silhouette_full']:.3f})",
                     fig_dir / "run3_lda.png")

    # --- 5-way MLP ---
    results_5way = run_mlp_full(X, y, mode_names, label="Run 3 5-way")
    all_results["5way"] = results_5way

    # --- 5-way permutation ---
    perm_5way = run_permutation(
        X, y, results_5way["silhouette_test_mean"],
        label="Run 3 5-way", n_permutations=50,
    )
    all_results["5way_permutation"] = perm_5way

    # --- 5-way CV stability ---
    stability_5way = run_cv_stability(X, y, label="Run 3 5-way", n_seeds=30)
    all_results["5way_cv_stability"] = stability_5way

    # --- 4-way (no compressed) ---
    compressed_idx = mode_names.index("compressed")
    mask_no_comp = y != compressed_idx
    X_4way = X[mask_no_comp]
    labels_4way = labels_str[mask_no_comp]

    le_4way = LabelEncoder()
    y_4way = le_4way.fit_transform(labels_4way)
    mode_names_4way = le_4way.classes_.tolist()

    logger.info(f"\n4-way subset: {X_4way.shape[0]} samples, modes: {mode_names_4way}")

    results_4way = run_mlp_full(X_4way, y_4way, mode_names_4way, label="Run 3 4-way (no compressed)")
    all_results["4way_no_compressed"] = results_4way

    # --- 4-way permutation ---
    perm_4way = run_permutation(
        X_4way, y_4way, results_4way["silhouette_test_mean"],
        label="Run 3 4-way", n_permutations=50,
    )
    all_results["4way_permutation"] = perm_4way

    # --- Summary ---
    elapsed = time.time() - t_start
    all_results["elapsed_seconds"] = float(elapsed)

    logger.info(f"\n{'='*60}")
    logger.info("RUN 3 CONTRASTIVE PROJECTION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Baseline silhouette:     {sil_raw:.4f}")
    logger.info(f"LDA sil (full/test):     {lda_results['silhouette_full']:.4f} / "
                f"{lda_results['silhouette_test_mean']:.4f}")
    logger.info(f"5-way MLP sil (test):    {results_5way['silhouette_test_mean']:.4f}")
    logger.info(f"5-way MLP kNN:           {results_5way['knn_accuracy_mean']:.2%}")
    logger.info(f"5-way permutation p:     {perm_5way['p_value']:.4f}")
    logger.info(f"5-way CV sil median:     {stability_5way['silhouette_median']:.4f}")
    logger.info(f"5-way CV kNN median:     {stability_5way['knn_median']:.2%}")
    logger.info(f"4-way MLP sil (test):    {results_4way['silhouette_test_mean']:.4f}")
    logger.info(f"4-way MLP kNN:           {results_4way['knn_accuracy_mean']:.2%}")
    logger.info(f"4-way permutation p:     {perm_4way['p_value']:.4f}")
    logger.info(f"Elapsed:                 {elapsed:.1f}s")

    logger.info(f"\nCross-run comparison:")
    logger.info(f"  {'Metric':<30} {'Run 3':>10} {'Run 4':>10}")
    logger.info(f"  {'5-way MLP sil':<30} {results_5way['silhouette_test_mean']:>10.4f} {'0.1755':>10}")
    logger.info(f"  {'5-way MLP kNN':<30} {results_5way['knn_accuracy_mean']:>10.2%} {'63.00%':>10}")
    logger.info(f"  {'4-way MLP sil':<30} {results_4way['silhouette_test_mean']:>10.4f} {'0.0471':>10}")
    logger.info(f"  {'4-way MLP kNN':<30} {results_4way['knn_accuracy_mean']:>10.2%} {'55.00%':>10}")
    logger.info(f"  {'5-way perm p':<30} {perm_5way['p_value']:>10.4f} {'0.0099':>10}")
    logger.info(f"  {'4-way perm p':<30} {perm_4way['p_value']:>10.4f} {'0.0196':>10}")

    out_path = OUTPUT_DIR / "contrastive_run3.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
