#!/usr/bin/env python3
"""
CV stability test: compute RF accuracy across 100 different CV random seeds.

Produces a distribution of observed accuracies to calibrate the true effect size,
addressing the 60% (permutation test) vs 70% (main analysis) discrepancy.

Usage:
    ANAMNESIS_RUN_NAME=run4_format_controlled python scripts/run_cv_stability.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Setup ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ExperimentConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

N_SEEDS = 100


def load_data(config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load Set A+B features and mode labels."""
    meta_path = config.metadata_path
    with open(meta_path) as f:
        raw = json.load(f)
    metadata = raw["generations"]

    # Filter to Set A+B
    ab_gens = [m for m in metadata if m["prompt_set"] in ("A", "B")]
    logger.info(f"Loaded {len(ab_gens)} Set A+B samples")

    features_list: list[np.ndarray] = []
    labels: list[str] = []

    for m in ab_gens:
        gid = m["generation_id"]
        sig_path = config.signatures_dir / f"gen_{gid:03d}.npz"
        if not sig_path.exists():
            logger.warning(f"Missing signature: {sig_path}")
            continue

        data = np.load(sig_path, allow_pickle=True)
        if "features" not in data:
            logger.warning(f"No 'features' key in {sig_path}")
            continue

        features_list.append(data["features"].flatten())
        labels.append(m["mode"])

    X = np.array(features_list)
    mode_names = sorted(set(labels))
    logger.info(f"Data: {X.shape[0]} samples, {X.shape[1]} features, {len(mode_names)} classes")
    logger.info(f"Modes: {mode_names}")

    return X, np.array(labels), mode_names


def main() -> None:
    config = ExperimentConfig()
    X, y_str, mode_names = load_data(config)

    le = LabelEncoder()
    y = le.fit_transform(y_str)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_classes = len(mode_names)
    logger.info(f"Running {N_SEEDS} CV iterations on {X_scaled.shape[0]} samples, "
                f"{X_scaled.shape[1]} features, {n_classes} classes")

    # Also run with shuffle=False (integer cv) to reproduce main analysis
    rf_noshuffle = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    noshuffle_scores = cross_val_score(rf_noshuffle, X_scaled, y, cv=5, scoring="accuracy")
    noshuffle_acc = float(noshuffle_scores.mean())
    logger.info(f"No-shuffle (main analysis style): {noshuffle_acc:.4f} "
                f"(folds: {[f'{s:.2f}' for s in noshuffle_scores]})")

    # Run with 100 different seeds
    accuracies: list[float] = []
    fold_scores_all: list[list[float]] = []

    for seed in range(N_SEEDS):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring="accuracy")
        acc = float(scores.mean())
        accuracies.append(acc)
        fold_scores_all.append([float(s) for s in scores])

        if (seed + 1) % 20 == 0:
            logger.info(f"  Seed {seed + 1}/{N_SEEDS}: running mean = {np.mean(accuracies):.4f}")

    accs = np.array(accuracies)

    results = {
        "n_seeds": N_SEEDS,
        "n_samples": int(X_scaled.shape[0]),
        "n_features": int(X_scaled.shape[1]),
        "n_classes": n_classes,
        "mode_names": mode_names,
        "noshuffle_accuracy": noshuffle_acc,
        "noshuffle_fold_scores": [float(s) for s in noshuffle_scores],
        "mean": float(accs.mean()),
        "std": float(accs.std()),
        "median": float(np.median(accs)),
        "min": float(accs.min()),
        "max": float(accs.max()),
        "p5": float(np.percentile(accs, 5)),
        "p25": float(np.percentile(accs, 25)),
        "p75": float(np.percentile(accs, 75)),
        "p95": float(np.percentile(accs, 95)),
        "ci_95_lower": float(np.percentile(accs, 2.5)),
        "ci_95_upper": float(np.percentile(accs, 97.5)),
        "all_accuracies": [float(a) for a in accs],
    }

    out_path = config.outputs_dir / "cv_stability.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {out_path}")

    print("\n" + "=" * 60)
    print("CV STABILITY TEST RESULTS")
    print("=" * 60)
    print(f"Samples: {X_scaled.shape[0]}, Features: {X_scaled.shape[1]}, Classes: {n_classes}")
    print(f"\nNo-shuffle CV (main analysis style): {noshuffle_acc:.1%}")
    print(f"  Fold scores: {[f'{s:.2f}' for s in noshuffle_scores]}")
    print(f"\n{N_SEEDS}-seed shuffled CV distribution:")
    print(f"  Mean:   {accs.mean():.1%}")
    print(f"  Median: {np.median(accs):.1%}")
    print(f"  Std:    {accs.std():.1%}")
    print(f"  Range:  [{accs.min():.1%}, {accs.max():.1%}]")
    print(f"  95% CI: [{np.percentile(accs, 2.5):.1%}, {np.percentile(accs, 97.5):.1%}]")
    print(f"  IQR:    [{np.percentile(accs, 25):.1%}, {np.percentile(accs, 75):.1%}]")


if __name__ == "__main__":
    main()
