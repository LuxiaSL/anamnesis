#!/usr/bin/env python3
"""Permutation test for Run 4 RF accuracy.

Tests whether 70% 5-way RF accuracy is significantly above chance given
587 features on 100 samples. Shuffles mode labels 1000 times and builds
a null distribution of RF accuracy under broken mode-feature association.

Usage:
    ANAMNESIS_RUN_NAME=run4_format_controlled python scripts/run_permutation_test.py
    ANAMNESIS_RUN_NAME=run4_format_controlled python scripts/run_permutation_test.py --n-permutations 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import OUTPUTS_DIR, FIGURES_DIR, ExperimentConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load Set A+B features and mode labels.

    Returns:
        (X_scaled, y, mode_names) where X_scaled is StandardScaler-transformed.
    """
    config = ExperimentConfig()

    with open(config.metadata_path) as f:
        raw = json.load(f)
    all_metadata: list[dict] = (
        raw["generations"] if isinstance(raw, dict) and "generations" in raw else raw
    )

    ab_meta = [m for m in all_metadata if m["prompt_set"] in ("A", "B")]

    gen_ids: list[int] = []
    features_list: list[np.ndarray] = []
    mode_labels: list[str] = []

    for m in ab_meta:
        gid = m["generation_id"]
        sig_path = config.signatures_dir / f"gen_{gid:03d}.npz"
        if not sig_path.exists():
            continue
        data = np.load(sig_path, allow_pickle=True)
        gen_ids.append(gid)
        features_list.append(data["features"])
        mode_labels.append(m["mode"])

    n = len(gen_ids)
    logger.info(f"Loaded {n} Set A+B samples, modes: {sorted(set(mode_labels))}")

    X = np.stack(features_list)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    le = LabelEncoder()
    y = le.fit_transform(mode_labels)
    mode_names = le.classes_.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, mode_names


def run_permutation_test(
    X_scaled: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 1000,
) -> tuple[dict, np.ndarray]:
    """Run permutation test on RF classification.

    Returns:
        (results_dict, null_accuracies_array)
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rng = np.random.RandomState(42)

    # Observed accuracy
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    observed_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring="accuracy")
    observed_acc = float(observed_scores.mean())
    logger.info(f"Observed RF accuracy: {observed_acc:.4f}")

    # Null distribution
    null_accuracies: list[float] = []
    t_start = time.time()

    for i in range(n_permutations):
        y_shuffled = rng.permutation(y)

        cv_perm = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rf_perm = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

        try:
            perm_scores = cross_val_score(
                rf_perm, X_scaled, y_shuffled, cv=cv_perm, scoring="accuracy"
            )
            null_accuracies.append(float(perm_scores.mean()))
        except Exception as e:
            logger.warning(f"Permutation {i} failed: {e}")
            continue

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            remaining = (n_permutations - i - 1) / rate
            logger.info(
                f"  Permutation {i + 1}/{n_permutations} "
                f"(null mean so far: {np.mean(null_accuracies):.4f}, "
                f"~{remaining:.0f}s remaining)"
            )

    null_arr = np.array(null_accuracies)
    elapsed_total = time.time() - t_start

    # Compute p-value: proportion of null >= observed
    n_above = int(np.sum(null_arr >= observed_acc))
    p_value = (n_above + 1) / (len(null_arr) + 1)  # +1 for continuity correction

    # Percentile of observed in null
    percentile = float(np.mean(null_arr < observed_acc) * 100)

    results = {
        "observed_accuracy": observed_acc,
        "observed_fold_scores": observed_scores.tolist(),
        "n_permutations": len(null_accuracies),
        "n_permutations_requested": n_permutations,
        "null_mean": float(null_arr.mean()),
        "null_std": float(null_arr.std()),
        "null_min": float(null_arr.min()),
        "null_max": float(null_arr.max()),
        "null_median": float(np.median(null_arr)),
        "null_p95": float(np.percentile(null_arr, 95)),
        "null_p99": float(np.percentile(null_arr, 99)),
        "p_value": p_value,
        "n_above_observed": n_above,
        "observed_percentile": percentile,
        "n_samples": X_scaled.shape[0],
        "n_features": X_scaled.shape[1],
        "n_classes": len(set(y)),
        "elapsed_seconds": round(elapsed_total, 1),
    }

    logger.info(f"Null distribution: {null_arr.mean():.4f} +/- {null_arr.std():.4f}")
    logger.info(f"Null range: [{null_arr.min():.4f}, {null_arr.max():.4f}]")
    logger.info(f"Null 95th percentile: {np.percentile(null_arr, 95):.4f}")
    logger.info(f"Null 99th percentile: {np.percentile(null_arr, 99):.4f}")
    logger.info(f"Observed {observed_acc:.4f} is at {percentile:.1f}th percentile of null")
    logger.info(f"p-value: {p_value:.4f}")

    return results, null_arr


def plot_null_distribution(
    null_accuracies: np.ndarray,
    observed_acc: float,
    p_value: float,
    n_samples: int,
    n_features: int,
    n_classes: int,
    output_path: Path,
) -> None:
    """Plot histogram of null distribution with observed accuracy line."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.hist(
        null_accuracies,
        bins=50,
        alpha=0.7,
        color="steelblue",
        edgecolor="white",
        label=f"Null distribution (n={len(null_accuracies)})",
    )
    ax.axvline(
        observed_acc,
        color="red",
        linewidth=2,
        linestyle="--",
        label=f"Observed: {observed_acc:.1%}",
    )

    # Chance level
    ax.axvline(
        1.0 / n_classes,
        color="gray",
        linewidth=1,
        linestyle=":",
        label=f"Chance: {1.0 / n_classes:.1%}",
    )

    ax.set_xlabel("RF Accuracy (5-fold CV)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Permutation Test: Observed {observed_acc:.1%} vs Null Distribution\n"
        f"p = {p_value:.4f}",
        fontsize=13,
    )
    ax.legend(fontsize=11)

    # Text annotation
    null_mean = float(np.mean(null_accuracies))
    null_std = float(np.std(null_accuracies))
    z_score = (observed_acc - null_mean) / null_std if null_std > 0 else float("inf")
    ax.text(
        0.97, 0.95,
        f"Null: {null_mean:.1%} +/- {null_std:.1%}\n"
        f"z = {z_score:.1f}\n"
        f"n_samples = {n_samples}, n_features = {n_features}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved permutation test plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Permutation test for RF accuracy")
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Number of permutation iterations (default: 1000)",
    )
    args = parser.parse_args()

    # Load data
    X_scaled, y, mode_names = load_data()
    logger.info(f"Data: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features, {len(mode_names)} classes")

    # Run permutation test (returns both summary and raw null array)
    results, null_arr = run_permutation_test(X_scaled, y, n_permutations=args.n_permutations)
    results["mode_names"] = mode_names

    # Save results
    output_path = OUTPUTS_DIR / "permutation_test.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {output_path}")

    # Plot using the null array we already computed
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = FIGURES_DIR / "permutation_test.png"
    plot_null_distribution(
        null_arr,
        results["observed_accuracy"],
        results["p_value"],
        results["n_samples"],
        results["n_features"],
        results["n_classes"],
        plot_path,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PERMUTATION TEST RESULTS")
    print("=" * 60)
    print(f"Samples: {results['n_samples']}, Features: {results['n_features']}, Classes: {results['n_classes']}")
    print(f"Observed RF accuracy: {results['observed_accuracy']:.1%}")
    print(f"Null distribution: {results['null_mean']:.1%} +/- {results['null_std']:.1%}")
    print(f"Null 95th percentile: {results['null_p95']:.1%}")
    print(f"Null 99th percentile: {results['null_p99']:.1%}")
    print(f"Observed at {results['observed_percentile']:.1f}th percentile")
    print(f"p-value: {results['p_value']:.4f}")
    print(f"Permutations: {results['n_permutations']}")
    print(f"Elapsed: {results['elapsed_seconds']}s")

    if results["p_value"] < 0.01:
        print("\nCONCLUSION: Observed accuracy is SIGNIFICANTLY above chance (p < 0.01)")
    elif results["p_value"] < 0.05:
        print("\nCONCLUSION: Observed accuracy is SIGNIFICANTLY above chance (p < 0.05)")
    else:
        print("\nCONCLUSION: Observed accuracy is NOT significantly above chance")
        print("  The feature-to-sample ratio concern may be validated.")


if __name__ == "__main__":
    main()
