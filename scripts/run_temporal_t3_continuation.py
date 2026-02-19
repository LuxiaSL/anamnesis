#!/usr/bin/env python3
"""Continuation: run T3 tier analysis only, merge into focused_partial.json.

T1 and T2.5 are already saved. This picks up where the interrupted run left off.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

# Import shared functions from the focused script
from run_temporal_prediction_focused import (
    build_trajectory_indices,
    extract_tier_early_late,
    compute_semantic_embeddings,
    run_tier_permutation_test,
    run_tier_cross_gen,
    run_tier_residual,
)
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase05" / "temporal_prediction"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    t_start = time.time()

    # Load existing partial results
    partial_path = OUTPUT_DIR / "focused_partial.json"
    with open(partial_path) as f:
        all_results = json.load(f)

    logger.info("Loaded partial results with T1 and T2.5 complete.")

    # Load data
    logger.info("Loading data...")
    data = np.load(PROJECT_ROOT / "review_pack" / "features.npz", allow_pickle=True)
    X_full = data["features"].astype(np.float64)
    X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
    feature_names = data["feature_names"]
    labels_str = data["labels"]
    topics = data["topics"]

    le = LabelEncoder()
    y = le.fit_transform(labels_str)

    unique_topics = np.unique(topics)
    topic_to_id = {t: i for i, t in enumerate(unique_topics)}
    groups = np.array([topic_to_id[t] for t in topics])

    # Parse trajectory structure
    traj_indices = build_trajectory_indices(feature_names)

    # Extract T3
    result = extract_tier_early_late(X_full, feature_names, traj_indices, "T3")
    if result is None:
        logger.error("No T3 trajectory features found!")
        return
    tier_early, tier_late, matched_names = result
    logger.info(f"T3: {tier_early.shape[1]} matched features")

    # Semantic embeddings
    metadata_path = (
        PROJECT_ROOT / "outputs" / "runs" / "run4_format_controlled" / "metadata.json"
    )
    sem_early, _ = compute_semantic_embeddings(metadata_path)

    # Get observed delta
    observed_delta = all_results["observed_deltas_from_main"]["T3"]
    logger.info(f"Observed T3 delta: {observed_delta:.4f}")

    # Run analyses
    logger.info(f"\n{'='*70}")
    logger.info("TIER: T3 (500 features)")
    logger.info(f"{'='*70}")

    tier_results: dict = {
        "n_features": int(tier_early.shape[1]),
        "matched_feature_names": matched_names[:20],  # truncate for readability
        "matched_feature_count": len(matched_names),
    }

    # Permutation test
    perm = run_tier_permutation_test(
        tier_early, tier_late, sem_early, groups,
        observed_delta=observed_delta,
        tier_name="T3",
        n_permutations=200,
    )
    null_deltas = perm.pop("null_deltas")
    tier_results["permutation_test"] = perm
    np.save(OUTPUT_DIR / "null_deltas_T3.npy", np.array(null_deltas))

    # Cross-generation control
    cross_gen = run_tier_cross_gen(tier_early, tier_late, y, groups, "T3")
    tier_results["cross_generation_control"] = cross_gen

    # Within-mode residual
    residual = run_tier_residual(tier_early, tier_late, y, groups, "T3")
    tier_results["within_mode_residual"] = residual

    # Merge into results
    all_results["T3"] = tier_results
    elapsed = time.time() - t_start
    all_results["t3_elapsed_seconds"] = float(elapsed)

    # Save
    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        if isinstance(o, list):
            return [convert(v) for v in o]
        return o

    out_path = OUTPUT_DIR / "focused_results.json"
    with open(out_path, "w") as f:
        json.dump(convert(all_results), f, indent=2, default=str)
    logger.info(f"\nResults saved to {out_path}")

    # Also update partial
    with open(partial_path, "w") as f:
        json.dump(convert(all_results), f, indent=2, default=str)

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("T3 RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"Permutation: delta={observed_delta:+.3f}, p={perm['p_value']:.4f}, null_mean={perm['null_mean']:+.3f}")
    logger.info(f"Cross-gen: within={cross_gen['within_gen_median_r2']:+.3f}, cross={cross_gen['cross_gen_median_r2']:+.3f}")
    logger.info(f"Residual: median R²={residual['median_r2']:+.3f}")
    logger.info(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
