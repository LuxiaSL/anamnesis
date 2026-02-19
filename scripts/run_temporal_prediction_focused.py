#!/usr/bin/env python3
"""Focused per-tier follow-up to the temporal prediction experiment.

The main experiment (run_temporal_prediction.py) revealed that overall Ridge
prediction is fatally underdetermined (583 features on 80 training samples).
The per-tier decomposition, motivated by the existing tier structure central
to the paper, tells a different story:

  T2.5 delta: +0.231  (KV cache self-predicts better than semantics predicts it)
  T1 delta:   +0.160  (activation norms same direction)
  T3 delta:   -0.100  (PCA features: semantics predicts better — content signal)

This script runs targeted follow-ups at the tier level where the feature-to-
sample ratio is tractable:
  - T2.5: 21 features on 80 samples (4:1 ratio)
  - T1:   62 features on 80 samples (1.3:1 ratio — still tight)
  - T3:   500 features on 80 samples (0.16:1 — still underdetermined, reported for completeness)

Analyses:
  1. Per-tier permutation test on delta (200 shuffles)
  2. Per-tier cross-generation control (LOO mode centroid)
  3. Per-tier within-mode residual prediction

Framing: This is a pilot constrained by data availability. Per-tier decomposition
is pre-motivated by the existing analytical framework (Phase 0.5 tier ablation),
not selected post-hoc. Results inform pipeline design for future temporal feature
extraction, not a paper capstone.

Usage:
    python scripts/run_temporal_prediction_focused.py
    python scripts/run_temporal_prediction_focused.py --n-permutations 500
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
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase05" / "temporal_prediction"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Tier boundaries in the 1837-dim feature vector
TIER_SLICES = {
    "T1": (0, 221),
    "T2": (221, 442),
    "T2.5": (442, 587),
    "T3": (587, 1837),
}


# ---------------------------------------------------------------------------
# Feature parsing (shared with main script)
# ---------------------------------------------------------------------------

def _feature_base_name(name: str) -> str:
    """Strip the trajectory index from a feature name to get its base identity."""
    base = re.sub(r"_?traj\d+_?", "_", name).strip("_")
    base = re.sub(r"_t\d+_", "_", base)
    return base


def build_trajectory_indices(
    feature_names: np.ndarray,
) -> dict[str, dict[str, list[int]]]:
    """Parse feature names to identify trajectory-indexed features by tier."""
    result: dict[str, dict[str, list[int]]] = {}
    for tier_name, (start, end) in TIER_SLICES.items():
        tier_traj: dict[str, list[int]] = {}
        for i in range(start, end):
            name = str(feature_names[i])
            m = re.search(r"traj(\d+)", name)
            if not m:
                m = re.search(r"_t(\d+)_", name)
            if m:
                t = int(m.group(1))
                key = f"traj{t}"
                tier_traj.setdefault(key, []).append(i)
        result[tier_name] = tier_traj
    return result


def extract_tier_early_late(
    X: np.ndarray,
    feature_names: np.ndarray,
    traj_indices: dict[str, dict[str, list[int]]],
    tier_name: str,
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Extract matched early/late features for a single tier.

    Returns (X_early_tier, X_late_tier, matched_names) or None if no trajectory
    features exist for this tier.
    """
    tier_data = traj_indices.get(tier_name, {})
    if not tier_data:
        return None

    available_traj = sorted(tier_data.keys())
    max_traj = max(int(k.replace("traj", "")) for k in available_traj)

    if max_traj >= 4:
        pairs = [("traj0", "traj3"), ("traj1", "traj4")]
    elif max_traj >= 3:
        pairs = [("traj0", "traj2"), ("traj1", "traj3")]
    else:
        return None

    early_idx: list[int] = []
    late_idx: list[int] = []
    matched_names: list[str] = []

    for early_key, late_key in pairs:
        e_indices = tier_data.get(early_key, [])
        l_indices = tier_data.get(late_key, [])

        e_by_base = {_feature_base_name(str(feature_names[idx])): idx for idx in e_indices}
        l_by_base = {_feature_base_name(str(feature_names[idx])): idx for idx in l_indices}

        common_bases = sorted(set(e_by_base) & set(l_by_base))
        for base in common_bases:
            early_idx.append(e_by_base[base])
            late_idx.append(l_by_base[base])
            matched_names.append(f"{tier_name}:{base}")

    if not early_idx:
        return None

    return X[:, early_idx], X[:, late_idx], matched_names


# ---------------------------------------------------------------------------
# Ridge prediction (shared logic)
# ---------------------------------------------------------------------------

def ridge_predict_cv(
    X_pred: np.ndarray,
    Y_target: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Ridge regression with GroupKFold: predict all target features jointly.

    Uses multi-output Ridge (single matrix solve per fold) instead of
    per-target fitting. Dramatically faster for permutation tests.
    Alpha is fixed (not CV-tuned) for speed — the relative comparison
    between conditions is what matters, not absolute R².
    """
    from sklearn.linear_model import Ridge

    gkf = GroupKFold(n_splits=n_splits)
    n_targets = Y_target.shape[1]

    fold_r2s: list[np.ndarray] = []

    for train_idx, test_idx in gkf.split(X_pred, groups=groups):
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        X_train = scaler_x.fit_transform(X_pred[train_idx])
        X_test = scaler_x.transform(X_pred[test_idx])
        Y_train = scaler_y.fit_transform(Y_target[train_idx])
        Y_test = scaler_y.transform(Y_target[test_idx])

        try:
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_train, Y_train)  # multi-output: one matrix solve
            Y_pred = ridge.predict(X_test)

            # Per-feature R²
            feature_r2s = np.array([
                r2_score(Y_test[:, j], Y_pred[:, j])
                if np.std(Y_test[:, j]) > 1e-10 else 0.0
                for j in range(n_targets)
            ])
        except Exception:
            feature_r2s = np.zeros(n_targets)

        fold_r2s.append(feature_r2s)

    r2_matrix = np.stack(fold_r2s)
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
        "per_feature_r2": mean_r2_per_feature.tolist(),
    }


# ---------------------------------------------------------------------------
# Semantic embeddings
# ---------------------------------------------------------------------------

def compute_semantic_embeddings(metadata_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Split generated text at midpoint, embed each half."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence-transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

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
# Analysis 1: Per-tier permutation test on delta
# ---------------------------------------------------------------------------

def run_tier_permutation_test(
    tier_early: np.ndarray,
    tier_late: np.ndarray,
    sem_early: np.ndarray,
    groups: np.ndarray,
    observed_delta: float,
    tier_name: str,
    n_permutations: int = 200,
) -> dict[str, Any]:
    """Permutation test: shuffle early↔late pairing, recompute tier delta.

    Under null (no temporal structure), both compute→compute and semantic→compute
    should be equally poor with broken pairing, so delta ≈ 0.

    Shuffles the sample order for late targets, keeping early predictors fixed.
    Both Task 1 and Task 2 are recomputed under each shuffle.
    """
    logger.info(f"  Permutation test for {tier_name} ({n_permutations} shuffles)...")

    rng = np.random.RandomState(42)
    null_deltas: list[float] = []

    for perm_i in range(n_permutations):
        shuffle_idx = rng.permutation(len(groups))
        tier_late_shuffled = tier_late[shuffle_idx]
        groups_shuffled = groups  # keep original topic structure for GroupKFold splits

        r2_compute = ridge_predict_cv(tier_early, tier_late_shuffled, groups_shuffled)
        r2_semantic = ridge_predict_cv(sem_early, tier_late_shuffled, groups_shuffled)
        null_delta = r2_compute["median_r2"] - r2_semantic["median_r2"]
        null_deltas.append(null_delta)

        if (perm_i + 1) % 25 == 0:
            n_above = sum(1 for d in null_deltas if d >= observed_delta)
            logger.info(
                f"    Perm {perm_i + 1}/{n_permutations}: "
                f"null_delta={null_delta:.4f}, "
                f"running p={n_above}/{len(null_deltas)}"
            )

    null_arr = np.array(null_deltas)
    n_above = int(np.sum(null_arr >= observed_delta))
    p_value = (n_above + 1) / (n_permutations + 1)

    logger.info(f"    Observed delta: {observed_delta:.4f}")
    logger.info(f"    Null mean: {np.mean(null_arr):.4f} +/- {np.std(null_arr):.4f}")
    logger.info(f"    Null max: {np.max(null_arr):.4f}")
    logger.info(f"    p-value: {p_value:.4f} ({n_above}/{n_permutations} >= observed)")

    return {
        "tier": tier_name,
        "observed_delta": float(observed_delta),
        "null_mean": float(np.mean(null_arr)),
        "null_std": float(np.std(null_arr)),
        "null_min": float(np.min(null_arr)),
        "null_max": float(np.max(null_arr)),
        "null_median": float(np.median(null_arr)),
        "n_above": n_above,
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "null_deltas": null_arr.tolist(),
    }


# ---------------------------------------------------------------------------
# Analysis 2: Per-tier cross-generation control
# ---------------------------------------------------------------------------

def run_tier_cross_gen(
    tier_early: np.ndarray,
    tier_late: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    tier_name: str,
) -> dict[str, Any]:
    """Cross-generation control at tier level.

    Replace each sample's early features with LOO mode centroid.
    Compare to within-generation prediction.
    """
    logger.info(f"  Cross-generation control for {tier_name}...")

    n_samples = tier_early.shape[0]
    unique_modes = np.unique(y)

    # Build LOO mode centroid for early features
    early_centroid = np.zeros_like(tier_early)
    for mode_idx in unique_modes:
        mode_mask = y == mode_idx
        mode_indices = np.where(mode_mask)[0]
        mode_early = tier_early[mode_mask]

        for local_i, global_i in enumerate(mode_indices):
            loo_mask = np.ones(len(mode_early), dtype=bool)
            loo_mask[local_i] = False
            early_centroid[global_i] = mode_early[loo_mask].mean(axis=0)

    within = ridge_predict_cv(tier_early, tier_late, groups)
    cross = ridge_predict_cv(early_centroid, tier_late, groups)

    delta = within["median_r2"] - cross["median_r2"]

    # Feature-to-sample ratio for context
    n_train_approx = int(n_samples * 0.8)
    ratio = tier_early.shape[1] / n_train_approx

    logger.info(f"    Feature:sample ratio = {tier_early.shape[1]}:{n_train_approx} ({ratio:.2f})")
    logger.info(f"    Within-gen median R²: {within['median_r2']:.4f}")
    logger.info(f"    Cross-gen median R²:  {cross['median_r2']:.4f}")
    logger.info(f"    Delta (within - cross): {delta:.4f}")

    if delta > 0.05:
        interp = "TEMPORAL_COHERENCE"
    elif abs(delta) < 0.02:
        interp = "MODE_FINGERPRINTING"
    else:
        interp = "MARGINAL"

    logger.info(f"    Interpretation: {interp}")

    return {
        "tier": tier_name,
        "n_features": int(tier_early.shape[1]),
        "feature_sample_ratio": float(ratio),
        "within_gen_median_r2": float(within["median_r2"]),
        "within_gen_mean_r2": float(within["mean_r2"]),
        "within_gen_frac_positive": float(within["frac_positive"]),
        "cross_gen_median_r2": float(cross["median_r2"]),
        "cross_gen_mean_r2": float(cross["mean_r2"]),
        "cross_gen_frac_positive": float(cross["frac_positive"]),
        "delta_median": float(delta),
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# Analysis 3: Per-tier within-mode residual
# ---------------------------------------------------------------------------

def run_tier_residual(
    tier_early: np.ndarray,
    tier_late: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    tier_name: str,
) -> dict[str, Any]:
    """Within-mode residual prediction at tier level."""
    logger.info(f"  Within-mode residual for {tier_name}...")

    early_resid = tier_early.copy()
    late_resid = tier_late.copy()

    for mode_idx in np.unique(y):
        mask = y == mode_idx
        early_resid[mask] -= tier_early[mask].mean(axis=0)
        late_resid[mask] -= tier_late[mask].mean(axis=0)

    result = ridge_predict_cv(early_resid, late_resid, groups)

    logger.info(f"    Residual median R²: {result['median_r2']:.4f}")
    logger.info(f"    Frac positive: {result['frac_positive']:.1%}")

    return {
        "tier": tier_name,
        "median_r2": float(result["median_r2"]),
        "mean_r2": float(result["mean_r2"]),
        "frac_positive": float(result["frac_positive"]),
        "n_positive": int(result["n_positive"]),
        "n_features": int(result["n_features"]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Focused per-tier temporal prediction follow-up")
    parser.add_argument("--n-permutations", type=int, default=200,
                        help="Permutation count per tier (default: 200)")
    args = parser.parse_args()

    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------

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

    unique_topics = np.unique(topics)
    topic_to_id = {t: i for i, t in enumerate(unique_topics)}
    groups = np.array([topic_to_id[t] for t in topics])

    logger.info(f"  {X_full.shape[0]} samples, {len(mode_names)} modes, {len(unique_topics)} topics")

    # -------------------------------------------------------------------
    # Parse trajectory structure and extract per-tier early/late
    # -------------------------------------------------------------------

    logger.info("Parsing trajectory structure...")
    traj_indices = build_trajectory_indices(feature_names)

    tier_data: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    for tier_name in ["T1", "T2.5", "T3"]:
        result = extract_tier_early_late(X_full, feature_names, traj_indices, tier_name)
        if result is not None:
            tier_data[tier_name] = result
            e, l, names = result
            n_train = int(e.shape[0] * 0.8)
            logger.info(
                f"  {tier_name}: {e.shape[1]} matched features, "
                f"feature:sample ratio = {e.shape[1]}:{n_train} ({e.shape[1]/n_train:.2f})"
            )

    # -------------------------------------------------------------------
    # Load observed deltas from main experiment
    # -------------------------------------------------------------------

    partial_path = OUTPUT_DIR / "temporal_prediction_partial.json"
    if partial_path.exists():
        with open(partial_path) as f:
            main_results = json.load(f)
        observed_deltas = main_results.get("delta_task1_minus_task2", {}).get("by_tier", {})
        logger.info(f"  Loaded observed deltas from main experiment: {observed_deltas}")
    else:
        logger.error("Main experiment results not found! Run run_temporal_prediction.py first.")
        return

    # -------------------------------------------------------------------
    # Compute semantic embeddings
    # -------------------------------------------------------------------

    metadata_path = (
        PROJECT_ROOT / "outputs" / "runs" / "run4_format_controlled" / "metadata.json"
    )
    sem_early, sem_late = compute_semantic_embeddings(metadata_path)
    logger.info(f"  Semantic embeddings: {sem_early.shape}")

    # -------------------------------------------------------------------
    # Run all three analyses for each tier
    # -------------------------------------------------------------------

    all_results: dict[str, Any] = {
        "experiment": "temporal_prediction_focused",
        "framing": (
            "Per-tier follow-up to main temporal prediction experiment. "
            "Overall prediction failed due to underdetermined Ridge regression "
            f"(583 features on ~80 training samples). Per-tier decomposition is "
            "pre-motivated by the existing tier structure (Phase 0.5 tier ablation), "
            "not selected post-hoc based on these results."
        ),
        "n_samples": int(X_full.shape[0]),
        "n_permutations": args.n_permutations,
        "observed_deltas_from_main": observed_deltas,
        "tiers_analyzed": list(tier_data.keys()),
    }

    for tier_name, (tier_early, tier_late, matched_names) in tier_data.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"TIER: {tier_name} ({tier_early.shape[1]} features)")
        logger.info(f"{'='*70}")

        tier_results: dict[str, Any] = {
            "n_features": int(tier_early.shape[1]),
            "matched_feature_names": matched_names,
        }

        # Get observed delta for this tier
        delta_key = tier_name
        observed_delta = observed_deltas.get(delta_key)
        if observed_delta is None:
            logger.warning(f"  No observed delta for {tier_name}, skipping permutation test")
        else:
            logger.info(f"  Observed delta: {observed_delta:.4f}")

        # --- Permutation test ---
        if observed_delta is not None:
            perm = run_tier_permutation_test(
                tier_early, tier_late, sem_early, groups,
                observed_delta=observed_delta,
                tier_name=tier_name,
                n_permutations=args.n_permutations,
            )
            # Don't save the full null distribution in the summary (save separately)
            null_deltas = perm.pop("null_deltas")
            tier_results["permutation_test"] = perm

            # Save null distribution separately for plotting
            np.save(
                OUTPUT_DIR / f"null_deltas_{tier_name}.npy",
                np.array(null_deltas),
            )

        # --- Cross-generation control ---
        cross_gen = run_tier_cross_gen(tier_early, tier_late, y, groups, tier_name)
        tier_results["cross_generation_control"] = cross_gen

        # --- Within-mode residual ---
        residual = run_tier_residual(tier_early, tier_late, y, groups, tier_name)
        tier_results["within_mode_residual"] = residual

        all_results[tier_name] = tier_results

        # Incremental save
        _save_json(all_results, OUTPUT_DIR / "focused_partial.json")
        logger.info(f"  [partial save after {tier_name}]")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------

    elapsed = time.time() - t_start
    all_results["elapsed_seconds"] = float(elapsed)

    logger.info(f"\n{'='*70}")
    logger.info("FOCUSED FOLLOW-UP SUMMARY")
    logger.info(f"{'='*70}")

    logger.info("\nPer-tier deltas and permutation p-values:")
    for tier_name in ["T1", "T2.5", "T3"]:
        if tier_name not in all_results:
            continue
        tr = all_results[tier_name]
        delta = observed_deltas.get(tier_name, float("nan"))
        perm = tr.get("permutation_test", {})
        p = perm.get("p_value", float("nan"))
        null_mean = perm.get("null_mean", float("nan"))
        logger.info(
            f"  {tier_name:5s}: delta={delta:+.3f}, "
            f"p={p:.4f}, null_mean={null_mean:+.3f} "
            f"({tr['n_features']} features)"
        )

    logger.info("\nCross-generation control (within-gen vs cross-gen R²):")
    for tier_name in ["T1", "T2.5", "T3"]:
        if tier_name not in all_results:
            continue
        cg = all_results[tier_name]["cross_generation_control"]
        logger.info(
            f"  {tier_name:5s}: within={cg['within_gen_median_r2']:+.3f}, "
            f"cross={cg['cross_gen_median_r2']:+.3f}, "
            f"delta={cg['delta_median']:+.3f} → {cg['interpretation']}"
        )

    logger.info("\nWithin-mode residual prediction:")
    for tier_name in ["T1", "T2.5", "T3"]:
        if tier_name not in all_results:
            continue
        res = all_results[tier_name]["within_mode_residual"]
        logger.info(
            f"  {tier_name:5s}: median R²={res['median_r2']:+.3f}, "
            f"frac_positive={res['frac_positive']:.1%}"
        )

    logger.info(f"\nElapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save final results
    out_path = OUTPUT_DIR / "focused_results.json"
    _save_json(all_results, out_path)
    logger.info(f"Results saved to {out_path}")


def _save_json(obj: Any, path: Path) -> None:
    """Save dict to JSON, handling numpy types."""
    def convert(o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        if isinstance(o, list):
            return [convert(v) for v in o]
        return o

    with open(path, "w") as f:
        json.dump(convert(obj), f, indent=2, default=str)


if __name__ == "__main__":
    main()
