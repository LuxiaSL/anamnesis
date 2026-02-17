#!/usr/bin/env python3
"""Phase 0.5, Experiment 2: Semantic Disambiguation.

Tests whether T2+T2.5 compute features carry information about processing mode
that is NOT accessible from the generated text. The decisive comparison is:
combined (semantic + compute) vs semantic + noise (dimensionality control).

Eight conditions with two CV strategies:
  1. semantic_only           - Sentence-transformer embeddings (384-dim)
  2. tfidf                   - TF-IDF + TruncatedSVD (384-dim)
  3. compute_only            - T2+T2.5 features (366-dim)
  4. combined                - semantic + compute (750-dim)
  5. semantic_noise           - semantic + random Gaussian (750-dim)
  6. combined_topic_shuffle   - semantic + compute shuffled within topic (750-dim)
  7. combined_global_shuffle  - semantic + compute shuffled globally (750-dim)
  8. residual_combined        - semantic + residual(compute|semantic) (750-dim)

Two CV strategies:
  Primary:   GroupKFold by topic (tests generalization to new topics)
  Secondary: StratifiedKFold (traditional stratified split)

Additional analyses:
  - Text→compute ridge regression (R² per feature)
  - Paired fold-wise Δ distributions (combined vs semantic+noise)
  - Permutation tests (conditions 1, 3, 4)
  - 100-seed CV stability (conditions 1, 3, 4, 5)

Decision criteria (pre-registered):
  - Sub-semantic CONFIRMED: Δ(#4 vs #5) > 5pp AND p < 0.05
  - Sub-semantic REFUTED:   Δ(#4 vs #5) < 3pp OR p > 0.10
  - Ambiguous:              3-5pp with 0.05 < p < 0.10

Usage:
    python scripts/run_semantic_disambiguation.py
    python scripts/run_semantic_disambiguation.py --skip-permutation --skip-stability
    python scripts/run_semantic_disambiguation.py --quick
    python scripts/run_semantic_disambiguation.py --n-permutations 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase05" / "semantic_disambiguation"
FIGURES_DIR = OUTPUT_DIR / "figures"

# ---------------------------------------------------------------------------
# Torch imports
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
except ImportError:
    logger.error("PyTorch required for this experiment")
    sys.exit(1)

from scripts.run_contrastive_projection import (
    _train_projection_mlp,
    run_mlp_permutation_test,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Condition ordering (static conditions — residual_combined handled separately)
CONDITION_ORDER = [
    "semantic_only", "tfidf", "compute_only", "combined",
    "semantic_noise", "combined_topic_shuffle", "combined_global_shuffle",
]

# Conditions that get permutation tests (expensive, primary CV only)
PERMUTATION_CONDITIONS = {"semantic_only", "compute_only", "combined"}

# Conditions that get CV stability analysis
STABILITY_CONDITIONS = {"semantic_only", "compute_only", "combined", "semantic_noise"}

# Pre-registered predictions
PRE_REGISTERED = {
    "advocate": {
        "semantic_mlp": "70-78% overall",
        "compute_only": "64-73%",
        "combined": "75-83%",
        "delta_combined_vs_noise": "+5-10pp",
        "shuffle_ordering": "combined >> topic_shuffle >> global_shuffle",
        "per_mode": "socratic Δ > 0, contrastive Δ ≈ 0",
        "text_compute_r2_median": "< 0.3",
    },
    "adversary": {
        "semantic_mlp": "65-72% overall",
        "compute_only": "64-66%",
        "combined": "68-75% (half the Δ is overfitting)",
        "delta_combined_vs_noise": "1-4pp (small, ambiguous)",
        "per_mode": "noisy and uninterpretable at N=20",
        "overall": "result will be mildly positive but equivocal",
    },
    "decision_criteria": {
        "confirmed": "Δ(#4 vs #5) > 5pp AND p < 0.05",
        "refuted": "Δ(#4 vs #5) < 3pp OR p > 0.10",
        "ambiguous": "3-5pp with 0.05 < p < 0.10",
    },
}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[
    list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]
]:
    """Load generated texts, T2+T2.5 features, labels, and topics.

    Returns:
        texts, X_compute, y, topics_str, topics_encoded, mode_names
    """
    features_path = PROJECT_ROOT / "review_pack" / "features.npz"
    data = np.load(features_path, allow_pickle=True)

    tier2 = data["tier2"].astype(np.float64)       # (100, 221)
    tier2_5 = data["tier2_5"].astype(np.float64)    # (100, 145)
    X_compute = np.hstack([tier2, tier2_5])         # (100, 366)
    X_compute = np.nan_to_num(X_compute, nan=0.0, posinf=0.0, neginf=0.0)

    labels_str = data["labels"]
    topics_str = data["topics"]
    gen_ids = data["generation_ids"]

    le_labels = LabelEncoder()
    y = le_labels.fit_transform(labels_str)
    mode_names = le_labels.classes_.tolist()

    le_topics = LabelEncoder()
    topics_encoded = le_topics.fit_transform(topics_str)

    # Load texts from signature JSONs
    sig_dir = (
        PROJECT_ROOT / "outputs" / "runs" / "run4_format_controlled" / "signatures"
    )
    texts: list[str] = []
    for gid in gen_ids:
        sig_path = sig_dir / f"gen_{int(gid):03d}.json"
        with open(sig_path) as f:
            sig_data = json.load(f)
        texts.append(sig_data["generated_text"])

    logger.info(
        f"Loaded {len(texts)} texts, {X_compute.shape[1]} compute features, "
        f"{len(mode_names)} modes: {mode_names}, "
        f"{len(np.unique(topics_str))} topics"
    )
    return texts, X_compute, y, topics_str, topics_encoded, mode_names


# ---------------------------------------------------------------------------
# Embedding Computation
# ---------------------------------------------------------------------------

def compute_semantic_embeddings(texts: list[str]) -> np.ndarray:
    """Compute sentence-transformer embeddings (all-MiniLM-L6-v2, 384-dim)."""
    from sentence_transformers import SentenceTransformer

    logger.info("Computing sentence-transformer embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    logger.info(f"  Semantic embeddings: {embeddings.shape}")
    return embeddings.astype(np.float64)


def compute_tfidf_svd(
    texts: list[str], n_components: int = 384
) -> tuple[np.ndarray, float]:
    """Compute TF-IDF + TruncatedSVD to match semantic embedding dim.

    Returns:
        (X_reduced, explained_variance_ratio)
    """
    logger.info(f"Computing TF-IDF + TruncatedSVD({n_components})...")
    tfidf = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 1), stop_words="english",
    )
    X_tfidf = tfidf.fit_transform(texts)
    logger.info(f"  Raw TF-IDF: {X_tfidf.shape}")

    n_comp_actual = min(n_components, X_tfidf.shape[1] - 1, X_tfidf.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_comp_actual, random_state=42)
    X_reduced = svd.fit_transform(X_tfidf)
    explained = float(svd.explained_variance_ratio_.sum())

    logger.info(
        f"  TF-IDF SVD: {X_reduced.shape}, explained variance: {explained:.2%}"
    )
    return X_reduced.astype(np.float64), explained


# ---------------------------------------------------------------------------
# Condition Building
# ---------------------------------------------------------------------------

def build_conditions(
    X_semantic: np.ndarray,
    X_tfidf: np.ndarray,
    X_compute: np.ndarray,
    topics_str: np.ndarray,
    rng_seed: int = 42,
) -> dict[str, np.ndarray]:
    """Build feature matrices for the 7 static conditions."""
    conditions: dict[str, np.ndarray] = {}

    # 1. Semantic only (384-dim)
    conditions["semantic_only"] = X_semantic

    # 2. TF-IDF (384-dim after SVD)
    conditions["tfidf"] = X_tfidf

    # 3. Compute only (366-dim)
    conditions["compute_only"] = X_compute

    # 4. Combined: semantic + compute (750-dim)
    conditions["combined"] = np.hstack([X_semantic, X_compute])

    # 5. Semantic + noise (750-dim) — Gaussian with per-feature mean/std of compute
    rng = np.random.RandomState(rng_seed)
    compute_mean = X_compute.mean(axis=0)
    compute_std = X_compute.std(axis=0) + 1e-10
    noise = rng.normal(loc=compute_mean, scale=compute_std, size=X_compute.shape)
    conditions["semantic_noise"] = np.hstack([X_semantic, noise])

    # 6. Combined with compute shuffled within topic (750-dim)
    #    Breaks compute-mode association while preserving compute-topic structure
    X_compute_topic_shuffled = X_compute.copy()
    rng_topic = np.random.RandomState(rng_seed + 1)
    for topic in np.unique(topics_str):
        topic_indices = np.where(topics_str == topic)[0]
        shuffled = rng_topic.permutation(topic_indices)
        X_compute_topic_shuffled[topic_indices] = X_compute[shuffled]
    conditions["combined_topic_shuffle"] = np.hstack(
        [X_semantic, X_compute_topic_shuffled]
    )

    # 7. Combined with compute shuffled globally (750-dim)
    #    Breaks both compute-mode and compute-topic associations
    rng_global = np.random.RandomState(rng_seed + 2)
    global_perm = rng_global.permutation(len(X_compute))
    conditions["combined_global_shuffle"] = np.hstack(
        [X_semantic, X_compute[global_perm]]
    )

    for name, X in conditions.items():
        logger.info(f"  Condition '{name}': {X.shape}")

    return conditions


# ---------------------------------------------------------------------------
# Core Evaluation
# ---------------------------------------------------------------------------

def evaluate_condition(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    topics_encoded: np.ndarray,
    mode_names: list[str],
    cv_type: str = "group",
    n_splits: int = 5,
    bottleneck_dim: int = 32,
    n_epochs: int = 200,
) -> dict[str, Any]:
    """Run contrastive MLP + kNN evaluation for a single condition.

    Returns dict with kNN accuracy, silhouette, per-mode recall, train-val gap.
    """
    if cv_type == "group":
        cv = GroupKFold(n_splits=n_splits)
        splits = list(cv.split(X, y, groups=topics_encoded))
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(cv.split(X, y))

    fold_sils: list[float] = []
    fold_knns: list[float] = []
    fold_train_knns: list[float] = []
    all_y_test: list[int] = []
    all_y_pred: list[int] = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        emb_train, emb_test, _loss = _train_projection_mlp(
            X_train, y_train, X_test,
            bottleneck_dim=bottleneck_dim, n_epochs=n_epochs,
            torch_seed=42 + fold_i,
        )

        if emb_test is not None and len(np.unique(y_test)) > 1:
            sil = silhouette_score(emb_test, y_test, metric="cosine")
            fold_sils.append(float(sil))

        knn = KNeighborsClassifier(n_neighbors=3, metric="cosine")
        knn.fit(emb_train, y_train)
        y_pred = knn.predict(emb_test)
        fold_knns.append(float(np.mean(y_pred == y_test)))
        fold_train_knns.append(float(knn.score(emb_train, y_train)))

        all_y_test.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

    # Per-mode recall (aggregated across folds)
    y_test_arr = np.array(all_y_test)
    y_pred_arr = np.array(all_y_pred)
    per_mode_recall: dict[str, float] = {}
    for i, mode in enumerate(mode_names):
        mask = y_test_arr == i
        if mask.sum() > 0:
            per_mode_recall[mode] = float(np.mean(y_pred_arr[mask] == y_test_arr[mask]))

    train_val_gaps = [t - v for t, v in zip(fold_train_knns, fold_knns)]

    result: dict[str, Any] = {
        "condition": name,
        "cv_type": cv_type,
        "n_features": int(X.shape[1]),
        "knn_accuracy_mean": float(np.mean(fold_knns)),
        "knn_accuracy_std": float(np.std(fold_knns)),
        "knn_accuracy_per_fold": fold_knns,
        "silhouette_test_mean": float(np.mean(fold_sils)) if fold_sils else 0.0,
        "silhouette_test_std": float(np.std(fold_sils)) if fold_sils else 0.0,
        "silhouette_test_per_fold": fold_sils,
        "per_mode_recall": per_mode_recall,
        "train_knn_per_fold": fold_train_knns,
        "train_val_gap_mean": float(np.mean(train_val_gaps)),
        "train_val_gap_per_fold": train_val_gaps,
    }

    logger.info(
        f"    {name:>28s} ({cv_type:>10s}): "
        f"kNN={result['knn_accuracy_mean']:.2%} ± {result['knn_accuracy_std']:.2%}, "
        f"sil={result['silhouette_test_mean']:.4f}, "
        f"gap={result['train_val_gap_mean']:.2%}"
    )
    return result


def evaluate_residual_condition(
    X_semantic: np.ndarray,
    X_compute: np.ndarray,
    y: np.ndarray,
    topics_encoded: np.ndarray,
    mode_names: list[str],
    cv_type: str = "group",
    n_splits: int = 5,
    bottleneck_dim: int = 32,
    n_epochs: int = 200,
    ridge_alpha: float = 1.0,
) -> dict[str, Any]:
    """Evaluate condition 8: semantic + residual(compute|semantic).

    Per-fold: fit ridge semantic→compute, compute residuals, concatenate,
    train MLP. No information leakage since ridge is fit on train only.
    """
    if cv_type == "group":
        cv = GroupKFold(n_splits=n_splits)
        splits = list(cv.split(X_semantic, y, groups=topics_encoded))
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(cv.split(X_semantic, y))

    fold_sils: list[float] = []
    fold_knns: list[float] = []
    fold_train_knns: list[float] = []
    fold_ridge_r2s: list[float] = []
    all_y_test: list[int] = []
    all_y_pred: list[int] = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        # Scale semantic and compute independently
        scaler_sem = StandardScaler()
        scaler_comp = StandardScaler()
        sem_train = scaler_sem.fit_transform(X_semantic[train_idx])
        sem_test = scaler_sem.transform(X_semantic[test_idx])
        comp_train = scaler_comp.fit_transform(X_compute[train_idx])
        comp_test = scaler_comp.transform(X_compute[test_idx])

        # Ridge: predict compute from semantic (train-only fit)
        ridge = Ridge(alpha=ridge_alpha)
        ridge.fit(sem_train, comp_train)
        fold_ridge_r2s.append(float(ridge.score(sem_test, comp_test)))

        # Residuals = actual - predicted
        resid_train = comp_train - ridge.predict(sem_train)
        resid_test = comp_test - ridge.predict(sem_test)

        # Concatenate semantic + residual, then re-standardize for MLP
        # (consistent with other conditions where StandardScaler is applied
        # to the full concatenated matrix before MLP training)
        X_concat_train = np.hstack([sem_train, resid_train])
        X_concat_test = np.hstack([sem_test, resid_test])
        scaler_combined = StandardScaler()
        X_train = scaler_combined.fit_transform(X_concat_train)
        X_test = scaler_combined.transform(X_concat_test)
        y_train, y_test = y[train_idx], y[test_idx]

        emb_train, emb_test, _loss = _train_projection_mlp(
            X_train, y_train, X_test,
            bottleneck_dim=bottleneck_dim, n_epochs=n_epochs,
            torch_seed=42 + fold_i,
        )

        if emb_test is not None and len(np.unique(y_test)) > 1:
            sil = silhouette_score(emb_test, y_test, metric="cosine")
            fold_sils.append(float(sil))

        knn = KNeighborsClassifier(n_neighbors=3, metric="cosine")
        knn.fit(emb_train, y_train)
        y_pred = knn.predict(emb_test)
        fold_knns.append(float(np.mean(y_pred == y_test)))
        fold_train_knns.append(float(knn.score(emb_train, y_train)))

        all_y_test.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

    # Per-mode recall
    y_test_arr = np.array(all_y_test)
    y_pred_arr = np.array(all_y_pred)
    per_mode_recall: dict[str, float] = {}
    for i, mode in enumerate(mode_names):
        mask = y_test_arr == i
        if mask.sum() > 0:
            per_mode_recall[mode] = float(np.mean(y_pred_arr[mask] == y_test_arr[mask]))

    train_val_gaps = [t - v for t, v in zip(fold_train_knns, fold_knns)]

    result: dict[str, Any] = {
        "condition": "residual_combined",
        "cv_type": cv_type,
        "n_features_semantic": int(X_semantic.shape[1]),
        "n_features_compute": int(X_compute.shape[1]),
        "n_features_total": int(X_semantic.shape[1] + X_compute.shape[1]),
        "knn_accuracy_mean": float(np.mean(fold_knns)),
        "knn_accuracy_std": float(np.std(fold_knns)),
        "knn_accuracy_per_fold": fold_knns,
        "silhouette_test_mean": float(np.mean(fold_sils)) if fold_sils else 0.0,
        "silhouette_test_std": float(np.std(fold_sils)) if fold_sils else 0.0,
        "silhouette_test_per_fold": fold_sils,
        "per_mode_recall": per_mode_recall,
        "train_knn_per_fold": fold_train_knns,
        "train_val_gap_mean": float(np.mean(train_val_gaps)),
        "train_val_gap_per_fold": train_val_gaps,
        "ridge_r2_per_fold": fold_ridge_r2s,
        "ridge_r2_mean": float(np.mean(fold_ridge_r2s)),
        "ridge_alpha": ridge_alpha,
    }

    logger.info(
        f"    {'residual_combined':>28s} ({cv_type:>10s}): "
        f"kNN={result['knn_accuracy_mean']:.2%} ± {result['knn_accuracy_std']:.2%}, "
        f"sil={result['silhouette_test_mean']:.4f}, "
        f"ridge_R²={result['ridge_r2_mean']:.4f}, "
        f"gap={result['train_val_gap_mean']:.2%}"
    )
    return result


# ---------------------------------------------------------------------------
# Paired Delta Analysis
# ---------------------------------------------------------------------------

def compute_paired_deltas(
    results_a: dict[str, Any],
    results_b: dict[str, Any],
    label: str = "",
) -> dict[str, Any]:
    """Compute paired fold-wise Δ (A − B) with sign-flip permutation test."""
    knns_a = results_a["knn_accuracy_per_fold"]
    knns_b = results_b["knn_accuracy_per_fold"]

    if len(knns_a) != len(knns_b):
        logger.warning(f"  Δ {label}: fold count mismatch ({len(knns_a)} vs {len(knns_b)})")
        return {"error": "fold_count_mismatch"}

    deltas = [a - b for a, b in zip(knns_a, knns_b)]
    delta_arr = np.array(deltas)

    # Sign-flip permutation test (10k iterations — fast)
    rng = np.random.RandomState(42)
    observed_mean = float(np.mean(delta_arr))
    n_perm = 10000
    null_means = np.array([
        float(np.mean(delta_arr * rng.choice([-1, 1], size=len(delta_arr))))
        for _ in range(n_perm)
    ])
    p_value = float(np.mean(np.abs(null_means) >= abs(observed_mean)))

    result = {
        "condition_a": results_a["condition"],
        "condition_b": results_b["condition"],
        "label": label,
        "delta_per_fold": [float(d) for d in deltas],
        "delta_mean": float(np.mean(delta_arr)),
        "delta_median": float(np.median(delta_arr)),
        "delta_std": float(np.std(delta_arr)),
        "paired_permutation_p": p_value,
        "n_folds": len(deltas),
    }
    logger.info(
        f"  Δ {label}: {result['delta_mean']:+.2%} ± {result['delta_std']:.2%}, "
        f"p={p_value:.4f}"
    )
    return result


# ---------------------------------------------------------------------------
# CV Stability
# ---------------------------------------------------------------------------

def run_cv_stability(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    topics_encoded: np.ndarray,
    n_seeds: int = 100,
    n_splits: int = 5,
    bottleneck_dim: int = 32,
    n_epochs: int = 200,
) -> dict[str, Any]:
    """CV stability: re-run with different GroupKFold topic assignments."""
    logger.info(f"  CV stability: {name} ({n_seeds} seeds)...")

    seed_knns: list[float] = []
    seed_sils: list[float] = []

    for seed_i in range(n_seeds):
        cv_seed = seed_i * 7 + 13

        # Remap topic IDs to create different GroupKFold assignments
        rng = np.random.RandomState(cv_seed)
        unique_topics = np.unique(topics_encoded)
        shuffled = rng.permutation(unique_topics)
        remap = {old: new for new, old in enumerate(shuffled)}
        remapped = np.array([remap[t] for t in topics_encoded])

        cv = GroupKFold(n_splits=n_splits)
        splits = list(cv.split(X, y, groups=remapped))

        fold_sils: list[float] = []
        fold_knns: list[float] = []

        for fold_i, (train_idx, test_idx) in enumerate(splits):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            y_train, y_test = y[train_idx], y[test_idx]

            emb_train, emb_test, _ = _train_projection_mlp(
                X_train, y_train, X_test,
                bottleneck_dim=bottleneck_dim, n_epochs=n_epochs,
                torch_seed=cv_seed + fold_i,
            )

            if emb_test is not None and len(np.unique(y_test)) > 1:
                fold_sils.append(
                    float(silhouette_score(emb_test, y_test, metric="cosine"))
                )

            knn = KNeighborsClassifier(n_neighbors=3, metric="cosine")
            knn.fit(emb_train, y_train)
            fold_knns.append(float(knn.score(emb_test, y_test)))

        seed_sils.append(float(np.mean(fold_sils)) if fold_sils else 0.0)
        seed_knns.append(float(np.mean(fold_knns)))

        if (seed_i + 1) % 20 == 0:
            logger.info(
                f"    Seed {seed_i + 1}/{n_seeds}: "
                f"kNN={seed_knns[-1]:.2%}, "
                f"running median={np.median(seed_knns):.2%}"
            )

    knn_arr = np.array(seed_knns)
    sil_arr = np.array(seed_sils)

    return {
        "condition": name,
        "n_seeds": n_seeds,
        "knn_mean": float(np.mean(knn_arr)),
        "knn_median": float(np.median(knn_arr)),
        "knn_std": float(np.std(knn_arr)),
        "knn_iqr": [
            float(np.percentile(knn_arr, 25)),
            float(np.percentile(knn_arr, 75)),
        ],
        "knn_p5_p95": [
            float(np.percentile(knn_arr, 5)),
            float(np.percentile(knn_arr, 95)),
        ],
        "knn_distribution": [float(k) for k in seed_knns],
        "silhouette_mean": float(np.mean(sil_arr)),
        "silhouette_median": float(np.median(sil_arr)),
        "silhouette_std": float(np.std(sil_arr)),
        "silhouette_distribution": [float(s) for s in seed_sils],
    }


# ---------------------------------------------------------------------------
# Text → Compute Regression
# ---------------------------------------------------------------------------

def run_text_compute_regression(
    X_semantic: np.ndarray,
    X_compute: np.ndarray,
    compute_feature_names: list[str],
    topics_encoded: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    ridge_alpha: float = 1.0,
) -> dict[str, Any]:
    """Ridge regression predicting each compute feature from semantic embeddings.

    Uses GroupKFold CV. Reports per-feature R² to identify text-predictable
    vs text-independent features.
    """
    logger.info("=== Text → Compute Ridge Regression ===")

    cv = GroupKFold(n_splits=n_splits)
    splits = list(cv.split(X_semantic, y, groups=topics_encoded))
    n_features = X_compute.shape[1]

    per_feature_r2_folds: list[np.ndarray] = []
    overall_r2_folds: list[float] = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        scaler_sem = StandardScaler()
        scaler_comp = StandardScaler()
        sem_train = scaler_sem.fit_transform(X_semantic[train_idx])
        sem_test = scaler_sem.transform(X_semantic[test_idx])
        comp_train = scaler_comp.fit_transform(X_compute[train_idx])
        comp_test = scaler_comp.transform(X_compute[test_idx])

        ridge = Ridge(alpha=ridge_alpha)
        ridge.fit(sem_train, comp_train)
        pred_test = ridge.predict(sem_test)

        # Per-feature R²
        per_feat_r2 = np.zeros(n_features)
        for j in range(n_features):
            ss_res = np.sum((comp_test[:, j] - pred_test[:, j]) ** 2)
            ss_tot = np.sum((comp_test[:, j] - np.mean(comp_test[:, j])) ** 2)
            per_feat_r2[j] = (
                1.0 - ss_res / (ss_tot + 1e-10) if ss_tot > 1e-10 else 0.0
            )
        per_feature_r2_folds.append(per_feat_r2)
        overall_r2_folds.append(float(ridge.score(sem_test, comp_test)))

    per_feature_r2 = np.mean(per_feature_r2_folds, axis=0)

    # Sort by R² for reporting
    top_idx = np.argsort(per_feature_r2)[::-1]
    top_predictable = [
        {"name": compute_feature_names[i], "r2": float(per_feature_r2[i]), "index": int(i)}
        for i in top_idx[:20]
    ]
    bottom_idx = np.argsort(per_feature_r2)
    top_independent = [
        {"name": compute_feature_names[i], "r2": float(per_feature_r2[i]), "index": int(i)}
        for i in bottom_idx[:20]
    ]

    median_r2 = float(np.median(per_feature_r2))
    mean_r2 = float(np.mean(per_feature_r2))

    results: dict[str, Any] = {
        "overall_r2_mean": float(np.mean(overall_r2_folds)),
        "overall_r2_per_fold": overall_r2_folds,
        "per_feature_r2_median": median_r2,
        "per_feature_r2_mean": mean_r2,
        "per_feature_r2_std": float(np.std(per_feature_r2)),
        "n_features_r2_above_0.5": int(np.sum(per_feature_r2 > 0.5)),
        "n_features_r2_above_0.3": int(np.sum(per_feature_r2 > 0.3)),
        "n_features_r2_below_0.1": int(np.sum(per_feature_r2 < 0.1)),
        "top_20_text_predictable": top_predictable,
        "top_20_text_independent": top_independent,
        "per_feature_r2_all": [float(r) for r in per_feature_r2],
        "ridge_alpha": ridge_alpha,
    }

    logger.info(f"  Overall R²: {results['overall_r2_mean']:.4f}")
    logger.info(f"  Per-feature R²: median={median_r2:.4f}, mean={mean_r2:.4f}")
    logger.info(
        f"  R² > 0.5: {results['n_features_r2_above_0.5']}/{n_features}, "
        f"R² < 0.1: {results['n_features_r2_below_0.1']}/{n_features}"
    )
    for f in top_predictable[:5]:
        logger.info(f"    text-predictable: {f['name']} R²={f['r2']:.4f}")
    for f in top_independent[:5]:
        logger.info(f"    text-independent: {f['name']} R²={f['r2']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_condition_comparison(
    conditions_results: dict[str, dict[str, Any]],
    cv_type: str,
    save_path: Path,
) -> None:
    """Bar chart of all 8 conditions — kNN accuracy ± std."""
    order = CONDITION_ORDER + ["residual_combined"]
    names: list[str] = []
    means: list[float] = []
    stds: list[float] = []

    for cond in order:
        key = f"{cond}_{cv_type}"
        if key in conditions_results:
            r = conditions_results[key]
            names.append(cond.replace("_", "\n"))
            means.append(r["knn_accuracy_mean"])
            stds.append(r["knn_accuracy_std"])

    if len(names) < 2:
        return

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    x = np.arange(len(names))

    colors = [
        "#2196F3", "#9E9E9E", "#4CAF50", "#FF5722",
        "#FFC107", "#795548", "#607D8B", "#9C27B0",
    ]
    bars = ax.bar(
        x, means, yerr=stds, capsize=4,
        color=colors[: len(names)], alpha=0.85, edgecolor="k", linewidth=0.5,
    )
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{m:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.axhline(0.20, color="gray", linestyle="--", alpha=0.5, label="chance (20%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("kNN Accuracy (5-way)")
    ax.set_ylim(0, 1.05)
    cv_label = "GroupKFold by topic" if cv_type == "group" else "StratifiedKFold"
    ax.set_title(f"Exp 2: Semantic Disambiguation — {cv_label}")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {save_path}")


def plot_per_mode_delta(
    conditions_results: dict[str, dict[str, Any]],
    mode_names: list[str],
    cv_type: str,
    save_path: Path,
) -> None:
    """Per-mode Δ(combined − semantic_only) recall."""
    combined = conditions_results.get(f"combined_{cv_type}", {})
    semantic = conditions_results.get(f"semantic_only_{cv_type}", {})
    c_recall = combined.get("per_mode_recall", {})
    s_recall = semantic.get("per_mode_recall", {})

    modes, deltas = [], []
    for m in mode_names:
        if m in c_recall and m in s_recall:
            modes.append(m)
            deltas.append(c_recall[m] - s_recall[m])

    if not modes:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.arange(len(modes))
    colors = ["#4CAF50" if d > 0 else "#F44336" for d in deltas]
    bars = ax.bar(x, deltas, color=colors, alpha=0.8, edgecolor="k", linewidth=0.5)
    for bar, d in zip(bars, deltas):
        offset = 0.01 if d >= 0 else -0.04
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
            f"{d:+.1%}", ha="center", fontsize=10,
        )

    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Δ kNN recall (combined − semantic)")
    cv_label = "GroupKFold" if cv_type == "group" else "Stratified"
    ax.set_title(
        f"Per-Mode Δ ({cv_label})\n"
        "EXPLORATORY — N=20/mode, underpowered for per-mode claims"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {save_path}")


def plot_text_compute_regression(
    regression_results: dict[str, Any], save_path: Path
) -> None:
    """Histogram of per-feature R² values."""
    r2_all = regression_results.get("per_feature_r2_all", [])
    if not r2_all:
        return

    r2_arr = np.array(r2_all)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(r2_arr, bins=30, color="steelblue", alpha=0.7, edgecolor="k", linewidth=0.5)
    ax.axvline(
        np.median(r2_arr), color="red", linewidth=2, linestyle="--",
        label=f"Median R² = {np.median(r2_arr):.3f}",
    )
    ax.axvline(
        0.5, color="orange", linewidth=1, linestyle=":",
        label="R² = 0.5 threshold",
    )
    ax.set_xlabel("Per-feature R² (semantic → compute)")
    ax.set_ylabel("Count")
    ax.set_title("Text→Compute Ridge Regression")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {save_path}")


def plot_cv_stability_multi(
    stability_results: dict[str, dict[str, Any]], save_path: Path
) -> None:
    """Overlaid histograms of kNN distributions for multiple conditions."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors_map = {
        "semantic_only": "#2196F3", "compute_only": "#4CAF50",
        "combined": "#FF5722", "semantic_noise": "#FFC107",
    }

    for name, results in stability_results.items():
        dist = results.get("knn_distribution", [])
        if not dist:
            continue
        color = colors_map.get(name, "gray")
        median_val = float(np.median(dist))
        ax.hist(
            dist, bins=20, alpha=0.35, color=color, edgecolor=color,
            linewidth=1, label=f"{name} (med={median_val:.1%})",
        )
        ax.axvline(median_val, color=color, linewidth=1.5, linestyle="--")

    ax.set_xlabel("Mean kNN Accuracy (5-way)")
    ax.set_ylabel("Count")
    ax.set_title("CV Stability (GroupKFold, topic-remapped)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Incremental save helper
# ---------------------------------------------------------------------------

def save_partial(results: dict[str, Any], path: Path) -> None:
    """Write partial results to disk (incremental save after each condition)."""
    try:
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        logger.warning(f"  Partial save failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 2: Semantic Disambiguation"
    )
    parser.add_argument("--skip-permutation", action="store_true")
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument("--skip-regression", action="store_true")
    parser.add_argument("--skip-stratified", action="store_true",
                        help="Skip secondary StratifiedKFold analysis")
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--n-stability-seeds", type=int, default=100)
    parser.add_argument("--quick", action="store_true",
                        help="Fast iteration: 50 permutations, 20 stability seeds")
    args = parser.parse_args()

    if args.quick:
        args.n_permutations = 50
        args.n_stability_seeds = 20

    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    partial_path = OUTPUT_DIR / "semantic_disambiguation_partial.json"
    final_path = OUTPUT_DIR / "semantic_disambiguation.json"

    # ==================================================================
    # 1. Load data
    # ==================================================================
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: SEMANTIC DISAMBIGUATION")
    logger.info("=" * 60)

    texts, X_compute, y, topics_str, topics_encoded, mode_names = load_data()

    # ==================================================================
    # 2. Compute embeddings
    # ==================================================================
    X_semantic = compute_semantic_embeddings(texts)
    X_tfidf, tfidf_explained_var = compute_tfidf_svd(texts, n_components=384)

    # ==================================================================
    # 3. Build conditions
    # ==================================================================
    logger.info("\nBuilding conditions...")
    conditions = build_conditions(
        X_semantic, X_tfidf, X_compute, topics_str
    )

    # ==================================================================
    # 4. Main analysis: 8 conditions × {GroupKFold, StratifiedKFold}
    # ==================================================================
    all_results: dict[str, Any] = {
        "experiment": "semantic_disambiguation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": int(len(y)),
        "modes": mode_names,
        "n_topics": int(len(np.unique(topics_str))),
        "dims": {
            "semantic": int(X_semantic.shape[1]),
            "tfidf_svd": int(X_tfidf.shape[1]),
            "compute": int(X_compute.shape[1]),
            "combined": int(X_semantic.shape[1] + X_compute.shape[1]),
        },
        "tfidf_explained_variance": tfidf_explained_var,
        "pre_registered_predictions": PRE_REGISTERED,
        "conditions": {},
        "paired_deltas": {},
    }

    cv_types = ["group"]
    if not args.skip_stratified:
        cv_types.append("stratified")

    for cv_type in cv_types:
        cv_label = (
            "GroupKFold by topic" if cv_type == "group" else "StratifiedKFold"
        )
        logger.info(f"\n{'=' * 60}")
        logger.info(f"CV: {cv_label}")
        logger.info(f"{'=' * 60}")

        # Static conditions (1-7)
        for cond_name in CONDITION_ORDER:
            X_cond = conditions[cond_name]
            result = evaluate_condition(
                cond_name, X_cond, y, topics_encoded, mode_names,
                cv_type=cv_type,
            )
            all_results["conditions"][f"{cond_name}_{cv_type}"] = result
            save_partial(all_results, partial_path)

        # Condition 8: residual combined
        result_resid = evaluate_residual_condition(
            X_semantic, X_compute, y, topics_encoded, mode_names,
            cv_type=cv_type,
        )
        all_results["conditions"][f"residual_combined_{cv_type}"] = result_resid
        save_partial(all_results, partial_path)

    # ==================================================================
    # 5. Paired Δ analysis
    # ==================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info("PAIRED DELTA ANALYSIS")
    logger.info(f"{'=' * 60}")

    delta_specs = [
        ("combined", "semantic_noise", "PRIMARY: combined vs noise"),
        ("combined", "semantic_only", "combined vs semantic"),
        ("residual_combined", "semantic_only", "residual vs semantic"),
        ("combined_topic_shuffle", "combined_global_shuffle", "topic vs global shuffle"),
        ("semantic_only", "tfidf", "semantic vs tfidf"),
    ]

    for cv_type in cv_types:
        for cond_a, cond_b, desc in delta_specs:
            key_a = f"{cond_a}_{cv_type}"
            key_b = f"{cond_b}_{cv_type}"
            if (
                key_a in all_results["conditions"]
                and key_b in all_results["conditions"]
            ):
                delta = compute_paired_deltas(
                    all_results["conditions"][key_a],
                    all_results["conditions"][key_b],
                    label=f"{desc} ({cv_type})",
                )
                delta_key = f"{cond_a}_vs_{cond_b}_{cv_type}"
                all_results["paired_deltas"][delta_key] = delta

    save_partial(all_results, partial_path)

    # ==================================================================
    # 6. Text → Compute Regression
    # ==================================================================
    if not args.skip_regression:
        # Build T2+T2.5 feature names
        features_data = np.load(
            PROJECT_ROOT / "review_pack" / "features.npz", allow_pickle=True
        )
        all_feat_names = features_data["feature_names"].tolist()
        compute_feat_names = all_feat_names[221:442] + all_feat_names[442:587]
        assert len(compute_feat_names) == X_compute.shape[1], (
            f"Feature name count mismatch: {len(compute_feat_names)} names "
            f"vs {X_compute.shape[1]} features"
        )

        regression = run_text_compute_regression(
            X_semantic, X_compute, compute_feat_names, topics_encoded, y,
        )
        all_results["text_compute_regression"] = regression
        save_partial(all_results, partial_path)

    # ==================================================================
    # 7. Permutation tests (primary CV only)
    # ==================================================================
    if not args.skip_permutation:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"PERMUTATION TESTS ({args.n_permutations} shuffles)")
        logger.info(f"{'=' * 60}")

        all_results["permutation_tests"] = {}

        for cond_name in sorted(PERMUTATION_CONDITIONS):
            X_cond = conditions.get(cond_name)
            if X_cond is None:
                continue

            primary_key = f"{cond_name}_group"
            observed_sil = all_results["conditions"].get(
                primary_key, {}
            ).get("silhouette_test_mean", 0.0)

            if observed_sil < 0.005:
                logger.info(
                    f"  Skipping {cond_name} (sil={observed_sil:.4f} too low)"
                )
                continue

            logger.info(
                f"  Permutation: {cond_name} (observed sil={observed_sil:.4f})"
            )
            perm = run_mlp_permutation_test(
                X_cond, y, observed_sil,
                n_permutations=args.n_permutations,
            )
            all_results["permutation_tests"][cond_name] = perm
            save_partial(all_results, partial_path)

    # ==================================================================
    # 8. CV Stability (GroupKFold, subset of conditions)
    # ==================================================================
    if not args.skip_stability:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"CV STABILITY ({args.n_stability_seeds} seeds)")
        logger.info(f"{'=' * 60}")

        all_results["cv_stability"] = {}

        for cond_name in sorted(STABILITY_CONDITIONS):
            X_cond = conditions.get(cond_name)
            if X_cond is None:
                continue

            stability = run_cv_stability(
                cond_name, X_cond, y, topics_encoded,
                n_seeds=args.n_stability_seeds,
            )
            all_results["cv_stability"][cond_name] = stability
            save_partial(all_results, partial_path)

        # Paired Δ from stability distributions (100 seeds)
        comb_stab = all_results["cv_stability"].get("combined")
        noise_stab = all_results["cv_stability"].get("semantic_noise")
        if comb_stab and noise_stab:
            c_dist = comb_stab["knn_distribution"]
            n_dist = noise_stab["knn_distribution"]
            if len(c_dist) == len(n_dist):
                deltas_stability = [c - n for c, n in zip(c_dist, n_dist)]
                d_arr = np.array(deltas_stability)
                p_one_sided = float(np.mean(d_arr <= 0))
                all_results["cv_stability"]["paired_delta_combined_vs_noise"] = {
                    "delta_mean": float(np.mean(d_arr)),
                    "delta_median": float(np.median(d_arr)),
                    "delta_std": float(np.std(d_arr)),
                    "delta_p5_p95": [
                        float(np.percentile(d_arr, 5)),
                        float(np.percentile(d_arr, 95)),
                    ],
                    "p_value_one_sided": p_one_sided,
                    "n_seeds": len(deltas_stability),
                }
                logger.info(
                    f"  Stability Δ (combined − noise): "
                    f"mean={np.mean(d_arr):+.2%}, "
                    f"median={np.median(d_arr):+.2%}, "
                    f"p={p_one_sided:.4f}"
                )

        save_partial(all_results, partial_path)

    # ==================================================================
    # 9. Visualization
    # ==================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info("VISUALIZATION")
    logger.info(f"{'=' * 60}")

    plot_condition_comparison(
        all_results["conditions"], "group",
        FIGURES_DIR / "condition_comparison_group.png",
    )
    if not args.skip_stratified:
        plot_condition_comparison(
            all_results["conditions"], "stratified",
            FIGURES_DIR / "condition_comparison_stratified.png",
        )

    plot_per_mode_delta(
        all_results["conditions"], mode_names, "group",
        FIGURES_DIR / "per_mode_delta_group.png",
    )

    if "text_compute_regression" in all_results:
        plot_text_compute_regression(
            all_results["text_compute_regression"],
            FIGURES_DIR / "text_compute_regression.png",
        )

    if "cv_stability" in all_results:
        stability_for_plot = {
            k: v for k, v in all_results["cv_stability"].items()
            if isinstance(v, dict) and "knn_distribution" in v
        }
        if stability_for_plot:
            plot_cv_stability_multi(
                stability_for_plot,
                FIGURES_DIR / "cv_stability_combined.png",
            )

    # ==================================================================
    # 10. Summary and Verdict
    # ==================================================================
    elapsed = time.time() - t_start
    all_results["elapsed_seconds"] = float(elapsed)

    logger.info(f"\n{'=' * 60}")
    logger.info("EXPERIMENT 2 SUMMARY")
    logger.info(f"{'=' * 60}")

    # Primary results table (GroupKFold)
    logger.info("\nPrimary (GroupKFold by topic):")
    logger.info(f"  {'Condition':<28s} {'kNN':>8s} {'Sil':>8s} {'Gap':>8s}")
    logger.info(f"  {'-' * 52}")

    for cond_name in CONDITION_ORDER + ["residual_combined"]:
        key = f"{cond_name}_group"
        r = all_results["conditions"].get(key)
        if r:
            logger.info(
                f"  {cond_name:<28s} "
                f"{r['knn_accuracy_mean']:>7.2%} "
                f"{r['silhouette_test_mean']:>8.4f} "
                f"{r['train_val_gap_mean']:>7.2%}"
            )

    # Key deltas
    logger.info("\nKey Δ comparisons (GroupKFold):")
    for delta_key, delta in all_results.get("paired_deltas", {}).items():
        if delta_key.endswith("_group") and "error" not in delta:
            logger.info(
                f"  {delta['label']}: "
                f"Δ={delta['delta_mean']:+.2%}, p={delta['paired_permutation_p']:.4f}"
            )

    # Verdict
    primary_delta = all_results.get("paired_deltas", {}).get(
        "combined_vs_semantic_noise_group"
    )
    if primary_delta and "error" not in primary_delta:
        delta_pp = primary_delta["delta_mean"] * 100
        p_val = primary_delta["paired_permutation_p"]

        if delta_pp > 5 and p_val < 0.05:
            verdict = "SUB-SEMANTIC CONFIRMED"
            detail = f"Δ={delta_pp:.1f}pp > 5pp, p={p_val:.4f} < 0.05"
        elif delta_pp < 3 or p_val > 0.10:
            verdict = "SUB-SEMANTIC REFUTED"
            detail = f"Δ={delta_pp:.1f}pp, p={p_val:.4f}"
        else:
            verdict = "AMBIGUOUS"
            detail = f"Δ={delta_pp:.1f}pp, p={p_val:.4f} — need more data"

        all_results["verdict"] = verdict
        all_results["verdict_detail"] = detail
        logger.info(f"\nVERDICT: {verdict}")
        logger.info(f"  {detail}")

    # Regression summary
    if "text_compute_regression" in all_results:
        reg = all_results["text_compute_regression"]
        logger.info(f"\nText→Compute R²: median={reg['per_feature_r2_median']:.4f}")
        if reg["per_feature_r2_median"] > 0.5:
            logger.info("  → Compute features substantially text-predictable")
        elif reg["per_feature_r2_median"] < 0.2:
            logger.info("  → Compute features carry text-inaccessible information")
        else:
            logger.info("  → Mixed: some text-predictable, some independent")

    logger.info(f"\nElapsed: {elapsed:.1f}s")

    # Final save
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {final_path}")


if __name__ == "__main__":
    main()
