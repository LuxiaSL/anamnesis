#!/usr/bin/env python3
"""Phase 0.5, Experiment 2 Follow-up: Statistical Hardening.

This script addresses specific methodological gaps identified through
adversarial review (MCP adversarial-review, outside agents x2) of the
initial Experiment 2 results.

## WHY each addition exists

1. **1000 permutations (parallelized)** — The initial run used 50 permutations,
   giving p=0.0196 which is at the resolution floor (1/51). 1000 permutations
   give resolution to p=0.001. Parallelized via ProcessPoolExecutor because
   each permutation is independent. (Adversary + primary analysis agree 50
   is too few.)

2. **100-seed CV stability** — Pre-registered decision criteria specify
   "aggregate, 100-seed" for the paired delta p-value. The initial run
   used 20 seeds. This is the pre-registered test. (All analyses agree.)

3. **McNemar's test** — Per-sample paired significance test. With 5 folds,
   the sign-flip test has quantized p-values (resolution floor ~0.031).
   McNemar's operates on 100 per-sample paired predictions, giving real
   statistical power without collecting more data. (Outside agents' suggestion.)

4. **Text-native baselines (logreg + GroupKFold)** — The contrastive MLP
   pipeline may systematically disadvantage text features because semantic
   space is organized by topic, not mode. Logreg can use individual features
   as mode indicators without requiring geometric proximity. This answers
   "can text classify modes?" separately from "can text support mode retrieval?"
   (Adversary + outside agents both recommend this.)

5. **Pure-noise baseline** — 366 Gaussian dimensions through the full MLP
   pipeline establishes the kNN floor in this dimensionality regime.
   (Adversary's point about dimensionality artifacts.)

6. **Updated verdict logic** — Uses the 100-seed paired delta as the primary
   decision test, matching the pre-registered intent.

## Usage

    python scripts/run_semantic_disambiguation_followup.py
    python scripts/run_semantic_disambiguation_followup.py --quick  # 200 perms, 20 seeds
    python scripts/run_semantic_disambiguation_followup.py --workers 8

Loads data and embeddings from the initial run's outputs where possible.
Results saved to outputs/phase05/semantic_disambiguation/followup_*.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Prevent tokenizer fork deadlocks — must be set before any HF imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Prevent each spawn worker from creating its own 24-thread OMP pool
# (14 workers × 24 threads = 336 threads thrashing on 16 cores)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
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
except ImportError:
    logger.error("PyTorch required for this experiment")
    sys.exit(1)

from scripts.run_contrastive_projection import _train_projection_mlp


# ---------------------------------------------------------------------------
# Data Loading (reuse from original script)
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


def compute_semantic_embeddings(texts: list[str]) -> np.ndarray:
    """Compute sentence-transformer embeddings (all-MiniLM-L6-v2, 384-dim)."""
    from sentence_transformers import SentenceTransformer

    logger.info("Computing sentence-transformer embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    logger.info(f"  Semantic embeddings: {embeddings.shape}")
    return embeddings.astype(np.float64)


# ---------------------------------------------------------------------------
# Parallelized Permutation Test
# ---------------------------------------------------------------------------

def _single_permutation(
    X: np.ndarray,
    y: np.ndarray,
    perm_seed: int,
    precomputed_folds: list[tuple[np.ndarray, np.ndarray]],
    bottleneck_dim: int = 32,
    n_epochs: int = 200,
) -> tuple[float, float]:
    """Run one permutation: shuffle labels, train MLP, return (sil, knn).

    Designed to be called from ProcessPoolExecutor. Must re-import torch
    inside the worker since torch tensors can't cross process boundaries.
    """
    # Enforce single-threaded in spawned workers BEFORE torch import
    # (env vars must be set before OMP/MKL are initialized)
    import os as _os
    _os.environ["OMP_NUM_THREADS"] = "1"
    _os.environ["MKL_NUM_THREADS"] = "1"

    # Re-import inside worker process (triggers torch import)
    from scripts.run_contrastive_projection import _train_projection_mlp

    # Belt-and-suspenders: also set torch thread count directly
    import torch
    torch.set_num_threads(1)
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(perm_seed)
    y_shuffled = rng.permutation(y)

    fold_sils: list[float] = []
    fold_knns: list[float] = []

    for fold_i, (train_idx, test_idx) in enumerate(precomputed_folds):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train = y_shuffled[train_idx]
        y_test = y_shuffled[test_idx]

        emb_train, emb_test, _ = _train_projection_mlp(
            X_train, y_train, X_test,
            bottleneck_dim=bottleneck_dim,
            n_epochs=n_epochs,
            torch_seed=42 + fold_i,
        )

        if len(np.unique(y_test)) > 1:
            sil = silhouette_score(emb_test, y_test, metric="cosine")
            fold_sils.append(sil)

        knn = KNeighborsClassifier(n_neighbors=3, metric="cosine")
        knn.fit(emb_train, y_train)
        fold_knns.append(knn.score(emb_test, y_test))

    mean_sil = float(np.mean(fold_sils)) if fold_sils else 0.0
    mean_knn = float(np.mean(fold_knns)) if fold_knns else 0.0
    return mean_sil, mean_knn


def run_parallel_permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    observed_sil: float,
    n_permutations: int = 1000,
    max_workers: int | None = None,
    bottleneck_dim: int = 32,
    n_epochs: int = 200,
) -> dict[str, Any]:
    """Parallelized permutation test using ProcessPoolExecutor.

    Each permutation shuffles labels independently and trains a fresh MLP.
    Workers are isolated processes, so no GIL contention.
    """
    # Pre-compute fold indices from ORIGINAL labels
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    precomputed_folds = list(cv.split(X, y))

    # Determine worker count
    if max_workers is None:
        cpu_count = os.cpu_count() or 4
        # Leave 2 cores free, and cap at n_permutations
        max_workers = min(max(1, cpu_count - 2), n_permutations, 14)

    logger.info(
        f"  Running {n_permutations} permutations with {max_workers} workers..."
    )

    null_silhouettes: list[float] = []
    null_knns: list[float] = []
    completed = 0

    # Use 'spawn' context to avoid fork-after-threading deadlocks with torch/tokenizers
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = {
            executor.submit(
                _single_permutation,
                X, y, perm_i * 11 + 17,  # deterministic seed, different base from CV stability
                precomputed_folds,
                bottleneck_dim, n_epochs,
            ): perm_i
            for perm_i in range(n_permutations)
        }

        for future in as_completed(futures):
            try:
                sil, knn = future.result()
                null_silhouettes.append(sil)
                null_knns.append(knn)
                completed += 1

                if completed == 1 or completed % 50 == 0:
                    n_above = sum(1 for s in null_silhouettes if s >= observed_sil)
                    logger.info(
                        f"    Permutation {completed}/{n_permutations}: "
                        f"running p={n_above}/{completed}, "
                        f"last_sil={sil:.4f}"
                    )
            except Exception as e:
                logger.warning(f"  Permutation failed: {e}")
                completed += 1

    n_above = sum(1 for s in null_silhouettes if s >= observed_sil)
    p_value = (n_above + 1) / (len(null_silhouettes) + 1)

    results = {
        "n_permutations": len(null_silhouettes),
        "n_requested": n_permutations,
        "observed_silhouette": float(observed_sil),
        "null_silhouette_mean": float(np.mean(null_silhouettes)),
        "null_silhouette_std": float(np.std(null_silhouettes)),
        "null_silhouette_max": float(np.max(null_silhouettes)),
        "null_silhouette_p95": float(np.percentile(null_silhouettes, 95)),
        "null_knn_mean": float(np.mean(null_knns)),
        "null_knn_std": float(np.std(null_knns)),
        "n_above_observed": int(n_above),
        "p_value": float(p_value),
        "max_workers": max_workers,
        "null_distribution": [float(s) for s in null_silhouettes],
    }

    logger.info(
        f"  Permutation test: p={p_value:.4f} "
        f"({n_above}/{len(null_silhouettes)} above observed {observed_sil:.4f})"
    )
    logger.info(
        f"  Null: mean={results['null_silhouette_mean']:.4f}, "
        f"std={results['null_silhouette_std']:.4f}, "
        f"max={results['null_silhouette_max']:.4f}"
    )
    return results


# ---------------------------------------------------------------------------
# McNemar's Test
# ---------------------------------------------------------------------------

def run_mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    label: str = "",
) -> dict[str, Any]:
    """McNemar's test comparing two classifiers on the same samples.

    Tests whether the disagreement pattern between classifiers A and B
    is symmetric. More powerful than comparing fold means because it
    operates on N=100 per-sample outcomes, not N=5 fold means.

    WHY: 5-fold sign-flip test has quantized p-values (resolution ~0.031).
    McNemar's gives real statistical power without new data.
    """
    correct_a = (y_pred_a == y_true).astype(int)
    correct_b = (y_pred_b == y_true).astype(int)

    # Contingency: a_right_b_wrong, a_wrong_b_right
    b01 = int(np.sum((correct_a == 1) & (correct_b == 0)))  # A right, B wrong
    b10 = int(np.sum((correct_a == 0) & (correct_b == 1)))  # A wrong, B right
    b00 = int(np.sum((correct_a == 0) & (correct_b == 0)))  # both wrong
    b11 = int(np.sum((correct_a == 1) & (correct_b == 1)))  # both right

    n_discordant = b01 + b10

    # Exact binomial test (more appropriate than chi-squared for small counts)
    from scipy.stats import binomtest

    if n_discordant == 0:
        p_value = 1.0
    else:
        result = binomtest(b01, n_discordant, 0.5, alternative="two-sided")
        p_value = float(result.pvalue)

    out = {
        "label": label,
        "a_right_b_wrong": b01,
        "a_wrong_b_right": b10,
        "both_right": b11,
        "both_wrong": b00,
        "n_discordant": n_discordant,
        "p_value": p_value,
        "accuracy_a": float(np.mean(correct_a)),
        "accuracy_b": float(np.mean(correct_b)),
    }

    tag = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    logger.info(
        f"  McNemar {label}: A={out['accuracy_a']:.0%} B={out['accuracy_b']:.0%}, "
        f"discordant={n_discordant} ({b01}:{b10}), p={p_value:.4f} {tag}"
    )
    return out


def collect_per_sample_predictions(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    topics_encoded: np.ndarray,
    n_splits: int = 5,
    bottleneck_dim: int = 32,
    n_epochs: int = 200,
) -> np.ndarray:
    """Run contrastive MLP + kNN, return per-sample predictions (N,).

    Uses GroupKFold so every sample gets exactly one out-of-fold prediction.
    """
    cv = GroupKFold(n_splits=n_splits)
    splits = list(cv.split(X, y, groups=topics_encoded))

    y_pred_all = np.full(len(y), -1, dtype=int)

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train = y[train_idx]

        emb_train, emb_test, _ = _train_projection_mlp(
            X_train, y_train, X_test,
            bottleneck_dim=bottleneck_dim,
            n_epochs=n_epochs,
            torch_seed=42 + fold_i,
        )

        knn = KNeighborsClassifier(n_neighbors=3, metric="cosine")
        knn.fit(emb_train, y_train)
        y_pred_all[test_idx] = knn.predict(emb_test)

    assert np.all(y_pred_all >= 0), "Some samples missing predictions"
    return y_pred_all


# ---------------------------------------------------------------------------
# Text-Native Baselines (logreg)
# ---------------------------------------------------------------------------

def run_text_native_baselines(
    texts: list[str],
    X_semantic: np.ndarray,
    X_compute: np.ndarray,
    y: np.ndarray,
    topics_encoded: np.ndarray,
    mode_names: list[str],
    n_splits: int = 5,
) -> dict[str, Any]:
    """Run logistic regression baselines with GroupKFold.

    WHY: The contrastive MLP pipeline may systematically disadvantage text
    features because semantic space is organized by topic, not mode. Logreg
    can use individual features as mode indicators without requiring geometric
    proximity. This answers "can text classify modes?" separately from
    "can text support mode retrieval?"

    Baselines:
      1. TF-IDF (no SVD) + logreg
      2. Semantic embeddings + logreg
      3. Compute features + logreg
      4. Combined (semantic + compute) + logreg
    """
    logger.info("=== Text-Native Baselines (Logreg + GroupKFold) ===")

    # Build TF-IDF without SVD
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 1), stop_words="english")
    X_tfidf_raw = tfidf.fit_transform(texts).toarray()
    logger.info(f"  TF-IDF raw: {X_tfidf_raw.shape}")

    cv = GroupKFold(n_splits=n_splits)

    conditions = {
        "tfidf_logreg": X_tfidf_raw,
        "semantic_logreg": X_semantic,
        "compute_logreg": X_compute,
        "combined_logreg": np.hstack([X_semantic, X_compute]),
    }

    results: dict[str, Any] = {}
    all_preds: dict[str, np.ndarray] = {}

    for name, X in conditions.items():
        fold_accs: list[float] = []
        fold_train_accs: list[float] = []
        y_pred_all = np.full(len(y), -1, dtype=int)

        for fold_i, (train_idx, test_idx) in enumerate(
            cv.split(X, y, groups=topics_encoded)
        ):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            y_train, y_test = y[train_idx], y[test_idx]

            clf = LogisticRegression(
                max_iter=2000, C=1.0, solver="lbfgs",
                multi_class="multinomial", random_state=42,
            )
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            y_pred_all[test_idx] = y_pred

            fold_accs.append(float(np.mean(y_pred == y_test)))
            fold_train_accs.append(float(clf.score(X_train, y_train)))

        # Per-mode recall
        per_mode: dict[str, float] = {}
        for i, mode in enumerate(mode_names):
            mask = y == i
            if mask.sum() > 0:
                per_mode[mode] = float(np.mean(y_pred_all[mask] == y[mask]))

        train_val_gaps = [t - v for t, v in zip(fold_train_accs, fold_accs)]

        results[name] = {
            "accuracy_mean": float(np.mean(fold_accs)),
            "accuracy_std": float(np.std(fold_accs)),
            "accuracy_per_fold": fold_accs,
            "train_accuracy_per_fold": fold_train_accs,
            "train_val_gap_mean": float(np.mean(train_val_gaps)),
            "per_mode_recall": per_mode,
            "n_features": int(X.shape[1]),
        }
        all_preds[name] = y_pred_all

        logger.info(
            f"  {name:25s}: {results[name]['accuracy_mean']:.1%} ± "
            f"{results[name]['accuracy_std']:.1%}, "
            f"gap={results[name]['train_val_gap_mean']:.1%}, "
            f"features={X.shape[1]}"
        )

    results["_predictions"] = {k: v.tolist() for k, v in all_preds.items()}
    return results


# ---------------------------------------------------------------------------
# CV Stability (reuse from original but with configurable seeds)
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
# Evaluation helper (for noise baseline + McNemar predictions)
# ---------------------------------------------------------------------------

def evaluate_condition(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    topics_encoded: np.ndarray,
    mode_names: list[str],
    n_splits: int = 5,
    bottleneck_dim: int = 32,
    n_epochs: int = 200,
) -> dict[str, Any]:
    """Run contrastive MLP + kNN evaluation (GroupKFold only)."""
    cv = GroupKFold(n_splits=n_splits)
    splits = list(cv.split(X, y, groups=topics_encoded))

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
        "condition": name,
        "knn_accuracy_mean": float(np.mean(fold_knns)),
        "knn_accuracy_std": float(np.std(fold_knns)),
        "knn_accuracy_per_fold": fold_knns,
        "silhouette_test_mean": float(np.mean(fold_sils)) if fold_sils else 0.0,
        "silhouette_test_std": float(np.std(fold_sils)) if fold_sils else 0.0,
        "per_mode_recall": per_mode_recall,
        "train_knn_per_fold": fold_train_knns,
        "train_val_gap_mean": float(np.mean(train_val_gaps)),
    }

    logger.info(
        f"  {name:>25s}: kNN={result['knn_accuracy_mean']:.1%} ± "
        f"{result['knn_accuracy_std']:.1%}, sil={result['silhouette_test_mean']:.4f}, "
        f"gap={result['train_val_gap_mean']:.1%}"
    )
    return result


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_text_native_comparison(
    native_results: dict[str, Any],
    mlp_results: dict[str, Any],
    save_path: Path,
) -> None:
    """Bar chart comparing logreg baselines vs MLP pipeline results."""
    conditions = [
        ("TF-IDF\n(logreg)", native_results.get("tfidf_logreg", {})),
        ("Semantic\n(logreg)", native_results.get("semantic_logreg", {})),
        ("Compute\n(logreg)", native_results.get("compute_logreg", {})),
        ("Combined\n(logreg)", native_results.get("combined_logreg", {})),
        ("Compute\n(MLP+kNN)", mlp_results.get("compute_only", {})),
        ("Combined\n(MLP+kNN)", mlp_results.get("combined", {})),
        ("Semantic\n(MLP+kNN)", mlp_results.get("semantic_only", {})),
    ]

    names, means, stds = [], [], []
    for label, r in conditions:
        acc_key = "accuracy_mean" if "accuracy_mean" in r else "knn_accuracy_mean"
        std_key = "accuracy_std" if "accuracy_std" in r else "knn_accuracy_std"
        if acc_key in r:
            names.append(label)
            means.append(r[acc_key])
            stds.append(r.get(std_key, 0))

    if len(names) < 2:
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    x = np.arange(len(names))
    colors = ["#FF9800"] * 4 + ["#4CAF50"] * 3  # orange=logreg, green=MLP
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85,
                  edgecolor="k", linewidth=0.5)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{m:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(0.20, color="gray", linestyle="--", alpha=0.5, label="chance (20%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Accuracy (5-way, GroupKFold by topic)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Classification vs Retrieval: Logreg Baselines vs MLP Pipeline")
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

    for name, r in stability_results.items():
        dist = r.get("knn_distribution", [])
        if not dist:
            continue
        color = colors_map.get(name, "gray")
        median_val = float(np.median(dist))
        ax.hist(dist, bins=20, alpha=0.35, color=color, edgecolor=color,
                linewidth=1, label=f"{name} (med={median_val:.1%})")
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
# Incremental save
# ---------------------------------------------------------------------------

def save_partial(results: dict[str, Any], path: Path) -> None:
    """Write partial results to disk."""
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
        description="Experiment 2 Follow-up: Statistical Hardening"
    )
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--n-stability-seeds", type=int, default=100)
    parser.add_argument("--workers", type=int, default=None,
                        help="Max workers for parallel permutation test")
    parser.add_argument("--skip-permutation", action="store_true")
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument("--skip-mcnemar", action="store_true")
    parser.add_argument("--skip-native", action="store_true",
                        help="Skip text-native logreg baselines")
    parser.add_argument("--quick", action="store_true",
                        help="Fast: 200 perms, 20 seeds")
    args = parser.parse_args()

    if args.quick:
        args.n_permutations = 200
        args.n_stability_seeds = 20

    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    partial_path = OUTPUT_DIR / "followup_partial.json"
    final_path = OUTPUT_DIR / "followup_results.json"

    logger.info("=" * 60)
    logger.info("EXPERIMENT 2 FOLLOW-UP: STATISTICAL HARDENING")
    logger.info("=" * 60)
    logger.info("Rationale: address methodological gaps from adversarial review")
    logger.info(f"  Permutations: {args.n_permutations}")
    logger.info(f"  Stability seeds: {args.n_stability_seeds}")
    logger.info(f"  Workers: {args.workers or 'auto'}")

    # ==================================================================
    # 1. Load data
    # ==================================================================
    texts, X_compute, y, topics_str, topics_encoded, mode_names = load_data()
    X_semantic = compute_semantic_embeddings(texts)

    all_results: dict[str, Any] = {
        "experiment": "semantic_disambiguation_followup",
        "rationale": (
            "Statistical hardening of Exp 2 results. Addresses: "
            "(1) 50-perm resolution floor → 1000 perms; "
            "(2) 20-seed → 100-seed stability (pre-registered intent); "
            "(3) McNemar's per-sample test (quantized fold-level p-values); "
            "(4) text-native baselines (logreg, classification vs retrieval); "
            "(5) pure-noise baseline (dimensionality control)."
        ),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": int(len(y)),
        "modes": mode_names,
    }

    # ==================================================================
    # 2. Pure-noise baseline (366 Gaussian dims through MLP pipeline)
    # ==================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info("PURE-NOISE BASELINE (366 Gaussian dims)")
    logger.info("=" * 60)
    logger.info("  WHY: Tests kNN floor in this dimensionality regime.")
    logger.info("  If noise > 20%, dimensionality alone inflates accuracy.")

    rng = np.random.RandomState(42)
    X_noise = rng.normal(size=(len(y), X_compute.shape[1]))
    noise_result = evaluate_condition(
        "pure_noise_366", X_noise, y, topics_encoded, mode_names,
    )
    all_results["pure_noise_baseline"] = noise_result
    save_partial(all_results, partial_path)

    # ==================================================================
    # 3. Text-native baselines (logreg + GroupKFold)
    # ==================================================================
    if not args.skip_native:
        logger.info(f"\n{'=' * 60}")
        logger.info("TEXT-NATIVE BASELINES (Logreg + GroupKFold)")
        logger.info("=" * 60)
        logger.info(
            "  WHY: Contrastive MLP may disadvantage text features because "
            "semantic space is organized by topic, not mode. Logreg can use "
            "individual features as mode indicators."
        )

        native_results = run_text_native_baselines(
            texts, X_semantic, X_compute, y, topics_encoded, mode_names,
        )
        # Don't save the raw predictions array to JSON (too large)
        preds_for_mcnemar = native_results.pop("_predictions", {})
        all_results["text_native_baselines"] = native_results
        save_partial(all_results, partial_path)

    # ==================================================================
    # 4. McNemar's test (per-sample paired significance)
    # ==================================================================
    if not args.skip_mcnemar:
        logger.info(f"\n{'=' * 60}")
        logger.info("McNEMAR'S TEST (per-sample paired outcomes)")
        logger.info("=" * 60)
        logger.info(
            "  WHY: 5-fold sign-flip has quantized p-values (resolution ~0.031). "
            "McNemar's uses N=100 per-sample outcomes for real statistical power."
        )

        # Collect per-sample predictions for MLP pipeline conditions
        logger.info("  Collecting per-sample predictions (MLP pipeline)...")
        conditions_for_mcnemar = {
            "compute_only": X_compute,
            "semantic_only": X_semantic,
            "combined": np.hstack([X_semantic, X_compute]),
            "semantic_noise": np.hstack([
                X_semantic,
                np.random.RandomState(42).normal(
                    loc=X_compute.mean(axis=0),
                    scale=X_compute.std(axis=0) + 1e-10,
                    size=X_compute.shape,
                ),
            ]),
        }

        mlp_preds: dict[str, np.ndarray] = {}
        for name, X_cond in conditions_for_mcnemar.items():
            logger.info(f"    Predictions for {name}...")
            mlp_preds[name] = collect_per_sample_predictions(
                name, X_cond, y, topics_encoded,
            )

        # Run McNemar's tests
        all_results["mcnemar_tests"] = {}

        mcnemar_pairs = [
            ("compute_only", "semantic_only", "compute vs semantic (MLP)"),
            ("combined", "semantic_noise", "combined vs noise (MLP) — PRIMARY"),
            ("compute_only", "combined", "compute vs combined (MLP)"),
            ("combined", "semantic_only", "combined vs semantic (MLP)"),
        ]

        for name_a, name_b, label in mcnemar_pairs:
            preds_a = mlp_preds.get(name_a)
            preds_b = mlp_preds.get(name_b)
            if preds_a is not None and preds_b is not None:
                mcn = run_mcnemar_test(y, preds_a, preds_b, label=label)
                all_results["mcnemar_tests"][f"{name_a}_vs_{name_b}_mlp"] = mcn

        # Also run McNemar's on logreg baselines if available
        if not args.skip_native and preds_for_mcnemar and len(preds_for_mcnemar) > 0:
            logger.info("  McNemar's on logreg baselines...")
            logreg_pairs = [
                ("compute_logreg", "semantic_logreg", "compute vs semantic (logreg)"),
                ("compute_logreg", "tfidf_logreg", "compute vs TF-IDF (logreg)"),
                ("combined_logreg", "semantic_logreg", "combined vs semantic (logreg)"),
            ]
            for name_a, name_b, label in logreg_pairs:
                pa = preds_for_mcnemar.get(name_a)
                pb = preds_for_mcnemar.get(name_b)
                if pa is not None and pb is not None:
                    preds_a = np.array(pa)
                    preds_b = np.array(pb)
                    mcn = run_mcnemar_test(y, preds_a, preds_b, label=label)
                    all_results["mcnemar_tests"][f"{name_a}_vs_{name_b}"] = mcn

        save_partial(all_results, partial_path)

    # ==================================================================
    # 5. Permutation tests (parallelized, 1000 shuffles)
    # ==================================================================
    if not args.skip_permutation:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"PERMUTATION TESTS ({args.n_permutations} shuffles, parallelized)")
        logger.info("=" * 60)
        logger.info(
            "  WHY: Initial 50 permutations gave p=0.0196 at resolution floor (1/51). "
            f"Running {args.n_permutations} for resolution to p={1/(args.n_permutations+1):.4f}."
        )

        # Load observed silhouettes from initial run
        initial_results_path = OUTPUT_DIR / "semantic_disambiguation.json"
        observed_sils: dict[str, float] = {}
        if initial_results_path.exists():
            with open(initial_results_path) as f:
                initial = json.load(f)
            for cond_name in ["compute_only", "combined"]:
                key = f"{cond_name}_group"
                sil = initial.get("conditions", {}).get(key, {}).get(
                    "silhouette_test_mean", None
                )
                if sil is not None and sil > 0.005:
                    observed_sils[cond_name] = sil
        else:
            logger.warning("  No initial results found, running fresh evaluations...")
            for cond_name, X_cond in [
                ("compute_only", X_compute),
                ("combined", np.hstack([X_semantic, X_compute])),
            ]:
                r = evaluate_condition(
                    cond_name, X_cond, y, topics_encoded, mode_names,
                )
                if r["silhouette_test_mean"] > 0.005:
                    observed_sils[cond_name] = r["silhouette_test_mean"]

        all_results["permutation_tests"] = {}

        for cond_name, obs_sil in observed_sils.items():
            if cond_name == "compute_only":
                X_cond = X_compute
            elif cond_name == "combined":
                X_cond = np.hstack([X_semantic, X_compute])
            else:
                continue

            logger.info(f"\n  Permutation: {cond_name} (observed sil={obs_sil:.4f})")
            perm = run_parallel_permutation_test(
                X_cond, y, obs_sil,
                n_permutations=args.n_permutations,
                max_workers=args.workers,
            )
            all_results["permutation_tests"][cond_name] = perm
            save_partial(all_results, partial_path)

    # ==================================================================
    # 6. CV Stability (100 seeds)
    # ==================================================================
    if not args.skip_stability:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"CV STABILITY ({args.n_stability_seeds} seeds)")
        logger.info("=" * 60)
        logger.info(
            "  WHY: Pre-registered criteria specify '100-seed' for paired delta. "
            "Initial run used 20 seeds. This is the pre-registered test."
        )

        stability_conditions = {
            "compute_only": X_compute,
            "combined": np.hstack([X_semantic, X_compute]),
            "semantic_noise": np.hstack([
                X_semantic,
                np.random.RandomState(42).normal(
                    loc=X_compute.mean(axis=0),
                    scale=X_compute.std(axis=0) + 1e-10,
                    size=X_compute.shape,
                ),
            ]),
            "semantic_only": X_semantic,
        }

        all_results["cv_stability"] = {}
        for cond_name, X_cond in stability_conditions.items():
            stability = run_cv_stability(
                cond_name, X_cond, y, topics_encoded,
                n_seeds=args.n_stability_seeds,
            )
            all_results["cv_stability"][cond_name] = stability
            save_partial(all_results, partial_path)

        # Paired delta from stability distributions
        comb_stab = all_results["cv_stability"].get("combined")
        noise_stab = all_results["cv_stability"].get("semantic_noise")
        if comb_stab and noise_stab:
            c_dist = comb_stab["knn_distribution"]
            n_dist = noise_stab["knn_distribution"]
            if len(c_dist) == len(n_dist):
                deltas = [c - n for c, n in zip(c_dist, n_dist)]
                d_arr = np.array(deltas)
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
                    "n_seeds": len(deltas),
                }
                logger.info(
                    f"  Paired Δ (combined − noise): "
                    f"mean={np.mean(d_arr):+.2%}, "
                    f"median={np.median(d_arr):+.2%}, "
                    f"p={p_one_sided:.6f}"
                )

        # Also compute paired delta for compute vs semantic
        comp_stab = all_results["cv_stability"].get("compute_only")
        sem_stab = all_results["cv_stability"].get("semantic_only")
        if comp_stab and sem_stab:
            c_dist = comp_stab["knn_distribution"]
            s_dist = sem_stab["knn_distribution"]
            if len(c_dist) == len(s_dist):
                deltas = [c - s for c, s in zip(c_dist, s_dist)]
                d_arr = np.array(deltas)
                p_one_sided = float(np.mean(d_arr <= 0))
                all_results["cv_stability"]["paired_delta_compute_vs_semantic"] = {
                    "delta_mean": float(np.mean(d_arr)),
                    "delta_median": float(np.median(d_arr)),
                    "delta_std": float(np.std(d_arr)),
                    "delta_p5_p95": [
                        float(np.percentile(d_arr, 5)),
                        float(np.percentile(d_arr, 95)),
                    ],
                    "p_value_one_sided": p_one_sided,
                    "n_seeds": len(deltas),
                }
                logger.info(
                    f"  Paired Δ (compute − semantic): "
                    f"mean={np.mean(d_arr):+.2%}, "
                    f"median={np.median(d_arr):+.2%}, "
                    f"p={p_one_sided:.6f}"
                )

        save_partial(all_results, partial_path)

    # ==================================================================
    # 7. Visualization
    # ==================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info("VISUALIZATION")
    logger.info("=" * 60)

    # Load original MLP results for comparison chart
    initial_results_path = OUTPUT_DIR / "semantic_disambiguation.json"
    mlp_results_for_plot: dict[str, Any] = {}
    if initial_results_path.exists():
        with open(initial_results_path) as f:
            initial = json.load(f)
        for cond in ["compute_only", "combined", "semantic_only"]:
            key = f"{cond}_group"
            if key in initial.get("conditions", {}):
                mlp_results_for_plot[cond] = initial["conditions"][key]

    if "text_native_baselines" in all_results and mlp_results_for_plot:
        plot_text_native_comparison(
            all_results["text_native_baselines"],
            mlp_results_for_plot,
            FIGURES_DIR / "classification_vs_retrieval.png",
        )

    if "cv_stability" in all_results:
        stab_for_plot = {
            k: v for k, v in all_results["cv_stability"].items()
            if isinstance(v, dict) and "knn_distribution" in v
        }
        if stab_for_plot:
            plot_cv_stability_multi(
                stab_for_plot,
                FIGURES_DIR / "cv_stability_100seed.png",
            )

    # ==================================================================
    # 8. Summary
    # ==================================================================
    elapsed = time.time() - t_start
    all_results["elapsed_seconds"] = float(elapsed)

    logger.info(f"\n{'=' * 60}")
    logger.info("FOLLOW-UP SUMMARY")
    logger.info("=" * 60)

    # Pure-noise baseline
    noise_acc = all_results.get("pure_noise_baseline", {}).get("knn_accuracy_mean", 0)
    logger.info(f"\n  Pure-noise baseline (366 Gaussian): {noise_acc:.1%}")
    if noise_acc > 0.25:
        logger.info("    WARNING: noise baseline above chance — dimensionality concern")
    else:
        logger.info("    OK: noise at/below chance — dimensionality not inflating results")

    # Text-native baselines
    if "text_native_baselines" in all_results:
        logger.info("\n  Text-native baselines (logreg, GroupKFold):")
        for name, r in all_results["text_native_baselines"].items():
            if isinstance(r, dict) and "accuracy_mean" in r:
                logger.info(f"    {name:25s}: {r['accuracy_mean']:.1%}")

    # McNemar's tests
    if "mcnemar_tests" in all_results:
        logger.info("\n  McNemar's tests:")
        for name, r in all_results["mcnemar_tests"].items():
            tag = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
            logger.info(
                f"    {r['label']:45s}: p={r['p_value']:.4f} {tag} "
                f"({r['a_right_b_wrong']}:{r['a_wrong_b_right']} discordant)"
            )

    # Permutation tests
    if "permutation_tests" in all_results:
        logger.info("\n  Permutation tests:")
        for name, r in all_results["permutation_tests"].items():
            logger.info(
                f"    {name:20s}: p={r['p_value']:.4f}, "
                f"obs_sil={r['observed_silhouette']:.4f}, "
                f"null_mean={r['null_silhouette_mean']:.4f}"
            )

    # CV stability
    if "cv_stability" in all_results:
        logger.info(f"\n  CV stability ({args.n_stability_seeds} seeds):")
        for name, r in all_results["cv_stability"].items():
            if isinstance(r, dict) and "knn_mean" in r:
                logger.info(
                    f"    {name:20s}: mean={r['knn_mean']:.1%}, "
                    f"median={r['knn_median']:.1%}, "
                    f"std={r['knn_std']:.1%}"
                )

        # Verdict using 100-seed paired delta
        delta_result = all_results["cv_stability"].get(
            "paired_delta_combined_vs_noise"
        )
        if delta_result:
            delta_pp = delta_result["delta_mean"] * 100
            p_val = delta_result["p_value_one_sided"]
            logger.info(
                f"\n  PRIMARY DECISION TEST (100-seed paired delta):"
            )
            logger.info(
                f"    Δ(combined - noise) = {delta_pp:+.1f}pp, p = {p_val:.6f}"
            )

            if delta_pp > 5 and p_val < 0.05:
                verdict = "SUB-SEMANTIC CONFIRMED"
                detail = f"Δ={delta_pp:.1f}pp > 5pp, p={p_val:.6f} < 0.05"
            elif delta_pp < 3 or p_val > 0.10:
                verdict = "SUB-SEMANTIC REFUTED"
                detail = f"Δ={delta_pp:.1f}pp, p={p_val:.6f}"
            else:
                verdict = "AMBIGUOUS"
                detail = f"Δ={delta_pp:.1f}pp, p={p_val:.6f} — need more data"

            all_results["verdict"] = verdict
            all_results["verdict_detail"] = detail
            logger.info(f"\n  VERDICT: {verdict}")
            logger.info(f"    {detail}")

    logger.info(f"\n  Elapsed: {elapsed:.1f}s")

    # Final save
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"  Results saved to {final_path}")


if __name__ == "__main__":
    main()
