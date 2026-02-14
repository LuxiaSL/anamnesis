#!/usr/bin/env python3
"""Phase 1, Experiment 1: Contrastive projection on existing Phase 0 data.

Tests whether the discriminative subspace found by Random Forest can be made
geometrically accessible via learned projections. Two stages:

  Stage A: Linear contrastive projection (metric learning / Mahalanobis distance)
  Stage B: Nonlinear contrastive projection (MLP with contrastive loss)

If linear works → subspace is a simple rotation/scaling.
If linear fails but nonlinear works → signal requires nonlinear access.
If both fail → RF signal may be spurious or requires more data.

Pre-registered predictions (from research/notes/phase1-plan.md):
  - Linear: silhouette 0.05-0.10 (60% confidence)
  - Nonlinear: silhouette 0.10-0.25 (65% confidence)
  - Analogical separates first; socratic/linear distinction is the test case

Usage:
    python scripts/run_contrastive_projection.py
    python scripts/run_contrastive_projection.py --skip-nonlinear
    python scripts/run_contrastive_projection.py --skip-robustness
    python scripts/run_contrastive_projection.py --bottleneck-dim 16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Phase 1 outputs live outside the Phase 0 run directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE1_DIR = PROJECT_ROOT / "outputs" / "phase1" / "contrastive_projection"
PHASE1_FIGURES_DIR = PHASE1_DIR / "figures"

# ---------------------------------------------------------------------------
# Torch imports (optional — graceful fallback if unavailable)
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared MLP components (used by multiple functions)
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    class ProjectionNet(nn.Module):
        """Contrastive projection network: input → hidden → bottleneck → L2-norm."""

        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, drop: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            return nn.functional.normalize(out, p=2, dim=1)


def mine_triplets(
    labels: np.ndarray, rng: np.random.RandomState, n_triplets: int = 500
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fully random triplet mining: random anchor, random positive, random negative.

    Intentionally not hard-negative or semi-hard — with 80 training samples,
    harder mining would accelerate memorization rather than improve generalization.
    """
    anchors, positives, negatives = [], [], []

    for _ in range(n_triplets):
        anchor_idx = rng.randint(len(labels))
        anchor_label = labels[anchor_idx]

        same_class = np.where(labels == anchor_label)[0]
        same_class = same_class[same_class != anchor_idx]
        if len(same_class) == 0:
            continue
        pos_idx = rng.choice(same_class)

        diff_class = np.where(labels != anchor_label)[0]
        neg_idx = rng.choice(diff_class)

        anchors.append(anchor_idx)
        positives.append(pos_idx)
        negatives.append(neg_idx)

    return np.array(anchors), np.array(positives), np.array(negatives)


def _train_projection_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray | None,
    bottleneck_dim: int = 32,
    n_epochs: int = 200,
    lr: float = 1e-3,
    margin: float = 1.0,
    dropout: float = 0.5,
    weight_decay: float = 1e-3,
    torch_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Train a projection MLP and return embeddings.

    Returns:
        (emb_train, emb_eval_or_None, final_loss)
    """
    torch.manual_seed(torch_seed)
    rng = np.random.RandomState(torch_seed)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_eval_t = torch.tensor(X_eval, dtype=torch.float32) if X_eval is not None else None

    model = ProjectionNet(X_train.shape[1], 256, bottleneck_dim, dropout)
    optimizer_obj = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin)

    final_loss = 0.0
    model.train()
    for _epoch in range(n_epochs):
        a_idx, p_idx, n_idx = mine_triplets(y_train, rng, n_triplets=300)
        if len(a_idx) < 10:
            continue

        emb = model(X_train_t)
        loss = triplet_loss_fn(emb[a_idx], emb[p_idx], emb[n_idx])

        optimizer_obj.zero_grad()
        loss.backward()
        optimizer_obj.step()
        final_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        emb_train = model(X_train_t).numpy()
        emb_eval = model(X_eval_t).numpy() if X_eval_t is not None else None

    return emb_train, emb_eval, final_loss


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_features() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Load features from review pack.

    Returns:
        (X, y_int, mode_names, labels_str)
    """
    review_pack = PROJECT_ROOT / "review_pack" / "features.npz"
    if review_pack.exists():
        data = np.load(review_pack, allow_pickle=True)
        X = data["features"].astype(np.float64)
        labels_str = data["labels"]
    else:
        # Fall back to loading from Run 4 signatures
        logger.info("review_pack not found, loading from Run 4 signatures...")
        run_name = os.environ.get("ANAMNESIS_RUN_NAME", "run4_format_controlled")
        run_dir = PROJECT_ROOT / "outputs" / "runs" / run_name
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

    logger.info(f"Loaded {X.shape[0]} samples, {X.shape[1]} features, "
                f"modes: {mode_names}")
    return X, y, mode_names, labels_str


# ---------------------------------------------------------------------------
# Stage A: Linear projections
# ---------------------------------------------------------------------------

def run_lda_projection(
    X: np.ndarray, y: np.ndarray, n_splits: int = 5
) -> tuple[dict[str, Any], np.ndarray]:
    """LDA as a linear discriminant projection — the simplest metric learning.

    LDA finds the linear subspace maximizing between-class / within-class
    variance ratio. With 5 classes, this gives a 4-dimensional projection.
    """
    logger.info("=== Stage A: LDA projection (linear) ===")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = StandardScaler()

    silhouettes_train: list[float] = []
    silhouettes_test: list[float] = []
    knn_accs: list[float] = []
    projections_test: list[np.ndarray] = []
    labels_test: list[np.ndarray] = []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        lda = LinearDiscriminantAnalysis()
        X_train_proj = lda.fit_transform(X_train, y_train)
        X_test_proj = lda.transform(X_test)

        sil_train = sil_test = float("nan")

        # Silhouette on train (are training samples geometrically separated?)
        if len(np.unique(y_train)) > 1:
            sil_train = silhouette_score(X_train_proj, y_train, metric="cosine")
            silhouettes_train.append(sil_train)

        # Silhouette on test (does it generalize?)
        if len(np.unique(y_test)) > 1:
            sil_test = silhouette_score(X_test_proj, y_test, metric="cosine")
            silhouettes_test.append(sil_test)

        # kNN accuracy in projected space
        knn = KNeighborsClassifier(n_neighbors=3, metric="cosine")
        knn.fit(X_train_proj, y_train)
        knn_acc = knn.score(X_test_proj, y_test)
        knn_accs.append(knn_acc)

        logger.info(f"  Fold {fold_i}: sil_train={sil_train:.4f}, "
                    f"sil_test={sil_test:.4f}, kNN={knn_acc:.2%}")

    # Full-data projection for visualization
    X_scaled = StandardScaler().fit_transform(X)
    lda_full = LinearDiscriminantAnalysis()
    X_proj_full = lda_full.fit_transform(X_scaled, y)
    sil_full = silhouette_score(X_proj_full, y, metric="cosine")

    results = {
        "method": "LDA",
        "projection_dim": int(X_proj_full.shape[1]),
        "silhouette_train_mean": float(np.mean(silhouettes_train)),
        "silhouette_train_std": float(np.std(silhouettes_train)),
        "silhouette_test_mean": float(np.mean(silhouettes_test)),
        "silhouette_test_std": float(np.std(silhouettes_test)),
        "silhouette_full": float(sil_full),
        "knn_accuracy_mean": float(np.mean(knn_accs)),
        "knn_accuracy_std": float(np.std(knn_accs)),
        "knn_accuracy_per_fold": [float(a) for a in knn_accs],
    }

    logger.info(f"  LDA summary: sil_full={sil_full:.4f}, "
                f"sil_test={results['silhouette_test_mean']:.4f} +/- "
                f"{results['silhouette_test_std']:.4f}, "
                f"kNN={results['knn_accuracy_mean']:.2%}")

    return results, X_proj_full


def run_nca_projection(
    X: np.ndarray, y: np.ndarray, n_components: int = 32, n_splits: int = 5
) -> dict[str, Any]:
    """Neighborhood Components Analysis — learned linear metric that directly
    optimizes kNN leave-one-out accuracy.

    This is a stronger linear baseline than LDA: it learns a Mahalanobis
    distance specifically for retrieval, not just class separation.
    """
    from sklearn.neighbors import NeighborhoodComponentsAnalysis

    logger.info(f"=== Stage A: NCA projection (linear, {n_components}d) ===")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = StandardScaler()

    silhouettes_test: list[float] = []
    knn_accs: list[float] = []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        # Cap components at min(features, train samples - 1)
        max_comp = min(n_components, X_train.shape[1], X_train.shape[0] - 1)

        try:
            nca = NeighborhoodComponentsAnalysis(
                n_components=max_comp,
                max_iter=500,
                random_state=42,
            )
            nca.fit(X_train, y_train)
            X_train_proj = nca.transform(X_train)
            X_test_proj = nca.transform(X_test)
        except Exception as e:
            logger.warning(f"  Fold {fold_i} NCA failed: {e}")
            continue

        if len(np.unique(y_test)) > 1:
            sil = silhouette_score(X_test_proj, y_test, metric="cosine")
            silhouettes_test.append(sil)

        knn = KNeighborsClassifier(n_neighbors=3, metric="cosine")
        knn.fit(X_train_proj, y_train)
        knn_acc = knn.score(X_test_proj, y_test)
        knn_accs.append(knn_acc)

        logger.info(f"  Fold {fold_i}: sil_test={sil:.4f}, kNN={knn_acc:.2%}")

    results = {
        "method": "NCA",
        "n_components": n_components,
        "silhouette_test_mean": float(np.mean(silhouettes_test)) if silhouettes_test else 0.0,
        "silhouette_test_std": float(np.std(silhouettes_test)) if silhouettes_test else 0.0,
        "knn_accuracy_mean": float(np.mean(knn_accs)) if knn_accs else 0.0,
        "knn_accuracy_std": float(np.std(knn_accs)) if knn_accs else 0.0,
        "knn_accuracy_per_fold": [float(a) for a in knn_accs],
        "folds_completed": len(knn_accs),
    }

    logger.info(f"  NCA summary: sil_test={results['silhouette_test_mean']:.4f}, "
                f"kNN={results['knn_accuracy_mean']:.2%}")

    return results


# ---------------------------------------------------------------------------
# Stage B: Nonlinear projection (contrastive MLP)
# ---------------------------------------------------------------------------

def run_contrastive_mlp(
    X: np.ndarray,
    y: np.ndarray,
    bottleneck_dim: int = 32,
    n_splits: int = 5,
    n_epochs: int = 200,
    lr: float = 1e-3,
    margin: float = 1.0,
    dropout: float = 0.5,
    weight_decay: float = 1e-3,
) -> dict[str, Any]:
    """Nonlinear contrastive projection via triplet-loss MLP.

    Architecture: input → 256 → dropout → bottleneck_dim → L2-normalize
    Loss: triplet margin loss (anchor, positive=same mode, negative=different mode)
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available, skipping nonlinear projection")
        return {"method": "contrastive_mlp", "error": "pytorch_not_available"}

    logger.info(f"=== Stage B: Contrastive MLP ({bottleneck_dim}d) ===")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = StandardScaler()

    silhouettes_train: list[float] = []
    silhouettes_test: list[float] = []
    knn_accs: list[float] = []
    final_losses: list[float] = []
    all_y_test: list[int] = []
    all_y_pred: list[int] = []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        emb_train, emb_test, final_loss = _train_projection_mlp(
            X_train, y_train, X_test,
            bottleneck_dim=bottleneck_dim, n_epochs=n_epochs, lr=lr,
            margin=margin, dropout=dropout, weight_decay=weight_decay,
            torch_seed=42 + fold_i,
        )
        final_losses.append(final_loss)

        sil_train = sil_test = float("nan")
        if len(np.unique(y_train)) > 1:
            sil_train = silhouette_score(emb_train, y_train, metric="cosine")
            silhouettes_train.append(sil_train)

        if len(np.unique(y_test)) > 1:
            sil_test = silhouette_score(emb_test, y_test, metric="cosine")
            silhouettes_test.append(sil_test)

        knn = KNeighborsClassifier(n_neighbors=3, metric="cosine")
        knn.fit(emb_train, y_train)
        y_pred = knn.predict(emb_test)
        knn_acc = float(np.mean(y_pred == y_test))
        knn_accs.append(knn_acc)

        all_y_test.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        logger.info(f"  Fold {fold_i}: loss={final_loss:.4f}, "
                    f"sil_train={sil_train:.4f}, sil_test={sil_test:.4f}, "
                    f"kNN={knn_acc:.2%}")

    # Per-mode confusion matrix from aggregated CV test predictions
    from sklearn.metrics import confusion_matrix
    y_test_arr = np.array(all_y_test)
    y_pred_arr = np.array(all_y_pred)
    n_classes = len(np.unique(y))
    cm = confusion_matrix(y_test_arr, y_pred_arr, labels=list(range(n_classes)))
    per_mode_recall = {}
    for i in range(n_classes):
        row_sum = cm[i].sum()
        per_mode_recall[i] = float(cm[i, i] / row_sum) if row_sum > 0 else 0.0

    results = {
        "method": "contrastive_mlp",
        "bottleneck_dim": bottleneck_dim,
        "n_epochs": n_epochs,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "margin": margin,
        "silhouette_train_mean": float(np.mean(silhouettes_train)) if silhouettes_train else 0.0,
        "silhouette_train_std": float(np.std(silhouettes_train)) if silhouettes_train else 0.0,
        "silhouette_test_mean": float(np.mean(silhouettes_test)) if silhouettes_test else 0.0,
        "silhouette_test_std": float(np.std(silhouettes_test)) if silhouettes_test else 0.0,
        "knn_accuracy_mean": float(np.mean(knn_accs)) if knn_accs else 0.0,
        "knn_accuracy_std": float(np.std(knn_accs)) if knn_accs else 0.0,
        "knn_accuracy_per_fold": [float(a) for a in knn_accs],
        "final_loss_per_fold": [float(fl) for fl in final_losses],
        "confusion_matrix": cm.tolist(),
        "per_mode_recall": per_mode_recall,
    }

    logger.info(f"  MLP summary: sil_test={results['silhouette_test_mean']:.4f} +/- "
                f"{results['silhouette_test_std']:.4f}, "
                f"kNN={results['knn_accuracy_mean']:.2%}")
    logger.info(f"  Per-mode recall (CV test): {per_mode_recall}")

    return results


# ---------------------------------------------------------------------------
# Robustness: Permutation test
# ---------------------------------------------------------------------------

def run_mlp_permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    observed_sil: float,
    n_permutations: int = 100,
    bottleneck_dim: int = 32,
    n_splits: int = 5,
    n_epochs: int = 200,
    lr: float = 1e-3,
    margin: float = 1.0,
    dropout: float = 0.5,
    weight_decay: float = 1e-3,
) -> dict[str, Any]:
    """Permutation test for MLP: train on shuffled labels, measure null distribution.

    Analogous to the Phase 0 permutation test (p=0.001 for RF). Tests whether
    the MLP's test silhouette could arise from noise in high-dimensional space.
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available, skipping permutation test")
        return {"error": "pytorch_not_available"}

    logger.info(f"=== Permutation test ({n_permutations} shuffles) ===")

    # Pre-compute fold indices from ORIGINAL labels — critical for correct
    # permutation test design. Each permutation should only change the
    # label-feature association, not the fold composition.
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
                bottleneck_dim=bottleneck_dim, n_epochs=n_epochs, lr=lr,
                margin=margin, dropout=dropout, weight_decay=weight_decay,
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
        null_silhouettes.append(mean_sil)
        null_knn_accs.append(mean_knn)

        if (perm_i + 1) % 10 == 0:
            n_above = sum(1 for s in null_silhouettes if s >= observed_sil)
            logger.info(f"  Permutation {perm_i + 1}/{n_permutations}: "
                        f"null_sil={mean_sil:.4f}, "
                        f"running p={n_above}/{len(null_silhouettes)}")

    n_above = sum(1 for s in null_silhouettes if s >= observed_sil)
    p_value = (n_above + 1) / (n_permutations + 1)  # +1 for continuity correction

    results = {
        "n_permutations": n_permutations,
        "observed_silhouette": float(observed_sil),
        "null_silhouette_mean": float(np.mean(null_silhouettes)),
        "null_silhouette_std": float(np.std(null_silhouettes)),
        "null_silhouette_max": float(np.max(null_silhouettes)),
        "null_silhouette_p95": float(np.percentile(null_silhouettes, 95)),
        "null_knn_mean": float(np.mean(null_knn_accs)),
        "null_knn_std": float(np.std(null_knn_accs)),
        "n_above_observed": int(n_above),
        "p_value": float(p_value),
        "null_distribution": [float(s) for s in null_silhouettes],
    }

    logger.info(f"  Permutation test: p={p_value:.4f} "
                f"({n_above}/{n_permutations} above observed {observed_sil:.4f})")
    logger.info(f"  Null distribution: mean={results['null_silhouette_mean']:.4f}, "
                f"std={results['null_silhouette_std']:.4f}, "
                f"max={results['null_silhouette_max']:.4f}, "
                f"p95={results['null_silhouette_p95']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Robustness: CV stability
# ---------------------------------------------------------------------------

def run_cv_stability(
    X: np.ndarray,
    y: np.ndarray,
    n_seeds: int = 50,
    bottleneck_dim: int = 32,
    n_splits: int = 5,
    n_epochs: int = 200,
    lr: float = 1e-3,
    margin: float = 1.0,
    dropout: float = 0.5,
    weight_decay: float = 1e-3,
) -> dict[str, Any]:
    """CV stability analysis: run MLP with different fold splits.

    Analogous to Phase 0's 100-seed CV stability (median 63.7%, 95% CI [55%, 73%]).
    Tests whether the result is robust to fold assignment or seed-dependent.
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available, skipping CV stability")
        return {"error": "pytorch_not_available"}

    logger.info(f"=== CV stability ({n_seeds} seeds) ===")

    seed_silhouettes: list[float] = []
    seed_knn_accs: list[float] = []

    for seed_i in range(n_seeds):
        cv_seed = seed_i * 7 + 13  # spread seeds to avoid clustering

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
                bottleneck_dim=bottleneck_dim, n_epochs=n_epochs, lr=lr,
                margin=margin, dropout=dropout, weight_decay=weight_decay,
                torch_seed=cv_seed + fold_i,
            )

            if len(np.unique(y_test)) > 1:
                sil = silhouette_score(emb_test, y_test, metric="cosine")
                fold_sils.append(sil)

            knn = KNeighborsClassifier(n_neighbors=3, metric="cosine")
            knn.fit(emb_train, y_train)
            fold_knns.append(knn.score(emb_test, y_test))

        mean_sil = float(np.mean(fold_sils)) if fold_sils else 0.0
        mean_knn = float(np.mean(fold_knns)) if fold_knns else 0.0
        seed_silhouettes.append(mean_sil)
        seed_knn_accs.append(mean_knn)

        if (seed_i + 1) % 10 == 0:
            logger.info(f"  Seed {seed_i + 1}/{n_seeds}: "
                        f"sil={mean_sil:.4f}, kNN={mean_knn:.2%}, "
                        f"running_median_sil={np.median(seed_silhouettes):.4f}")

    sil_arr = np.array(seed_silhouettes)
    knn_arr = np.array(seed_knn_accs)

    results = {
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
        "silhouette_distribution": [float(s) for s in seed_silhouettes],
        "knn_distribution": [float(k) for k in seed_knn_accs],
    }

    logger.info(f"  CV stability: sil median={results['silhouette_median']:.4f}, "
                f"95% CI=[{results['silhouette_ci95'][0]:.4f}, "
                f"{results['silhouette_ci95'][1]:.4f}]")
    logger.info(f"  CV stability: kNN median={results['knn_median']:.2%}, "
                f"95% CI=[{results['knn_ci95'][0]:.2%}, "
                f"{results['knn_ci95'][1]:.2%}]")

    return results


# ---------------------------------------------------------------------------
# Per-mode MLP analysis
# ---------------------------------------------------------------------------

def run_full_data_mlp(
    X: np.ndarray,
    y: np.ndarray,
    mode_names: list[str],
    bottleneck_dim: int = 32,
    n_epochs: int = 200,
    lr: float = 1e-3,
    margin: float = 1.0,
    dropout: float = 0.5,
    weight_decay: float = 1e-3,
) -> dict[str, Any]:
    """Train MLP on all data and compute per-mode silhouette in learned space.

    Note: this is trained on all data, so silhouette is inflated (training data).
    The per-mode *ranking* is still informative — which modes does the contrastive
    loss find easiest to separate?
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available, skipping full-data MLP")
        return {"error": "pytorch_not_available"}

    logger.info("=== Per-mode MLP analysis (full data) ===")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    emb_full, _, final_loss = _train_projection_mlp(
        X_scaled, y, None,
        bottleneck_dim=bottleneck_dim, n_epochs=n_epochs, lr=lr,
        margin=margin, dropout=dropout, weight_decay=weight_decay,
        torch_seed=42,
    )

    sil_full = silhouette_score(emb_full, y, metric="cosine")
    sil_samples_arr = silhouette_samples(emb_full, y, metric="cosine")

    per_mode = {}
    for i, name in enumerate(mode_names):
        mask = y == i
        per_mode[name] = float(np.mean(sil_samples_arr[mask]))

    # Pairwise mode distances in embedding space (centroid-to-centroid)
    centroids = {}
    for i, name in enumerate(mode_names):
        mask = y == i
        centroids[name] = emb_full[mask].mean(axis=0)

    pairwise_distances = {}
    for i, name_a in enumerate(mode_names):
        for j, name_b in enumerate(mode_names):
            if j <= i:
                continue
            dist = float(1.0 - np.dot(centroids[name_a], centroids[name_b]) /
                         (np.linalg.norm(centroids[name_a]) *
                          np.linalg.norm(centroids[name_b]) + 1e-10))
            pairwise_distances[f"{name_a}_vs_{name_b}"] = dist

    results = {
        "silhouette_full": float(sil_full),
        "per_mode_silhouette": per_mode,
        "pairwise_cosine_distances": pairwise_distances,
        "final_loss": float(final_loss),
        "note": "Trained on all data — silhouette inflated, per-mode ranking informative",
    }

    logger.info(f"  Full-data MLP silhouette: {sil_full:.4f}")
    for name, sil in sorted(per_mode.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"    {name}: {sil:.4f}")

    return results


# ---------------------------------------------------------------------------
# Per-mode analysis (raw / LDA)
# ---------------------------------------------------------------------------

def per_mode_silhouette(
    X_proj: np.ndarray, y: np.ndarray, mode_names: list[str]
) -> dict[str, float]:
    """Per-mode silhouette contribution: average silhouette of samples in each mode."""
    sil_samples_arr = silhouette_samples(X_proj, y, metric="cosine")
    per_mode = {}
    for i, name in enumerate(mode_names):
        mask = y == i
        per_mode[name] = float(np.mean(sil_samples_arr[mask]))
    return per_mode


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_projections(
    X_proj: np.ndarray,
    y: np.ndarray,
    mode_names: list[str],
    title: str,
    save_path: Path,
) -> None:
    """2D scatter of projected features, colored by mode."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, len(mode_names)))
    for i, name in enumerate(mode_names):
        mask = y == i
        ax.scatter(
            X_proj[mask, 0], X_proj[mask, 1],
            c=[colors[i]], label=name, alpha=0.7, s=50, edgecolors="k", linewidth=0.3
        )

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {save_path}")


def plot_comparison(results: dict[str, Any], save_path: Path) -> None:
    """Bar chart comparing methods on silhouette and kNN accuracy."""
    methods = []
    sil_test = []
    knn_acc = []

    for key in ["baseline", "lda", "nca", "contrastive_mlp"]:
        if key not in results:
            continue
        r = results[key]
        methods.append(r.get("method", key))
        sil_test.append(r.get("silhouette_test_mean", r.get("silhouette_full", 0.0)))
        knn_acc.append(r.get("knn_accuracy_mean", 0.0))

    if len(methods) < 2:
        return

    x = np.arange(len(methods))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(x, sil_test, width, color="steelblue")
    ax1.set_ylabel("Silhouette (cosine)")
    ax1.set_title("Geometric Separability")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15)
    ax1.axhline(y=0.10, color="red", linestyle="--", alpha=0.5, label="target (0.10)")
    ax1.legend()

    ax2.bar(x, knn_acc, width, color="coral")
    ax2.set_ylabel("kNN Accuracy (5-way)")
    ax2.set_title("Retrieval Accuracy")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=15)
    ax2.axhline(y=0.20, color="gray", linestyle="--", alpha=0.5, label="chance")
    ax2.axhline(y=0.50, color="red", linestyle="--", alpha=0.5, label="target (50%)")
    ax2.legend()

    fig.suptitle("Contrastive Projection: Raw vs Learned Embeddings", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {save_path}")


def plot_permutation_null(
    null_dist: list[float], observed: float, p_value: float, save_path: Path
) -> None:
    """Histogram of null distribution with observed value marked."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.hist(null_dist, bins=30, color="steelblue", alpha=0.7, edgecolor="k", linewidth=0.5)
    ax.axvline(observed, color="red", linewidth=2, linestyle="--",
               label=f"Observed ({observed:.3f})")
    ax.set_xlabel("Mean test silhouette (shuffled labels)")
    ax.set_ylabel("Count")
    ax.set_title(f"MLP Permutation Test (p={p_value:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {save_path}")


def plot_cv_stability(
    sil_dist: list[float], knn_dist: list[float], save_path: Path
) -> None:
    """Histogram of CV stability distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(sil_dist, bins=20, color="steelblue", alpha=0.7, edgecolor="k", linewidth=0.5)
    ax1.axvline(np.median(sil_dist), color="red", linewidth=2, linestyle="--",
                label=f"Median ({np.median(sil_dist):.3f})")
    ax1.set_xlabel("Mean test silhouette")
    ax1.set_ylabel("Count")
    ax1.set_title("Silhouette stability across CV seeds")
    ax1.legend()

    ax2.hist(knn_dist, bins=20, color="coral", alpha=0.7, edgecolor="k", linewidth=0.5)
    ax2.axvline(np.median(knn_dist), color="red", linewidth=2, linestyle="--",
                label=f"Median ({np.median(knn_dist):.2%})")
    ax2.set_xlabel("Mean kNN accuracy")
    ax2.set_ylabel("Count")
    ax2.set_title("kNN accuracy stability across CV seeds")
    ax2.legend()

    fig.suptitle("CV Stability Analysis", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Contrastive projection experiment")
    parser.add_argument("--skip-nonlinear", action="store_true",
                        help="Skip Stage B (MLP)")
    parser.add_argument("--skip-robustness", action="store_true",
                        help="Skip permutation test and CV stability")
    parser.add_argument("--bottleneck-dim", type=int, default=32,
                        help="MLP bottleneck dimensionality (default: 32)")
    parser.add_argument("--nca-dim", type=int, default=32,
                        help="NCA projection dimensionality (default: 32)")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="MLP training epochs per fold (default: 200)")
    parser.add_argument("--n-permutations", type=int, default=100,
                        help="Number of permutations for null test (default: 100)")
    parser.add_argument("--n-stability-seeds", type=int, default=50,
                        help="Number of CV seeds for stability (default: 50)")
    args = parser.parse_args()

    t_start = time.time()

    # Load data
    X, y, mode_names, labels_str = load_features()

    # Output directory — Phase 1, separate from Phase 0 run dirs
    out_dir = PHASE1_DIR
    fig_dir = PHASE1_FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {
        "experiment": "contrastive_projection",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "modes": mode_names,
        "pre_registered_predictions": {
            "linear_silhouette": "0.05-0.10 (60% confidence)",
            "nonlinear_silhouette": "0.10-0.25 (65% confidence)",
        },
    }

    # Baseline: raw feature silhouette
    X_scaled_full = StandardScaler().fit_transform(X)
    sil_raw = silhouette_score(X_scaled_full, y, metric="cosine")
    logger.info(f"Baseline silhouette (raw, cosine): {sil_raw:.4f}")
    all_results["baseline"] = {
        "method": "raw_features",
        "silhouette_full": float(sil_raw),
        "per_mode": per_mode_silhouette(X_scaled_full, y, mode_names),
    }

    # Stage A: LDA
    lda_results, X_proj_lda = run_lda_projection(X, y)
    lda_results["per_mode"] = per_mode_silhouette(X_proj_lda, y, mode_names)
    all_results["lda"] = lda_results

    plot_projections(X_proj_lda, y, mode_names,
                     f"LDA Projection (sil={lda_results['silhouette_full']:.3f})",
                     fig_dir / "contrastive_lda.png")

    # Stage A: NCA
    nca_results = run_nca_projection(X, y, n_components=args.nca_dim)
    all_results["nca"] = nca_results

    # Stage B: Contrastive MLP
    mlp_results: dict[str, Any] = {}
    if not args.skip_nonlinear:
        mlp_results = run_contrastive_mlp(
            X, y,
            bottleneck_dim=args.bottleneck_dim,
            n_epochs=args.n_epochs,
        )
        all_results["contrastive_mlp"] = mlp_results

    # Comparison plot
    plot_comparison(all_results, fig_dir / "contrastive_comparison.png")

    # Map per-mode recall from integer keys to mode names
    if mlp_results and "per_mode_recall" in mlp_results:
        named_recall = {mode_names[int(k)]: v
                        for k, v in mlp_results["per_mode_recall"].items()}
        mlp_results["per_mode_recall"] = named_recall
        logger.info("  Per-mode recall (CV test, named):")
        for name, recall in sorted(named_recall.items(),
                                   key=lambda x: x[1], reverse=True):
            logger.info(f"    {name}: {recall:.2%}")

    # Per-mode MLP analysis (full data — pairwise centroid distances)
    if not args.skip_nonlinear:
        mlp_per_mode = run_full_data_mlp(
            X, y, mode_names,
            bottleneck_dim=args.bottleneck_dim,
            n_epochs=args.n_epochs,
        )
        all_results["mlp_per_mode"] = mlp_per_mode

    # Robustness analyses
    if not args.skip_nonlinear and not args.skip_robustness and "error" not in mlp_results:
        observed_sil = mlp_results.get("silhouette_test_mean", 0.0)

        # Permutation test
        perm_results = run_mlp_permutation_test(
            X, y, observed_sil,
            n_permutations=args.n_permutations,
            bottleneck_dim=args.bottleneck_dim,
            n_epochs=args.n_epochs,
        )
        all_results["permutation_test"] = perm_results

        if "null_distribution" in perm_results:
            plot_permutation_null(
                perm_results["null_distribution"],
                observed_sil,
                perm_results["p_value"],
                fig_dir / "contrastive_permutation.png",
            )

        # CV stability
        stability_results = run_cv_stability(
            X, y,
            n_seeds=args.n_stability_seeds,
            bottleneck_dim=args.bottleneck_dim,
            n_epochs=args.n_epochs,
        )
        all_results["cv_stability"] = stability_results

        if "silhouette_distribution" in stability_results:
            plot_cv_stability(
                stability_results["silhouette_distribution"],
                stability_results["knn_distribution"],
                fig_dir / "contrastive_cv_stability.png",
            )

    # Summary
    elapsed = time.time() - t_start
    all_results["elapsed_seconds"] = float(elapsed)

    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Baseline silhouette:  {sil_raw:.4f}")
    logger.info(f"LDA silhouette (full): {lda_results['silhouette_full']:.4f}")
    logger.info(f"LDA kNN accuracy:     {lda_results['knn_accuracy_mean']:.2%}")
    if "nca" in all_results and all_results["nca"].get("folds_completed", 0) > 0:
        logger.info(f"NCA silhouette (test): {nca_results['silhouette_test_mean']:.4f}")
        logger.info(f"NCA kNN accuracy:     {nca_results['knn_accuracy_mean']:.2%}")
    if mlp_results and "error" not in mlp_results:
        logger.info(f"MLP silhouette (test): {mlp_results['silhouette_test_mean']:.4f}")
        logger.info(f"MLP kNN accuracy:     {mlp_results['knn_accuracy_mean']:.2%}")
    if "permutation_test" in all_results:
        pt = all_results["permutation_test"]
        logger.info(f"Permutation p-value:  {pt['p_value']:.4f}")
    if "cv_stability" in all_results:
        cs = all_results["cv_stability"]
        logger.info(f"CV stability (sil):   median={cs['silhouette_median']:.4f}, "
                    f"95% CI=[{cs['silhouette_ci95'][0]:.4f}, "
                    f"{cs['silhouette_ci95'][1]:.4f}]")
        logger.info(f"CV stability (kNN):   median={cs['knn_median']:.2%}, "
                    f"95% CI=[{cs['knn_ci95'][0]:.2%}, {cs['knn_ci95'][1]:.2%}]")
    logger.info(f"Elapsed: {elapsed:.1f}s")

    # Verdict
    best_sil = max(
        lda_results.get("silhouette_test_mean", 0),
        nca_results.get("silhouette_test_mean", 0) if "nca" in all_results else 0,
        mlp_results.get("silhouette_test_mean", 0) if mlp_results else 0,
    )
    if best_sil > 0.10:
        verdict = "PASS — discriminative subspace is geometrically accessible"
    elif best_sil > 0.05:
        verdict = "WEAK PASS — some geometric structure, may need more data"
    else:
        verdict = "FAIL — signal not geometrically accessible via tested projections"
    all_results["verdict"] = verdict
    logger.info(f"\nVerdict: {verdict}")

    # Save
    out_path = out_dir / "contrastive_projection.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
