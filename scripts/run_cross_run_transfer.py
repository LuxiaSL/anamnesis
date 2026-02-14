#!/usr/bin/env python3
"""Phase 1, Experiment 1.5: Cross-run transfer test.

Tests whether contrastive projections learned on one mode set transfer to
a different mode set — evidence for computational topology vs prompt-specific
features.

Design:
  1. Train contrastive MLP on ALL Run 4 data (100 samples, 5 modes)
  2. Freeze model, embed Run 3 data (75 samples, 5 modes) through same projection
  3. Compute nearest Run 4 mode centroid for each Run 3 sample
  4. Check mapping against pre-registered predictions

  Also:
  5. LDA direction projection test: fit LDA on Run 3, project Run 4 data onto
     those discriminant directions. If silhouette at chance → linear directions
     separating Run 3 carry zero Run 4 information.

Pre-registered mapping predictions (from session-handoff-2026-02-14.md):
  - deliberative (R3) → dialectical (R4): both propose-challenge-revise
  - pedagogical (R3) → socratic (R4): both interactive explanation through questions
  - associative (R3) → analogical (R4): both non-sequential, connection-driven
  - structured (R3) → linear (R4): both sequential organized exposition

  Key test case: pedagogical → socratic (least obvious pairing).
  Compressed (R3) has no pre-registered mapping — interesting to see where it lands.

Usage:
    python scripts/run_cross_run_transfer.py
    python scripts/run_cross_run_transfer.py --n-seeds 20
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase1" / "cross_run_transfer"
FIGURES_DIR = OUTPUT_DIR / "figures"

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    logger.error("PyTorch required for this experiment")
    sys.exit(1)

from scripts.run_contrastive_projection import ProjectionNet, mine_triplets


# ---------------------------------------------------------------------------
# Pre-registered predictions
# ---------------------------------------------------------------------------

PREDICTED_MAPPING: dict[str, str] = {
    "deliberative": "dialectical",
    "pedagogical": "socratic",
    "associative": "analogical",
    "structured": "linear",
}
# compressed has no predicted mapping — it's the wildcard

# Reverse mapping: same predictions, opposite direction
PREDICTED_MAPPING_REVERSE: dict[str, str] = {
    "dialectical": "deliberative",
    "socratic": "pedagogical",
    "analogical": "associative",
    "linear": "structured",
}
# contrastive has no predicted mapping in reverse


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run4_features() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Load Run 4 features from review pack."""
    review_pack = PROJECT_ROOT / "review_pack" / "features.npz"
    data = np.load(review_pack, allow_pickle=True)
    X = data["features"].astype(np.float64)
    labels_str = data["labels"]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    le = LabelEncoder()
    y = le.fit_transform(labels_str)
    mode_names = le.classes_.tolist()

    logger.info(f"Run 4: {X.shape[0]} samples, {X.shape[1]} features, modes: {mode_names}")
    return X, y, mode_names, labels_str


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

    logger.info(f"Run 3: {X.shape[0]} samples, {X.shape[1]} features, modes: {mode_names}")
    return X, y, mode_names, labels_str


# ---------------------------------------------------------------------------
# MLP training (full-data, returns model for transfer)
# ---------------------------------------------------------------------------

def train_full_data_mlp(
    X: np.ndarray,
    y: np.ndarray,
    bottleneck_dim: int = 32,
    n_epochs: int = 200,
    lr: float = 1e-3,
    margin: float = 1.0,
    dropout: float = 0.5,
    weight_decay: float = 1e-3,
    torch_seed: int = 42,
) -> tuple[ProjectionNet, float]:
    """Train contrastive MLP on full dataset and return the model.

    Unlike _train_projection_mlp, this returns the model itself so we can
    embed new data through it.
    """
    torch.manual_seed(torch_seed)
    rng = np.random.RandomState(torch_seed)

    X_t = torch.tensor(X, dtype=torch.float32)

    model = ProjectionNet(X.shape[1], 256, bottleneck_dim, dropout)
    optimizer_obj = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin)

    final_loss = 0.0
    model.train()
    for _epoch in range(n_epochs):
        a_idx, p_idx, n_idx = mine_triplets(y, rng, n_triplets=300)
        if len(a_idx) < 10:
            continue

        emb = model(X_t)
        loss = triplet_loss_fn(emb[a_idx], emb[p_idx], emb[n_idx])

        optimizer_obj.zero_grad()
        loss.backward()
        optimizer_obj.step()
        final_loss = float(loss.item())

    return model, final_loss


def embed_with_model(model: ProjectionNet, X: np.ndarray) -> np.ndarray:
    """Embed data through a frozen model."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        emb = model(X_t).numpy()
    return emb


# ---------------------------------------------------------------------------
# Cross-run transfer analysis
# ---------------------------------------------------------------------------

def compute_centroid_matrix(
    emb_r4: np.ndarray,
    y_r4: np.ndarray,
    modes_r4: list[str],
    emb_r3: np.ndarray,
    y_r3: np.ndarray,
    modes_r3: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Compute centroids for each mode in both runs, return cosine similarity matrix.

    Returns:
        (r4_centroids, r3_centroids, sim_matrix[r3_modes x r4_modes])
    """
    r4_centroids: dict[str, np.ndarray] = {}
    for i, name in enumerate(modes_r4):
        mask = y_r4 == i
        r4_centroids[name] = emb_r4[mask].mean(axis=0)

    r3_centroids: dict[str, np.ndarray] = {}
    for i, name in enumerate(modes_r3):
        mask = y_r3 == i
        r3_centroids[name] = emb_r3[mask].mean(axis=0)

    # Cosine similarity matrix: R3 modes (rows) × R4 modes (cols)
    sim_matrix = np.zeros((len(modes_r3), len(modes_r4)))
    for i, r3_name in enumerate(modes_r3):
        for j, r4_name in enumerate(modes_r4):
            c3 = r3_centroids[r3_name]
            c4 = r4_centroids[r4_name]
            norm_product = np.linalg.norm(c3) * np.linalg.norm(c4) + 1e-10
            sim_matrix[i, j] = float(np.dot(c3, c4) / norm_product)

    return r4_centroids, r3_centroids, sim_matrix


def compute_nearest_centroid_mapping(
    emb_r3: np.ndarray,
    y_r3: np.ndarray,
    modes_r3: list[str],
    r4_centroids: dict[str, np.ndarray],
    modes_r4: list[str],
) -> tuple[np.ndarray, dict[str, dict[str, int]]]:
    """For each Run 3 sample, find the nearest Run 4 mode centroid.

    Returns:
        (assignment_matrix[r3_modes x r4_modes], per_sample_assignments)
    """
    # Stack Run 4 centroids in order
    centroid_stack = np.stack([r4_centroids[name] for name in modes_r4])

    # Cosine similarity of each Run 3 sample to each Run 4 centroid
    # emb_r3 is L2-normalized, centroids may not be
    centroid_norms = np.linalg.norm(centroid_stack, axis=1, keepdims=True) + 1e-10
    centroid_normed = centroid_stack / centroid_norms
    similarities = emb_r3 @ centroid_normed.T  # (n_r3, n_r4_modes)

    nearest_r4_idx = similarities.argmax(axis=1)

    # Build assignment matrix
    assignment_matrix = np.zeros((len(modes_r3), len(modes_r4)), dtype=int)
    for i, r3_name in enumerate(modes_r3):
        mask = y_r3 == i
        for r4_idx in nearest_r4_idx[mask]:
            assignment_matrix[i, r4_idx] += 1

    return assignment_matrix, nearest_r4_idx


def evaluate_mapping_accuracy(
    assignment_matrix: np.ndarray,
    modes_r3: list[str],
    modes_r4: list[str],
    predicted_mapping: dict[str, str],
) -> dict[str, Any]:
    """Evaluate how well the nearest-centroid mapping matches pre-registered predictions."""
    total_mapped = 0
    correct_mapped = 0
    per_mode: dict[str, dict[str, Any]] = {}

    for i, r3_name in enumerate(modes_r3):
        if r3_name not in predicted_mapping:
            # No pre-registered prediction (e.g., compressed)
            actual_nearest = modes_r4[assignment_matrix[i].argmax()]
            per_mode[r3_name] = {
                "predicted": None,
                "actual_nearest": actual_nearest,
                "samples": int(assignment_matrix[i].sum()),
                "assignment_distribution": {
                    modes_r4[j]: int(assignment_matrix[i, j])
                    for j in range(len(modes_r4))
                },
            }
            continue

        predicted_r4 = predicted_mapping[r3_name]
        predicted_r4_idx = modes_r4.index(predicted_r4)
        n_correct = int(assignment_matrix[i, predicted_r4_idx])
        n_total = int(assignment_matrix[i].sum())
        actual_nearest = modes_r4[assignment_matrix[i].argmax()]

        total_mapped += n_total
        correct_mapped += n_correct

        per_mode[r3_name] = {
            "predicted": predicted_r4,
            "actual_nearest": actual_nearest,
            "correct": n_correct,
            "total": n_total,
            "accuracy": float(n_correct / n_total) if n_total > 0 else 0.0,
            "match": actual_nearest == predicted_r4,
            "assignment_distribution": {
                modes_r4[j]: int(assignment_matrix[i, j])
                for j in range(len(modes_r4))
            },
        }

    overall_accuracy = float(correct_mapped / total_mapped) if total_mapped > 0 else 0.0
    n_modes_correct = sum(
        1 for v in per_mode.values()
        if v.get("match", False)
    )

    return {
        "overall_accuracy": overall_accuracy,
        "correct_mapped": correct_mapped,
        "total_mapped": total_mapped,
        "modes_correct": n_modes_correct,
        "modes_total": len(predicted_mapping),
        "chance_accuracy": 1.0 / len(modes_r4),
        "per_mode": per_mode,
    }


# ---------------------------------------------------------------------------
# LDA direction projection test
# ---------------------------------------------------------------------------

def run_lda_direction_test(
    X_r3: np.ndarray,
    y_r3: np.ndarray,
    modes_r3: list[str],
    X_r4: np.ndarray,
    y_r4: np.ndarray,
    modes_r4: list[str],
) -> dict[str, Any]:
    """Fit LDA on Run 3, project Run 4 data onto those discriminant directions.

    If Run 4 silhouette is at chance in Run 3's LDA space → the linear
    directions separating Run 3 carry zero Run 4 information.
    """
    logger.info("=== LDA direction projection test ===")

    scaler = StandardScaler()
    X_r3_scaled = scaler.fit_transform(X_r3)
    X_r4_scaled = scaler.transform(X_r4)  # same scaler

    lda = LinearDiscriminantAnalysis()
    X_r3_proj = lda.fit_transform(X_r3_scaled, y_r3)
    X_r4_proj = lda.transform(X_r4_scaled)

    # Run 3 silhouette in its own LDA space (sanity check)
    sil_r3 = silhouette_score(X_r3_proj, y_r3, metric="cosine")
    knn_r3 = KNeighborsClassifier(n_neighbors=3, metric="cosine")
    knn_r3.fit(X_r3_proj, y_r3)
    knn_r3_acc = knn_r3.score(X_r3_proj, y_r3)  # training accuracy as sanity check

    # Run 4 silhouette in Run 3's LDA space
    sil_r4 = silhouette_score(X_r4_proj, y_r4, metric="cosine")
    sil_r4_samples = silhouette_samples(X_r4_proj, y_r4, metric="cosine")

    per_mode_r4_sil = {}
    for i, name in enumerate(modes_r4):
        mask = y_r4 == i
        per_mode_r4_sil[name] = float(np.mean(sil_r4_samples[mask]))

    # kNN: train on Run 3 projected, test on Run 4 projected
    # This asks: do Run 4 modes land near the "right" Run 3 modes?
    # Not directly meaningful for 5-way (different label sets), but the
    # prediction distribution tells us something about the geometry.
    knn_cross = KNeighborsClassifier(n_neighbors=3, metric="cosine")
    knn_cross.fit(X_r3_proj, y_r3)
    r4_predictions = knn_cross.predict(X_r4_proj)

    # Build cross-run prediction matrix: for each Run 4 mode, what Run 3 mode
    # does the kNN assign?
    cross_prediction_matrix = np.zeros((len(modes_r4), len(modes_r3)), dtype=int)
    for i, r4_mode in enumerate(modes_r4):
        mask = y_r4 == i
        for pred in r4_predictions[mask]:
            cross_prediction_matrix[i, pred] += 1

    logger.info(f"  Run 3 sil in own LDA space: {sil_r3:.4f} (sanity check)")
    logger.info(f"  Run 4 sil in Run 3 LDA space: {sil_r4:.4f}")
    logger.info(f"  Per-mode Run 4 sil in Run 3 LDA:")
    for name, sil in sorted(per_mode_r4_sil.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"    {name}: {sil:.4f}")

    logger.info(f"  Cross-run kNN predictions (R4 mode → R3 mode):")
    for i, r4_mode in enumerate(modes_r4):
        preds = {modes_r3[j]: int(cross_prediction_matrix[i, j]) for j in range(len(modes_r3))}
        logger.info(f"    {r4_mode}: {preds}")

    results = {
        "run3_sil_own_lda": float(sil_r3),
        "run4_sil_in_run3_lda": float(sil_r4),
        "run4_per_mode_sil": per_mode_r4_sil,
        "cross_prediction_matrix": cross_prediction_matrix.tolist(),
        "cross_prediction_rows": modes_r4,
        "cross_prediction_cols": modes_r3,
        "projection_dim": int(X_r3_proj.shape[1]),
        "interpretation": (
            "If Run 4 sil is near zero or negative in Run 3's LDA space, "
            "the linear directions separating Run 3 modes carry zero information "
            "about Run 4 modes. Confirms the 'directions vs manifolds' hypothesis."
        ),
    }

    return results, X_r3_proj, X_r4_proj


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_transfer_scatter(
    emb_train: np.ndarray,
    y_train: np.ndarray,
    modes_train: list[str],
    emb_test: np.ndarray,
    y_test: np.ndarray,
    modes_test: list[str],
    title: str,
    save_path: Path,
    train_label: str = "train",
    test_label: str = "test",
) -> None:
    """2D scatter of combined train + test embeddings.

    Uses TSNE to reduce the 32-d embeddings to 2D for visualization.
    Train modes: circles. Test modes: triangles.
    """
    from sklearn.manifold import TSNE

    combined = np.vstack([emb_train, emb_test])
    n_train = len(emb_train)

    # t-SNE on combined embeddings
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined) - 1))
    coords = tsne.fit_transform(combined)

    coords_train = coords[:n_train]
    coords_test = coords[n_train:]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Color palette: train modes get Set2, test modes get Set1
    colors_train = plt.cm.Set2(np.linspace(0, 0.8, len(modes_train)))
    colors_test = plt.cm.Set1(np.linspace(0, 0.8, len(modes_test)))

    # Plot train modes as circles
    for i, name in enumerate(modes_train):
        mask = y_train == i
        ax.scatter(
            coords_train[mask, 0], coords_train[mask, 1],
            c=[colors_train[i]], marker="o", s=60,
            label=f"{train_label}: {name}", alpha=0.7, edgecolors="k", linewidth=0.5,
        )

    # Plot test modes as triangles
    for i, name in enumerate(modes_test):
        mask = y_test == i
        ax.scatter(
            coords_test[mask, 0], coords_test[mask, 1],
            c=[colors_test[i]], marker="^", s=80,
            label=f"{test_label}: {name}", alpha=0.7, edgecolors="k", linewidth=0.5,
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {save_path}")


def plot_similarity_matrix(
    sim_matrix: np.ndarray,
    row_modes: list[str],
    col_modes: list[str],
    title: str,
    save_path: Path,
    predicted_mapping: dict[str, str] | None = None,
) -> None:
    """Heatmap of centroid cosine similarities. Rows=test modes, cols=train modes."""
    if predicted_mapping is None:
        predicted_mapping = PREDICTED_MAPPING

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(col_modes)))
    ax.set_xticklabels(col_modes, rotation=45, ha="right")
    ax.set_yticks(range(len(row_modes)))
    ax.set_yticklabels(row_modes)
    ax.set_xlabel("Train modes (centroids)")
    ax.set_ylabel("Test modes")
    ax.set_title(title)

    # Annotate cells
    for i in range(len(row_modes)):
        for j in range(len(col_modes)):
            row_name = row_modes[i]
            col_name = col_modes[j]
            is_predicted = predicted_mapping.get(row_name) == col_name
            fontweight = "bold" if is_predicted else "normal"
            color = "white" if abs(sim_matrix[i, j]) > 0.5 else "black"
            ax.text(
                j, i, f"{sim_matrix[i, j]:.2f}",
                ha="center", va="center", fontsize=9,
                fontweight=fontweight, color=color,
            )

    fig.colorbar(im, ax=ax, label="Cosine similarity")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {save_path}")


def plot_assignment_matrix(
    assignment_matrix: np.ndarray,
    row_modes: list[str],
    col_modes: list[str],
    title: str,
    save_path: Path,
    predicted_mapping: dict[str, str] | None = None,
) -> None:
    """Heatmap of per-sample nearest-centroid assignments. Rows=test, cols=train."""
    if predicted_mapping is None:
        predicted_mapping = PREDICTED_MAPPING

    # Normalize rows to proportions
    row_sums = assignment_matrix.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1.0
    prop_matrix = assignment_matrix.astype(float) / row_sums

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    im = ax.imshow(prop_matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(col_modes)))
    ax.set_xticklabels(col_modes, rotation=45, ha="right")
    ax.set_yticks(range(len(row_modes)))
    ax.set_yticklabels(row_modes)
    ax.set_xlabel("Nearest centroid (train modes)")
    ax.set_ylabel("Test modes")
    ax.set_title(title)

    # Annotate with counts and proportions
    for i in range(len(row_modes)):
        for j in range(len(col_modes)):
            count = assignment_matrix[i, j]
            prop = prop_matrix[i, j]
            row_name = row_modes[i]
            col_name = col_modes[j]
            is_predicted = predicted_mapping.get(row_name) == col_name
            fontweight = "bold" if is_predicted else "normal"
            color = "white" if prop > 0.5 else "black"
            ax.text(
                j, i, f"{count}\n({prop:.0%})",
                ha="center", va="center", fontsize=9,
                fontweight=fontweight, color=color,
            )

    fig.colorbar(im, ax=ax, label="Proportion")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def run_transfer_multi_seed(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    modes_train: list[str],
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    modes_test: list[str],
    predicted_mapping: dict[str, str],
    direction_label: str = "forward",
    wildcard_mode: str | None = None,
    n_seeds: int = 10,
    bottleneck_dim: int = 32,
    n_epochs: int = 200,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Run cross-run transfer with multiple seeds, aggregate results.

    Trains MLP on train data, freezes, embeds test data. Evaluates nearest-centroid
    mapping of test modes to train mode centroids against predicted_mapping.

    Args:
        X_train_scaled: Scaled training features
        y_train: Training labels (integer-encoded)
        modes_train: Training mode names (in label order)
        X_test_scaled: Scaled test features (same scaler as training)
        y_test: Test labels (integer-encoded)
        modes_test: Test mode names (in label order)
        predicted_mapping: {test_mode: train_mode} predictions
        direction_label: Label for logging ("forward" or "reverse")
        wildcard_mode: Test mode with no prediction to track separately
        n_seeds: Number of random seeds
        bottleneck_dim: MLP bottleneck dimensionality
        n_epochs: MLP training epochs

    Returns:
        (results_dict, best_emb_train, best_emb_test)
    """
    logger.info(f"=== Cross-run transfer [{direction_label}] ({n_seeds} seeds) ===")

    all_mapping_accuracies: list[float] = []
    all_modes_correct: list[int] = []
    all_per_mode_accuracies: dict[str, list[float]] = {
        name: [] for name in predicted_mapping
    }
    all_sim_matrices: list[np.ndarray] = []
    all_assignment_matrices: list[np.ndarray] = []

    # Track wildcard mode assignments across seeds
    wildcard_assignments: dict[str, int] | None = None
    if wildcard_mode and wildcard_mode in modes_test:
        wildcard_assignments = {name: 0 for name in modes_train}

    best_seed = -1
    best_accuracy = -1.0
    best_emb_train: np.ndarray | None = None
    best_emb_test: np.ndarray | None = None

    for seed_i in range(n_seeds):
        torch_seed = seed_i * 7 + 42

        model, final_loss = train_full_data_mlp(
            X_train_scaled, y_train,
            bottleneck_dim=bottleneck_dim, n_epochs=n_epochs,
            torch_seed=torch_seed,
        )

        emb_train = embed_with_model(model, X_train_scaled)
        emb_test = embed_with_model(model, X_test_scaled)

        # Compute centroids and mapping
        # sim_matrix: test_modes (rows) × train_modes (cols)
        train_centroids, test_centroids, sim_matrix = compute_centroid_matrix(
            emb_train, y_train, modes_train, emb_test, y_test, modes_test,
        )
        all_sim_matrices.append(sim_matrix)

        assignment_matrix, _ = compute_nearest_centroid_mapping(
            emb_test, y_test, modes_test, train_centroids, modes_train,
        )
        all_assignment_matrices.append(assignment_matrix)

        mapping_eval = evaluate_mapping_accuracy(
            assignment_matrix, modes_test, modes_train, predicted_mapping,
        )

        all_mapping_accuracies.append(mapping_eval["overall_accuracy"])
        all_modes_correct.append(mapping_eval["modes_correct"])

        for test_name, info in mapping_eval["per_mode"].items():
            if test_name in all_per_mode_accuracies and "accuracy" in info:
                all_per_mode_accuracies[test_name].append(info["accuracy"])

        # Track wildcard mode assignments
        if wildcard_assignments is not None and wildcard_mode in modes_test:
            wc_idx = modes_test.index(wildcard_mode)
            wc_nearest = modes_train[assignment_matrix[wc_idx].argmax()]
            wildcard_assignments[wc_nearest] += 1

        if mapping_eval["overall_accuracy"] > best_accuracy:
            best_accuracy = mapping_eval["overall_accuracy"]
            best_seed = seed_i
            best_emb_train = emb_train
            best_emb_test = emb_test

        logger.info(
            f"  Seed {seed_i}: mapping_acc={mapping_eval['overall_accuracy']:.2%}, "
            f"modes_correct={mapping_eval['modes_correct']}/{mapping_eval['modes_total']}, "
            f"loss={final_loss:.4f}"
        )

    # Aggregate
    acc_arr = np.array(all_mapping_accuracies)
    modes_arr = np.array(all_modes_correct)

    # Median similarity matrix and mean assignment matrix
    median_sim = np.median(np.stack(all_sim_matrices), axis=0)
    mean_assignment = np.mean(np.stack(all_assignment_matrices).astype(float), axis=0)

    results: dict[str, Any] = {
        "direction": direction_label,
        "n_seeds": n_seeds,
        "mapping_accuracy_mean": float(np.mean(acc_arr)),
        "mapping_accuracy_median": float(np.median(acc_arr)),
        "mapping_accuracy_std": float(np.std(acc_arr)),
        "mapping_accuracy_per_seed": [float(a) for a in all_mapping_accuracies],
        "modes_correct_mean": float(np.mean(modes_arr)),
        "modes_correct_per_seed": [int(m) for m in all_modes_correct],
        "chance_accuracy": 1.0 / len(modes_train),
        "per_mode_accuracy": {
            name: {
                "mean": float(np.mean(accs)),
                "std": float(np.std(accs)),
                "per_seed": [float(a) for a in accs],
            }
            for name, accs in all_per_mode_accuracies.items()
        },
        "median_similarity_matrix": median_sim.tolist(),
        "mean_assignment_matrix": mean_assignment.tolist(),
        "similarity_rows": modes_test,
        "similarity_cols": modes_train,
        "best_seed": best_seed,
        "predicted_mapping": predicted_mapping,
    }

    if wildcard_assignments is not None:
        results["wildcard_mode"] = wildcard_mode
        results["wildcard_assignments"] = wildcard_assignments

    logger.info(f"\n  [{direction_label}] Aggregate: "
                f"mapping_acc={results['mapping_accuracy_mean']:.2%} "
                f"+/- {results['mapping_accuracy_std']:.2%} "
                f"(chance={results['chance_accuracy']:.2%})")
    logger.info(f"  Modes correct (mean): {results['modes_correct_mean']:.1f}/{len(predicted_mapping)}")
    for name, info in results["per_mode_accuracy"].items():
        predicted = predicted_mapping[name]
        logger.info(f"    {name} → {predicted}: {info['mean']:.2%} +/- {info['std']:.2%}")
    if wildcard_assignments is not None:
        logger.info(f"  {wildcard_mode} lands on: {wildcard_assignments}")

    assert best_emb_train is not None and best_emb_test is not None
    return results, best_emb_train, best_emb_test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-run transfer experiment")
    parser.add_argument("--n-seeds", type=int, default=10,
                        help="Number of random seeds for MLP training (default: 10)")
    parser.add_argument("--bottleneck-dim", type=int, default=32,
                        help="MLP bottleneck dimensionality (default: 32)")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="MLP training epochs (default: 200)")
    args = parser.parse_args()

    t_start = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    X_r4, y_r4, modes_r4, labels_r4 = load_run4_features()
    X_r3, y_r3, modes_r3, labels_r3 = load_run3_features()

    assert X_r4.shape[1] == X_r3.shape[1], (
        f"Feature dimensions must match: Run 4={X_r4.shape[1]}, Run 3={X_r3.shape[1]}"
    )

    # Scale using Run 4 statistics (the training distribution)
    scaler = StandardScaler()
    X_r4_scaled = scaler.fit_transform(X_r4)
    X_r3_scaled = scaler.transform(X_r3)

    all_results: dict[str, Any] = {
        "experiment": "cross_run_transfer",
        "run4_n_samples": int(X_r4.shape[0]),
        "run3_n_samples": int(X_r3.shape[0]),
        "n_features": int(X_r4.shape[1]),
        "run4_modes": modes_r4,
        "run3_modes": modes_r3,
        "predicted_mapping": PREDICTED_MAPPING,
    }

    # --- Part 1a: Forward transfer (train Run 4, embed Run 3) ---
    fwd_results, best_emb_r4_fwd, best_emb_r3_fwd = run_transfer_multi_seed(
        X_r4_scaled, y_r4, modes_r4,
        X_r3_scaled, y_r3, modes_r3,
        predicted_mapping=PREDICTED_MAPPING,
        direction_label="forward (train R4 → embed R3)",
        wildcard_mode="compressed",
        n_seeds=args.n_seeds,
        bottleneck_dim=args.bottleneck_dim,
        n_epochs=args.n_epochs,
    )
    all_results["transfer_forward"] = fwd_results

    # --- Part 1b: Reverse transfer (train Run 3, embed Run 4) ---
    # Scaler for reverse: fit on Run 3 (training distribution)
    scaler_rev = StandardScaler()
    X_r3_scaled_rev = scaler_rev.fit_transform(X_r3)
    X_r4_scaled_rev = scaler_rev.transform(X_r4)

    rev_results, best_emb_r3_rev, best_emb_r4_rev = run_transfer_multi_seed(
        X_r3_scaled_rev, y_r3, modes_r3,
        X_r4_scaled_rev, y_r4, modes_r4,
        predicted_mapping=PREDICTED_MAPPING_REVERSE,
        direction_label="reverse (train R3 → embed R4)",
        wildcard_mode="contrastive",
        n_seeds=args.n_seeds,
        bottleneck_dim=args.bottleneck_dim,
        n_epochs=args.n_epochs,
    )
    all_results["transfer_reverse"] = rev_results

    # --- Detailed analysis of best forward seed ---
    logger.info(f"\n=== Forward: detailed analysis of best seed ({fwd_results['best_seed']}) ===")
    r4_centroids, r3_centroids, sim_matrix_fwd = compute_centroid_matrix(
        best_emb_r4_fwd, y_r4, modes_r4, best_emb_r3_fwd, y_r3, modes_r3,
    )
    assignment_fwd, _ = compute_nearest_centroid_mapping(
        best_emb_r3_fwd, y_r3, modes_r3, r4_centroids, modes_r4,
    )
    best_eval_fwd = evaluate_mapping_accuracy(
        assignment_fwd, modes_r3, modes_r4, PREDICTED_MAPPING,
    )
    all_results["best_seed_detail_forward"] = {
        "similarity_matrix": sim_matrix_fwd.tolist(),
        "assignment_matrix": assignment_fwd.tolist(),
        "mapping_evaluation": best_eval_fwd,
    }

    logger.info(f"  Assignment matrix (R3 mode → nearest R4 centroid):")
    logger.info(f"  {'':>15} " + " ".join(f"{m:>12}" for m in modes_r4))
    for i, r3_name in enumerate(modes_r3):
        row = " ".join(f"{assignment_fwd[i, j]:>12}" for j in range(len(modes_r4)))
        predicted = PREDICTED_MAPPING.get(r3_name, "?")
        logger.info(f"  {r3_name:>15} {row}  (predicted: {predicted})")

    # --- Detailed analysis of best reverse seed ---
    logger.info(f"\n=== Reverse: detailed analysis of best seed ({rev_results['best_seed']}) ===")
    r3_centroids_rev, r4_centroids_rev, sim_matrix_rev = compute_centroid_matrix(
        best_emb_r3_rev, y_r3, modes_r3, best_emb_r4_rev, y_r4, modes_r4,
    )
    assignment_rev, _ = compute_nearest_centroid_mapping(
        best_emb_r4_rev, y_r4, modes_r4, r3_centroids_rev, modes_r3,
    )
    best_eval_rev = evaluate_mapping_accuracy(
        assignment_rev, modes_r4, modes_r3, PREDICTED_MAPPING_REVERSE,
    )
    all_results["best_seed_detail_reverse"] = {
        "similarity_matrix": sim_matrix_rev.tolist(),
        "assignment_matrix": assignment_rev.tolist(),
        "mapping_evaluation": best_eval_rev,
    }

    logger.info(f"  Assignment matrix (R4 mode → nearest R3 centroid):")
    logger.info(f"  {'':>15} " + " ".join(f"{m:>12}" for m in modes_r3))
    for i, r4_name in enumerate(modes_r4):
        row = " ".join(f"{assignment_rev[i, j]:>12}" for j in range(len(modes_r3)))
        predicted = PREDICTED_MAPPING_REVERSE.get(r4_name, "?")
        logger.info(f"  {r4_name:>15} {row}  (predicted: {predicted})")

    # --- Visualizations (forward) ---
    plot_similarity_matrix(
        sim_matrix_fwd, modes_r3, modes_r4,
        "Forward: centroid similarity (best seed)",
        FIGURES_DIR / "transfer_similarity_forward.png",
        predicted_mapping=PREDICTED_MAPPING,
    )

    plot_assignment_matrix(
        assignment_fwd, modes_r3, modes_r4,
        "Forward: nearest-centroid mapping (best seed)",
        FIGURES_DIR / "transfer_assignment_forward.png",
        predicted_mapping=PREDICTED_MAPPING,
    )

    plot_transfer_scatter(
        best_emb_r4_fwd, y_r4, modes_r4,
        best_emb_r3_fwd, y_r3, modes_r3,
        "Forward: Run 4 (trained) + Run 3 (embedded)",
        FIGURES_DIR / "transfer_scatter_forward.png",
        train_label="R4", test_label="R3",
    )

    # --- Visualizations (reverse) ---
    plot_similarity_matrix(
        sim_matrix_rev, modes_r4, modes_r3,
        "Reverse: centroid similarity (best seed)",
        FIGURES_DIR / "transfer_similarity_reverse.png",
        predicted_mapping=PREDICTED_MAPPING_REVERSE,
    )

    plot_assignment_matrix(
        assignment_rev, modes_r4, modes_r3,
        "Reverse: nearest-centroid mapping (best seed)",
        FIGURES_DIR / "transfer_assignment_reverse.png",
        predicted_mapping=PREDICTED_MAPPING_REVERSE,
    )

    plot_transfer_scatter(
        best_emb_r3_rev, y_r3, modes_r3,
        best_emb_r4_rev, y_r4, modes_r4,
        "Reverse: Run 3 (trained) + Run 4 (embedded)",
        FIGURES_DIR / "transfer_scatter_reverse.png",
        train_label="R3", test_label="R4",
    )

    # --- Part 2: LDA direction projection test ---
    lda_results, X_r3_lda, X_r4_lda = run_lda_direction_test(
        X_r3, y_r3, modes_r3,
        X_r4, y_r4, modes_r4,
    )
    all_results["lda_direction_test"] = lda_results

    # LDA scatter: Run 4 in Run 3's LDA space
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Run 3 in its own LDA space
    colors_r3 = plt.cm.Set1(np.linspace(0, 0.8, len(modes_r3)))
    for i, name in enumerate(modes_r3):
        mask = y_r3 == i
        axes[0].scatter(
            X_r3_lda[mask, 0], X_r3_lda[mask, 1],
            c=[colors_r3[i]], label=name, alpha=0.7, s=50,
            edgecolors="k", linewidth=0.3,
        )
    axes[0].set_title(f"Run 3 in own LDA (sil={lda_results['run3_sil_own_lda']:.3f})")
    axes[0].set_xlabel("LDA 1")
    axes[0].set_ylabel("LDA 2")
    axes[0].legend(fontsize=8)

    # Right: Run 4 in Run 3's LDA space
    colors_r4 = plt.cm.Set2(np.linspace(0, 0.8, len(modes_r4)))
    for i, name in enumerate(modes_r4):
        mask = y_r4 == i
        axes[1].scatter(
            X_r4_lda[mask, 0], X_r4_lda[mask, 1],
            c=[colors_r4[i]], label=name, alpha=0.7, s=50,
            edgecolors="k", linewidth=0.3,
        )
    axes[1].set_title(f"Run 4 in Run 3 LDA (sil={lda_results['run4_sil_in_run3_lda']:.3f})")
    axes[1].set_xlabel("LDA 1")
    axes[1].set_ylabel("LDA 2")
    axes[1].legend(fontsize=8)

    fig.suptitle("LDA Direction Projection Test", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "lda_direction_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {FIGURES_DIR / 'lda_direction_test.png'}")

    # --- Summary ---
    elapsed = time.time() - t_start
    all_results["elapsed_seconds"] = float(elapsed)

    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT 1.5 SUMMARY")
    logger.info(f"{'='*60}")

    for label, res, mapping in [
        ("Forward (train R4 → embed R3)", fwd_results, PREDICTED_MAPPING),
        ("Reverse (train R3 → embed R4)", rev_results, PREDICTED_MAPPING_REVERSE),
    ]:
        logger.info(f"\n{label}:")
        logger.info(f"  Mapping accuracy: {res['mapping_accuracy_mean']:.2%} "
                    f"+/- {res['mapping_accuracy_std']:.2%} "
                    f"(chance={res['chance_accuracy']:.2%})")
        logger.info(f"  Modes correct (mean): {res['modes_correct_mean']:.1f}/{len(mapping)}")
        logger.info(f"  Per-mode mapping accuracy:")
        for name, info in res["per_mode_accuracy"].items():
            predicted = mapping[name]
            logger.info(f"    {name} → {predicted}: {info['mean']:.2%}")
        if "wildcard_assignments" in res:
            logger.info(f"  {res['wildcard_mode']} lands on: {res['wildcard_assignments']}")

    logger.info(f"\nLDA direction projection test:")
    logger.info(f"  Run 3 sil in own LDA: {lda_results['run3_sil_own_lda']:.4f}")
    logger.info(f"  Run 4 sil in Run 3 LDA: {lda_results['run4_sil_in_run3_lda']:.4f}")
    logger.info(f"  Interpretation: {'PASS' if lda_results['run4_sil_in_run3_lda'] < 0.05 else 'UNEXPECTED'} "
                f"— Run 3 linear directions {'do NOT' if lda_results['run4_sil_in_run3_lda'] < 0.05 else 'DO'} "
                f"carry Run 4 mode information")

    logger.info(f"\nElapsed: {elapsed:.1f}s")

    # --- Save ---
    out_path = OUTPUT_DIR / "cross_run_transfer.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
