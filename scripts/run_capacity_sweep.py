#!/usr/bin/env python3
"""Capacity sensitivity check: does test performance depend on hidden dim?

Tests hidden_dim = [64, 128, 256, 512] with same dropout/WD/architecture.
If test sil/kNN is stable across capacities, the result reflects genuine
manifold structure, not a capacity-dependent artifact.

Runs on T2+T2.5 (366 features) and combined (1837 features).
Expected runtime: ~2 minutes on CPU.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class ProjectionNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, drop: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.net(x), p=2, dim=1)


def mine_triplets(
    labels: np.ndarray, rng: np.random.RandomState, n_triplets: int = 300
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def train_and_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_dim: int = 256,
    bottleneck_dim: int = 32,
    n_epochs: int = 200,
    torch_seed: int = 42,
) -> dict[str, float]:
    """Train MLP and return test metrics."""
    torch.manual_seed(torch_seed)
    rng = np.random.RandomState(torch_seed)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    model = ProjectionNet(X_train.shape[1], hidden_dim, bottleneck_dim, drop=0.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.TripletMarginLoss(margin=1.0)

    n_params = sum(p.numel() for p in model.parameters())

    model.train()
    final_loss = 0.0
    for _ in range(n_epochs):
        a, p, n = mine_triplets(y_train, rng)
        if len(a) < 10:
            continue
        emb = model(X_train_t)
        loss = loss_fn(emb[a], emb[p], emb[n])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        emb_train = model(X_train_t).numpy()
        emb_test = model(X_test_t).numpy()

    sil_train = float(silhouette_score(emb_train, y_train, metric="cosine"))
    sil_test = float("nan")
    if len(np.unique(y_test)) > 1:
        sil_test = float(silhouette_score(emb_test, y_test, metric="cosine"))

    knn = KNeighborsClassifier(n_neighbors=3, metric="cosine")
    knn.fit(emb_train, y_train)
    knn_acc = float(knn.score(emb_test, y_test))

    return {
        "sil_train": sil_train,
        "sil_test": sil_test,
        "knn_acc": knn_acc,
        "final_loss": final_loss,
        "n_params": n_params,
    }


def run_capacity_sweep(
    X: np.ndarray,
    y: np.ndarray,
    label: str,
    hidden_dims: list[int],
) -> dict[str, Any]:
    """Run the same CV across multiple hidden dims."""
    logger.info(f"\n=== Capacity sweep: {label} ({X.shape[1]} features) ===")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results_by_dim: dict[int, dict[str, Any]] = {}

    for hdim in hidden_dims:
        fold_sils: list[float] = []
        fold_knns: list[float] = []
        fold_sils_train: list[float] = []
        fold_losses: list[float] = []

        for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            y_train, y_test = y[train_idx], y[test_idx]

            metrics = train_and_eval(
                X_train, y_train, X_test, y_test,
                hidden_dim=hdim, torch_seed=42 + fold_i,
            )
            fold_sils.append(metrics["sil_test"])
            fold_knns.append(metrics["knn_acc"])
            fold_sils_train.append(metrics["sil_train"])
            fold_losses.append(metrics["final_loss"])

        mean_sil = float(np.nanmean(fold_sils))
        mean_knn = float(np.mean(fold_knns))
        std_sil = float(np.nanstd(fold_sils))
        std_knn = float(np.std(fold_knns))
        mean_sil_train = float(np.mean(fold_sils_train))
        n_params = int((X.shape[1] * hdim) + hdim + (hdim * 32) + 32)

        results_by_dim[hdim] = {
            "hidden_dim": hdim,
            "n_params": n_params,
            "param_sample_ratio": f"{n_params / 80:.0f}:1",
            "mean_sil_test": mean_sil,
            "std_sil_test": std_sil,
            "mean_knn": mean_knn,
            "std_knn": std_knn,
            "mean_sil_train": mean_sil_train,
            "fold_sils": [float(s) for s in fold_sils],
            "fold_knns": [float(k) for k in fold_knns],
        }

        logger.info(
            f"  hidden={hdim:4d} | params={n_params:>7,} ({n_params//80:>5}:1) | "
            f"train_sil={mean_sil_train:.3f} | "
            f"test_sil={mean_sil:.3f} +/- {std_sil:.3f} | "
            f"kNN={mean_knn:.1%} +/- {std_knn:.1%}"
        )

    return {
        "feature_set": label,
        "n_features": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "hidden_dims_tested": hidden_dims,
        "results": {str(k): v for k, v in results_by_dim.items()},
    }


def main() -> None:
    start_time = time.time()

    data = np.load(PROJECT_ROOT / "review_pack" / "features.npz", allow_pickle=True)
    X_full = np.nan_to_num(data["features"].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X_t2 = np.nan_to_num(data["tier2"].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X_t25 = np.nan_to_num(data["tier2_5"].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X_t2_t25 = np.hstack([X_t2, X_t25])

    le = LabelEncoder()
    y = le.fit_transform(data["labels"])
    mode_names = le.classes_.tolist()

    hidden_dims = [64, 128, 256, 512]

    logger.info(f"Samples: {X_full.shape[0]}, Modes: {mode_names}")

    all_results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hidden_dims_tested": hidden_dims,
        "bottleneck_dim": 32,
        "dropout": 0.5,
        "weight_decay": 1e-3,
        "n_epochs": 200,
        "n_folds": 5,
    }

    # T2+T2.5
    sweep_t2t25 = run_capacity_sweep(X_t2_t25, y, "T2+T2.5", hidden_dims)
    all_results["T2_T25"] = sweep_t2t25

    # Combined
    sweep_full = run_capacity_sweep(X_full, y, "combined", hidden_dims)
    all_results["combined"] = sweep_full

    elapsed = time.time() - start_time
    all_results["elapsed_seconds"] = elapsed

    output_dir = PROJECT_ROOT / "outputs" / "phase1" / "reviewer_checks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "capacity_sweep.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")
    logger.info(f"Total elapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
