"""Section 8: Contrastive projection (MLP + triplet loss)."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from .signature_io import AnalysisData, BASELINE_TIERS, ENGINEERED_TIERS
from .schemas import (
    CapacitySweepEntry,
    ContrastiveAblationEntry,
    ContrastivePairwiseEntry,
    ContrastiveResult,
    ContrastiveSuperAdditivity,
    ContrastiveTierAblation,
    ContrastiveTierResult,
    LinearBaselineEntry,
)

# Try to import torch — this section is optional
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _build_topic_folds(
    topics: NDArray, n_folds: int = 5, seed: int = 42,
) -> list[tuple[NDArray[np.bool_], NDArray[np.bool_]]]:
    """Create topic-heldout train/test masks."""
    unique_topics = sorted(set(topics))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_topics))
    fold_size = len(unique_topics) // n_folds
    folds = []
    for i in range(n_folds):
        test_topic_idx = perm[i * fold_size : (i + 1) * fold_size]
        test_topics = {unique_topics[j] for j in test_topic_idx}
        test_mask = np.array([t in test_topics for t in topics])
        train_mask = ~test_mask
        folds.append((train_mask, test_mask))
    return folds


def _train_contrastive_mlp(
    X_train: NDArray, y_train: NDArray,
    hidden_dim: int = 256, bottleneck_dim: int = 32,
    n_epochs: int = 200, lr: float = 1e-3, margin: float = 1.0,
    seed: int = 42,
) -> tuple:
    """Train MLP with triplet loss, return model weights for inference."""
    if not HAS_TORCH:
        raise ImportError("PyTorch required for contrastive projection")

    torch.manual_seed(seed)
    input_dim = X_train.shape[1]

    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.5),  # Aggressive regularization for n≈80
        nn.Linear(hidden_dim, bottleneck_dim),
    )
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    triplet_loss = nn.TripletMarginLoss(margin=margin)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    unique_labels = sorted(set(y_train))
    label_to_idx: dict[str, list[int]] = {l: [] for l in unique_labels}
    for i, l in enumerate(y_train):
        label_to_idx[l].append(i)

    rng = np.random.default_rng(seed)

    for epoch in range(n_epochs):
        anchors, positives, negatives = [], [], []
        for _ in range(min(256, len(X_train) * 2)):
            label = unique_labels[rng.integers(len(unique_labels))]
            if len(label_to_idx[label]) < 2:
                continue
            a_idx, p_idx = rng.choice(label_to_idx[label], size=2, replace=False)
            neg_label = unique_labels[rng.integers(len(unique_labels))]
            while neg_label == label:
                neg_label = unique_labels[rng.integers(len(unique_labels))]
            n_idx = rng.choice(label_to_idx[neg_label])
            anchors.append(a_idx)
            positives.append(p_idx)
            negatives.append(n_idx)

        if not anchors:
            continue

        a = model(X_t[anchors])
        p = model(X_t[positives])
        n = model(X_t[negatives])

        a = nn.functional.normalize(a, dim=1)
        p = nn.functional.normalize(p, dim=1)
        n = nn.functional.normalize(n, dim=1)

        loss = triplet_loss(a, p, n)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def _embed(model, X: NDArray) -> NDArray:
    """Project data through trained model."""
    if not HAS_TORCH:
        raise ImportError("PyTorch required")
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        emb = model(X_t)
        emb = nn.functional.normalize(emb, dim=1)
    return emb.numpy()


def run_contrastive(data: AnalysisData) -> ContrastiveResult:
    """Run contrastive projection analysis."""
    if not HAS_TORCH:
        return ContrastiveResult(error="PyTorch not installed — skipping contrastive projection")

    tier_results: dict[str, ContrastiveTierResult] = {}

    for tier_name in ["T2+T2.5", "combined"]:
        print(f"    Contrastive: {tier_name}")
        X = data.get_tier(tier_name)
        X_scaled = StandardScaler().fit_transform(X)

        folds = _build_topic_folds(data.topics, n_folds=5, seed=42)

        fold_accs: list[float] = []
        fold_sils: list[float] = []

        for train_mask, test_mask in folds:
            y_train = data.modes[train_mask]
            y_test = data.modes[test_mask]

            try:
                model = _train_contrastive_mlp(X_scaled[train_mask], y_train)
                emb_train = _embed(model, X_scaled[train_mask])
                emb_test = _embed(model, X_scaled[test_mask])

                knn = KNeighborsClassifier(n_neighbors=3)
                knn.fit(emb_train, y_train)
                y_pred = knn.predict(emb_test)
                fold_accs.append(float(np.mean(y_pred == y_test)))

                if len(set(y_test)) > 1:
                    fold_sils.append(float(silhouette_score(emb_test, y_test)))
            except Exception as e:
                fold_accs.append(0.0)
                print(f"      Fold failed: {e}")

        tier_results[tier_name] = ContrastiveTierResult(
            knn_accuracy_mean=float(np.mean(fold_accs)) if fold_accs else 0.0,
            knn_accuracy_std=float(np.std(fold_accs)) if fold_accs else 0.0,
            knn_fold_accs=fold_accs,
            silhouette_mean=float(np.mean(fold_sils)) if fold_sils else None,
        )

    # Capacity sweep (T2+T2.5 only)
    print("    Capacity sweep...")
    X_key = StandardScaler().fit_transform(data.get_tier("T2+T2.5"))
    folds = _build_topic_folds(data.topics, n_folds=5, seed=42)
    capacities = [64, 128, 256, 512]
    capacity_results: dict[str, CapacitySweepEntry] = {}

    for hidden_dim in capacities:
        fold_accs = []
        fold_sils = []
        for train_mask, test_mask in folds:
            try:
                model = _train_contrastive_mlp(
                    X_key[train_mask], data.modes[train_mask],
                    hidden_dim=hidden_dim, bottleneck_dim=32,
                )
                emb_train = _embed(model, X_key[train_mask])
                emb_test = _embed(model, X_key[test_mask])

                knn = KNeighborsClassifier(n_neighbors=3)
                knn.fit(emb_train, data.modes[train_mask])
                y_pred = knn.predict(emb_test)
                fold_accs.append(float(np.mean(y_pred == data.modes[test_mask])))

                if len(set(data.modes[test_mask])) > 1:
                    fold_sils.append(float(silhouette_score(emb_test, data.modes[test_mask])))
            except Exception:
                fold_accs.append(0.0)

        capacity_results[str(hidden_dim)] = CapacitySweepEntry(
            knn_accuracy=float(np.mean(fold_accs)) if fold_accs else 0.0,
            silhouette=float(np.mean(fold_sils)) if fold_sils else None,
        )

    # Contrastive tier ablation (per-tier + pairwise)
    print("    Contrastive tier ablation...")
    ablation = _run_contrastive_tier_ablation(data)

    # Linear projection baselines (LDA / NCA) — compare to nonlinear MLP
    print("    Linear projection baselines...")
    baselines = _run_linear_baselines(data)

    return ContrastiveResult(
        T2_T2_5=tier_results["T2+T2.5"],
        combined=tier_results["combined"],
        capacity_sweep=capacity_results,
        tier_ablation=ablation,
        linear_baselines=baselines,
    )


def _run_linear_baselines(data: AnalysisData) -> dict[str, LinearBaselineEntry]:
    """LDA and NCA projection baselines on T2+T2.5."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.neighbors import NeighborhoodComponentsAnalysis

    X = data.get_tier("T2+T2.5")
    X_scaled = StandardScaler().fit_transform(X)
    folds = _build_topic_folds(data.topics, n_folds=5, seed=42)
    n_components = min(len(data.unique_modes) - 1, X.shape[1])

    results: dict[str, LinearBaselineEntry] = {}

    for name, ProjectorClass, projector_kwargs in [
        ("LDA", LinearDiscriminantAnalysis, {}),
        ("NCA", NeighborhoodComponentsAnalysis, {
            "n_components": n_components, "max_iter": 500, "random_state": 42,
        }),
    ]:
        print(f"      {name}...")
        fold_accs: list[float] = []
        fold_sils: list[float] = []

        for train_mask, test_mask in folds:
            try:
                proj = ProjectorClass(**projector_kwargs)
                X_train_proj = proj.fit_transform(
                    X_scaled[train_mask], data.modes[train_mask],
                )
                X_test_proj = proj.transform(X_scaled[test_mask])

                knn = KNeighborsClassifier(n_neighbors=3)
                knn.fit(X_train_proj, data.modes[train_mask])
                y_pred = knn.predict(X_test_proj)
                fold_accs.append(float(np.mean(y_pred == data.modes[test_mask])))

                if len(set(data.modes[test_mask])) > 1:
                    fold_sils.append(float(silhouette_score(
                        X_test_proj, data.modes[test_mask],
                    )))
            except Exception as e:
                fold_accs.append(0.0)
                print(f"        {name} fold failed: {e}")

        results[name] = LinearBaselineEntry(
            knn_accuracy=float(np.mean(fold_accs)) if fold_accs else 0.0,
            knn_std=float(np.std(fold_accs)) if fold_accs else 0.0,
            silhouette=float(np.mean(fold_sils)) if fold_sils else None,
            n_components=n_components,
        )

    return results


def _eval_contrastive_tier(
    X: NDArray, modes: NDArray, topics: NDArray, seed: int = 42,
) -> ContrastiveAblationEntry:
    """Run contrastive MLP + kNN evaluation on a single feature set."""
    X_scaled = StandardScaler().fit_transform(X)
    folds = _build_topic_folds(topics, n_folds=5, seed=seed)

    fold_accs: list[float] = []
    fold_sils: list[float] = []

    for train_mask, test_mask in folds:
        try:
            model = _train_contrastive_mlp(
                X_scaled[train_mask], modes[train_mask], seed=seed,
            )
            emb_train = _embed(model, X_scaled[train_mask])
            emb_test = _embed(model, X_scaled[test_mask])

            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(emb_train, modes[train_mask])
            y_pred = knn.predict(emb_test)
            fold_accs.append(float(np.mean(y_pred == modes[test_mask])))

            if len(set(modes[test_mask])) > 1:
                fold_sils.append(float(silhouette_score(emb_test, modes[test_mask])))
        except Exception as e:
            fold_accs.append(0.0)
            print(f"        Fold failed: {e}")

    return ContrastiveAblationEntry(
        knn_accuracy=float(np.mean(fold_accs)) if fold_accs else 0.0,
        knn_std=float(np.std(fold_accs)) if fold_accs else 0.0,
        silhouette=float(np.mean(fold_sils)) if fold_sils else None,
        n_features=int(X.shape[1]),
    )


def _run_contrastive_tier_ablation(data: AnalysisData) -> ContrastiveTierAblation:
    """Contrastive MLP tier ablation: individual tiers, all pairs, key combos."""
    present_baseline = [t for t in BASELINE_TIERS if t in data.run4.tier_features]
    present_engineered = [t for t in ENGINEERED_TIERS if t in data.run4.tier_features]
    all_individual = present_baseline + present_engineered

    # Individual tiers
    individual: dict[str, ContrastiveAblationEntry] = {}
    for tier in all_individual:
        print(f"      Individual: {tier}")
        individual[tier] = _eval_contrastive_tier(data.get_tier(tier), data.modes, data.topics)

    # Pairwise combinations (within baseline only — bounded)
    pairwise: dict[str, ContrastivePairwiseEntry] = {}
    for t1, t2 in combinations(present_baseline, 2):
        key = f"{t1}+{t2}"
        print(f"      Pair: {key}")
        X_pair = np.concatenate([data.get_tier(t1), data.get_tier(t2)], axis=1)
        result = _eval_contrastive_tier(X_pair, data.modes, data.topics)
        best_individual = max(
            individual[t1].knn_accuracy,
            individual[t2].knn_accuracy,
        )
        pairwise[key] = ContrastivePairwiseEntry(
            knn_accuracy=result.knn_accuracy,
            knn_std=result.knn_std,
            silhouette=result.silhouette,
            n_features=result.n_features,
            best_individual_knn=best_individual,
            gain_over_best_individual=result.knn_accuracy - best_individual,
        )

    print("      Combo: T2+T2.5")
    t2t25 = _eval_contrastive_tier(data.get_tier("T2+T2.5"), data.modes, data.topics)
    print("      Combo: combined")
    combined = _eval_contrastive_tier(data.get_tier("combined"), data.modes, data.topics)

    t2_knn = individual["T2"].knn_accuracy
    t25_knn = individual["T2.5"].knn_accuracy
    t2t25_knn = pairwise["T2+T2.5"].knn_accuracy
    super_add = ContrastiveSuperAdditivity.model_validate({
        "T2_alone": t2_knn,
        "T2.5_alone": t25_knn,
        "T2+T2.5_pair": t2t25_knn,
        "best_individual": max(t2_knn, t25_knn),
        "gain": t2t25_knn - max(t2_knn, t25_knn),
        "combined_knn": combined.knn_accuracy,
        "T2+T2.5_beats_combined": t2t25_knn > combined.knn_accuracy,
    })

    return ContrastiveTierAblation.model_validate({
        "individual": individual,
        "pairwise": pairwise,
        "T2+T2.5": t2t25,
        "combined": combined,
        "super_additivity": super_add,
    })
