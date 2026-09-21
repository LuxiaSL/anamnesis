"""Section 8: Contrastive projection (MLP + triplet loss).

The network and the law it is trained under are not defined here: they live in
:mod:`anamnesis.analysis.contrastive_mlp`, which holds both training laws this
instrument uses side by side and says why they differ. This section trains under
the analysis law — :func:`anamnesis.analysis.contrastive_mlp.train_embedding` —
because its embedding is a lens read once per fold and never banked as weights.

What this module owns is the reading built on top of that fit: topic-held-out
folds, kNN and silhouette on the held-out topics, a capacity sweep over the
network's width, a block ablation, and linear projection baselines to sit the
nonlinear number against. :func:`build_topic_folds` is public because section 9
compares its own feature types under the same folds, and a fold definition that
two sections disagreed about would make their numbers incomparable.
"""

from __future__ import annotations

import importlib.util
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from anamnesis.analysis.contrastive_mlp import embed, train_embedding

from .signature_io import (
    ALL_CORE,
    ATTENTION_AND_CACHE,
    ATTENTION_AND_DELTAS,
    CACHE_AND_KEYS,
    CORE_BLOCKS,
    FAMILY_BLOCKS,
    AnalysisData,
)
from .utils import absence_reason, topic_fold_partition
from .schemas import (
    CapacitySweepEntry,
    ContrastiveAblationEntry,
    ContrastivePairwiseEntry,
    ContrastiveResult,
    ContrastiveSuperAdditivity,
    ContrastiveBlockAblation,
    ContrastiveBlockResult,
    LinearBaselineEntry,
)

N_TOPIC_FOLDS = 5
"""How many topics this section holds out at a time. One number, because the
capacity sweep, the ablation grid and the linear baselines are compared with each
other and a fold count that differed between them would make them incomparable."""

HAS_TORCH = importlib.util.find_spec("torch") is not None
"""Whether a trainer can run at all. Probed rather than imported: this section is
optional, and a pass that reports its absence should not pay for loading torch to
find that out."""


def build_topic_folds(
    topics: NDArray, n_folds: int = N_TOPIC_FOLDS, seed: int = 42,
) -> list[tuple[NDArray[np.bool_], NDArray[np.bool_]]]:
    """Row masks for ``n_folds`` topic-held-out splits, over every topic.

    The topic partition is
    :func:`anamnesis.analysis.gauntlet.utils.topic_fold_partition`, shared with
    section 5, so two sections asked for the same seed hold out the same topics.
    Every topic is held out exactly once: a split that took ``n // n_folds``
    topics per fold would leave the remainder in no test mask at all.

    Raises
    ------
    InsufficientTopicsError
        When more folds are asked for than there are topics.
    """
    unique_topics = sorted(set(topics))
    rng = np.random.default_rng(seed)
    folds = []
    for test_topic_idx in topic_fold_partition(len(unique_topics), n_folds, rng):
        test_topics = {unique_topics[j] for j in test_topic_idx}
        test_mask = np.array([t in test_topics for t in topics])
        train_mask = ~test_mask
        folds.append((train_mask, test_mask))
    return folds


def run_contrastive(data: AnalysisData) -> ContrastiveResult:
    """Run contrastive projection analysis."""
    if not HAS_TORCH:
        return ContrastiveResult(error="PyTorch not installed — skipping contrastive projection")

    absent = absence_reason(data, ATTENTION_AND_CACHE, ALL_CORE)
    if absent is not None:
        return ContrastiveResult(error=f"contrastive projection reads {absent}")

    # Every reading below holds out topics, so a corpus with fewer topics than
    # folds is stated here rather than refused from inside the partitioner.
    n_topics = len(set(data.topics))
    if n_topics < N_TOPIC_FOLDS:
        return ContrastiveResult(
            error=(
                f"contrastive projection holds out {N_TOPIC_FOLDS} topic folds and "
                f"this corpus has {n_topics} topics"
            ),
        )

    block_results: dict[str, ContrastiveBlockResult] = {}

    for block_name in [ATTENTION_AND_CACHE, ALL_CORE]:
        print(f"    Contrastive: {block_name}")
        X = data.get_block(block_name)
        X_scaled = StandardScaler().fit_transform(X)

        folds = build_topic_folds(data.topics, n_folds=N_TOPIC_FOLDS, seed=42)

        fold_accs: list[float] = []
        fold_sils: list[float] = []

        for train_mask, test_mask in folds:
            y_train = data.modes[train_mask]
            y_test = data.modes[test_mask]

            try:
                model, _loss = train_embedding(X_scaled[train_mask], y_train)
                emb_train = embed(model, X_scaled[train_mask])
                emb_test = embed(model, X_scaled[test_mask])

                knn = KNeighborsClassifier(n_neighbors=3)
                knn.fit(emb_train, y_train)
                y_pred = knn.predict(emb_test)
                fold_accs.append(float(np.mean(y_pred == y_test)))

                if len(set(y_test)) > 1:
                    fold_sils.append(float(silhouette_score(emb_test, y_test)))
            except Exception as e:
                fold_accs.append(0.0)
                print(f"      Fold failed: {e}")

        block_results[block_name] = ContrastiveBlockResult(
            knn_accuracy_mean=float(np.mean(fold_accs)) if fold_accs else 0.0,
            knn_accuracy_std=float(np.std(fold_accs)) if fold_accs else 0.0,
            knn_fold_accs=fold_accs,
            silhouette_mean=float(np.mean(fold_sils)) if fold_sils else None,
        )

    # Capacity sweep (the attention-and-cache union only)
    print("    Capacity sweep...")
    X_key = StandardScaler().fit_transform(data.get_block(ATTENTION_AND_CACHE))
    folds = build_topic_folds(data.topics, n_folds=N_TOPIC_FOLDS, seed=42)
    capacities = [64, 128, 256, 512]
    capacity_results: dict[str, CapacitySweepEntry] = {}

    for hidden_dim in capacities:
        fold_accs = []
        fold_sils = []
        for train_mask, test_mask in folds:
            try:
                model, _loss = train_embedding(
                    X_key[train_mask], data.modes[train_mask], hidden_dim=hidden_dim,
                )
                emb_train = embed(model, X_key[train_mask])
                emb_test = embed(model, X_key[test_mask])

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

    # Contrastive block ablation (per-block + pairwise)
    print("    Contrastive block ablation...")
    ablation = _run_contrastive_block_ablation(data)

    # Linear projection baselines (LDA / NCA) — compare to nonlinear MLP
    print("    Linear projection baselines...")
    baselines = _run_linear_baselines(data)

    return ContrastiveResult(
        attention_and_cache=block_results[ATTENTION_AND_CACHE],
        combined=block_results[ALL_CORE],
        capacity_sweep=capacity_results,
        block_ablation=ablation,
        linear_baselines=baselines,
    )


def _run_linear_baselines(data: AnalysisData) -> dict[str, LinearBaselineEntry]:
    """LDA and NCA projection baselines on the attention-and-cache union."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.neighbors import NeighborhoodComponentsAnalysis

    X = data.get_block(ATTENTION_AND_CACHE)
    X_scaled = StandardScaler().fit_transform(X)
    folds = build_topic_folds(data.topics, n_folds=N_TOPIC_FOLDS, seed=42)
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


def _eval_contrastive_block(
    X: NDArray, modes: NDArray, topics: NDArray, seed: int = 42,
) -> ContrastiveAblationEntry:
    """Run contrastive MLP + kNN evaluation on a single feature set."""
    X_scaled = StandardScaler().fit_transform(X)
    folds = build_topic_folds(topics, n_folds=N_TOPIC_FOLDS, seed=seed)

    fold_accs: list[float] = []
    fold_sils: list[float] = []

    for train_mask, test_mask in folds:
        try:
            model, _loss = train_embedding(
                X_scaled[train_mask], modes[train_mask], seed=seed,
            )
            emb_train = embed(model, X_scaled[train_mask])
            emb_test = embed(model, X_scaled[test_mask])

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


def _run_contrastive_block_ablation(data: AnalysisData) -> ContrastiveBlockAblation:
    """Contrastive MLP block ablation: individual blocks, all pairs, key combos."""
    present_core = [t for t in CORE_BLOCKS if t in data.run4.block_features]
    present_families = [t for t in FAMILY_BLOCKS if t in data.run4.block_features]
    all_individual = present_core + present_families

    # Individual blocks
    individual: dict[str, ContrastiveAblationEntry] = {}
    for block in all_individual:
        print(f"      Individual: {block}")
        individual[block] = _eval_contrastive_block(data.get_block(block), data.modes, data.topics)

    # Pairwise combinations (within baseline only — bounded)
    pairwise: dict[str, ContrastivePairwiseEntry] = {}
    for left, right in combinations(present_core, 2):
        key = f"{left}+{right}"
        print(f"      Pair: {key}")
        X_pair = np.concatenate([data.get_block(left), data.get_block(right)], axis=1)
        result = _eval_contrastive_block(X_pair, data.modes, data.topics)
        best_individual = max(
            individual[left].knn_accuracy,
            individual[right].knn_accuracy,
        )
        pairwise[key] = ContrastivePairwiseEntry(
            knn_accuracy=result.knn_accuracy,
            knn_std=result.knn_std,
            silhouette=result.silhouette,
            n_features=result.n_features,
            best_individual_knn=best_individual,
            gain_over_best_individual=result.knn_accuracy - best_individual,
        )

    print("      Combo: attention and cache")
    attention_and_cache = _eval_contrastive_block(data.get_block(ATTENTION_AND_CACHE), data.modes, data.topics)
    print("      Combo: combined")
    combined = _eval_contrastive_block(data.get_block(ALL_CORE), data.modes, data.topics)

    attention_knn = individual[ATTENTION_AND_DELTAS].knn_accuracy
    cache_knn = individual[CACHE_AND_KEYS].knn_accuracy
    # The pair is keyed by its two members, which is not the union's own label.
    attention_and_cache_knn = pairwise[f"{ATTENTION_AND_DELTAS}+{CACHE_AND_KEYS}"].knn_accuracy
    super_add = ContrastiveSuperAdditivity(
        attention_alone=attention_knn,
        cache_alone=cache_knn,
        attention_and_cache_pair=attention_and_cache_knn,
        best_individual=max(attention_knn, cache_knn),
        gain=attention_and_cache_knn - max(attention_knn, cache_knn),
        combined_knn=combined.knn_accuracy,
        attention_and_cache_beats_combined=(
            attention_and_cache_knn > combined.knn_accuracy
        ),
    )

    return ContrastiveBlockAblation.model_validate({
        "individual": individual,
        "pairwise": pairwise,
        ATTENTION_AND_CACHE: attention_and_cache,
        ALL_CORE: combined,
        "super_additivity": super_add,
    })
