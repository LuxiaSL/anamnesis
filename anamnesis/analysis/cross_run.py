"""Cross-run transfer: does a mode learned in one corpus name a mode in another?

Two runs describe processing with different vocabularies. One names five
format-controlled modes; the other names five process modes with no format
constraint. If the signature axis is about *how* a generation was produced rather
than about which prompt produced it, then a projection learned on one run's modes
should place the other run's modes somewhere meaningful — and the predicted
pairs say where: a propose-challenge-revise mode should land on the dialectical
one, an interactive-explanation mode on the socratic one, and so on.

Both vocabularies, the pairs and the comparison they are read beside are rows of
:data:`anamnesis.modes.MODE_SETS_FILE`, reached by the name
:data:`anamnesis.modes.DEFAULT_MODE_MAPPING`. So every label this module reports is
one a reader can look up — including the five process modes, whose prompts are not in
this package and whose glosses are therefore what the registry can say about them. A second pair of
corpora is compared by adding a mapping row, not by editing this module.

Three readouts, and they disagree on purpose:

* **Transfer** trains the contrastive projection on one run and embeds the other,
  then asks which train-mode centroid each test sample is nearest. Scored against
  the predicted map, over several seeds, because one seed of a small network
  is one draw.
* **The wildcard** is the mode with no predicted partner. Where it lands is a
  finding rather than an error, so it is counted separately and never enters the
  accuracy.
* **The LDA direction test** asks the same question linearly: fit discriminants on
  one run, project the other into that space, and read its silhouette. A near-zero
  or negative silhouette while the transfer works is the dissociation this test
  exists for — the runs share a manifold, not a set of directions.

Both directions are run. They are not symmetric: which corpus trains is which
vocabulary the other is described in, and a pair that transfers one way and not the
other is a property of the two mode sets rather than a failed measurement.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from anamnesis.analysis.contrastive_mlp import (
    ANALYSIS_EPOCHS,
    BOTTLENECK_DIM,
    embed,
    train_embedding,
)
from anamnesis.config import outputs_root
from anamnesis.modes import (
    DEFAULT_MODE_MAPPING,
    ModeMapping,
    mapping_wildcards,
    mode_mapping,
)

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]

N_SEEDS = 10
SEED_STRIDE = 7
SEED_BASE = 42
"""Seed layout: ``i * SEED_STRIDE + SEED_BASE``. Fixed so a reported spread is
reproducible rather than merely quoted."""

KNN_NEIGHBOURS = 3
NORM_FLOOR = 1e-10

_LAYER_RE = re.compile(r"_L(\d+)(?:_|$)")
"""How a feature name carries its layer: ``_L13_`` mid-name or ``_L31`` at the end."""


def signatures_dir(run_name: str, outputs_base: Path | None = None) -> Path:
    """Where a run banks its signatures, under the configured outputs root."""
    base = Path(outputs_base) if outputs_base is not None else outputs_root()
    return base / "runs" / run_name / "signatures"


def load_feature_names(run_name: str, outputs_base: Path | None = None) -> NDArray[np.str_]:
    """The feature names a run's signatures carry, from its first banked vector."""
    sig_dir = signatures_dir(run_name, outputs_base)
    first = next(iter(sorted(sig_dir.glob("gen_*.npz"))), None)
    if first is None:
        raise FileNotFoundError(f"no gen_*.npz under {sig_dir}")
    with np.load(first, allow_pickle=True) as data:
        if "feature_names" not in data.files:
            raise KeyError(f"{first.name} carries no feature_names")
        return np.array(data["feature_names"], copy=True)


def build_layer_indices(
    feature_names: Sequence[Any],
    layers: set[int] | None = None,
    include_unlayered: bool = False,
) -> list[int]:
    """Column indices of the features belonging to the named layers.

    ``layers=None`` keeps every feature that carries a layer tag at all, which is
    the filter that drops the global output-source statistics; ``include_unlayered``
    keeps those too.
    """
    indices: list[int] = []
    for index, name in enumerate(feature_names):
        match = _LAYER_RE.search(str(name))
        if match is not None:
            if layers is None or int(match.group(1)) in layers:
                indices.append(index)
        elif include_unlayered:
            indices.append(index)
    return indices


def load_run_features(
    run_name: str,
    outputs_base: Path | None = None,
    feature_key: str | Sequence[str] = "features",
    feature_indices: Sequence[int] | None = None,
) -> tuple[F32, NDArray[np.int64], list[str], NDArray[np.str_]]:
    """One run's feature matrix and encoded mode labels, in banked order.

    ``feature_key`` names the npz key, or several keys to concatenate in the order
    given — which is how a subset of blocks is read without recomputing anything.
    ``feature_indices`` selects columns *after* that concatenation, so an index
    means a position in the vector the caller asked for.
    """
    sig_dir = signatures_dir(run_name, outputs_base)
    if not sig_dir.is_dir():
        raise FileNotFoundError(f"signatures dir not found: {sig_dir}")
    keys = [feature_key] if isinstance(feature_key, str) else list(feature_key)

    rows: list[F32] = []
    labels: list[str] = []
    for json_path in sorted(sig_dir.glob("gen_*.json")):
        npz_path = json_path.with_suffix(".npz")
        if not npz_path.exists():
            logger.warning(f"  no .npz beside {json_path.name} — skipped")
            continue
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        with np.load(npz_path, allow_pickle=True) as data:
            parts: list[F32] = []
            for key in keys:
                if key not in data.files:
                    raise KeyError(
                        f"feature key {key!r} not in {npz_path.name}; has {sorted(data.files)}"
                    )
                parts.append(np.asarray(data[key], dtype=np.float32))
        rows.append(
            np.concatenate(parts, axis=0).astype(np.float32) if len(parts) > 1 else parts[0]
        )
        labels.append(str(meta["mode"]))

    if not rows:
        raise RuntimeError(f"no feature vectors under {sig_dir}")
    X = np.nan_to_num(np.stack(rows).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if feature_indices is not None:
        selection = np.asarray(feature_indices, dtype=np.int64)
        if selection.size == 0:
            raise ValueError("feature_indices is empty")
        X = X[:, selection]

    label_array = np.array(labels)
    encoder = LabelEncoder()
    y = encoder.fit_transform(label_array).astype(np.int64)
    mode_names = [str(m) for m in encoder.classes_]
    logger.info(
        f"[{run_name}] n={X.shape[0]} samples, d={X.shape[1]} features, modes={mode_names} "
        f"(key={feature_key!r}, sliced={feature_indices is not None})"
    )
    return X, y, mode_names, label_array


def compute_centroids(
    embedded: F32, y: NDArray[np.int64], mode_names: Sequence[str]
) -> dict[str, F32]:
    """Each mode's mean embedding, keyed by its name."""
    return {name: embedded[y == index].mean(axis=0) for index, name in enumerate(mode_names)}


def similarity_matrix(
    row_centroids: dict[str, F32],
    col_centroids: dict[str, F32],
    row_order: Sequence[str],
    col_order: Sequence[str],
) -> F32:
    """Cosine similarity between two sets of centroids, rows by columns."""
    out = np.zeros((len(row_order), len(col_order)), dtype=np.float32)
    for i, row in enumerate(row_order):
        for j, col in enumerate(col_order):
            a, b = row_centroids[row], col_centroids[col]
            out[i, j] = float(
                np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + NORM_FLOOR)
            )
    return out


def nearest_centroid_assignment(
    embedded_test: F32,
    y_test: NDArray[np.int64],
    modes_test: Sequence[str],
    centroids_train: dict[str, F32],
    modes_train: Sequence[str],
) -> NDArray[np.int64]:
    """Counts of test samples per (test mode, nearest train mode).

    The embedding is already on the unit sphere, so the nearest centroid by cosine
    is the largest dot product after normalizing the centroids.
    """
    stack = np.stack([centroids_train[m] for m in modes_train])
    stack = stack / (np.linalg.norm(stack, axis=1, keepdims=True) + NORM_FLOOR)
    nearest = (embedded_test @ stack.T).argmax(axis=1)
    counts = np.zeros((len(modes_test), len(modes_train)), dtype=np.int64)
    for index in range(len(modes_test)):
        for assigned in nearest[y_test == index]:
            counts[index, int(assigned)] += 1
    return counts


class MappingScore(BaseModel):
    """One seed's scoring of an assignment against the predicted map."""

    model_config = ConfigDict(extra="forbid")

    overall_accuracy: float
    correct_mapped: int
    total_mapped: int
    modes_correct: int
    modes_total: int
    chance_accuracy: float
    per_mode: dict[str, dict[str, Any]]


def score_mapping(
    assignment: NDArray[np.int64],
    modes_test: Sequence[str],
    modes_train: Sequence[str],
    predicted: dict[str, str],
) -> MappingScore:
    """Score an assignment: how much of each predicted pair actually landed there.

    A test mode with no prediction — the wildcard — is reported with where it went
    and contributes to neither numerator nor denominator, because scoring it would
    mean scoring a hypothesis nobody registered.
    """
    total = correct = 0
    per_mode: dict[str, dict[str, Any]] = {}
    for index, test_mode in enumerate(modes_test):
        distribution = {
            modes_train[j]: int(assignment[index, j]) for j in range(len(modes_train))
        }
        n_total = int(assignment[index].sum())
        actual = modes_train[int(assignment[index].argmax())] if n_total > 0 else None
        if test_mode not in predicted:
            per_mode[test_mode] = {
                "predicted": None,
                "actual_nearest": actual,
                "samples": n_total,
                "assignment_distribution": distribution,
            }
            continue
        prediction = predicted[test_mode]
        n_correct = int(assignment[index, list(modes_train).index(prediction)])
        total += n_total
        correct += n_correct
        per_mode[test_mode] = {
            "predicted": prediction,
            "actual_nearest": actual,
            "correct": n_correct,
            "total": n_total,
            "accuracy": float(n_correct / n_total) if n_total > 0 else 0.0,
            "match": actual == prediction,
            "assignment_distribution": distribution,
        }
    return MappingScore(
        overall_accuracy=float(correct / total) if total > 0 else 0.0,
        correct_mapped=correct,
        total_mapped=total,
        modes_correct=sum(1 for v in per_mode.values() if v.get("match", False)),
        modes_total=len(predicted),
        chance_accuracy=1.0 / len(modes_train),
        per_mode=per_mode,
    )


def run_transfer(
    X_train: F32,
    y_train: NDArray[np.int64],
    modes_train: Sequence[str],
    X_test: F32,
    y_test: NDArray[np.int64],
    modes_test: Sequence[str],
    *,
    predicted: dict[str, str],
    wildcard_mode: str | None,
    direction_label: str,
    n_seeds: int = N_SEEDS,
    bottleneck_dim: int = BOTTLENECK_DIM,
    n_epochs: int = ANALYSIS_EPOCHS,
) -> dict[str, Any]:
    """Train on one run, embed the other, score the map — once per seed, then pool.

    The similarity matrices are pooled by **median** and the assignments by mean: a
    median similarity is robust to one seed's collapsed embedding, while a mean
    assignment is the expected count, which is the quantity the per-mode rates are
    already means of.
    """
    logger.info(f"=== transfer [{direction_label}] over {n_seeds} seeds ===")
    accuracies: list[float] = []
    modes_correct: list[int] = []
    per_mode: dict[str, list[float]] = {name: [] for name in predicted}
    similarities: list[F32] = []
    assignments: list[NDArray[np.int64]] = []
    wildcard_counts: dict[str, int] | None = (
        {name: 0 for name in modes_train}
        if wildcard_mode and wildcard_mode in modes_test
        else None
    )

    for index in range(n_seeds):
        seed = index * SEED_STRIDE + SEED_BASE
        model, final_loss = train_embedding(
            X_train, y_train, bottleneck_dim=bottleneck_dim, n_epochs=n_epochs, seed=seed
        )
        embedded_train = embed(model, X_train)
        embedded_test = embed(model, X_test)
        train_centroids = compute_centroids(embedded_train, y_train, modes_train)
        test_centroids = compute_centroids(embedded_test, y_test, modes_test)
        similarities.append(
            similarity_matrix(test_centroids, train_centroids, modes_test, modes_train)
        )
        assignment = nearest_centroid_assignment(
            embedded_test, y_test, modes_test, train_centroids, modes_train
        )
        assignments.append(assignment)
        scored = score_mapping(assignment, modes_test, modes_train, predicted)
        accuracies.append(scored.overall_accuracy)
        modes_correct.append(scored.modes_correct)
        for name, info in scored.per_mode.items():
            if name in per_mode and "accuracy" in info:
                per_mode[name].append(float(info["accuracy"]))
        if wildcard_counts is not None and wildcard_mode is not None:
            row = list(modes_test).index(wildcard_mode)
            wildcard_counts[modes_train[int(assignment[row].argmax())]] += 1
        logger.info(
            f"  seed {index}: acc={scored.overall_accuracy:.2%}, "
            f"modes_correct={scored.modes_correct}/{scored.modes_total}, loss={final_loss:.4f}"
        )

    results: dict[str, Any] = {
        "direction": direction_label,
        "n_seeds": n_seeds,
        "mapping_accuracy_mean": float(np.mean(accuracies)),
        "mapping_accuracy_median": float(np.median(accuracies)),
        "mapping_accuracy_std": float(np.std(accuracies)),
        "mapping_accuracy_per_seed": [float(a) for a in accuracies],
        "modes_correct_mean": float(np.mean(modes_correct)),
        "modes_correct_per_seed": [int(m) for m in modes_correct],
        "chance_accuracy": 1.0 / len(modes_train),
        "per_mode_accuracy": {
            name: {
                "mean": float(np.mean(values)) if values else float("nan"),
                "std": float(np.std(values)) if values else float("nan"),
                "per_seed": [float(v) for v in values],
            }
            for name, values in per_mode.items()
        },
        "median_similarity_matrix": np.median(np.stack(similarities), axis=0).tolist(),
        "mean_assignment_matrix": np.mean(
            np.stack(assignments).astype(np.float64), axis=0
        ).tolist(),
        "similarity_rows": list(modes_test),
        "similarity_cols": list(modes_train),
        "predicted_mapping": dict(predicted),
    }
    if wildcard_counts is not None:
        results["wildcard_mode"] = wildcard_mode
        results["wildcard_assignments"] = wildcard_counts
    logger.info(
        f"  [{direction_label}] acc={results['mapping_accuracy_mean']:.2%} "
        f"+/- {results['mapping_accuracy_std']:.2%} "
        f"(chance={results['chance_accuracy']:.2%})"
    )
    return results


def lda_direction_test(
    X_a: F32,
    y_a: NDArray[np.int64],
    modes_a: Sequence[str],
    X_b: F32,
    y_b: NDArray[np.int64],
    modes_b: Sequence[str],
) -> dict[str, Any]:
    """Fit discriminants on run A, project run B into them, read B's silhouette.

    The interpretation is stated in the result rather than left to the reader: a
    near-zero or negative silhouette for B means A's linear directions carry no
    information about B's modes, which is the directions-versus-manifolds
    dissociation. A's own silhouette in its own space is the sanity leg — without it
    a low number for B could just mean the fit failed.
    """
    logger.info("=== LDA direction projection test ===")
    scaler = StandardScaler()
    A_scaled = scaler.fit_transform(X_a)
    B_scaled = scaler.transform(X_b)

    lda = LinearDiscriminantAnalysis()
    A_projected = lda.fit_transform(A_scaled, y_a)
    B_projected = lda.transform(B_scaled)

    a_own = float(silhouette_score(A_projected, y_a, metric="cosine"))
    b_in_a = float(silhouette_score(B_projected, y_b, metric="cosine"))
    per_sample = silhouette_samples(B_projected, y_b, metric="cosine")
    per_mode = {
        name: float(per_sample[y_b == index].mean()) for index, name in enumerate(modes_b)
    }

    knn = KNeighborsClassifier(n_neighbors=KNN_NEIGHBOURS, metric="cosine")
    knn.fit(A_projected, y_a)
    predictions = knn.predict(B_projected)
    cross = np.zeros((len(modes_b), len(modes_a)), dtype=np.int64)
    for index in range(len(modes_b)):
        for prediction in predictions[y_b == index]:
            cross[index, int(prediction)] += 1

    logger.info(f"  A silhouette in its own LDA space: {a_own:.4f}")
    logger.info(f"  B silhouette in A's LDA space:      {b_in_a:.4f}")
    for name, value in sorted(per_mode.items(), key=lambda kv: -kv[1]):
        logger.info(f"    B per-mode silhouette: {name}: {value:.4f}")

    return {
        "r2_sil_own_lda": a_own,
        "r3_sil_in_r2_lda": b_in_a,
        "r3_per_mode_sil": per_mode,
        "cross_prediction_matrix": cross.tolist(),
        "cross_prediction_rows": list(modes_b),
        "cross_prediction_cols": list(modes_a),
        "projection_dim": int(A_projected.shape[1]),
        "interpretation": (
            "A silhouette near zero or negative for the projected run means the linear "
            "directions separating the fitted run's modes carry no information about the "
            "projected run's modes — the directions-versus-manifolds dissociation."
        ),
    }


def cross_run_transfer(
    *,
    train_run: str,
    test_run: str,
    feature_key: str = "features",
    mapping: str | ModeMapping = DEFAULT_MODE_MAPPING,
    n_seeds: int = N_SEEDS,
    bottleneck_dim: int = BOTTLENECK_DIM,
    n_epochs: int = ANALYSIS_EPOCHS,
    outputs_base: Path | None = None,
) -> dict[str, Any]:
    """Both directions and the LDA test, as one banked document.

    ``mapping`` names the registry row holding the predicted pairs, so comparing
    another pair of corpora is a row rather than an edit here. The test run supplies
    the mapping's source labels and the train run its target labels: the forward
    direction trains on the target vocabulary and embeds the source one.

    The two runs must have the same feature width: transfer means one projection
    reading both, so a width mismatch is two different instruments and is refused
    rather than truncated to the shorter.
    """
    row = mapping if isinstance(mapping, ModeMapping) else mode_mapping(mapping)
    forward_wildcard, reverse_wildcard = mapping_wildcards(row.name)
    X_train, y_train, modes_train, _ = load_run_features(
        train_run, outputs_base, feature_key=feature_key
    )
    X_test, y_test, modes_test, _ = load_run_features(
        test_run, outputs_base, feature_key=feature_key
    )
    if X_train.shape[1] != X_test.shape[1]:
        raise RuntimeError(
            f"feature widths must match (same pipeline, same calibration): "
            f"{train_run}={X_train.shape[1]}, {test_run}={X_test.shape[1]}"
        )

    results: dict[str, Any] = {
        "experiment": "cross_run_transfer",
        "train_run": train_run,
        "test_run": test_run,
        "feature_key": feature_key,
        "train_n_samples": int(X_train.shape[0]),
        "test_n_samples": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "train_modes": modes_train,
        "test_modes": modes_test,
        "mapping": row.name,
        "source_vocabulary": row.source,
        "target_vocabulary": row.target,
        "forward_mapping": dict(row.pairs),
        "reverse_mapping": row.reverse_pairs(),
        "reference": row.reference.model_dump() if row.reference is not None else None,
    }

    forward_scaler = StandardScaler()
    forward_train = forward_scaler.fit_transform(X_train).astype(np.float32)
    forward_test = forward_scaler.transform(X_test).astype(np.float32)
    results["transfer_forward"] = run_transfer(
        forward_train, y_train, modes_train,
        forward_test, y_test, modes_test,
        predicted=row.pairs,
        wildcard_mode=forward_wildcard,
        direction_label=f"forward (train {train_run} -> embed {test_run})",
        n_seeds=n_seeds, bottleneck_dim=bottleneck_dim, n_epochs=n_epochs,
    )

    reverse_scaler = StandardScaler()
    reverse_test = reverse_scaler.fit_transform(X_test).astype(np.float32)
    reverse_train = reverse_scaler.transform(X_train).astype(np.float32)
    results["transfer_reverse"] = run_transfer(
        reverse_test, y_test, modes_test,
        reverse_train, y_train, modes_train,
        predicted=row.reverse_pairs(),
        wildcard_mode=reverse_wildcard,
        direction_label=f"reverse (train {test_run} -> embed {train_run})",
        n_seeds=n_seeds, bottleneck_dim=bottleneck_dim, n_epochs=n_epochs,
    )

    results["lda_direction_test"] = lda_direction_test(
        X_test.astype(np.float32), y_test, modes_test,
        X_train.astype(np.float32), y_train, modes_train,
    )
    return results


def headline(results: dict[str, Any]) -> list[str]:
    """The two pairs and the two silhouettes, each beside the mapping's comparison.

    Which pair is quoted and what it is compared against come from the mapping row,
    so the comparison travels with the number instead of being looked up separately.
    A mapping carrying no reference reports its own numbers alone.
    """
    forward = results["transfer_forward"]["per_mode_accuracy"]
    reverse = results["transfer_reverse"]["per_mode_accuracy"]
    lda = results["lda_direction_test"]
    reference = results.get("reference")
    pairs: dict[str, str] = results["forward_mapping"]
    reverse_pairs: dict[str, str] = results["reverse_mapping"]

    if reference is None:
        forward_label = next(iter(pairs), "")
        reverse_label = next(iter(reverse_pairs), "")
        forward_rate = float(forward.get(forward_label, {}).get("mean", float("nan")))
        reverse_rate = float(reverse.get(reverse_label, {}).get("mean", float("nan")))
        return [
            f"forward {forward_label} -> {pairs.get(forward_label, '')}: {forward_rate:.2%}",
            f"reverse {reverse_label} -> {reverse_pairs.get(reverse_label, '')}: "
            f"{reverse_rate:.2%}",
            f"projected silhouette in the fitted run's LDA space: {lda['r3_sil_in_r2_lda']:+.4f}",
            f"fitted run's own silhouette (sanity):               {lda['r2_sil_own_lda']:+.4f}",
        ]

    model = reference["model"]
    forward_label = reference["forward_pair"]
    reverse_label = reference["reverse_pair"]
    forward_rate = float(forward.get(forward_label, {}).get("mean", float("nan")))
    reverse_rate = float(reverse.get(reverse_label, {}).get("mean", float("nan")))
    forward_reference = float(reference["forward_pair_accuracy"])
    reverse_reference = float(reference["reverse_pair_accuracy"])
    return [
        f"forward {forward_label} -> {pairs[forward_label]}: {forward_rate:.2%}  "
        f"({model} {forward_reference:.2%}, delta {forward_rate - forward_reference:+.2%})",
        f"reverse {reverse_label} -> {reverse_pairs[reverse_label]}: {reverse_rate:.2%}  "
        f"({model} {reverse_reference:.2%}, delta {reverse_rate - reverse_reference:+.2%})",
        f"projected silhouette in the fitted run's LDA space: "
        f"{lda['r3_sil_in_r2_lda']:+.4f}  ({model} {reference['projected_silhouette']:+.4f})",
        f"fitted run's own silhouette (sanity):               "
        f"{lda['r2_sil_own_lda']:+.4f}  ({model} {reference['fitted_silhouette']:+.4f})",
    ]


__all__ = [
    "MappingScore",
    "N_SEEDS",
    "build_layer_indices",
    "compute_centroids",
    "cross_run_transfer",
    "headline",
    "lda_direction_test",
    "load_feature_names",
    "load_run_features",
    "nearest_centroid_assignment",
    "run_transfer",
    "score_mapping",
    "signatures_dir",
    "similarity_matrix",
]
