"""The contrastive projection: one network shape, two laws for training it.

A linear probe asks whether modes are separable by a hyperplane. This asks
whether they are separable at all: a small network trained on triplets — an
anchor, another sample of its own class, one of a different class — learns an
embedding where same-mode states are close and different-mode states are not. It
is the rung above the linear probe on the ladder, and the readout it earns is
"the signal is on a curved manifold" rather than "the signal is absent".

The network is the same in both uses and is deliberately small:
``Linear(P, 256) -> ReLU -> Dropout -> Linear(256, k)``, L2-normalized, so the
embedding lives on a sphere and cosine distance is the metric it was trained
under. What differs is the **law the training follows**, and the two laws are
named rather than merged because they answer different questions:

* :func:`train_projection` fits a projection that will be **banked as a
  calibration artifact** and applied at feature-extraction time. It holds out
  whole *groups* — every sample of one generation goes to one side of the split,
  because the several layer-and-time samples of one generation are not independent
  — validates by kNN on the held-out groups, keeps the best weights and stops
  early. It returns numpy arrays, because inference runs in the feature family with
  no torch installed.
* :func:`train_embedding` fits a projection **used once, inside an analysis**, over
  the full data with no validation split, for a fixed number of epochs. It is the
  right law there and the wrong one for an artifact: nothing is held out, so its
  embedding describes the data it was fitted on and is never applied to another
  bank under the same weights. Several seeds are what make its readout a number
  with a spread rather than one draw.

Mining differs with the law. The banked fit samples a **class first and then two
of its members**, so every class contributes equally however imbalanced the corpus
is; the analysis fit samples an **anchor uniformly**, so the embedding reflects the
corpus as it stands. Both are here, under their own names.

This lives in the analysis layer rather than in extraction because fitting is
learned-probe machinery and because the numeric anchor's purity is guarded: the
family that *applies* these weights is pure numpy, and a torch import anywhere in
its closure would fail that guard.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]

HIDDEN_DIM = 256
BOTTLENECK_DIM = 32
DROPOUT = 0.5
MARGIN = 1.0
WEIGHT_DECAY = 1e-3
LEARNING_RATE = 1e-3

BANKED_EPOCHS = 300
BANKED_TRIPLETS = 512
BANKED_VAL_FRACTION = 0.2
BANKED_PATIENCE = 50
VALIDATE_EVERY = 10
KNN_NEIGHBOURS = 5

ANALYSIS_EPOCHS = 200
ANALYSIS_TRIPLETS = 300
MIN_TRIPLETS = 10
"""Below this many mined triplets an epoch is skipped rather than taken on a
handful: a triplet loss over three pairs is noise with a gradient."""

SCALE_FLOOR = 1e-12
"""A feature whose training-set standard deviation is below this is left unscaled
rather than divided by nearly zero."""


def _torch() -> tuple[Any, Any, Any]:
    """Torch and the two submodules the trainers use, imported at call time."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError as exc:  # pragma: no cover - torch is a hard dependency here
        raise ImportError(
            "training a contrastive projection needs torch; inference does not, "
            "which is why the family that applies these weights is pure numpy"
        ) from exc
    return torch, nn, optim


def projection_network(input_dim: int, *, hidden_dim: int = HIDDEN_DIM,
                       output_dim: int = BOTTLENECK_DIM, dropout: float = DROPOUT) -> Any:
    """The network both laws train, as a plain ``Sequential``.

    The L2 normalization is applied by the callers rather than being a layer, so the
    same module's ``state_dict`` keys are the two linear layers and nothing else —
    which is what lets the banked weights be four numpy arrays.
    """
    _torch_mod, nn, _optim = _torch()
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


def _weights_from(state: Any) -> dict[str, F32]:
    """The four arrays the banked artifact carries, out of a state dict."""
    return {
        "w1": state["0.weight"].numpy().copy(),
        "b1": state["0.bias"].numpy().copy(),
        "w2": state["3.weight"].numpy().copy(),
        "b2": state["3.bias"].numpy().copy(),
    }


def mine_triplets(
    labels: NDArray[Any],
    rng: np.random.RandomState,
    n_triplets: int = ANALYSIS_TRIPLETS,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Anchor-uniform mining: a random sample, a same-class other, a different one.

    An anchor whose class has no second member contributes no triplet, so the
    returned arrays can be shorter than ``n_triplets`` and the caller reads the
    length rather than assuming it.
    """
    anchors: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    for _ in range(n_triplets):
        anchor = rng.randint(len(labels))
        same = np.where(labels == labels[anchor])[0]
        same = same[same != anchor]
        if len(same) == 0:
            continue
        different = np.where(labels != labels[anchor])[0]
        if len(different) == 0:
            continue
        anchors.append(anchor)
        positives.append(int(rng.choice(same)))
        negatives.append(int(rng.choice(different)))
    return (
        np.array(anchors, dtype=np.int64),
        np.array(positives, dtype=np.int64),
        np.array(negatives, dtype=np.int64),
    )


def mine_balanced_triplets(
    by_label: dict[str, list[int]],
    labels_in_play: Sequence[str],
    rng: np.random.Generator,
    n_triplets: int,
) -> tuple[list[int], list[int], list[int]]:
    """Class-first mining: every class is drawn equally often as the anchor's.

    ``by_label`` maps a label to its row indices and ``labels_in_play`` names the
    labels with at least two members — a class with one member can be a negative
    but never an anchor, and filtering it out here is what keeps the draw uniform
    over the classes that can actually supply a pair.
    """
    if len(labels_in_play) < 2:
        raise ValueError("balanced mining needs at least two labels with two members each")
    anchors: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    for _ in range(n_triplets):
        label = labels_in_play[int(rng.integers(len(labels_in_play)))]
        anchor, positive = rng.choice(by_label[label], size=2, replace=False)
        negative_label = label
        while negative_label == label:
            negative_label = labels_in_play[int(rng.integers(len(labels_in_play)))]
        anchors.append(int(anchor))
        positives.append(int(positive))
        negatives.append(int(rng.choice(by_label[negative_label])))
    return anchors, positives, negatives


def embed(model: Any, X: F32) -> F32:
    """Project rows through a trained network, L2-normalized, as numpy."""
    torch, nn, _optim = _torch()
    model.eval()
    with torch.no_grad():
        out = nn.functional.normalize(model(torch.tensor(X, dtype=torch.float32)), p=2, dim=1)
    return out.numpy().astype(np.float32)


def train_embedding(
    X: F32,
    y: NDArray[np.int64],
    *,
    bottleneck_dim: int = BOTTLENECK_DIM,
    n_epochs: int = ANALYSIS_EPOCHS,
    n_triplets: int = ANALYSIS_TRIPLETS,
    lr: float = LEARNING_RATE,
    margin: float = MARGIN,
    dropout: float = DROPOUT,
    weight_decay: float = WEIGHT_DECAY,
    seed: int = 42,
) -> tuple[Any, float]:
    """The analysis law: full data, anchor-uniform mining, fixed epochs, no holdout.

    Returns the trained network and its last loss. Nothing is held out on purpose —
    the embedding is a lens this analysis looks through once, and the claim it
    supports comes from repeating the whole fit under several seeds rather than from
    a validation number inside one.
    """
    torch, nn, optim = _torch()
    if len(np.unique(y)) < 2:
        raise ValueError("a triplet loss needs at least two classes; one class has no negatives")
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    rows = torch.tensor(X, dtype=torch.float32)
    model = projection_network(X.shape[1], output_dim=bottleneck_dim, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.TripletMarginLoss(margin=margin)

    final_loss = 0.0
    model.train()
    for _ in range(n_epochs):
        anchors, positives, negatives = mine_triplets(y, rng, n_triplets=n_triplets)
        if len(anchors) < MIN_TRIPLETS:
            continue
        embedded = nn.functional.normalize(model(rows), p=2, dim=1)
        loss = loss_fn(embedded[anchors], embedded[positives], embedded[negatives])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
    return model, final_loss


def train_projection(
    X: F32,
    y: NDArray[Any],
    groups: NDArray[Any] | None = None,
    *,
    hidden_dim: int = HIDDEN_DIM,
    bottleneck_dim: int = BOTTLENECK_DIM,
    n_epochs: int = BANKED_EPOCHS,
    lr: float = LEARNING_RATE,
    margin: float = MARGIN,
    weight_decay: float = WEIGHT_DECAY,
    batch_triplets: int = BANKED_TRIPLETS,
    val_fraction: float = BANKED_VAL_FRACTION,
    seed: int = 42,
    standardize: bool = True,
) -> dict[str, F32]:
    """The banked law: grouped holdout, kNN validation, early stopping, numpy out.

    ``groups`` is the unit the split respects — a generation id, where the rows are
    its several layer-and-time samples. Splitting by row instead puts samples of one
    generation on both sides and the validation kNN reads its own training data,
    which is why the group split is the default path and the by-row split is only
    the degenerate fallback when one side would come out empty.

    The returned dictionary is what the feature family loads: two weight matrices,
    two biases, and the standardization the inputs were trained under.
    """
    torch, nn, optim = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    n_samples = len(X)
    if groups is not None:
        group_array = np.asarray(groups)
        unique_groups = np.unique(group_array)
        order = rng.permutation(len(unique_groups))
        n_val_groups = max(1, int(len(unique_groups) * val_fraction))
        held_out = set(unique_groups[order[:n_val_groups]].tolist())
        is_val = np.array([g in held_out for g in group_array])
        val_index = np.flatnonzero(is_val)
        train_index = np.flatnonzero(~is_val)
        if len(train_index) == 0 or len(val_index) == 0:
            order = rng.permutation(n_samples)
            n_val = max(1, int(n_samples * val_fraction))
            val_index, train_index = order[:n_val], order[n_val:]
    else:
        order = rng.permutation(n_samples)
        n_val = max(1, int(n_samples * val_fraction))
        val_index, train_index = order[:n_val], order[n_val:]

    scaler_mean: F32 | None = None
    scaler_scale: F32 | None = None
    if standardize:
        scaler_mean = X[train_index].mean(axis=0).astype(np.float32)
        scaler_scale = X[train_index].std(axis=0).astype(np.float32)
        scaler_scale = np.where(scaler_scale < SCALE_FLOOR, 1.0, scaler_scale)
        scaled = ((X - scaler_mean) / scaler_scale).astype(np.float32)
    else:
        scaled = X

    X_train, y_train = scaled[train_index], y[train_index]
    X_val, y_val = scaled[val_index], y[val_index]
    logger.info(
        f"training on {len(X_train)} samples, validating on {len(X_val)}, "
        f"input_dim={X_train.shape[1]}, hidden={hidden_dim}, bottleneck={bottleneck_dim}"
    )

    model = projection_network(X_train.shape[1], hidden_dim=hidden_dim, output_dim=bottleneck_dim)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.TripletMarginLoss(margin=margin)
    train_rows = torch.tensor(X_train, dtype=torch.float32)

    by_label: dict[str, list[int]] = {}
    for index, label in enumerate(y_train):
        by_label.setdefault(str(label), []).append(index)
    labels_in_play = [label for label in sorted(by_label) if len(by_label[label]) >= 2]

    from sklearn.neighbors import KNeighborsClassifier

    best_accuracy = 0.0
    best_weights: dict[str, F32] = {}
    since_improvement = 0

    for epoch in range(n_epochs):
        anchors, positives, negatives = mine_balanced_triplets(
            by_label, labels_in_play, rng, batch_triplets
        )
        if not anchors:
            continue
        anchor_out = nn.functional.normalize(model(train_rows[anchors]), dim=1)
        positive_out = nn.functional.normalize(model(train_rows[positives]), dim=1)
        negative_out = nn.functional.normalize(model(train_rows[negatives]), dim=1)
        loss = loss_fn(anchor_out, positive_out, negative_out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % VALIDATE_EVERY != 0:
            continue
        model.eval()
        with torch.no_grad():
            train_embedded = nn.functional.normalize(model(train_rows), dim=1).numpy()
            val_embedded = nn.functional.normalize(
                model(torch.tensor(X_val, dtype=torch.float32)), dim=1
            ).numpy()
        knn = KNeighborsClassifier(n_neighbors=KNN_NEIGHBOURS, metric="cosine")
        knn.fit(train_embedded, y_train)
        accuracy = float(knn.score(val_embedded, y_val))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            since_improvement = 0
            best_weights = _weights_from(model.state_dict())
        else:
            since_improvement += VALIDATE_EVERY
        if (epoch + 1) % 50 == 0:
            logger.info(
                f"  epoch {epoch + 1}/{n_epochs}: loss={loss.item():.4f}, "
                f"val_kNN={accuracy:.3f} (best={best_accuracy:.3f})"
            )
        model.train()
        if since_improvement >= BANKED_PATIENCE:
            logger.info(f"  early stop at epoch {epoch + 1}")
            break

    if not best_weights:
        best_weights = _weights_from(model.state_dict())
    logger.info(f"best validation kNN accuracy: {best_accuracy:.3f}")

    weights = dict(best_weights)
    if scaler_mean is not None and scaler_scale is not None:
        weights["scaler_mean"] = scaler_mean
        weights["scaler_scale"] = scaler_scale
    return weights




# ── The training corpus for a banked fit ──────────────────────────────────────
TEMPORAL_SAMPLES = 5
"""Evenly-spaced generation positions sampled per generation. The projection is
applied at feature-extraction time at these same positions, so the fit sees the
distribution it will be used on."""

EMBEDDING_OFFSET = 1
"""Hidden states are indexed with the embedding output first, so layer ``l`` is at
array index ``l + 1``. Every layer-indexed read in this instrument carries this
offset, and getting it wrong trains on a different depth than it names."""


def temporal_indices(n_steps: int, n_samples: int = TEMPORAL_SAMPLES) -> list[int]:
    """Evenly-spaced step indices, endpoints included.

    One sample is the first step rather than the middle: a single-sample fit is a
    prompt-end read, and saying so is better than averaging over a choice.
    """
    if n_samples <= 1:
        return [0]
    return [int(round(i * (n_steps - 1) / (n_samples - 1))) for i in range(n_samples)]


def load_hidden_state_samples(
    raw_dir: Path,
    metadata_dir: Path,
    layer_indices: Sequence[int],
    *,
    temporal_samples: int = TEMPORAL_SAMPLES,
    positional_means_path: Path | None = None,
    exclude_prompt_swap: bool = True,
) -> tuple[F32, NDArray[Any], NDArray[Any]]:
    """Hidden states, mode labels and generation groups, from banked raw tensors.

    One row per (generation, layer, position). The **group** is the generation, which
    is what the banked fit's split has to hold out whole: the several rows of one
    generation share its prompt, its seed and most of its trajectory.

    Positional correction is applied here when a means file is given, at the absolute
    position each sampled step sits at — the same subtraction the corrected features
    use. A fit on raw states whose projection is then applied to corrected ones is a
    fit-and-apply mismatch, which is why the correction is an argument rather than a
    later step.

    Prompt-swap generations are excluded by default: their mode label names the system
    prompt, and the projection is being fitted to separate executions.
    """
    from anamnesis.extraction.raw_saver import list_raw_tensor_ids, load_raw_tensors

    positional_means: F32 | None = None
    if positional_means_path is not None and Path(positional_means_path).exists():
        with np.load(positional_means_path) as means:
            positional_means = means["positional_means"].astype(np.float32)
        logger.info(f"positional means loaded: {positional_means.shape}")

    gen_ids = list_raw_tensor_ids(Path(raw_dir))
    logger.info(f"{len(gen_ids)} banked raw captures under {raw_dir}")

    rows: list[F32] = []
    labels: list[str] = []
    groups: list[int] = []
    skipped = 0

    for gen_id in gen_ids:
        meta_path = Path(metadata_dir) / f"gen_{gen_id:03d}.json"
        if not meta_path.exists():
            skipped += 1
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            skipped += 1
            continue
        mode = str(meta.get("mode", ""))
        condition = str(meta.get("condition", "standard"))
        if exclude_prompt_swap and (condition.startswith("prompt_swap") or mode.startswith("swap_")):
            continue
        try:
            data = load_raw_tensors(gen_id, Path(raw_dir), surfaces=("hidden",))
        except (OSError, ValueError, KeyError) as exc:
            logger.warning(f"gen_{gen_id:03d}: not loaded ({exc})")
            skipped += 1
            continue
        n_steps = len(data.hidden_states)
        if n_steps < 2:
            skipped += 1
            continue
        for layer in layer_indices:
            for step in temporal_indices(n_steps, temporal_samples):
                if step >= n_steps:
                    continue
                state = data.hidden_states[step][layer + EMBEDDING_OFFSET].copy().astype(np.float32)
                if positional_means is not None:
                    position = data.prompt_length + step
                    if (
                        layer + EMBEDDING_OFFSET < positional_means.shape[0]
                        and position < positional_means.shape[1]
                    ):
                        state = state - positional_means[layer + EMBEDDING_OFFSET, position]
                rows.append(state)
                labels.append(mode)
                groups.append(gen_id)

    if skipped:
        logger.info(f"{skipped} generations skipped (no metadata, unreadable, or too short)")
    if not rows:
        raise FileNotFoundError(
            f"no training rows from {raw_dir}: every generation lacked metadata, raw "
            f"hidden states, or enough steps to sample"
        )
    X = np.stack(rows, axis=0)
    y = np.array(labels)
    logger.info(f"training corpus: {X.shape[0]} rows of {X.shape[1]} dims over {len(set(groups))} generations")
    for mode in sorted(set(labels)):
        logger.info(f"  {mode}: {labels.count(mode)} rows")
    return X, y, np.array(groups)



__all__ = [
    "ANALYSIS_EPOCHS",
    "ANALYSIS_TRIPLETS",
    "BANKED_EPOCHS",
    "BANKED_TRIPLETS",
    "BOTTLENECK_DIM",
    "HIDDEN_DIM",
    "MIN_TRIPLETS",
    "embed",
    "mine_balanced_triplets",
    "mine_triplets",
    "projection_network",
    "TEMPORAL_SAMPLES",
    "load_hidden_state_samples",
    "temporal_indices",
    "train_embedding",
    "train_projection",
]
