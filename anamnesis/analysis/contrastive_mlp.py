"""The contrastive probe: a small network on triplets, as a nonlinear readout.

A linear probe asks whether modes are separable by a hyperplane. This asks
whether they are separable at all: a small network trained on triplets — an
anchor, another sample of its own class, one of a different class — learns an
embedding where same-mode states are close and different-mode states are not. It
is the rung above the linear probe on the ladder, and the readout it earns is
"the signal is on a curved manifold" rather than "the signal is absent".

The network is deliberately small: ``Linear(P, 256) -> ReLU -> Dropout ->
Linear(256, k)``, L2-normalized, so the embedding lives on a sphere and cosine
distance is the metric it was trained under.

There is one law for training it, and its shape is what makes the embedding a
*reading* rather than an artifact: the fit runs over the full data with no
validation split, for a fixed number of epochs, mining an **anchor uniformly** so
that the embedding reflects the corpus as it stands. Nothing is held out on
purpose — the embedding is a lens one analysis looks through once, never weights
carried to another corpus — and what turns its number into a number with a spread
is repeating the whole fit under several seeds. The callers are
:mod:`anamnesis.analysis.gauntlet.contrastive`,
:mod:`anamnesis.analysis.gauntlet.semantic` and
:mod:`anamnesis.analysis.cross_run`, each of which holds out its own folds around
the fit rather than inside it.

This lives in the analysis layer rather than in extraction because fitting is
learned-probe machinery. Torch is imported at call time, so importing this module
costs nothing to a caller that only reads features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

F32 = NDArray[np.float32]

HIDDEN_DIM = 256
BOTTLENECK_DIM = 32
DROPOUT = 0.5
MARGIN = 1.0
WEIGHT_DECAY = 1e-3
LEARNING_RATE = 1e-3

ANALYSIS_EPOCHS = 200
ANALYSIS_TRIPLETS = 300
MIN_TRIPLETS = 10
"""Below this many mined triplets an epoch is skipped rather than taken on a
handful: a triplet loss over three pairs is noise with a gradient."""


def _torch() -> tuple[Any, Any, Any]:
    """Torch and the two submodules the trainer uses, imported at call time."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError as exc:  # pragma: no cover - torch is a hard dependency here
        raise ImportError(
            "fitting a contrastive probe needs torch; the import is deferred to the "
            "call so that importing this module does not require it"
        ) from exc
    return torch, nn, optim


def projection_network(input_dim: int, *, hidden_dim: int = HIDDEN_DIM,
                       output_dim: int = BOTTLENECK_DIM, dropout: float = DROPOUT) -> Any:
    """The network :func:`train_embedding` fits, as a plain ``Sequential``.

    The L2 normalization is not a layer here: the trainer and :func:`embed` each
    apply it to this module's output, which is what lets one network serve both and
    keeps the projection itself a pair of linear maps — two weights and two biases,
    and nothing else in its ``state_dict``.
    """
    _torch_mod, nn, _optim = _torch()
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


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


def embed(model: Any, X: F32) -> F32:
    """Project rows through a trained network, L2-normalized, as numpy."""
    torch, nn, _optim = _torch()
    model.eval()
    with torch.no_grad():
        out = nn.functional.normalize(model(torch.tensor(X, dtype=torch.float32)), p=2, dim=1)
    return out.numpy().astype(np.float32)


def train_embedding(
    X: F32,
    y: NDArray[Any],
    *,
    hidden_dim: int = HIDDEN_DIM,
    bottleneck_dim: int = BOTTLENECK_DIM,
    n_epochs: int = ANALYSIS_EPOCHS,
    n_triplets: int = ANALYSIS_TRIPLETS,
    lr: float = LEARNING_RATE,
    margin: float = MARGIN,
    dropout: float = DROPOUT,
    weight_decay: float = WEIGHT_DECAY,
    seed: int = 42,
) -> tuple[Any, float]:
    """Fit the probe: full data, anchor-uniform mining, fixed epochs, no holdout.

    ``y`` carries whatever labels the analysis holds — mode names as strings are the
    usual case — because mining only ever compares labels for equality.

    ``hidden_dim`` is an argument rather than the constant because one of the
    callers runs a capacity sweep (:mod:`anamnesis.analysis.gauntlet.contrastive`):
    the same fit at four widths is the reading, so the width has to move while
    nothing else does.

    Returns the trained network and its last loss. Nothing is held out on purpose —
    the embedding is a lens this analysis looks through once, and the claim it
    supports comes from repeating the whole fit under several seeds rather than from
    a validation number inside one.

    Raises
    ------
    ValueError
        When fewer than two classes are present: a triplet loss over one class has
        no negatives to push against, and a network returned untrained from that
        would read as a fitted embedding.
    """
    torch, nn, optim = _torch()
    if len(np.unique(y)) < 2:
        raise ValueError("a triplet loss needs at least two classes; one class has no negatives")
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    rows = torch.tensor(X, dtype=torch.float32)
    model = projection_network(
        X.shape[1], hidden_dim=hidden_dim, output_dim=bottleneck_dim, dropout=dropout
    )
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


__all__ = [
    "ANALYSIS_EPOCHS",
    "ANALYSIS_TRIPLETS",
    "BOTTLENECK_DIM",
    "HIDDEN_DIM",
    "MIN_TRIPLETS",
    "embed",
    "mine_triplets",
    "projection_network",
    "train_embedding",
]
