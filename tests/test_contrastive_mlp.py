"""The contrastive probe's training law, and the network it fits.

One law, and the tests are about what makes it the law it is:

  * mining draws the anchor uniformly, so the triplets follow the corpus as it
    stands, and a class with one member can be a negative but never an anchor;
  * a single-class corpus is refused, because a triplet loss has no negative to pull
    against;
  * the embedding comes back on the unit sphere, which is the metric it was trained
    under;
  * the width of the network is an argument, because the capacity sweep in the
    gauntlet's section 8 is the same law read at four widths;
  * this is the only law in the package. The gauntlet's sections train under the one
    defined here rather than carrying a trainer each, and the check is over their
    source: a second implementation passes every test this one has, and then reports
    numbers under a law nothing here describes.

CPU only; the networks are tiny and trained for a handful of epochs.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from anamnesis.analysis.contrastive_mlp import (
    HIDDEN_DIM,
    embed,
    mine_triplets,
    projection_network,
    train_embedding,
)

GAUNTLET = Path(__file__).resolve().parent.parent / "anamnesis" / "analysis" / "gauntlet"


def two_class_corpus(n: int = 24, d: int = 6, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = np.array(["linear" if i % 2 == 0 else "socratic" for i in range(n)])
    X[y == "socratic", 0] += 4.0
    return X, y


def test_anchor_uniform_mining_skips_an_anchor_with_no_sibling() -> None:
    labels = np.array(["a", "a", "b", "c"])
    anchors, positives, negatives = mine_triplets(labels, np.random.RandomState(0), 200)
    assert len(anchors) == len(positives) == len(negatives)
    assert len(anchors) < 200, "the two single-member classes cannot anchor a triplet"
    for anchor, positive, negative in zip(anchors, positives, negatives):
        assert labels[anchor] == labels[positive] and labels[anchor] != labels[negative]


def test_the_law_refuses_a_single_class_corpus() -> None:
    X, _ = two_class_corpus()
    with pytest.raises(ValueError, match="at least two classes"):
        train_embedding(X, np.zeros(len(X), dtype=np.int64), n_epochs=5)


def test_the_embedding_lives_on_the_unit_sphere() -> None:
    X, y = two_class_corpus()
    codes = np.array([0 if label == "linear" else 1 for label in y], dtype=np.int64)
    model, loss = train_embedding(X, codes, n_epochs=20, n_triplets=32, bottleneck_dim=4, seed=4)
    embedded = embed(model, X)
    assert embedded.shape == (len(X), 4)
    assert np.allclose(np.linalg.norm(embedded, axis=1), 1.0, atol=1e-5)
    assert loss >= 0.0


def test_the_law_trains_at_the_width_it_is_given() -> None:
    """The capacity sweep needs the width to move and nothing else to."""
    X, y = two_class_corpus()
    narrow, _loss = train_embedding(X, y, hidden_dim=8, n_epochs=3, n_triplets=16, seed=5)
    default, _loss = train_embedding(X, y, n_epochs=3, n_triplets=16, seed=5)
    assert narrow[0].out_features == 8
    assert default[0].out_features == HIDDEN_DIM


def test_the_gauntlet_sections_carry_no_trainer_of_their_own() -> None:
    """One implementation of the law, named once, imported where it is needed.

    The markers are the two ways a second trainer comes back: the loss it would have
    to construct, and the name the duplicate went by. A section that grows its own
    fit would report numbers under a law nothing here describes.
    """
    offenders = [
        f"{path.name} names {marker}"
        for path in sorted(GAUNTLET.glob("*.py"))
        for marker in ("TripletMarginLoss", "_train_contrastive_mlp")
        if marker in path.read_text(encoding="utf-8")
    ]
    assert offenders == [], (
        f"{offenders}: the contrastive law lives in anamnesis.analysis.contrastive_mlp"
    )


def test_no_gauntlet_section_reaches_into_a_sibling_sections_private_names() -> None:
    """A private name is a module's own business; across modules it is a coupling.

    Reaching for one makes two sections share an implementation that neither
    documents and that a reader of either would not know was load-bearing.
    """
    reaches: list[str] = []
    for path in sorted(GAUNTLET.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            reaches.extend(
                f"{path.name} imports {alias.name} from {'.' * node.level}{node.module or ''}"
                for alias in node.names
                if alias.name.startswith("_")
            )
    assert reaches == [], str(reaches)


def test_the_network_is_two_linear_layers_and_nothing_else() -> None:
    """Normalization sits outside the module, so its state is the projection alone."""
    network = projection_network(7, hidden_dim=16, output_dim=3)
    keys = dict(network.state_dict())
    assert set(keys) == {"0.weight", "0.bias", "3.weight", "3.bias"}
