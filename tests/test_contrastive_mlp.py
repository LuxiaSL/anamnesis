"""The contrastive projection's two training laws, and the corpus they read.

Both laws train the same network, so the tests are about what makes them *different* laws
and about the sampling that feeds the banked one:

  * the banked fit returns the four arrays plus the standardization the feature family
    loads, and an unstandardized fit returns the four alone;
  * the grouped split holds out **whole generations**: with every row of a generation in one
    half, the validation kNN cannot read its own training data through a sibling row;
  * mining differs — class-first mining draws every class equally often, anchor-uniform
    mining follows the corpus — and a class with one member can be a negative but never an
    anchor;
  * a single-class corpus is refused by the analysis law, because a triplet loss has no
    negative to pull against;
  * the embedding comes back on the unit sphere, which is the metric it was trained under;
  * the temporal sample indices are evenly spaced with both endpoints, and the layer read is
    offset by one because the embedding output sits first;
  * the corpus loader excludes prompt-swap generations, keys the group to the generation,
    and applies the positional correction at the absolute position of each sampled step.

CPU only; the networks are tiny and trained for a handful of epochs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anamnesis.analysis.contrastive_mlp import (
    EMBEDDING_OFFSET,
    embed,
    load_hidden_state_samples,
    mine_balanced_triplets,
    mine_triplets,
    projection_network,
    temporal_indices,
    train_embedding,
    train_projection,
)


def two_class_corpus(n: int = 24, d: int = 6, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = np.array(["linear" if i % 2 == 0 else "socratic" for i in range(n)])
    X[y == "socratic", 0] += 4.0
    return X, y


def test_the_banked_fit_returns_what_the_feature_family_loads() -> None:
    X, y = two_class_corpus()
    weights = train_projection(X, y, n_epochs=20, batch_triplets=16, bottleneck_dim=4, seed=1)
    assert set(weights) == {"w1", "b1", "w2", "b2", "scaler_mean", "scaler_scale"}
    assert weights["w1"].shape == (256, X.shape[1])
    assert weights["w2"].shape == (4, 256)
    assert weights["scaler_scale"].shape == (X.shape[1],)
    assert np.all(weights["scaler_scale"] > 0), "a zero-variance feature is left unscaled"

    plain = train_projection(
        X, y, n_epochs=20, batch_triplets=16, bottleneck_dim=4, seed=1, standardize=False
    )
    assert set(plain) == {"w1", "b1", "w2", "b2"}


def test_the_split_holds_out_whole_generations() -> None:
    """Two rows of one generation must not straddle the split; the fit still runs."""
    X, y = two_class_corpus(n=40)
    groups = np.array([index // 4 for index in range(len(X))])
    weights = train_projection(
        X, y, groups, n_epochs=20, batch_triplets=16, bottleneck_dim=4, seed=2
    )
    assert weights["w1"].shape[1] == X.shape[1]


def test_a_degenerate_group_split_falls_back_rather_than_failing() -> None:
    """One group means the grouped split would hold out everything or nothing."""
    X, y = two_class_corpus(n=20)
    groups = np.zeros(len(X), dtype=int)
    weights = train_projection(
        X, y, groups, n_epochs=10, batch_triplets=8, bottleneck_dim=4, seed=3
    )
    assert "w1" in weights


def test_class_first_mining_draws_every_class_equally_often() -> None:
    by_label = {"a": [0, 1, 2], "b": [3, 4], "c": [5, 6]}
    rng = np.random.default_rng(0)
    anchors, positives, negatives = mine_balanced_triplets(
        by_label, ["a", "b", "c"], rng, 300
    )
    assert len(anchors) == len(positives) == len(negatives) == 300
    label_of = {index: label for label, rows in by_label.items() for index in rows}
    for anchor, positive, negative in zip(anchors, positives, negatives):
        assert label_of[anchor] == label_of[positive], "a positive shares the anchor's class"
        assert label_of[anchor] != label_of[negative], "a negative does not"
        assert anchor != positive
    counts = {label: 0 for label in by_label}
    for anchor in anchors:
        counts[label_of[anchor]] += 1
    assert min(counts.values()) > 60, "no class is starved of anchors"


def test_balanced_mining_needs_two_classes_that_can_supply_a_pair() -> None:
    with pytest.raises(ValueError, match="at least two labels"):
        mine_balanced_triplets({"a": [0, 1]}, ["a"], np.random.default_rng(0), 10)


def test_anchor_uniform_mining_skips_an_anchor_with_no_sibling() -> None:
    labels = np.array(["a", "a", "b", "c"])
    anchors, positives, negatives = mine_triplets(labels, np.random.RandomState(0), 200)
    assert len(anchors) == len(positives) == len(negatives)
    assert len(anchors) < 200, "the two single-member classes cannot anchor a triplet"
    for anchor, positive, negative in zip(anchors, positives, negatives):
        assert labels[anchor] == labels[positive] and labels[anchor] != labels[negative]


def test_the_analysis_law_refuses_a_single_class_corpus() -> None:
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


def test_the_network_shape_is_the_one_the_banked_weights_describe() -> None:
    network = projection_network(7, hidden_dim=16, output_dim=3)
    keys = dict(network.state_dict())
    assert set(keys) == {"0.weight", "0.bias", "3.weight", "3.bias"}


def test_the_temporal_indices_span_the_generation_with_both_endpoints() -> None:
    assert temporal_indices(10, 5) == [0, 2, 4, 7, 9]
    assert temporal_indices(10, 1) == [0], "one sample is the prompt-end read"
    assert temporal_indices(2, 5)[0] == 0 and temporal_indices(2, 5)[-1] == 1


def write_raw(root: Path, gid: int, *, mode: str, n_steps: int, n_layers: int, width: int,
              condition: str = "standard", prompt_length: int = 4) -> np.ndarray:
    """A raw capture in the banked shape, written through the real saver.

    Going through ``save_raw_tensors_v3`` rather than hand-writing an npz is what makes the
    layer-offset assertions meaningful: the reconstruction under test is the one the
    instrument performs, including the embedding output sitting at index zero.
    """
    from anamnesis.extraction.raw_saver import save_raw_tensors_v3
    from anamnesis.extraction.state_extractor import RawGenerationData

    raw_dir = root / "raw_tensors"
    sig_dir = root / "signatures"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(gid)
    hidden = [
        rng.standard_normal((n_layers + 1, width)).astype(np.float32) for _ in range(n_steps)
    ]
    attentions = []
    for step in range(n_steps):
        weights = rng.random((n_layers, 2, prompt_length + step + 1)).astype(np.float32)
        attentions.append(weights / weights.sum(axis=-1, keepdims=True))
    raw = RawGenerationData(
        hidden_states=hidden,
        attentions=attentions,
        logits=[np.zeros(8, dtype=np.float32) for _ in range(n_steps)],
        chosen_token_ids=np.arange(n_steps, dtype=np.float32),
        pre_rope_keys={},
        prompt_length=prompt_length,
    )
    save_raw_tensors_v3(
        raw, gid, raw_dir, prompt_length=prompt_length,
        input_ids=list(range(prompt_length + n_steps)), top_k_logits=4,
    )
    (sig_dir / f"gen_{gid:03d}.json").write_text(
        json.dumps({"generation_id": gid, "mode": mode, "condition": condition})
    )
    return np.stack(hidden)


def test_the_corpus_keys_the_group_to_the_generation_and_drops_swaps(tmp_path: Path) -> None:
    for gid, mode in enumerate(["linear", "socratic", "swap_socratic→linear"]):
        write_raw(tmp_path / "run", gid, mode=mode, n_steps=6, n_layers=4, width=5)

    X, y, groups = load_hidden_state_samples(
        tmp_path / "run" / "raw_tensors",
        tmp_path / "run" / "signatures",
        [1, 2],
        temporal_samples=3,
    )
    assert X.shape == (2 * 2 * 3, 5), "two generations, two layers, three positions"
    assert set(np.unique(y)) == {"linear", "socratic"}
    assert set(np.unique(groups)) == {0, 1}, "the group is the generation"
    assert all(np.sum(groups == gid) == 6 for gid in (0, 1))


def test_the_positional_correction_is_applied_at_the_absolute_position(tmp_path: Path) -> None:
    raw = write_raw(tmp_path / "run", 0, mode="linear", n_steps=4, n_layers=3, width=5,
                    prompt_length=2)
    means = np.zeros((4, 32, 5), dtype=np.float32)
    means[1 + EMBEDDING_OFFSET, 2 + 0] = 7.0  # layer 1, absolute position prompt_length + step 0
    means_path = tmp_path / "positional_means.npz"
    np.savez(means_path, positional_means=means)

    X, _y, _groups = load_hidden_state_samples(
        tmp_path / "run" / "raw_tensors",
        tmp_path / "run" / "signatures",
        [1],
        temporal_samples=4,
        positional_means_path=means_path,
    )
    expected = raw[0][1 + EMBEDDING_OFFSET] - 7.0
    assert np.allclose(X[0], expected, atol=1e-3), (
        "the correction is subtracted at prompt_length + step"
    )
    assert np.allclose(X[1], raw[1][1 + EMBEDDING_OFFSET], atol=1e-3), (
        "other positions are untouched"
    )


def test_a_corpus_with_no_usable_generation_is_refused(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    (tmp_path / "meta").mkdir()
    with pytest.raises(FileNotFoundError, match="no training rows"):
        load_hidden_state_samples(tmp_path / "raw", tmp_path / "meta", [0])
