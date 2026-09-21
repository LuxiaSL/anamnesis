"""Holding out topics: every topic held out once, and a refusal when it cannot be.

Sections 5, 8 and 9 all generalize across topics, so all three partition the topics
into held-out groups, and two of them compare their numbers with each other. One
partitioner serves all three, and these are the properties the sections depend on:

  * every topic lands in exactly one held-out group, so a variant is computed over
    the topics it reports and not a subset of them;
  * group sizes differ by at most one, which is what makes the folds comparable;
  * more folds than topics is refused by name rather than producing empty held-out
    groups that fail later inside a scaler;
  * the partition is unchanged, group for group, wherever the fold count divides
    the topic count — which is the case for every banked corpus, so no banked
    number moves.

CPU only; the partitioner takes a count and a generator, not data.
"""

from __future__ import annotations

import numpy as np
import pytest

from anamnesis.analysis.gauntlet.contrastive import build_topic_folds
from anamnesis.analysis.gauntlet.geometry import _generate_topic_folds
from anamnesis.analysis.gauntlet.utils import (
    InsufficientTopicsError,
    topic_fold_partition,
)

# The fold counts section 5 asks for, and the topic count every banked corpus has.
CCGP_FOLD_COUNTS = (4, 5, 10, 20)
BANKED_TOPIC_COUNT = 20


def old_stride_folds(n_topics: int, n_folds: int, rng: np.random.Generator) -> list[list[int]]:
    """Equal slices of one permutation: the partition where the count divides evenly.

    This is the construction to compare against, written out here so the claim
    "no banked number moves" is checked rather than asserted. It is not a partition
    in general — with a remainder it covers fewer topics than it was given, which
    is why it is not what the sections use.
    """
    perm = rng.permutation(n_topics)
    fold_size = n_topics // n_folds
    return [perm[i * fold_size:(i + 1) * fold_size].tolist() for i in range(n_folds)]


@pytest.mark.parametrize("n_folds", CCGP_FOLD_COUNTS)
def test_the_banked_fold_counts_partition_exactly_as_equal_slices_did(n_folds: int) -> None:
    """At 20 topics every fold count divides evenly, so the groups are the same groups.

    Same seed, same permutation, same group boundaries — which is the construction
    behind "no banked CCGP number changes": every banked pass loaded 20 topics.
    """
    assert BANKED_TOPIC_COUNT % n_folds == 0, "the premise of this comparison"
    new = [
        group.tolist()
        for group in topic_fold_partition(
            BANKED_TOPIC_COUNT, n_folds, np.random.default_rng(42),
        )
    ]
    old = old_stride_folds(BANKED_TOPIC_COUNT, n_folds, np.random.default_rng(42))
    assert new == old


@pytest.mark.parametrize("n_topics,n_folds", [(20, 4), (20, 20), (24, 20), (7, 5), (5, 5), (13, 4)])
def test_every_topic_is_held_out_exactly_once(n_topics: int, n_folds: int) -> None:
    groups = topic_fold_partition(n_topics, n_folds, np.random.default_rng(7))
    assert len(groups) == n_folds
    held_out = [int(i) for group in groups for i in group]
    assert sorted(held_out) == list(range(n_topics)), "a topic is in one group, and in one"
    sizes = sorted(len(group) for group in groups)
    assert sizes[-1] - sizes[0] <= 1, "group sizes differ by at most one"
    assert sizes[0] >= 1, "no held-out group is empty"


def test_a_remainder_is_distributed_rather_than_dropped() -> None:
    """24 topics over 20 folds: the four that equal slices left out are held out here."""
    groups = topic_fold_partition(24, 20, np.random.default_rng(0))
    assert sum(len(group) for group in groups) == 24
    dropped = 24 - sum(len(group) for group in old_stride_folds(24, 20, np.random.default_rng(0)))
    assert dropped == 4, "the construction being replaced covered 20 of 24 topics"


@pytest.mark.parametrize("n_topics,n_folds", [(5, 10), (5, 20), (1, 4), (19, 20)])
def test_more_folds_than_topics_is_refused_by_name(n_topics: int, n_folds: int) -> None:
    with pytest.raises(InsufficientTopicsError, match="would be empty"):
        topic_fold_partition(n_topics, n_folds, np.random.default_rng(0))


def test_a_fold_count_below_one_is_refused() -> None:
    with pytest.raises(InsufficientTopicsError, match="at least 1"):
        topic_fold_partition(10, 0, np.random.default_rng(0))


def test_the_two_sections_hold_out_the_same_topics_for_the_same_seed() -> None:
    """Section 5 takes topic names and section 8 row masks, over one partition."""
    topics = np.array([f"topic_{i // 3}" for i in range(30)])
    unique = sorted(set(topics))

    ccgp_folds = _generate_topic_folds(
        unique, n_folds=5, rng=np.random.default_rng(42),
    )
    contrastive_folds = build_topic_folds(topics, n_folds=5, seed=42)

    assert len(ccgp_folds) == len(contrastive_folds) == 5
    for (_, held_out_names), (_, test_mask) in zip(ccgp_folds, contrastive_folds):
        assert set(held_out_names) == set(topics[test_mask].tolist())

    # And between them they cover every topic once, which is the property a number
    # compared across the two sections rests on.
    covered = [name for _, held_out in ccgp_folds for name in held_out]
    assert sorted(covered) == unique


def test_section_five_folds_train_on_everything_it_does_not_hold_out() -> None:
    unique = [f"topic_{i}" for i in range(13)]
    folds = _generate_topic_folds(unique, n_folds=4, rng=np.random.default_rng(3))
    assert len(folds) == 4
    for train, held_out in folds:
        assert set(train).isdisjoint(held_out)
        assert set(train) | set(held_out) == set(unique)
    assert sorted(name for _, held_out in folds for name in held_out) == sorted(unique)


def test_section_eight_refuses_a_fold_count_its_corpus_cannot_support() -> None:
    topics = np.array([f"topic_{i}" for i in range(4)])
    with pytest.raises(InsufficientTopicsError):
        build_topic_folds(topics, n_folds=5, seed=42)


def test_the_two_topic_sections_state_a_corpus_too_narrow_to_hold_out() -> None:
    """Sections 8 and 9 hold out a fixed number of folds, so a narrow corpus is stated.

    The partitioner refuses rather than returning empty held-out groups, and the
    sections say so in their own result instead of letting that refusal end the
    pass — the same rule as an absent block.
    """
    from anamnesis.analysis.gauntlet.contrastive import N_TOPIC_FOLDS
    from anamnesis.analysis.gauntlet.semantic import _contrastive_topic_heldout

    topics = np.array([f"topic_{i}" for i in range(N_TOPIC_FOLDS - 1)])
    modes = np.array(["linear", "socratic"] * 2)
    X = np.zeros((len(topics), 3), dtype=np.float32)

    comparison = _contrastive_topic_heldout(X, X, None, modes, topics)
    assert comparison.error is not None
    assert str(N_TOPIC_FOLDS) in comparison.error or "torch" in comparison.error
