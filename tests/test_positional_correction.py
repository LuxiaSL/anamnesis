"""The positional correction's edges, in the anchor and in the reference alike.

``_correct_hidden_state`` exists twice: in :mod:`anamnesis.extraction.state_extractor`,
the numeric anchor, and in :mod:`anamnesis.extraction.state_extractor_reference`, the
unoptimized statement of the same math. Every case below runs against both, and the
last one asserts they agree everywhere on a grid that includes the edges, so a guard
added to one copy and not the other fails here.

The edges are where numpy's indexing is wrong for this table: a negative index reads
from the end. A negative position would be corrected by the mean of the last
position, and a negative layer by the last layer's mean.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from anamnesis.extraction import state_extractor, state_extractor_reference

Correct = Callable[[np.ndarray, int, int, np.ndarray | None], np.ndarray]

COPIES: dict[str, Correct] = {
    "anchor": state_extractor._correct_hidden_state,
    "reference": state_extractor_reference._correct_hidden_state,
}

LAYERS, POSITIONS, UNITS = 3, 5, 4


def _means() -> np.ndarray:
    """Each row is distinguishable: value = 100 * layer + position."""
    layer = np.arange(LAYERS)[:, None, None]
    position = np.arange(POSITIONS)[None, :, None]
    return np.broadcast_to(100.0 * layer + position, (LAYERS, POSITIONS, UNITS)).astype(
        np.float32
    )


@pytest.fixture(params=sorted(COPIES))
def correct(request: pytest.FixtureRequest) -> Correct:
    return COPIES[request.param]


def test_a_position_inside_the_table_subtracts_its_own_row(correct: Correct) -> None:
    h = np.zeros(UNITS, dtype=np.float32)
    assert np.array_equal(correct(h, 1, 2, _means()), -_means()[1, 2])


def test_a_position_past_the_table_is_corrected_by_its_last_row(correct: Correct) -> None:
    h = np.zeros(UNITS, dtype=np.float32)
    assert np.array_equal(correct(h, 2, POSITIONS + 7, _means()), -_means()[2, POSITIONS - 1])


def test_a_negative_position_is_corrected_by_row_zero_not_the_last(correct: Correct) -> None:
    h = np.zeros(UNITS, dtype=np.float32)
    corrected = correct(h, 1, -1, _means())
    assert np.array_equal(corrected, -_means()[1, 0])
    assert not np.array_equal(corrected, -_means()[1, POSITIONS - 1])


@pytest.mark.parametrize("layer", [-1, LAYERS])
def test_a_layer_outside_the_table_is_refused(correct: Correct, layer: int) -> None:
    with pytest.raises(IndexError, match="outside the positional means"):
        correct(np.zeros(UNITS, dtype=np.float32), layer, 0, _means())


def test_no_means_leaves_the_state_alone(correct: Correct) -> None:
    h = np.arange(UNITS, dtype=np.float32)
    assert correct(h, -5, -5, None) is h


def test_the_two_copies_agree_across_the_grid_and_its_edges() -> None:
    means = _means()
    h = np.random.default_rng(3).normal(size=UNITS).astype(np.float32)
    for layer in range(LAYERS):
        for position in range(-3, POSITIONS + 3):
            anchor = COPIES["anchor"](h, layer, position, means)
            reference = COPIES["reference"](h, layer, position, means)
            assert np.array_equal(anchor, reference), (layer, position)
    for layer in (-1, LAYERS):
        for copy in COPIES.values():
            with pytest.raises(IndexError):
                copy(h, layer, 0, means)
