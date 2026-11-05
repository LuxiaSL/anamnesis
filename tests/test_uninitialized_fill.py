"""The batched reduction without torch's fill of new allocations: the same bytes.

Under ``torch.use_deterministic_algorithms`` torch writes NaN into every new ``empty``
allocation. The batched reduction turns that fill off while it runs
(:func:`uninitialized_fill`), because it writes everything it reads. These checks hold it
to that: rows reduced with the fill on and off, alone and batched, must be the same bytes,
and with the fill on no feature may be NaN, which is where a read of unwritten memory
shows. A planted read of an unwritten buffer must fail the check. The setting is restored
after the reduction, error or not.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anamnesis.extraction.fast import batched_families
from anamnesis.extraction.fast.engine_reducer import EngineReducer, uninitialized_fill
from anamnesis.extraction.fast.external import ExternalCaptureError
from test_engine_reducer import ENGINE, signed_example


@pytest.fixture(autouse=True)
def deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def made_rows(counts=(16, 16, 13, 9, 16, 11)):
    lane, rows, ids = None, [], []
    for i, steps in enumerate(counts):
        row_lane, row, token_ids, _, _ = signed_example(steps=steps, start=4, sampled=(0, 1, 2),
                                                        seed=i)
        lane = lane or row_lane
        rows.append(row)
        ids.append(token_ids)
    return lane, rows, ids


def reduce(lane, rows, ids, fill: bool) -> np.ndarray:
    reducer = EngineReducer(lane_for=lambda steps: lane, batched=True, fill_uninitialized=fill)
    return np.stack([r.features for r in reducer.reduce(rows, token_ids=ids, engine=ENGINE)])


def assert_fill_free(lane, rows, ids) -> None:
    """The rows' bytes with the fill on and off, batched and each alone, are the same, and
    with the fill on none is NaN."""
    filled = reduce(lane, rows, ids, True)
    assert not np.isnan(filled).any(), "a feature read memory nothing wrote"
    assert reduce(lane, rows, ids, False).tobytes() == filled.tobytes()
    for i in range(len(rows)):
        alone = reduce(lane, rows[i:i + 1], ids[i:i + 1], False)
        assert alone.tobytes() == filled[i:i + 1].tobytes()


def test_the_reduction_has_the_same_bytes_without_the_fill():
    assert_fill_free(*made_rows())


def test_the_check_bites_on_a_read_of_unwritten_memory(monkeypatch):
    """A family that returns a buffer it never wrote: the fill puts NaN in its features, which
    the reduction refuses as nonfinite (or, past that guard, the NaN check would catch)."""
    original = batched_families._eff_dims

    def unwritten(x, counts):
        return torch.empty_like(original(x, counts))

    monkeypatch.setattr(batched_families, "_eff_dims", unwritten)
    with pytest.raises((AssertionError, ExternalCaptureError)):
        assert_fill_free(*made_rows((16, 13)))


def test_the_fill_is_off_inside_and_restored_after():
    det = torch.utils.deterministic
    before = det.fill_uninitialized_memory
    with uninitialized_fill(False):
        assert det.fill_uninitialized_memory is False
    assert det.fill_uninitialized_memory == before
    with pytest.raises(RuntimeError, match="inside"):
        with uninitialized_fill(not before):
            raise RuntimeError("inside")
    assert det.fill_uninitialized_memory == before


def test_the_reduction_restores_the_fill_when_a_group_raises(monkeypatch):
    from anamnesis.extraction.fast import engine_reducer

    det = torch.utils.deterministic
    before = det.fill_uninitialized_memory
    seen: list[bool] = []

    def failing(*args, **kwargs):
        seen.append(det.fill_uninitialized_memory)
        raise RuntimeError("a group failed")

    monkeypatch.setattr(engine_reducer, "_reduce_uniform", failing)
    lane, rows, ids = made_rows((16,))
    reducer = EngineReducer(lane_for=lambda steps: lane, batched=True)
    with pytest.raises(RuntimeError, match="a group failed"):
        reducer.reduce(rows, token_ids=ids, engine=ENGINE)
    assert seen == [False]
    assert det.fill_uninitialized_memory == before
