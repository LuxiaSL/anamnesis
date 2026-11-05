"""The spectral Laplacians of every sampled layer are solved together.

The attention family cuts its sampled layers into chunks bounded by the working set. Each
chunk builds its rows' spectral Laplacians, and the eigenvalues they need are solved once for
the whole batch: one ``eigvalsh_sized`` call, which groups the rows of all chunks by size. A
row's features must be the same bytes whether its layers run as one chunk or one per layer, and
the solves must not multiply with the chunks.
"""

from __future__ import annotations

import pytest
import torch

from anamnesis.extraction.fast import batched_attention, batched_stats
from test_batched_stats import reduce, signed_row


@pytest.fixture(autouse=True)
def deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def _rows():
    made = [signed_row(n, 3 + n % 7, 400 + n, 8) for n in (16, 13, 9, 16, 11)]
    return [r for r, _ in made], [i for _, i in made]


SOLVE = batched_attention.eigvalsh_sized


def _features(monkeypatch, working_bytes: int):
    monkeypatch.setattr(batched_stats, "WORKING_BYTES", working_bytes)
    monkeypatch.setattr(batched_attention, "WORKING_BYTES", working_bytes)
    calls: list[int] = []
    solve = SOLVE

    def counted(lap, k):
        calls.append(lap.shape[0])
        return solve(lap, k)

    monkeypatch.setattr(batched_attention, "eigvalsh_sized", counted)
    rows, ids = _rows()
    out = reduce("cpu", 8, rows, ids, batched=True)
    return [r.features.tobytes() for r in out], calls


def test_one_solve_for_every_chunk_and_the_same_bytes(monkeypatch):
    whole, whole_calls = _features(monkeypatch, 1 << 40)   # every sampled layer in one chunk
    split, split_calls = _features(monkeypatch, 1)         # one chunk per sampled layer
    assert split == whole
    rows_x_layers = 5 * 3  # five rows at three sampled layers
    assert whole_calls == split_calls == [rows_x_layers]  # one solve per batch, every row of every layer


def test_counts_concatenate_in_order():
    a = batched_stats.Counts.of([3, 5], "cpu", 8, padded=True)
    b = batched_stats.Counts.of([7], "cpu", 8, padded=True)
    c = batched_stats.Counts.concat([a, b])
    assert c.host == (3, 5, 7) and c.dev.tolist() == [3, 5, 7] and (c.width, c.padded) == (8, True)
    with pytest.raises(ValueError):
        batched_stats.Counts.concat([a, batched_stats.Counts.of([1], "cpu", 4, padded=True)])
