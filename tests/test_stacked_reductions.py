"""Stacked passes: several same-shaped batches reduced in one pass give each row its own bytes.

The batched families reduce many ``[B, m]`` series that share their step counts: the
windowed operators and spectrum of every novelty, drift and alignment series, and the
routing family's summaries and means. :func:`stacked` stacks such series along the rows,
tiles the counts to match, and splits the result back. Every reduction is batchmate-invariant,
so a row's result must be the same bytes in a pass of many as in a call of its own. These
checks hold :func:`stacked`, :func:`_temporal_many` and the deferred operator columns to that,
with ragged, padded, shifted and uniform counts, across chunk boundaries, and they bite when
the counts are tiled in the wrong order.
"""

from __future__ import annotations

import pytest
import torch

from anamnesis.extraction.fast import batched_stats
from anamnesis.extraction.fast.batched_stats import (
    Counts,
    _Columns,
    _operator_names,
    _temporal_many,
    _temporal_rows,
    stacked,
)

N_WINDOWS = 3


def ragged(batch: int, width: int, seed: int) -> Counts:
    """Padded counts of ``batch`` rows in a bucket of ``width``, from two to ``width`` steps."""
    g = torch.Generator().manual_seed(seed)
    host = torch.randint(2, width + 1, (batch,), generator=g).tolist()
    return Counts.of(host, "cpu", width, padded=True)


def series(batch: int, width: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, width, generator=g, dtype=torch.float64)


def cases() -> list[tuple[torch.Tensor, Counts | None]]:
    """Series of two widths under four kinds of counts, several sharing each counts object."""
    padded, uniform = ragged(5, 40, 0), Counts.of([40] * 5, "cpu")
    shifted = padded.minus(1)
    short = ragged(5, 12, 1)
    out: list[tuple[torch.Tensor, Counts | None]] = []
    for i in range(3):
        out += [(series(5, 40, 10 + i), padded), (series(5, 39, 20 + i), shifted),
                (series(5, 40, 30 + i), uniform), (series(5, 40, 40 + i), None),
                (series(5, 12, 50 + i), short)]
    return out


def test_tiled_counts_are_the_counts_batch_after_batch():
    counts = ragged(4, 16, 3)
    tiled = counts.tile(3)
    assert tiled.host == counts.host * 3
    assert torch.equal(tiled.dev, counts.dev.repeat(3))
    assert (tiled.width, tiled.padded) == (counts.width, counts.padded)
    assert counts.tile(3) is tiled  # built once


@pytest.mark.parametrize("include_stft", [True, False])
def test_many_series_have_the_bytes_of_one_call_each(include_stft):
    items = cases()
    many = _temporal_many(items, N_WINDOWS, include_stft)
    for (x, counts), got in zip(items, many, strict=True):
        alone = _temporal_rows(x, N_WINDOWS, include_stft, counts)
        assert got.dtype == alone.dtype and got.shape == alone.shape
        assert got.numpy().tobytes() == alone.numpy().tobytes()


@pytest.mark.parametrize("budget", [1, 5 * 2 * batched_stats._temporal_row_bytes(40, True)])
def test_chunked_passes_have_the_same_bytes(monkeypatch, budget):
    """One series per pass, and two per pass with a third left over."""
    items = cases()
    whole = _temporal_many(items, N_WINDOWS, True)
    monkeypatch.setattr(batched_stats, "WORKING_BYTES", budget)
    for got, want in zip(_temporal_many(items, N_WINDOWS, True), whole, strict=True):
        assert got.numpy().tobytes() == want.numpy().tobytes()


def test_the_check_bites_when_the_counts_are_tiled_in_the_wrong_order(monkeypatch):
    """Counts interleaved row by row instead of batch after batch give rows other rows'
    counts: the byte check must catch it."""
    monkeypatch.setattr(Counts, "tile", Counts.repeat)
    items = [(series(5, 40, i), ragged(5, 40, 0)) for i in range(3)]
    items = [(x, items[0][1]) for x, _ in items]  # one counts object, so they stack
    many = _temporal_many(items, N_WINDOWS, True)
    assert any(got.numpy().tobytes() != _temporal_rows(x, N_WINDOWS, True, c).numpy().tobytes()
               for (x, c), got in zip(items, many, strict=True))


def test_stacked_calls_one_at_a_time_when_the_shapes_differ():
    calls: list[int] = []

    def fn(x: torch.Tensor, counts: Counts | None) -> torch.Tensor:
        calls.append(x.shape[0])
        return x.sum(dim=1)

    xs = [series(3, 8, 0), series(3, 9, 1)]
    got = stacked(fn, xs, None, 64)
    assert calls == [3, 3]
    assert all(torch.equal(g, x.sum(dim=1)) for g, x in zip(got, xs, strict=True))
    calls.clear()
    stacked(fn, [series(3, 8, 0), series(3, 8, 1)], None, 64)
    assert calls == [6]


def test_deferred_operator_columns_are_the_immediate_ones():
    counts = ragged(5, 40, 0)
    one = counts.minus(1)
    inputs = [("a", series(5, 40, 1), counts), ("b", series(5, 39, 2), one),
              ("c", series(5, 40, 3), counts)]
    deferred, immediate = _Columns(), _Columns()
    for i, (prefix, x, c) in enumerate(inputs):
        deferred.put(f"plain{i}", x[:, 0])
        immediate.put(f"plain{i}", x[:, 0])
        deferred.operators(prefix, x, N_WINDOWS, True, c)
        immediate.block(_operator_names(prefix, N_WINDOWS, True),
                        _temporal_rows(x, N_WINDOWS, True, c))
    assert deferred.names == immediate.names
    assert len(deferred) == len(immediate)
    got, want = deferred.stack(5, "cpu"), immediate.stack(5, "cpu")
    assert got.numpy().tobytes() == want.numpy().tobytes()


def test_zeroing_columns_resolves_the_deferred_ones_first():
    counts = ragged(4, 16, 2)
    cols = _Columns()
    cols.put("before", torch.ones(4, dtype=torch.float64))
    start = len(cols)
    cols.operators("x", series(4, 16, 0), N_WINDOWS, False, counts)
    rows = torch.tensor([True, False, True, False])
    cols.where_since(start, rows)
    values = cols.stack(4, "cpu")
    assert torch.equal(values[:, 0], torch.ones(4, dtype=torch.float64))
    assert torch.all(values[rows, start:] == 0)
    want = _temporal_rows(series(4, 16, 0), N_WINDOWS, False, counts).double()
    assert torch.equal(values[~rows, start:], want[~rows])
