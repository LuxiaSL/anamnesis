"""Per-row indices from per-bucket device tables.

A padded bucket's per-row positions (trajectory points, strided samples, spectral steps)
are a function of each row's count alone. They're built once per bucket as a table over
every count the bucket admits, and a batch gathers its rows from it on the device.
Checked: the gathered index and sizes equal the per-batch lists they replace, and a new
batch of a bucket uploads nothing to build them.
"""

from __future__ import annotations

import pytest
import torch

from anamnesis.extraction.fast import batched_stats
from anamnesis.extraction.fast.batched_stats import Counts, canonical_size, per_row_index
from anamnesis.extraction.fast.ops import trajectory_indices


def _old_index(counts: Counts, key, picks):
    """The per-batch construction the table replaces."""
    lists = [list(picks(n)) for n in counts.host]
    width = canonical_size(("index", key), counts.width, lambda n: len(list(picks(n))))
    padded = [p + [p[-1] if p else 0] * (width - len(p)) for p in lists]
    return torch.tensor(padded, dtype=torch.long), [len(p) for p in lists]


PICKS = {
    "traj5": lambda n: trajectory_indices(n, 5),
    "traj17": lambda n: trajectory_indices(n, 17),
    "stride": lambda n: list(range(0, n, max(1, n // 30) if n > 30 else 1)),
    "empty-below-3": lambda n: list(range(n)) if n >= 3 else [],
}


@pytest.mark.parametrize("name", sorted(PICKS))
def test_the_table_gives_the_per_batch_index(name):
    g = torch.Generator().manual_seed(len(name))
    width = 64
    host = torch.randint(1, width + 1, (11,), generator=g).tolist()
    counts = Counts.of(host, "cpu", width, padded=True)
    index, sized = per_row_index(counts, ("test", name), PICKS[name])
    want, sizes = _old_index(counts, ("test", name), PICKS[name])
    assert torch.equal(index, want)
    assert sized.host == tuple(sizes) and sized.dev.tolist() == sizes
    assert (sized.width, sized.padded) == (want.shape[1], True)


def test_a_new_batch_uploads_nothing_to_build_its_index(monkeypatch):
    picks = PICKS["traj17"]
    per_row_index(Counts.of([5, 9], "cpu", 32, padded=True), ("upload", 1), picks)  # builds the table
    calls: list[int] = []
    original = batched_stats.host_tensor

    def counted(values, dtype, device):
        calls.append(len(values))
        return original(values, dtype, device)

    monkeypatch.setattr(batched_stats, "host_tensor", counted)
    counts = Counts([7, 31, 12], torch.tensor([7, 31, 12]), 32, padded=True)  # counts already on the device
    index, sized = per_row_index(counts, ("upload", 1), picks)
    assert calls == []
    want, sizes = _old_index(counts, ("upload", 1), picks)
    assert torch.equal(index, want) and sized.dev.tolist() == sizes
