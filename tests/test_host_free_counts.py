"""Device-only step counts: a padded bucket reduced without reading a row's count on the host.

A captured graph replays its launches with whatever counts sit in its input buffer, so nothing between a batch's checks
and its finish may branch on, or build a table from, one batch's counts on the host. Under
:func:`~anamnesis.extraction.fast.engine_reducer.host_free_counts` a padded bucket's counts are device-only
(:meth:`~anamnesis.extraction.fast.batched_stats.Counts.device_only`). Checked here:

- a device-only count refuses every host read, and so does every count derived from it;
- the effective dimension's two Gram matrices, formed for every row and selected per row, are the bytes of the host
  split, including the rows each side of the split and buckets narrower than the matrix;
- a reduce under device-only counts has the host reduce's bytes, lane id and receipt, over many count vectors in each
  bucket (short rows with their placeholders included), alone and beside mates;
- the falsifier: a host read planted inside a family raises instead of passing quietly.

``ANAMNESIS_TEST_HOST_FREE=1`` runs the whole suite under device-only counts (see conftest).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anamnesis.extraction.fast import batched_families
from anamnesis.extraction.fast.batched_attention import mean_rows_cover
from anamnesis.extraction.fast.batched_families import _eff_dims
from anamnesis.extraction.fast.batched_stats import (
    BUCKET_FLOOR,
    Counts,
    HostCountsError,
    bucket_width,
    per_row_index,
    positional_means_cover,
)
from anamnesis.extraction.fast.engine_reducer import host_free_counts
from anamnesis.extraction.fast.ops import trajectory_indices
from test_step_buckets import BUCKETED, DEVICES, MAX_START, SHORT, reduce, signed_row


@pytest.fixture(autouse=True)
def deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def _bucket(host, width=16):
    return Counts.of(host, "cpu", width, padded=True)


def test_device_only_counts_refuse_every_host_read():
    counts = _bucket([3, 9, 16, 5]).device_only()
    for read in (lambda c: c.host, lambda c: c.lo, lambda c: c.hi):
        with pytest.raises(HostCountsError):
            read(counts)
    assert counts.reach == 16 and counts.rows == 4 and counts.is_device_only
    assert counts.dev.tolist() == [3, 9, 16, 5]
    derived = [counts.minus(1), counts.repeat(3), counts.tile(2), counts.every(1, 2),
               counts.derive(counts.dev * 2, lambda c: c * 2),
               Counts.concat([counts, counts]),
               per_row_index(counts, ("traj", 5), lambda n: trajectory_indices(n, 5))[1]]
    for d in derived:
        assert d.is_device_only
        with pytest.raises(HostCountsError):
            d.host
    assert derived[0].dev.tolist() == [2, 8, 15, 4] and derived[3].dev.tolist() == [9, 5]


def test_only_a_padded_bucket_goes_device_only_and_never_below_its_floor():
    with pytest.raises(ValueError):
        Counts.of([4, 4], "cpu", 4).device_only()
    with pytest.raises(ValueError):
        Counts.of([4, 4], "cpu", None, padded=True).device_only()
    with pytest.raises(ValueError, match="floor"):
        _bucket([1, 9]).device_only(BUCKET_FLOOR)


def test_below_is_exact_on_the_host_and_the_floor_on_the_device():
    host = _bucket([5, 9])
    dev = host.device_only(2)
    assert [host.below(k) for k in (2, 5, 6)] == [False, False, True]
    assert [dev.below(k) for k in (2, 3, 6)] == [False, True, True]
    assert dev.minus(1).below(2)  # a derived count's floor moves with it


def test_the_cover_checks_skip_only_device_only_counts():
    counts = _bucket([5, 9])
    assert mean_rows_cover(12, [3, 3], counts) and not mean_rows_cover(11, [3, 3], counts)
    assert not mean_rows_cover(100, [0, 3], counts.device_only())  # a start is still checked
    pm = torch.zeros(2, 12, 4)
    assert positional_means_cover(pm, [3, 3], counts) and not positional_means_cover(pm, [4, 4], counts)
    assert positional_means_cover(pm, [40, 40], counts.device_only())


@pytest.mark.parametrize("n,width", [(6, 16), (6, 8), (12, 8), (3, 16)],
                         ids=["both-sides", "mixed-narrow", "all-steps", "all-columns"])
def test_the_masked_gram_choice_is_the_split_choice(n, width):
    """Rows of fewer steps than columns take the Gram matrix over their steps, the rest over the columns: the masked
    form computes what the bucket may need for every row, and each row keeps its own."""
    g = torch.Generator().manual_seed(n * 100 + width)
    for trial in range(6):
        host = torch.randint(BUCKET_FLOOR, width + 1, (7,), generator=g).tolist()
        x = torch.randn(7, width, n, generator=g, dtype=torch.float64)
        for b, c in enumerate(host):
            x[b, c:] = 0  # the padded steps, as a stacked bucket has them
        counts = _bucket(host, width)
        split = _eff_dims(x, counts)
        masked = _eff_dims(x, _bucket(host, width).device_only())
        assert split.numpy().tobytes() == masked.numpy().tobytes(), (trial, host)


def _batches(rng, hidden):
    """Batches whose rows fill each bucket in different ways: short rows in the first bucket, rows straddling the
    toy thresholds above it, and both together."""
    pool = BUCKETED + SHORT
    out = []
    for size in (1, 3, 9):
        for _ in range(2):
            steps = [int(rng.choice(pool)) for _ in range(size)]
            out.append([signed_row(n, int(rng.integers(1, MAX_START + 1)), int(rng.integers(1 << 30)), hidden)
                        for n in steps])
    return out


@pytest.mark.parametrize("device,hidden", DEVICES)
def test_a_host_free_reduce_has_the_host_reduces_bytes(device, hidden):
    rng = np.random.default_rng(29)
    for batch in _batches(rng, hidden):
        with host_free_counts(False):
            host = reduce(device, hidden, batch)
        with host_free_counts():
            free = reduce(device, hidden, batch)
        for h, f, (row, _) in zip(host, free, batch, strict=True):
            assert not isinstance(f, Exception), f
            assert f.features.tobytes() == h.features.tobytes(), row.positions
            assert f.lane_id == h.lane_id and f.meta["feature_names"] == h.meta["feature_names"]
            fresh = ("replay_id",)  # a new id per reduce
            assert ({k: v for k, v in f.meta["receipt"].items() if k not in fresh}
                    == {k: v for k, v in h.meta["receipt"].items() if k not in fresh}), row.positions
            assert f.meta["receipt"]["bucket_width"] == bucket_width(row.positions, "pow2")


def test_the_switch_is_scoped():
    from anamnesis.extraction.fast.engine_reducer import _HOST_FREE

    before = _HOST_FREE.get()  # on under ANAMNESIS_TEST_HOST_FREE=1
    with host_free_counts():
        assert _HOST_FREE.get()
        with host_free_counts(False):
            assert not _HOST_FREE.get()
        assert _HOST_FREE.get()
    assert _HOST_FREE.get() == before


def test_a_host_read_inside_a_family_raises_under_device_only_counts(monkeypatch):
    """The falsifier: a family that reads its counts on the host passes in the host reduce and raises under
    device-only counts, never quietly baking one batch's counts into the launches."""
    original = batched_families._halves

    def peeks(x, counts):
        sum(counts.host)
        return original(x, counts)
    monkeypatch.setattr(batched_families, "_halves", peeks)
    batch = [signed_row(n, 3, 900 + n, 8) for n in (9, 13)]
    with host_free_counts(False):
        assert all(not isinstance(r, Exception) for r in reduce("cpu", 8, batch))
    with host_free_counts(), pytest.raises(HostCountsError):
        reduce("cpu", 8, batch)
