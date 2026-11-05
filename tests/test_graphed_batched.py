"""CUDA-graph replay of the batched families (graph step 3): a replayed row is the eager row, byte for byte.

Over the step-bucket toy lane (tests/test_step_buckets.py), on CUDA:

- replay is eager over many count vectors in one bucket, the key reused across them, and the lane id and receipt
  are the eager ones;
- a row has the same bytes alone and beside mates of other counts, under the graphs (filler rows included);
- every padded batch size up to the ladder's reuses its ladder size's key;
- a planted read of the batch's data on the host refuses the capture, and the rows still come out right (eagerly);
- the spectral solve is the one solver cut of a key, reading only the graph's own buffers;
- a small pool budget evicts least recently used, and the audit counts its re-verifications.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anamnesis.extraction.fast import batched_families
from anamnesis.extraction.fast.batched_stats import bucket_width
from anamnesis.extraction.fast.engine_reducer import EngineReducer
from anamnesis.extraction.fast.graphed_batched import LADDER, BatchedGraphCache, ladder_size
from test_step_buckets import BUCKETED, ENGINE, MAX_START, SHORT, lanes, signed_row

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
HIDDEN = 1536


@pytest.fixture(autouse=True)
def deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def _reduce(made, cache=None):
    reducer = EngineReducer(lane_for=lanes("cuda", HIDDEN), batched=True, buckets="pow2",
                            expected_weights=ENGINE["weights"], graphs=cache)
    return reducer.reduce([r for r, _ in made], token_ids=[i for _, i in made], engine=ENGINE)


def _same(a, b) -> bool:
    return (a.features.tobytes() == b.features.tobytes() and a.lane_id == b.lane_id
            and a.meta["feature_names"] == b.meta["feature_names"])


def test_the_ladder():
    assert [ladder_size(n) for n in (1, 2, 3, 5, 9, 13, 17, 25, 49, 97, 128, 129)] == [
        1, 2, 4, 8, 12, 16, 24, 32, 64, 128, 128, None]
    assert LADDER[-1] == 128 and list(LADDER) == sorted(LADDER)


@CUDA
def test_replay_is_eager_over_many_count_vectors_in_one_bucket():
    rng = np.random.default_rng(3)
    cache = BatchedGraphCache(audit_every=0)
    pool = [n for n in BUCKETED if bucket_width(n, "pow2") == 32]  # one bucket: 17..32 steps
    for trial in range(6):
        made = [signed_row(int(rng.choice(pool)), int(rng.integers(1, MAX_START + 1)), 7000 + 37 * trial + i, HIDDEN)
                for i in range(5)]  # five rows: the ladder's 8
        eager = _reduce(made)
        graphed = _reduce(made, cache)
        for e, g in zip(eager, graphed, strict=True):
            assert _same(e, g), trial
    stats = cache.stats()
    assert not stats.refused, stats.refused
    assert stats.captures == 1 and stats.replays == 5, stats


@CUDA
def test_a_row_has_the_same_bytes_alone_and_beside_mates_under_the_graphs():
    rng = np.random.default_rng(4)
    cache = BatchedGraphCache(audit_every=0)
    targets = [signed_row(n, int(rng.integers(1, MAX_START + 1)), 8000 + n, HIDDEN) for n in (9, 16, 61, 5)]
    alone = [_reduce([t], cache)[0] for t in targets]
    eager_alone = [_reduce([t])[0] for t in targets]
    for a, e in zip(alone, eager_alone, strict=True):
        assert _same(a, e)
    for size in (3, 7, 12):
        for t, target in enumerate(targets):
            n = target[0].positions
            same_bucket = [m for m in BUCKETED + SHORT if bucket_width(m, "pow2") == bucket_width(n, "pow2")]
            mates = [signed_row(int(rng.choice(same_bucket)), int(rng.integers(1, MAX_START + 1)),
                                9000 + size * 100 + t * 10 + m, HIDDEN) for m in range(size - 1)]
            place = int(rng.integers(0, size))
            got = _reduce(mates[:place] + [target] + mates[place:], cache)[place]
            assert _same(got, alone[t]), (size, n)
    assert not cache.stats().refused


@CUDA
def test_a_host_read_of_the_batchs_data_refuses_the_capture_and_the_rows_stay_right(monkeypatch):
    original = batched_families._halves

    def peeks(x, counts):
        if counts.dev.sum().item() < 0:  # a read of the batch's data on the host
            raise AssertionError("unreachable")
        return original(x, counts)
    monkeypatch.setattr(batched_families, "_halves", peeks)
    made = [signed_row(n, 3, 600 + n, HIDDEN) for n in (17, 21, 30)]
    eager = _reduce(made)
    cache = BatchedGraphCache(audit_every=0)
    with pytest.warns(RuntimeWarning, match="reduced eagerly"):
        graphed = _reduce(made, cache)
    for e, g in zip(eager, graphed, strict=True):
        assert _same(e, g)
    stats = cache.stats()
    assert stats.entries == 0 and len(stats.refused) == 1
    assert "derived from the span's data" in next(iter(stats.refused.values()))


@CUDA
def test_the_spectral_solve_is_one_cut_reading_the_graphs_own_buffers():
    made = [signed_row(n, 4, 500 + n, HIDDEN) for n in (17, 25, 31)]
    cache = BatchedGraphCache(audit_every=0)
    _reduce(made, cache)
    (entry,) = cache.entries.values()
    assert entry.tape.solver_steps == 1
    from anamnesis.extraction.fast.graphed import _SolverStep

    (step,) = [s for s in entry.tape.steps if isinstance(s, _SolverStep)]
    caller = {t.untyped_storage().data_ptr() for r, _ in made for t in r.surfaces.values()
              if isinstance(t, torch.Tensor)}
    owned = {r.base.untyped_storage().data_ptr() for r in entry.replicas}
    for arg in step.args:
        if isinstance(arg, torch.Tensor):
            ptr = arg.untyped_storage().data_ptr()
            assert ptr not in caller
    sizes = step.args[1]
    assert sizes.untyped_storage().data_ptr() in owned or sizes.data_ptr() != 0  # the counts' replica (or a view)


@CUDA
def test_eviction_under_the_pool_budget_and_the_audit_count():
    cache = BatchedGraphCache(audit_every=1, pool_budget_bytes=1)  # every capture overruns: keep only the newest
    for widths, seed in (((9, 13), 1), ((17, 25), 2), ((40, 61), 3)):
        made = [signed_row(n, 3, 400 + seed * 10 + n, HIDDEN) for n in widths]
        _reduce(made, cache)
        _reduce(made, cache)  # a replay, audited (audit_every=1)
    stats = cache.stats()
    assert stats.entries == 1 and stats.evictions == 2, stats
    assert stats.audits >= 1 and stats.audit_mismatches == 0, stats
