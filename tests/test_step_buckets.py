"""Step-count buckets: rows of different step counts reduced together, batchmate-invariant.

Signed rows are random but well-formed, over a toy lane whose thresholds fall inside the
buckets: the decay fit (ten steps), epochs (window 8, stride 4: twelve steps), the surprise
window (twenty), the entropy subsampling (thirty and sixty) and the spectral graph's size.
Under the ``pow2`` rule a row of ``n >= 2`` steps is padded to the next power of two (at
least eight) and reduced beside rows of other step counts in its bucket, the short-span
placeholders (the residual trajectory below three steps, per-head below four, the windows
and spectrum below four and eight) per row; a one-step row, whose schema is its own, keeps
an exact-step group.

The checks:

* a row reduced alone and inside batches of mates of other step counts, in its bucket and
  across buckets, has the same bytes, in this process and in a fresh one;
* on CPU, a bucketed row has the bytes of the same row reduced in an exact-step group; on
  every device, its names and their order are the exact-step path's, and it agrees with the
  per-row reduction as the exact-step path does;
* two falsifiers: a reducer that pads each batch to its longest row and ignores the masks
  moves rows with their batchmates, and one that ignores the masks under the bucket rule
  leaves the exact-step bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from anamnesis.config import ExtractionConfig, FeaturePipelineConfig
from anamnesis.extraction.fast import batched_stats, engine_reducer
from anamnesis.extraction.fast.batched_stats import (
    BUCKET_FLOOR,
    BUCKET_RULES,
    HYBRID_LANE_PREFIX,
    bucket_width,
    hybrid_lane_id,
)
from anamnesis.extraction.fast.engine_reducer import EngineReducer
from anamnesis.extraction.fast.features import GpuFeatureLane
from anamnesis.extraction.fast.schema import resolve_gpu_schema

pytestmark = pytest.mark.xdist_group("step-buckets")
"""One worker runs this module (with ``--dist loadgroup``): its batches peak at several GB, and
two of them side by side exceed a CI runner's memory."""

LAYERS, QUERY_HEADS, KV_HEADS, HEAD_DIM, GATE = 6, 4, 2, 2, 12
SAMPLED, TRAJECTORY, PCA = (1, 3, 4), (2, 3), (0, 3)
MAX_START, MAX_STEPS = 12, 70
ROW_WIDTH = MAX_START + MAX_STEPS + 2
"""The toy engine's ``mean_rows`` width: one width for every row, as an engine has."""
ENGINE = {"producer": "synthetic-engine", "width": 128, "weights": "w" * 64}

BUCKETED = (9, 11, 13, 16, 17, 19, 21, 31, 33, 47, 61, 64, 65, 70)
"""Step counts at or above the floor, straddling each toy threshold inside a bucket."""
SHORT = (2, 3, 4, 5, 7)
"""Step counts below eight, which share the first bucket and take their placeholders per row."""
ALONE = (1,)
"""Step counts below the floor, whose schema is their own: exact-step groups."""

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEVICES = [pytest.param("cpu", 8, id="cpu"),
           pytest.param("cuda", 1536, id="cuda", marks=CUDA)]

TOLERANCE = dict(rtol=1e-5, atol=1e-6)
EIGEN = ("spectral_fiedler_", "spectral_hfer_", "spectral_spectral_entropy_")


@pytest.fixture(autouse=True)
def deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


@dataclass
class SignedRow:
    """The engine's signed row, by the attributes the reducer reads."""

    positions: int
    kept: int
    start: int
    lever_slot: int
    steered: bool
    route_sources: tuple[tuple[int, int], ...]
    final_sources: int
    weights: str = ""
    surfaces: dict[str, Any] = field(default_factory=dict)


_LANES: dict[tuple[str, int], Any] = {}


def lanes(device: str, hidden: int):
    """``lane_for(steps)``: one toy lane per step count, sharing one calibration."""
    key = (device, hidden)
    if key in _LANES:
        return _LANES[key]
    g = np.random.default_rng(7)
    extraction = ExtractionConfig(sampled_layers=list(SAMPLED), pca_layers=list(PCA),
                                  pca_components=3, early_layer_cutoff=2, late_layer_cutoff=4,
                                  epoch_window_size=8, epoch_stride=4,
                                  spectral_subsample_step=4)
    families = FeaturePipelineConfig(
        include_core_blocks=True, enable_residual_trajectory=True,
        trajectory_layers=list(TRAJECTORY), enable_attention_flow=True,
        enable_gate_features=True, enable_per_head=True, enable_value_geometry=True,
        enable_qk_geometry=True, enable_kv_cka=True, contrastive_layers=[1])
    components = g.standard_normal((3, hidden)).astype(np.float32)
    pca_mean = g.standard_normal(hidden).astype(np.float32) * 0.1
    pm = g.standard_normal((LAYERS + 1, MAX_START + MAX_STEPS + 8, hidden)).astype(np.float32)
    built: dict[int, GpuFeatureLane] = {}

    def lane_for(steps: int) -> GpuFeatureLane:
        steps = min(int(steps), 8)
        if steps not in built:
            names = resolve_gpu_schema(LAYERS, steps, extraction, families,
                                       components).feature_names
            built[steps] = GpuFeatureLane(extraction, families, list(names), pm, components,
                                          pca_mean, device=device, calibration_sha256="a" * 64)
        return built[steps]

    _LANES[key] = lane_for
    return lane_for


def signed_row(steps: int, start: int, seed: int, hidden: int):
    """A random, well-formed signed row and its token ids."""
    g = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.rand(*shape, generator=g)

    def randn(*shape):
        return torch.randn(*shape, generator=g)

    n_s = len(SAMPLED)
    route_sources = tuple((layer, layer + 1) for layer in range(LAYERS))
    route_width = LAYERS + 2
    route = torch.zeros(steps, LAYERS, 2, route_width)
    for layer, (n_attn, n_mlp) in enumerate(route_sources):
        for point, n in ((0, n_attn), (1, n_mlp)):
            if n:
                w = rand(steps, n) + 1e-3
                route[:, layer, point, :n] = w / w.sum(1, keepdim=True)
    route_final = rand(steps, route_width) + 1e-3
    rows = torch.zeros(steps, n_s, ROW_WIDTH)
    for j in range(steps):
        live = start + j + 1
        w = rand(n_s, live) + 1e-4
        rows[j, :, :live] = w / w.sum(1, keepdim=True)
    logit_stats = torch.stack([rand(steps) * 5, rand(steps), rand(steps),
                               (rand(steps) * 50).floor(), rand(steps), -rand(steps) * 5],
                              dim=1).double()
    surfaces = dict(
        depth=randn(steps, LAYERS, hidden),
        q=randn(steps, n_s, QUERY_HEADS, HEAD_DIM), k=randn(steps, n_s, KV_HEADS, HEAD_DIM),
        v=randn(steps, n_s, KV_HEADS, HEAD_DIM), gate=randn(steps, n_s, GATE),
        head_entropy=rand(steps, LAYERS, QUERY_HEADS) * float(np.log(start + steps + 1)),
        mean_entropy=rand(steps, LAYERS) * float(np.log(start + steps + 1)),
        head_scalars=rand(steps, n_s, 3, QUERY_HEADS),
        mean_rows=rows, logit_stats=logit_stats, route=route,
        route_final=route_final / route_final.sum(1, keepdim=True),
        committed=[randn(steps, 8) for _ in range(4)])
    token_ids = [int(t) for t in torch.randint(3, 1000, (start + steps + 1,), generator=g)]
    row = SignedRow(positions=steps, kept=steps, start=start, lever_slot=0, steered=False,
                    route_sources=route_sources, final_sources=route_width,
                    weights=ENGINE["weights"], surfaces=surfaces)
    return row, token_ids


def reduce(device: str, hidden: int, made, *, buckets: str = "pow2", batched: bool = True):
    reducer = EngineReducer(lane_for=lanes(device, hidden), batched=batched, buckets=buckets,
                            expected_weights=ENGINE["weights"])
    return reducer.reduce([r for r, _ in made], token_ids=[i for _, i in made], engine=ENGINE)


def digests(device: str, hidden: int, steps, seeds, starts, buckets: str = "pow2") -> list[str]:
    """sha256 of each row's features, the rows reduced together."""
    made = [signed_row(n, s, seed, hidden) for n, seed, s in zip(steps, seeds, starts)]
    return [hashlib.sha256(r.features.tobytes()).hexdigest()
            for r in reduce(device, hidden, made, buckets=buckets)]


def test_the_bucket_rules():
    assert [bucket_width(n, "pow2") for n in (1, 2, 7, 8, 9, 16, 17, 129, 256, 257)] == [
        1, 8, 8, 8, 16, 16, 32, 256, 256, 512]
    assert [bucket_width(n, "pow4") for n in (1, 2, 8, 9, 32, 33, 129)] == [1, 8, 8, 32, 32, 128, 512]
    assert [bucket_width(n, "tail256") for n in (1, 5, 8, 200, 256, 300)] == [1, 256, 256, 256, 256, 512]
    assert [bucket_width(n, "exact-steps") for n in (1, 9, 200)] == [1, 9, 200]
    assert set(BUCKET_RULES) == {"exact-steps", "pow2", "pow4", "tail256"}
    with pytest.raises(ValueError):
        bucket_width(9, "pow3")
    with pytest.raises(ValueError):
        EngineReducer(lane_for=lanes("cpu", 8), batched=True, buckets="pow3")


@pytest.mark.parametrize("device,hidden", DEVICES)
def test_a_row_has_the_same_bytes_alone_and_beside_rows_of_other_step_counts(device, hidden):
    rng = np.random.default_rng(11)
    targets = [signed_row(n, int(rng.integers(1, MAX_START + 1)), 2000 + n, hidden)
               for n in (9, 16, 61, 70, 5, 2, 1)]
    alone = [reduce(device, hidden, [t])[0] for t in targets]
    pool = BUCKETED + SHORT + ALONE
    for size in (2, 12):
        for t, target in enumerate(targets):
            n = target[0].positions
            same_bucket = [m for m in pool if bucket_width(m, "pow2") == bucket_width(n, "pow2")]
            for choices in (same_bucket, pool):  # mates in its bucket, then from every bucket
                mates = [signed_row(int(rng.choice(choices)), int(rng.integers(1, MAX_START + 1)),
                                    size * 1000 + t * 50 + m, hidden) for m in range(size - 1)]
                place = int(rng.integers(0, size))
                batch = mates[:place] + [target] + mates[place:]
                got = reduce(device, hidden, batch)[place]
                assert got.features.tobytes() == alone[t].features.tobytes(), (size, n)
                assert got.lane_id == alone[t].lane_id
                assert got.meta["receipt"]["bucket_width"] == bucket_width(n, "pow2")


@pytest.mark.parametrize("device,hidden", DEVICES)
def test_a_bucketed_row_matches_the_exact_step_path(device, hidden):
    """Names, order and placeholders are the exact-step path's; on CPU the bytes are too. On
    a GPU, every coordinate agrees to rounding, and the count that differ is printed."""
    rng = np.random.default_rng(5)
    made = [signed_row(n, int(rng.integers(1, MAX_START + 1)), 300 + n, hidden)
            for n in BUCKETED + SHORT + ALONE]
    bucketed = reduce(device, hidden, made)
    exact = reduce(device, hidden, made, buckets="exact-steps")
    moved = 0
    for b, e, (row, _) in zip(bucketed, exact, made, strict=True):
        assert b.meta["feature_names"] == e.meta["feature_names"]
        assert b.names_digest == e.names_digest
        assert b.lane_id != e.lane_id and b.lane_id.startswith(HYBRID_LANE_PREFIX)
        assert b.meta["receipt"]["batched"]["bucket_rule"] == "pow2"
        assert e.meta["receipt"]["batched"]["bucket_rule"] == "exact-steps"
        assert b.meta["mean_logprob"] == e.meta["mean_logprob"] or device == "cuda"
        if device == "cpu":
            assert b.features.tobytes() == e.features.tobytes(), row.positions
        else:
            np.testing.assert_allclose(b.features, e.features, **TOLERANCE)
            moved += int((b.features != e.features).sum())
    print(json.dumps({"device": device, "coordinates_moved_vs_exact": moved,
                      "rows": len(made)}))


@pytest.mark.parametrize("device,hidden", DEVICES)
def test_a_bucketed_row_agrees_with_the_per_row_reduction(device, hidden):
    rng = np.random.default_rng(6)
    made = [signed_row(n, int(rng.integers(1, MAX_START + 1)), 400 + n, hidden)
            for n in BUCKETED]
    bucketed = reduce(device, hidden, made)
    per_row = reduce(device, hidden, made, batched=False)
    for b, p in zip(bucketed, per_row, strict=True):
        names = p.meta["feature_names"]
        assert b.meta["feature_names"] == names
        eigen = np.array([n.startswith(EIGEN) for n in names])
        eff = np.array([n.endswith("_eff_dim") or "_eff_dim_L" in n for n in names])
        topk = np.array([n.endswith("_topk_overlap_mean") for n in names])
        rest = ~eigen & ~eff & ~topk
        np.testing.assert_allclose(b.features[rest], p.features[rest], **TOLERANCE)
        np.testing.assert_allclose(b.features[eff], p.features[eff], **TOLERANCE)
        np.testing.assert_allclose(b.features[eigen], p.features[eigen], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(b.features[topk], p.features[topk], rtol=1e-6, atol=1e-7)
        epochs = np.array([n.startswith("epoch_") for n in names])
        if b.meta["steps"] < 12:  # no epochs: the placeholders are exact zeros in both
            assert not b.features[epochs].any() and not p.features[epochs].any()
        else:
            assert b.features[epochs].any()


def test_the_bucket_rule_joins_the_lane_id():
    lane = lanes("cpu", 8)(16)
    ids = {rule: hybrid_lane_id(lane, ENGINE, rule) for rule in BUCKET_RULES}
    assert len(set(ids.values())) == len(BUCKET_RULES)
    identities = {EngineReducer(lane_for=lanes("cpu", 8), batched=True, buckets=rule).reducer_identity
                  for rule in BUCKET_RULES}
    assert len(identities) == len(BUCKET_RULES)


_RESTART = """
import json, sys, torch
torch.use_deterministic_algorithms(True)
sys.path.insert(0, {tests!r})
from test_step_buckets import digests
print(json.dumps(digests({device!r}, {hidden}, {steps!r}, {seeds!r}, {starts!r})))
"""
RESTART_BATCH = dict(steps=[9, 16, 21, 47, 70, 5, 2, 1], seeds=[1, 2, 3, 4, 5, 6, 7, 8],
                     starts=[3, 12, 1, 7, 2, 9, 4, 5])


@pytest.mark.parametrize("device,hidden", DEVICES)
def test_two_processes_give_the_same_bytes(device, hidden):
    tests = str(Path(__file__).parent)
    code = _RESTART.format(tests=tests, device=device, hidden=hidden, **RESTART_BATCH)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(Path(__file__).parent.parent), tests, os.environ.get("PYTHONPATH", "")]))
    runs = [json.loads(subprocess.run([sys.executable, "-c", code], env=env, check=True,
                                      capture_output=True, text=True).stdout.strip()
                       .splitlines()[-1]) for _ in range(2)]
    assert runs[0] == runs[1]
    assert runs[0] == digests(device, hidden, **RESTART_BATCH)


def _ignore_masks(monkeypatch) -> None:
    """Every reduction over a row's steps runs over the whole padded width."""
    monkeypatch.setattr(batched_stats.Counts, "lengths", lambda self, x: None)


def _pad_to_the_longest(monkeypatch) -> None:
    """Each batch is padded to its longest row, not to the bucket's canonical width."""
    original = engine_reducer._reduce_uniform

    def padded(lane, rows, spans, engine, width=None, bucket_rule="pow2", lens=None):
        return original(lane, rows, spans, engine, max(int(r.positions) for r in rows),
                        bucket_rule, lens)
    monkeypatch.setattr(engine_reducer, "_reduce_uniform", padded)


def test_the_invariance_check_bites_on_unmasked_padding_to_the_longest_row(monkeypatch):
    """The falsifier: without the masks, a batch padded to its longest row gives a row
    different bytes beside a longer mate than alone."""
    _ignore_masks(monkeypatch)
    _pad_to_the_longest(monkeypatch)
    target = signed_row(13, 4, 77, 8)
    alone = reduce("cpu", 8, [target])[0]
    beside = reduce("cpu", 8, [target, signed_row(16, 2, 78, 8)])[0]
    assert beside.features.tobytes() != alone.features.tobytes()
    moved = int((beside.features != alone.features).sum())
    print(json.dumps({"falsifier": "unmasked-pad-to-longest", "coordinates_moved": moved}))
    assert moved > 50


def test_the_exact_step_check_bites_on_ignored_masks(monkeypatch):
    """The falsifier: with the masks ignored under the bucket rule, a padded row's bytes
    leave the exact-step path's."""
    made = [signed_row(n, 3, 900 + n, 8) for n in (9, 13, 21, 47)]
    exact = reduce("cpu", 8, made, buckets="exact-steps")
    _ignore_masks(monkeypatch)
    bucketed = reduce("cpu", 8, made)
    for b, e in zip(bucketed, exact, strict=True):
        assert b.features.tobytes() != e.features.tobytes()


def _launches(monkeypatch) -> list[tuple]:
    """Every primitive call from here on, as (operation, width, masked by lengths, masked by
    a boolean mask, dtype)."""
    from anamnesis.extraction.fast import invariant_ops

    calls: list[tuple] = []
    original = invariant_ops._reduce

    def recorded(op, x, y, lengths, mask, backend):
        calls.append((op, x.shape[-1], lengths is not None, mask is not None, str(x.dtype)))
        return original(op, x, y, lengths, mask, backend)
    monkeypatch.setattr(invariant_ops, "_reduce", recorded)
    return calls


@pytest.mark.parametrize("rule", ["pow2", "tail256"])
@pytest.mark.parametrize("target", [3, 13, 16, 61, 64], ids=lambda n: f"steps{n}")
def test_a_rows_kernels_do_not_depend_on_its_bucketmates(monkeypatch, target, rule):
    """A row's reductions run at widths, with masks and through branches fixed by its own
    count and its bucket: a GPU kernel's bytes may follow its width, so batchmates of other
    step counts must not change any width a row is reduced at, nor which reductions run.
    Checked as the sequence of primitive calls for the row alone and for the row beside the
    longest and shortest counts of its bucket."""
    calls = _launches(monkeypatch)
    row = signed_row(target, 4, 31, 8)
    reduce("cpu", 8, [row], buckets=rule)
    alone = list(calls)
    width = bucket_width(target, rule)
    # Mates at the shortest and longest counts of the row's bucket (tail256's one bucket spans
    # every count the fixtures build, so its mates are the floor and the longest pooled count).
    bucket = ({BUCKET_FLOOR, max(BUCKETED)} if rule == "tail256"
              else {max(BUCKET_FLOOR, width // 2 + 1), width})
    for mate in sorted(bucket - {target}):
        calls.clear()
        reduce("cpu", 8, [row, signed_row(mate, 2, 32, 8)], buckets=rule)
        assert calls == alone, (target, mate)


def test_a_group_reduced_in_chunks_has_the_same_bytes(monkeypatch):
    """``max_group_rows`` splits a bucket's group into chunks; batchmate invariance means each
    row keeps its bytes and its lane id, and no chunk holds more rows than the cap."""
    made = [signed_row(n, 1 + n % MAX_START, 700 + n, 8) for n in (9, 11, 13, 16, 2, 5, 7, 1)]
    whole = reduce("cpu", 8, made)
    sizes: list[int] = []
    original = engine_reducer._reduce_uniform

    def counted(lane, rows, *a, **kw):
        sizes.append(len(rows))
        return original(lane, rows, *a, **kw)

    monkeypatch.setattr(engine_reducer, "_reduce_uniform", counted)
    reducer = EngineReducer(lane_for=lanes("cpu", 8), batched=True, buckets="pow2",
                            expected_weights=ENGINE["weights"], max_group_rows=2)
    chunked = reducer.reduce([r for r, _ in made], token_ids=[i for _, i in made], engine=ENGINE)
    assert max(sizes) <= 2 and len(sizes) > 2
    for w, c in zip(whole, chunked, strict=True):
        assert c.features.tobytes() == w.features.tobytes()
        assert c.lane_id == w.lane_id
    with pytest.raises(ValueError, match="max_group_rows"):
        EngineReducer(lane_for=lanes("cpu", 8), max_group_rows=0)
