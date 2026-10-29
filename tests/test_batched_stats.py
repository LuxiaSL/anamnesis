"""The hybrid engine reduction: the block-output family batched, batchmate-invariant.

Signed rows are random but well-formed, in the engine's layout, over a toy lane with the
whole declared battery (sampled, trajectory and PCA layers). A row reduced alone and the
same row inside batches of random mates of the same step count and other prompt lengths
must give the same bytes for the whole vector, in this process and in a fresh one. Against
the per-row reduction, the names are the same, every family that stays per row is byte
for byte the same, and the block-output family agrees to rounding. On a GPU the same
checks run at a width the primitives cut into two tiles.
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
from anamnesis.extraction.fast import invariant_ops
from anamnesis.extraction.fast.batched_stats import (
    BATCHED_SOURCE_SHA256,
    HYBRID_LANE_PREFIX,
    hybrid_lane_id,
)
from anamnesis.extraction.fast.engine_reducer import (
    EngineReducer,
    reduce_signed_rows_hybrid,
)
from anamnesis.extraction.fast.external import ExternalCaptureError
from anamnesis.extraction.fast.external_stats import LANE_PREFIX, stats_lane_id
from anamnesis.extraction.fast.features import GpuFeatureLane
from anamnesis.extraction.fast.schema import resolve_gpu_schema

LAYERS, QUERY_HEADS, KV_HEADS, HEAD_DIM, GATE = 6, 4, 2, 2, 12
SAMPLED, TRAJECTORY, PCA = (1, 3, 4), (2, 3), (0, 3)
MAX_START, MAX_STEPS = 24, 40
ENGINE = {"producer": "synthetic-engine", "width": 128, "weights": "w" * 64}
PORTED = ("activation_norm_", "delta_norm_", "delta_cosine_", "spectral_smoothness_",
          "res_traj_", "pca_")
"""Name prefixes of the block-output family, the features the hybrid computes batched."""

TOLERANCE = dict(rtol=1e-5, atol=1e-6)
"""Hybrid against per-row on the block-output family: float32 rounding of reordered sums."""

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEVICES = [pytest.param("cpu", 8, id="cpu"),
           pytest.param("cpu", 1100, id="cpu-two-tiles", marks=pytest.mark.slow),
           pytest.param("cuda", 1536, id="cuda", marks=CUDA)]


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


def lanes(device: str, hidden: int):
    """``lane_for(steps)``: one toy lane per step count, sharing one calibration."""
    g = np.random.default_rng(7)
    extraction = ExtractionConfig(sampled_layers=list(SAMPLED), pca_layers=list(PCA),
                                  pca_components=3, early_layer_cutoff=2, late_layer_cutoff=4)
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
        if steps not in built:
            names = resolve_gpu_schema(LAYERS, steps, extraction, families,
                                       components).feature_names
            built[steps] = GpuFeatureLane(extraction, families, list(names), pm, components,
                                          pca_mean, device=device, calibration_sha256="a" * 64)
        return built[steps]

    return lane_for


def signed_row(steps: int, start: int, seed: int, hidden: int):
    """A random, well-formed signed row and its token ids."""
    g = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.rand(*shape, generator=g)

    def randn(*shape):
        return torch.randn(*shape, generator=g)

    width = start + steps + 1
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
    rows = torch.zeros(steps, n_s, width)
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
        head_entropy=rand(steps, LAYERS, QUERY_HEADS) * float(np.log(width)),
        mean_entropy=rand(steps, LAYERS) * float(np.log(width)),
        head_scalars=rand(steps, n_s, 3, QUERY_HEADS),
        mean_rows=rows, logit_stats=logit_stats, route=route,
        route_final=route_final / route_final.sum(1, keepdim=True),
        committed=[randn(steps, 8) for _ in range(4)])
    token_ids = [int(t) for t in torch.randint(3, 1000, (start + steps + 1,), generator=g)]
    row = SignedRow(positions=steps, kept=steps, start=start, lever_slot=0, steered=False,
                    route_sources=route_sources, final_sources=route_width,
                    weights=ENGINE["weights"], surfaces=surfaces)
    return row, token_ids


def reduce(device: str, hidden: int, rows, ids, *, batched: bool, lane_for=None):
    reducer = EngineReducer(lane_for=lane_for or lanes(device, hidden), batched=batched,
                            expected_weights=ENGINE["weights"])
    return reducer.reduce(rows, token_ids=ids, engine=ENGINE)


def batch_digests(device: str, hidden: int, steps: int, seeds, starts) -> list[str]:
    """sha256 of each row's features, the rows reduced together through the hybrid path."""
    made = [signed_row(steps, start, seed, hidden) for seed, start in zip(seeds, starts)]
    reduced = reduce(device, hidden, [r for r, _ in made], [i for _, i in made], batched=True)
    return [hashlib.sha256(r.features.tobytes()).hexdigest() for r in reduced]


STEP_COUNTS = [1, 3, 9, 16, 40]


@pytest.mark.parametrize("device,hidden", DEVICES)
@pytest.mark.parametrize("steps", STEP_COUNTS, ids=[f"steps{s}" for s in STEP_COUNTS])
def test_a_row_has_the_same_bytes_alone_and_in_any_batch(device, hidden, steps):
    lane_for = lanes(device, hidden)
    rng = np.random.default_rng(steps)
    targets = [signed_row(steps, int(start), 1000 + i, hidden)
               for i, start in enumerate((1, 7, MAX_START))]
    alone = [reduce(device, hidden, [row], [ids], batched=True, lane_for=lane_for)[0]
             for row, ids in targets]
    for size in (2, 8, 32):
        mates = [signed_row(steps, int(rng.integers(1, MAX_START + 1)), size * 100 + m, hidden)
                 for m in range(size - 1)]
        for t, (row, ids) in enumerate(targets):
            place = int(rng.integers(0, size))
            batch = mates[:place] + [(row, ids)] + mates[place:]
            got = reduce(device, hidden, [r for r, _ in batch], [i for _, i in batch],
                         batched=True, lane_for=lane_for)[place]
            assert got.features.tobytes() == alone[t].features.tobytes(), (size, t)
            assert got.lane_id == alone[t].lane_id
            assert got.meta["receipt"]["batch_size"] == size


@pytest.mark.parametrize("device,hidden", DEVICES)
@pytest.mark.parametrize("steps", STEP_COUNTS, ids=[f"steps{s}" for s in STEP_COUNTS])
def test_the_hybrid_agrees_with_the_per_row_reduction(device, hidden, steps):
    lane_for = lanes(device, hidden)
    made = [signed_row(steps, start, 50 + start, hidden) for start in (1, 5, 12, MAX_START)]
    rows, ids = [r for r, _ in made], [i for _, i in made]
    hybrid = reduce(device, hidden, rows, ids, batched=True, lane_for=lane_for)
    per_row = reduce(device, hidden, rows, ids, batched=False, lane_for=lane_for)
    report: dict[str, tuple[float, float]] = {}
    for h, p in zip(hybrid, per_row, strict=True):
        names = p.meta["feature_names"]
        assert h.meta["feature_names"] == names
        assert h.names_digest == p.names_digest
        ported = np.array([n.startswith(PORTED) for n in names])
        assert ported.sum() > 100
        assert h.features[~ported].tobytes() == p.features[~ported].tobytes()
        np.testing.assert_allclose(h.features[ported], p.features[ported], **TOLERANCE)
        if steps < 3:  # the trajectory's placeholders are exact zeros in both
            traj = np.array([n.startswith("res_traj_") for n in names])
            assert not h.features[traj].any() and not p.features[traj].any()
        for prefix in PORTED:
            sel = np.array([n.startswith(prefix) for n in names])
            a, b = h.features[sel].astype(np.float64), p.features[sel].astype(np.float64)
            diff = np.abs(a - b)
            rel = diff / np.maximum(np.abs(b), 1e-30)
            prev = report.get(prefix, (0.0, 0.0))
            report[prefix] = (max(prev[0], float(diff.max(initial=0))),
                              max(prev[1], float(rel[np.abs(b) > 1e-6].max(initial=0))))
    print(json.dumps({"device": device, "steps": steps,
                      "max_abs_rel": {k: [f"{v[0]:.3g}", f"{v[1]:.3g}"]
                                      for k, v in report.items()}}))


@pytest.mark.parametrize("device,hidden", DEVICES)
def test_the_hybrid_has_its_own_lane_id_and_receipt(device, hidden):
    lane_for = lanes(device, hidden)
    row, ids = signed_row(16, 4, 3, hidden)
    hybrid = reduce(device, hidden, [row], [ids], batched=True, lane_for=lane_for)[0]
    per_row = reduce(device, hidden, [row], [ids], batched=False, lane_for=lane_for)[0]
    lane = lane_for(16)
    receipt = hybrid.meta["receipt"]
    assert hybrid.lane_id.startswith(HYBRID_LANE_PREFIX)
    assert hybrid.lane_id.startswith(LANE_PREFIX)
    assert hybrid.lane_id == hybrid_lane_id(lane, ENGINE) != per_row.lane_id
    assert per_row.lane_id == stats_lane_id(lane, ENGINE)
    assert receipt["stats_lane_id"] == per_row.lane_id
    assert receipt["reductions"]["block_outputs"] == "batched"
    assert receipt["batched"]["families"] == ["block_outputs"]
    assert receipt["batched"]["source_sha256"] == BATCHED_SOURCE_SHA256
    assert receipt["batched"]["row_backend"] == (
        "triton" if device == "cuda" and invariant_ops.HAVE_TRITON else "torch")
    assert "block_outputs" not in per_row.meta["receipt"]["reductions"]
    assert hybrid.meta["reducer_identity"] != per_row.meta["reducer_identity"]


def test_a_bad_row_in_a_batch_is_refused_naming_it():
    lane_for = lanes("cpu", 8)
    row, ids = signed_row(16, 4, 0, 8)
    other, other_ids = signed_row(16, 6, 1, 8)
    reducer = EngineReducer(lane_for=lane_for, batched=True, expected_weights=ENGINE["weights"])

    def refused(match, bad_row, bad_ids=other_ids):
        with pytest.raises(ExternalCaptureError, match=match):
            reducer.reduce([row, bad_row], token_ids=[ids, bad_ids], engine=ENGINE)

    refused(r"row 1: incomplete span", SignedRow(**{**other.__dict__, "kept": 15}))
    refused(r"row 1: the row has", other, other_ids + [7])
    refused(r"row 1: weights digest", SignedRow(**{**other.__dict__, "weights": "x" * 64}))
    surfaces = {k: v for k, v in other.surfaces.items() if k != "mean_rows"}
    refused(r"row 1: the signed row lacks surfaces \['mean_rows'\]",
            SignedRow(**{**other.__dict__, "surfaces": surfaces}))
    bad_depth = dict(other.surfaces, depth=other.surfaces["depth"].clone())
    bad_depth["depth"][3, 2, 1] = float("nan")
    refused(r"row 1: the capture holds a nonfloating or nonfinite tensor",
            SignedRow(**{**other.__dict__, "surfaces": bad_depth}))
    # A refused row changes nothing for the row beside it.
    alone = reducer.reduce([row], token_ids=[ids], engine=ENGINE)[0]
    results = reduce_signed_rows_hybrid(
        lane_for(16), [row, SignedRow(**{**other.__dict__, "surfaces": bad_depth})],
        [ids, other_ids], ENGINE)
    assert isinstance(results[1], ExternalCaptureError)
    assert np.asarray(results[0].features, np.float32).tobytes() == alone.features.tobytes()


def test_rows_of_different_step_counts_reduce_in_their_own_groups():
    lane_for = lanes("cpu", 8)
    made = [signed_row(steps, 3, steps, 8) for steps in (16, 9, 16, 3)]
    together = reduce("cpu", 8, [r for r, _ in made], [i for _, i in made], batched=True,
                      lane_for=lane_for)
    for (row, ids), got in zip(made, together, strict=True):
        alone = reduce("cpu", 8, [row], [ids], batched=True, lane_for=lane_for)[0]
        assert got.meta["steps"] == row.positions
        assert got.features.tobytes() == alone.features.tobytes()


_RESTART = """
import json, sys, torch
torch.use_deterministic_algorithms(True)
sys.path.insert(0, {tests!r})
from test_batched_stats import batch_digests
print(json.dumps(batch_digests({device!r}, {hidden}, 16, [5, 6, 7], [2, 9, 20])))
"""


@pytest.mark.parametrize("device,hidden", DEVICES)
def test_two_processes_give_the_same_bytes(device, hidden):
    tests = str(Path(__file__).parent)
    code = _RESTART.format(tests=tests, device=device, hidden=hidden)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(Path(__file__).parent.parent), tests, os.environ.get("PYTHONPATH", "")]))
    runs = [json.loads(subprocess.run([sys.executable, "-c", code], env=env, check=True,
                                      capture_output=True, text=True).stdout.strip()
                       .splitlines()[-1]) for _ in range(2)]
    assert runs[0] == runs[1]
    assert runs[0] == batch_digests(device, hidden, 16, [5, 6, 7], [2, 9, 20])


def _torch_reductions(monkeypatch):
    """The batched module's primitives replaced by torch's own last-dimension reductions."""
    from anamnesis.extraction.fast import batched_stats

    def mean(x, **_):
        return x.mean(dim=-1)

    def dev(x, **_):
        return x.std(dim=-1, correction=0)

    for name, fn in dict(row_sum=lambda x, **_: x.sum(dim=-1),
                         row_sumsq=lambda x, **_: x.square().sum(dim=-1),
                         row_dot=lambda a, b, **_: (a * b).sum(dim=-1),
                         row_mean=mean, row_std=dev,
                         row_norm=lambda x, **_: x.norm(dim=-1)).items():
        monkeypatch.setattr(batched_stats, name, fn)


@CUDA
def test_the_invariance_check_bites_on_torch_reductions(monkeypatch):
    """The falsifier: with torch's reductions in place of the primitives, some row's bytes
    move with its batchmates on a GPU, so the invariance test can fail."""
    _torch_reductions(monkeypatch)
    lane_for = lanes("cuda", 1536)
    target = signed_row(40, 7, 1000, 1536)
    alone = reduce("cuda", 1536, [target[0]], [target[1]], batched=True, lane_for=lane_for)[0]
    moved = 0
    for size in (8, 32):
        mates = [signed_row(40, 1 + m % MAX_START, size * 100 + m, 1536) for m in range(size - 1)]
        batch = [target] + mates
        got = reduce("cuda", 1536, [r for r, _ in batch], [i for _, i in batch], batched=True,
                     lane_for=lane_for)[0]
        moved += int((got.features != alone.features).sum())
    assert moved > 0
