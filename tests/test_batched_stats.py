"""The batched engine reduction: every family batched, batchmate-invariant.

Signed rows are random but well-formed, in the engine's layout, over a toy lane with the
whole declared battery (sampled, trajectory and PCA layers, epochs, the surprise window).
Rows of one engine share its ``mean_rows`` width, whatever their prompt lengths. A row
reduced alone and the same row inside batches of random mates of the same step count and
other prompt lengths must give the same bytes for the whole vector, in this process and in a
fresh one. Against the per-row reduction, the names are the same and every family agrees to
rounding; three groups are held apart: the effective dimensions (Frobenius norms against
singular values), the eigen-derived spectral features (a batched eigensolve of Laplacians
built by row sums), and the gate top-k Jaccard, whose ties the batched lane breaks by the
lowest index. On a GPU the same checks run at a width the primitives cut into two tiles, and
the batched families' and the attention family's float64 values are checked at tsumugi's
widths for batches up to 128 rows.
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
    ATTENTION_SOURCE_SHA256,
    BATCHED_FAMILIES,
    BATCHED_SOURCE_SHA256,
    FAMILIES_SOURCE_SHA256,
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
ROW_WIDTH = MAX_START + MAX_STEPS + 2
"""The toy engine's ``mean_rows`` width: one width for every row, as an engine has."""
ENGINE = {"producer": "synthetic-engine", "width": 128, "weights": "w" * 64}
EIGEN = ("spectral_fiedler_", "spectral_hfer_", "spectral_spectral_entropy_")
"""Name prefixes of the features read from the spectral Laplacian's eigenvalues."""
ATTENTION = ("attn_entropy_", "head_agreement_", "cache_", "attn_flow_")
"""Name prefixes of the rest of the attention family."""

TOLERANCE = dict(rtol=1e-5, atol=1e-6)
"""Batched against per-row: float32 rounding of reordered sums."""

EFF_DIM_TOLERANCE = dict(rtol=1e-5, atol=1e-6)
"""Batched (Frobenius norms) against per-row (singular values) on the effective dimensions."""

EIGEN_TOLERANCE = dict(rtol=1e-4, atol=1e-5)
"""Batched against per-row on the eigen-derived features: the eigenvalues of Laplacians whose
Gram matrices were summed in another order."""


def name_group(name: str) -> str:
    """``attention``, ``eigen``, ``gate_topk``, ``eff_dim`` (the key, value and query effective
    dimensions) or the name's family prefix."""
    if name.startswith(EIGEN):
        return "eigen"
    if name.startswith(ATTENTION):
        return "attention"
    if name.endswith("_topk_overlap_mean"):
        return "gate_topk"
    if name.endswith("_eff_dim") or "_eff_dim_L" in name:
        return "eff_dim"
    for prefix in ("activation_norm_", "delta_norm_", "delta_cosine_", "spectral_smoothness_",
                   "res_traj_", "pca_", "kv_key_cka", "kv_value_cka", "kv_key_", "ph_",
                   "value_", "qk_", "q_crosslayer", "gate_", "epoch_", "attnres_",
                   "surprise_boundary"):
        if name.startswith(prefix):
            return prefix
    return "outputs"

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
                                  pca_components=3, early_layer_cutoff=2, late_layer_cutoff=4,
                                  epoch_window_size=8, epoch_stride=4)
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

    width = ROW_WIDTH
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


def agreement(hybrid, per_row) -> dict[str, list[float]]:
    """Per name group, the largest absolute and relative difference between the hybrid's
    and the per-row reduction's features; asserts the names agree."""
    report: dict[str, list[float]] = {}
    for h, p in zip(hybrid, per_row, strict=True):
        names = p.meta["feature_names"]
        assert h.meta["feature_names"] == names
        assert h.names_digest == p.names_digest
        groups = np.array([name_group(n) for n in names])
        for group in set(groups):
            sel = groups == group
            a, b = h.features[sel].astype(np.float64), p.features[sel].astype(np.float64)
            diff = np.abs(a - b)
            rel = diff / np.maximum(np.abs(b), 1e-30)
            acc = report.setdefault(group, [0.0, 0.0])
            acc[0] = max(acc[0], float(diff.max(initial=0)))
            acc[1] = max(acc[1], float(rel[np.abs(b) > 1e-6].max(initial=0)))
    return report


@pytest.mark.parametrize("device,hidden", DEVICES)
@pytest.mark.parametrize("steps", STEP_COUNTS, ids=[f"steps{s}" for s in STEP_COUNTS])
def test_the_hybrid_agrees_with_the_per_row_reduction(device, hidden, steps):
    lane_for = lanes(device, hidden)
    made = [signed_row(steps, start, 50 + start, hidden) for start in (1, 5, 12, MAX_START)]
    rows, ids = [r for r, _ in made], [i for _, i in made]
    hybrid = reduce(device, hidden, rows, ids, batched=True, lane_for=lane_for)
    per_row = reduce(device, hidden, rows, ids, batched=False, lane_for=lane_for)
    for h, p in zip(hybrid, per_row, strict=True):
        names = p.meta["feature_names"]
        groups = np.array([name_group(n) for n in names])
        eff, eigen, topk = groups == "eff_dim", groups == "eigen", groups == "gate_topk"
        rest = ~eff & ~eigen & ~topk
        assert (groups == "attention").sum() > 10 and rest.sum() > 100 and eigen.sum() == 9
        np.testing.assert_allclose(h.features[rest], p.features[rest], **TOLERANCE)
        np.testing.assert_allclose(h.features[eff], p.features[eff], **EFF_DIM_TOLERANCE)
        np.testing.assert_allclose(h.features[eigen], p.features[eigen], **EIGEN_TOLERANCE)
        # The toy gates are continuous: no ties, so the two top-k definitions take one set.
        np.testing.assert_allclose(h.features[topk], p.features[topk], rtol=1e-6, atol=1e-7)
        assert abs(h.meta["mean_logprob"] - p.meta["mean_logprob"]) < 1e-12
        if steps < 3:  # the trajectory's placeholders are exact zeros in both
            traj = np.array([n.startswith("res_traj_") for n in names])
            assert not h.features[traj].any() and not p.features[traj].any()
        if steps < 2:  # so are the attention flow's
            flow = np.array([n.startswith("attn_flow_") for n in names])
            assert not h.features[flow].any() and not p.features[flow].any()
        if steps >= 16:  # the toy lane's epochs (window 8, stride 4) are live
            assert any(n.startswith("epoch_") for n in names)
    report = agreement(hybrid, per_row)
    print(json.dumps({"device": device, "steps": steps,
                      "max_abs_rel": {k: [f"{v[0]:.3g}", f"{v[1]:.3g}"]
                                      for k, v in sorted(report.items())}}))


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
    for family in BATCHED_FAMILIES:
        assert receipt["reductions"][family] == "batched"
    assert receipt["reductions"]["attention"] == "batched"
    assert per_row.meta["receipt"]["reductions"]["attention"] == "stats"
    assert receipt["batched"]["families"] == list(BATCHED_FAMILIES)
    assert {"attention", "keys", "routing"} <= set(BATCHED_FAMILIES)
    assert receipt["batched"]["source_sha256"] == BATCHED_SOURCE_SHA256
    assert receipt["batched"]["families_source_sha256"] == FAMILIES_SOURCE_SHA256
    assert receipt["batched"]["attention_source_sha256"] == ATTENTION_SOURCE_SHA256
    assert receipt["batched"]["eff_dim"] == "norms"
    assert receipt["batched"]["topk_ties"] == "lowest-index"
    assert receipt["batched"]["eigvalsh"] == ("batched" if device == "cuda" else "per-matrix")
    assert receipt["batched"]["per_row"] == []
    for field_name in ("attention_layers_reduced", "retained_attention_summary_bytes",
                       "feature_schema_sha256", "input_tokens_sha256", "span_start", "span_end"):
        assert receipt[field_name] == per_row.meta["receipt"][field_name], field_name
    assert receipt["batched"]["row_backend"] == (
        "triton" if device == "cuda" and invariant_ops.HAVE_TRITON else "torch")
    assert receipt["routing_family"] and len(hybrid.features) == len(per_row.features)
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


def block_inputs(lane, steps: int, seeds, hidden: int):
    """``block_output_features``'s inputs for rows of the given seeds: block outputs, starts
    and, per sampled layer, a random symmetric Laplacian over the spectral steps."""
    stride = lane.config.spectral_subsample_step
    spectral = list(range(0, steps, stride))
    if len(spectral) < 3:
        spectral = list(range(steps))
    depth, starts, laps = [], [], []
    for seed in seeds:
        g = torch.Generator().manual_seed(seed)
        depth.append(torch.randn(steps, LAYERS, hidden, generator=g))
        starts.append(1 + seed % MAX_START)
        a = torch.randn(len(SAMPLED), len(spectral), len(spectral), generator=g, dtype=torch.float64)
        laps.append(a + a.transpose(1, 2))
    stacked = torch.stack(laps)
    laplacians = {layer: (spectral, stacked[:, j].to(lane.device))
                  for j, layer in enumerate(SAMPLED)}
    return torch.stack(depth).to(lane.device), starts, laplacians


def moved_rows(lane, steps: int, hidden: int, sizes=(2, 8, 32, 128)) -> int:
    """Rows of the batch whose float64 block-output values differ from the row alone."""
    from anamnesis.extraction.fast.batched_stats import block_output_features

    moved = 0
    seeds = list(range(max(sizes)))
    depth, starts, laps = block_inputs(lane, steps, seeds, hidden)
    alone = [block_output_features(lane, depth[b:b + 1], starts[b:b + 1],
                                   {k: (s, v[b:b + 1]) for k, (s, v) in laps.items()}).values[0]
             for b in seeds]
    for size in sizes:
        together = block_output_features(lane, depth[:size], starts[:size],
                                         {k: (s, v[:size]) for k, (s, v) in laps.items()}).values
        moved += sum(int(not torch.equal(together[b], alone[b])) for b in range(size))
    return moved


@pytest.mark.parametrize("device,hidden", DEVICES)
@pytest.mark.parametrize("steps", [3, 40], ids=["steps3", "steps40"])
def test_the_block_outputs_are_batchmate_invariant_in_float64(device, hidden, steps):
    """Before the collector's float32 cast: every row's float64 values, alone and in
    batches of 2, 8, 32 and 128."""
    assert moved_rows(lanes(device, hidden)(steps), steps, hidden) == 0


def _torch_reductions(monkeypatch):
    """The batched module's primitives replaced by torch's own last-dimension reductions,
    masks honoured."""
    from anamnesis.extraction.fast import batched_stats

    def kept(x, lengths=None, mask=None):
        if lengths is not None:
            mask = torch.arange(x.shape[-1], device=x.device) < lengths[..., None]
        return (x, None) if mask is None else (torch.where(mask, x, x.new_zeros(())), mask)

    def count(x, mask):
        return (x.new_full(x.shape[:-1], x.shape[-1]) if mask is None
                else mask.sum(dim=-1).clamp_min(1).to(x.dtype))

    def mean(x, lengths=None, mask=None, **_):
        x, mask = kept(x, lengths, mask)
        return x.sum(dim=-1) / count(x, mask)

    def dev(x, lengths=None, mask=None, **_):
        x, keep = kept(x, lengths, mask)
        d = x - mean(x, mask=keep)[..., None]
        d = d if keep is None else torch.where(keep, d, d.new_zeros(()))
        return (d.square().sum(dim=-1) / count(x, keep)).sqrt()

    for name, fn in dict(row_sum=lambda x, lengths=None, mask=None, **_: kept(x, lengths, mask)[0].sum(dim=-1),
                         row_sumsq=lambda x, lengths=None, mask=None, **_: kept(x, lengths, mask)[0].square().sum(dim=-1),
                         row_dot=lambda a, b, lengths=None, mask=None, **_: kept(a * b, lengths, mask)[0].sum(dim=-1),
                         row_mean=mean, row_std=dev,
                         row_norm=lambda x, lengths=None, mask=None, **_: kept(x, lengths, mask)[0].norm(dim=-1)).items():
        monkeypatch.setattr(batched_stats, name, fn)


@CUDA
@pytest.mark.parametrize("steps", [3, 40], ids=["steps3", "steps40"])
def test_the_invariance_check_bites_on_torch_reductions(monkeypatch, steps):
    """The falsifier: with torch's reductions in place of the primitives, some row's float64
    values move with its batchmates on a GPU, so the invariance check can fail."""
    _torch_reductions(monkeypatch)
    lane = lanes("cuda", 1536)(steps)
    assert moved_rows(lane, steps, 1536) > 0


def test_the_device_gather_is_the_host_slice_byte_for_byte():
    """Each row's positional means, gathered from the device copy of the lane's table, are
    the host table's own slice: the gather copies, it does not compute."""
    from anamnesis.extraction.fast.batched_stats import device_positional_means

    rng = np.random.default_rng(0)
    pm = rng.standard_normal((9, 40, 6)).astype(np.float32)
    starts, steps, first, last = [3, 11, 0, 20], 7, 2, 6
    table = device_positional_means(pm, "cpu")
    positions = torch.as_tensor([[s + t for t in range(steps)] for s in starts])
    gathered = table[1 + first:1 + last][:, positions].permute(1, 2, 0, 3)
    host = np.stack([np.ascontiguousarray(pm[1 + first:1 + last, s:s + steps].transpose(1, 0, 2))
                     for s in starts])
    assert gathered.contiguous().numpy().tobytes() == host.tobytes()
    assert device_positional_means(pm, "cpu") is table   # one upload per host table


TSUMUGI_WIDTHS = dict(query_heads=12, kv_heads=6, head_dim=128, gate=4096, committed=1536)
"""Tsumugi's per-layer widths: the batched families' float64 values are checked at these on a GPU."""

TOY_WIDTHS = dict(query_heads=QUERY_HEADS, kv_heads=KV_HEADS, head_dim=HEAD_DIM, gate=GATE,
                  committed=8)


def family_inputs(steps: int, seeds, widths: dict[str, int], device: str) -> dict[str, Any]:
    """The batched families' inputs for rows of the given seeds, stacked on ``device``."""
    n_s = len(SAMPLED)
    rows = []
    for seed in seeds:
        g = torch.Generator().manual_seed(seed)
        heads = widths["query_heads"]
        route = torch.rand(steps, LAYERS, 2, LAYERS + 2, generator=g) + 1e-3
        rows.append(dict(
            k=torch.randn(steps, n_s, widths["kv_heads"], widths["head_dim"], generator=g),
            v=torch.randn(steps, n_s, widths["kv_heads"], widths["head_dim"], generator=g),
            q=torch.randn(steps, n_s, heads, widths["head_dim"], generator=g),
            gate=torch.randn(steps, n_s, widths["gate"], generator=g),
            ent=torch.rand(steps, n_s, heads, generator=g, dtype=torch.float64),
            sink=torch.rand(steps, n_s, heads, generator=g, dtype=torch.float64),
            logit_stats=torch.stack([torch.rand(steps, generator=g) * 5,
                                     torch.rand(steps, generator=g),
                                     torch.rand(steps, generator=g),
                                     (torch.rand(steps, generator=g) * 50).floor(),
                                     torch.rand(steps, generator=g),
                                     -torch.rand(steps, generator=g) * 5], dim=1).double(),
            route=route / route.sum(-1, keepdim=True),
            committed=torch.randn(4, steps, widths["committed"], generator=g)))
    return {name: torch.stack([r[name] for r in rows]).to(device) for name in rows[0]}


def family_values(lane, inputs: dict[str, Any], rows: slice) -> torch.Tensor:
    """The key/value/query/gate, per-head, cross-layer, output and routing families of the
    selected rows, as one float64 ``[rows, features]``."""
    from anamnesis.extraction.fast.batched_families import (
        kvqg_features,
        outputs_features,
        routing_features,
    )

    x = {name: value[rows] for name, value in inputs.items()}
    per_head = {layer: (x["ent"][:, :, j], x["sink"][:, :, j]) for j, layer in enumerate(SAMPLED)}
    kvqg = kvqg_features(lane, x["k"], x["v"], x["q"], x["gate"], per_head)
    outputs, logprob = outputs_features(lane, x["logit_stats"])
    points = []
    for layer in range(LAYERS):
        if layer:
            points.append(("pre_attn", layer, x["route"][:, :, layer, 0, :layer]))
        points.append(("pre_mlp", layer, x["route"][:, :, layer, 1, :layer + 1]))
    points.append(("final", -1, x["route"][:, :, -1, 1]))
    _, routing = routing_features(SAMPLED, points, list(x["committed"].unbind(1)))
    return torch.cat([kvqg.values, outputs.values, logprob[:, None], routing.double()], dim=1)


def moved_family_rows(lane, steps: int, widths, device: str, sizes=(2, 8, 32, 128)) -> int:
    """Rows of the batch whose float64 batched-family values differ from the row alone."""
    inputs = family_inputs(steps, range(max(sizes)), widths, device)
    alone = [family_values(lane, inputs, slice(b, b + 1))[0] for b in range(max(sizes))]
    moved = 0
    for size in sizes:
        together = family_values(lane, inputs, slice(0, size))
        moved += sum(int(not torch.equal(together[b], alone[b])) for b in range(size))
    return moved


@pytest.mark.parametrize("steps", [1, 3, 40], ids=["steps1", "steps3", "steps40"])
def test_the_batched_families_are_batchmate_invariant_in_float64(steps):
    """Before the collector's float32 cast: every row's float64 values of the key, value,
    query, gate, per-head, cross-layer, output and routing families, alone and in batches
    of 2, 8, 32 and 128."""
    assert moved_family_rows(lanes("cpu", 8)(steps), steps, TOY_WIDTHS, "cpu") == 0


@CUDA
@pytest.mark.slow
@pytest.mark.parametrize("steps", [3, 23, 40], ids=["steps3", "steps23", "steps40"])
def test_the_batched_families_are_batchmate_invariant_at_tsumugi_widths(steps):
    assert moved_family_rows(lanes("cuda", 64)(steps), steps, TSUMUGI_WIDTHS, "cuda") == 0


def _torch_family_reductions(monkeypatch):
    """Both batched modules' primitives replaced by torch's own last-dimension reductions."""
    from anamnesis.extraction.fast import batched_families

    _torch_reductions(monkeypatch)
    from anamnesis.extraction.fast import batched_stats

    for name in ("row_sum", "row_sumsq", "row_dot", "row_mean", "row_std", "row_norm"):
        monkeypatch.setattr(batched_families, name, getattr(batched_stats, name))


@CUDA
@pytest.mark.slow
def test_the_family_invariance_check_bites_on_torch_reductions(monkeypatch):
    """The falsifier: with torch's reductions in place of the primitives, some row's float64
    family values move with its batchmates on a GPU at tsumugi's widths."""
    _torch_family_reductions(monkeypatch)
    lane = lanes("cuda", 64)(23)
    assert moved_family_rows(lane, 23, TSUMUGI_WIDTHS, "cuda") > 0


def test_rows_of_one_step_count_and_different_shapes_reduce_in_their_own_batches():
    """A row whose routing plan differs from its step-count mates is batched apart from
    them; every row keeps its bytes alone."""
    lane_for = lanes("cpu", 8)
    made = [signed_row(16, 3 + i, 20 + i, 8) for i in range(4)]
    odd, odd_ids = made[1]
    plan = tuple((n_attn, max(1, n_mlp - 1)) for n_attn, n_mlp in odd.route_sources)
    made[1] = (SignedRow(**{**odd.__dict__, "route_sources": plan}), odd_ids)
    together = reduce("cpu", 8, [r for r, _ in made], [i for _, i in made], batched=True,
                      lane_for=lane_for)
    sizes = [r.meta["receipt"]["batch_size"] for r in together]
    assert sizes == [3, 1, 3, 3]
    for (row, ids), got in zip(made, together, strict=True):
        alone = reduce("cpu", 8, [row], [ids], batched=True, lane_for=lane_for)[0]
        assert got.features.tobytes() == alone.features.tobytes()


@pytest.mark.parametrize("surface,match", [
    ("mean_rows", r"row 1: mean_rows holds a nonfinite value"),
    ("head_scalars", r"row 1: head_recency_mass holds a nonfinite value"),
    ("head_entropy", r"row 1: head_entropy holds a nonfinite value"),
    ("logit_stats", r"row 1: logit_stats must be finite"),
    ("route", r"row 1: the routing weights hold a nonfinite value"),
    ("gate", r"row 1: the capture holds a nonfloating or nonfinite tensor"),
])
def test_a_nonfinite_surface_refuses_only_its_row(surface, match):
    lane_for = lanes("cpu", 8)
    row, ids = signed_row(16, 4, 0, 8)
    other, other_ids = signed_row(16, 6, 1, 8)
    bad = dict(other.surfaces, **{surface: other.surfaces[surface].clone()})
    bad[surface].view(-1)[5] = float("nan")
    bad_row = SignedRow(**{**other.__dict__, "surfaces": bad})
    reducer = EngineReducer(lane_for=lane_for, batched=True, expected_weights=ENGINE["weights"])
    with pytest.raises(ExternalCaptureError, match=match):
        reducer.reduce([row, bad_row], token_ids=[ids, other_ids], engine=ENGINE)
    results = reduce_signed_rows_hybrid(lane_for(16), [row, bad_row], [ids, other_ids], ENGINE)
    alone = reducer.reduce([row], token_ids=[ids], engine=ENGINE)[0]
    assert isinstance(results[1], ExternalCaptureError)
    assert results[0].metadata["batch_size"] == 1
    assert np.asarray(results[0].features, np.float32).tobytes() == alone.features.tobytes()


@pytest.mark.parametrize("surface,match", [
    ("mean_rows", r"row 1: mean_rows holds a nonfinite value"),
    ("head_scalars", r"row 1: head_recency_mass holds a nonfinite value"),
])
def test_the_per_row_reduction_words_an_attention_refusal_the_same(surface, match):
    """The batched refusal of a nonfinite attention surface is the per-row reduction's."""
    lane_for = lanes("cpu", 8)
    row, ids = signed_row(16, 4, 0, 8)
    other, other_ids = signed_row(16, 6, 1, 8)
    bad = dict(other.surfaces, **{surface: other.surfaces[surface].clone()})
    bad[surface].view(-1)[5] = float("nan")
    bad_row = SignedRow(**{**other.__dict__, "surfaces": bad})
    reducer = EngineReducer(lane_for=lane_for, expected_weights=ENGINE["weights"])
    with pytest.raises(ExternalCaptureError, match=match):
        reducer.reduce([row, bad_row], token_ids=[ids, other_ids], engine=ENGINE)


TSUMUGI_ATTENTION = dict(layers=72, heads=12, sampled=10, width=1048)
"""Tsumugi's attention statistics: layers, query heads, sampled layers, and an engine's
``mean_rows`` width that covers a 1024-token prompt and its 24-token completion."""

TOY_ATTENTION = dict(layers=LAYERS, heads=QUERY_HEADS, sampled=len(SAMPLED), width=ROW_WIDTH)


def attention_inputs(steps: int, seeds, shapes: dict[str, int], device: str,
                     max_start: int) -> tuple[dict[str, torch.Tensor], list[int]]:
    """The attention family's stacked inputs for rows of the given seeds, of prompt lengths
    spread over ``[1, max_start]``, and those prompt lengths."""
    layers, heads, n_s, width = (shapes[k] for k in ("layers", "heads", "sampled", "width"))
    out: dict[str, list[torch.Tensor]] = {k: [] for k in ("head_entropy", "mean_entropy",
                                                          "head_scalars", "mean_rows")}
    starts = []
    for seed in seeds:
        g = torch.Generator().manual_seed(seed)
        start = 1 + (seed * 37) % max_start
        starts.append(start)
        rows = torch.zeros(steps, n_s, width)
        for j in range(steps):
            live = start + j + 1
            w = torch.rand(n_s, live, generator=g) + 1e-4
            w = w * (torch.rand(n_s, live, generator=g) < 0.7)  # some exact zeros
            rows[j, :, :live] = w / w.sum(1, keepdim=True).clamp_min(1e-12)
        out["head_entropy"].append(torch.rand(steps, layers, heads, generator=g) * 6)
        out["mean_entropy"].append(torch.rand(steps, layers, generator=g) * 6)
        out["head_scalars"].append(torch.rand(steps, n_s, 3, heads, generator=g))
        out["mean_rows"].append(rows)
    return {k: torch.stack(v).to(device) for k, v in out.items()}, starts


def tsumugi_like_lane(device: str, layers: int, sampled: int):
    """What the attention family reads of a lane, with tsumugi's battery: ``layers`` layers,
    ``sampled`` of them spread over the depth."""
    from types import SimpleNamespace

    picked = sorted({round(i * (layers - 1) / (sampled - 1)) for i in range(sampled)})
    return SimpleNamespace(config=ExtractionConfig(sampled_layers=picked),
                           families=FeaturePipelineConfig(enable_attention_flow=True),
                           device=torch.device(device))


def attention_values(lane, inputs, starts, rows: slice) -> torch.Tensor:
    """The attention family of the selected rows, with each sampled layer's normalised
    Laplacian and per-head summaries, as one float64 ``[rows, ...]``."""
    from anamnesis.extraction.fast.batched_attention import attention_features

    x = {k: v[rows] for k, v in inputs.items()}
    got = attention_features(lane, x["head_entropy"], x["mean_entropy"], x["head_scalars"],
                             x["mean_rows"], starts[rows])
    batch = got.values.shape[0]
    parts = [got.values]
    for layer in sorted(got.laplacians):
        parts.append(got.laplacians[layer][1].reshape(batch, -1))
        ent, sink = got.per_head[layer]
        parts += [ent.reshape(batch, -1), sink.reshape(batch, -1)]
    return torch.cat(parts, dim=1)


def moved_attention_rows(lane, steps: int, shapes, device: str, max_start: int,
                         sizes=(2, 8, 32, 128)) -> int:
    """Rows of the batch whose float64 attention values differ from the row alone."""
    inputs, starts = attention_inputs(steps, range(max(sizes)), shapes, device, max_start)
    alone = [attention_values(lane, inputs, starts, slice(b, b + 1))[0] for b in range(max(sizes))]
    moved = 0
    for size in sizes:
        together = attention_values(lane, inputs, starts, slice(0, size))
        moved += sum(int(not torch.equal(together[b], alone[b])) for b in range(size))
    return moved


@pytest.mark.parametrize("steps", [1, 3, 12, 40], ids=["steps1", "steps3", "steps12", "steps40"])
def test_the_attention_family_is_batchmate_invariant_in_float64(steps):
    """Every row's float64 attention values, its Laplacians and per-head summaries, alone and
    in batches of 2, 8, 32 and 128 rows of other prompt lengths."""
    lane = lanes("cpu", 8)(steps)
    assert moved_attention_rows(lane, steps, TOY_ATTENTION, "cpu", MAX_START) == 0


@CUDA
@pytest.mark.slow
@pytest.mark.parametrize("steps", [23, 127], ids=["steps23", "steps127"])
def test_the_attention_family_is_batchmate_invariant_at_tsumugi_widths(steps):
    shapes = dict(TSUMUGI_ATTENTION, width=1024 + steps + 1)
    lane = tsumugi_like_lane("cuda", shapes["layers"], shapes["sampled"])
    assert moved_attention_rows(lane, steps, shapes, "cuda", 1024) == 0


def _torch_attention_reductions(monkeypatch):
    """The attention module's primitives replaced by torch's own last-dimension reductions."""
    from anamnesis.extraction.fast import batched_attention, batched_stats

    _torch_reductions(monkeypatch)
    for name in ("row_sum", "row_mean", "row_std", "row_norm"):
        monkeypatch.setattr(batched_attention, name, getattr(batched_stats, name))


@CUDA
@pytest.mark.slow
def test_the_attention_invariance_check_bites_on_torch_reductions(monkeypatch):
    """The falsifier: with torch's reductions in place of the primitives, some row's float64
    attention values move with its batchmates on a GPU at tsumugi's widths."""
    _torch_attention_reductions(monkeypatch)
    shapes = TSUMUGI_ATTENTION
    lane = tsumugi_like_lane("cuda", shapes["layers"], shapes["sampled"])
    assert moved_attention_rows(lane, 23, shapes, "cuda", 1024) > 0


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("k", [1, 3, 13, 33, 52])
def test_a_laplacians_eigenvalues_are_the_same_alone_and_in_a_batch(device, k):
    """The eigensolve the attention family uses treats each matrix independently: a
    Laplacian's eigenvalues have the same bytes alone and in any batch."""
    from anamnesis.extraction.fast.batched_attention import eigvalsh_rows

    g = torch.Generator().manual_seed(k)
    x = torch.rand(300, k, 64, generator=g, dtype=torch.float64)
    x = x / x.norm(dim=-1, keepdim=True)
    graph = x @ x.transpose(1, 2)
    graph.diagonal(dim1=1, dim2=2).zero_()
    lap = (torch.diag_embed(graph.sum(-1)) - graph + torch.eye(k, dtype=torch.float64) * 1e-10).to(device)
    alone = torch.cat([eigvalsh_rows(lap[i:i + 1]) for i in range(8)])
    for size in (2, 8, 37, 300):
        together = eigvalsh_rows(lap[:size])
        assert torch.equal(together[: min(size, 8)], alone[: min(size, 8)]), size
    assert torch.equal(eigvalsh_rows(torch.cat([lap[100:], lap[:8]]))[-8:], alone)
    np.testing.assert_allclose(alone.cpu().numpy(),
                               np.linalg.eigvalsh(lap[:8].cpu().numpy()), rtol=0, atol=1e-12)


def stable_topk_reference(pre_silu: np.ndarray) -> float:
    """The gate top-k Jaccard of one ``[steps, width]`` surface with the lane's tie rule, in
    NumPy: the per-row lane's SiLU, then a stable sort descending by value, so ties at the
    k-th value go to the lowest column index."""
    a = pre_silu.astype(np.float64)
    a = np.abs(a * (1 / (1 + np.exp(-np.clip(a, -88, 88)))))
    k = min(100, a.shape[1] // 10)
    top = np.argsort(-a, axis=1, kind="stable")[:, :k] if k else np.tile(np.arange(a.shape[1]),
                                                                         (len(a), 1))
    member = np.zeros(a.shape, dtype=bool)
    np.put_along_axis(member, top, True, axis=1)
    shared = np.count_nonzero(member[:-1] & member[1:], axis=1)
    union = 2 * top.shape[1] - shared
    return float(np.mean(np.where(union > 0, shared / np.maximum(union, 1), 0.0)))


def tied_gates(batch: int, steps: int, width: int, seed: int) -> torch.Tensor:
    """Pre-SiLU gates rounded to bf16 and drawn from few levels, so the k-th value ties."""
    g = torch.Generator().manual_seed(seed)
    levels = torch.randn(64, generator=g)
    x = levels[torch.randint(0, 64, (batch, steps, width), generator=g)]
    return x.to(torch.bfloat16).float()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("width", [9, 12, 4096])
def test_the_gate_topk_breaks_ties_by_the_lowest_index(device, width):
    """The batched gate top-k is the stable, lowest-index-first top-k: on gates that tie at
    the k-th value it matches the NumPy reference to rounding of the mean, and every row has
    the same bytes alone and in a batch."""
    from anamnesis.extraction.fast.batched_families import gate_topk_overlap_rows

    gates = tied_gates(32, 9, width, width)
    got = gate_topk_overlap_rows(gates.to(device)).cpu()
    expected = torch.tensor([stable_topk_reference(gates[b].numpy()) for b in range(32)],
                            dtype=torch.float64)
    np.testing.assert_allclose(got.numpy(), expected.numpy(), rtol=1e-12, atol=0)
    alone = torch.cat([gate_topk_overlap_rows(gates[b:b + 1].to(device)).cpu() for b in range(32)])
    assert torch.equal(got, alone)


def test_the_tie_rule_is_where_the_gate_topk_leaves_the_per_row_lane():
    """On tied gates the per-row lane (NumPy's argsort) and the batched lane (lowest index
    first) can take different sets; without ties they agree. The count is printed: it is
    what the tie rule costs against the per-row lane."""
    from anamnesis.extraction.fast.batched_families import gate_topk_overlap_rows
    from anamnesis.extraction.fast.families import gate_topk_overlap

    gates = tied_gates(16, 9, 4096, 3)
    batched = gate_topk_overlap_rows(gates).numpy()
    per_row = np.array([gate_topk_overlap(gates[b].numpy()) for b in range(16)])
    moved = np.abs(batched - per_row)
    print(json.dumps({"tied_rows": 16, "moved": int((moved > 1e-12).sum()),
                      "max_abs": float(moved.max())}))
    continuous = torch.randn(16, 9, 4096, generator=torch.Generator().manual_seed(4))
    np.testing.assert_allclose(gate_topk_overlap_rows(continuous).numpy(),
                               [gate_topk_overlap(continuous[b].numpy()) for b in range(16)],
                               rtol=1e-12, atol=0)
