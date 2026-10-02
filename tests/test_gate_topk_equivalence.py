"""The gate's top-k Jaccard, and the source equivalences that keep the fast lane's id.

The gate reducer's host-side top-k runs its SiLU's ufuncs in place and the sampled
layers concurrently. The feature must not move a byte: ``_historical_topk_overlap``
below is the earlier form of the computation, frozen verbatim, and the shipped
function must equal it exactly on surfaces where ties straddle the top-k cut, which
is where NumPy's sort order is the feature.

The reference runs beside the shipped code on the same machine rather than against
pinned bytes: NumPy's ``argsort`` dispatches by CPU instruction set, and on tied rows
AVX-512, AVX2 and scalar builds choose different top-k sets, so a pinned vector would
pin the test machine's CPU rather than the code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from anamnesis.extraction.fast import features as features_mod
from anamnesis.extraction.fast.families import FamilyReducer, gate_topk_overlap
from anamnesis.extraction.fast.ops import FeatureCollector

# The digests the lane identity hashes; every current file canonicalizes to them.
IDENTITY_SOURCES = {
    "features.py": "dd59f0cec50c827de5a70d04930c1ba4000d37624c18f0529b97d6c295ff0815",
    "ops.py": "e6920e84ef2713817071da67d3ef2f8eb4d2427c7cc961ad1c700951738ef536",
    "attention.py": "4ea56d87439a0e8979ea9f1935a9746e30232ab1a899623bc3ed34ff2c031a8c",
    "families.py": "a9d80a60350e52f1c8c158c6b51fba1f4882f9a8f3669c86c032935bc07c5dea",
    "batch_layout.py": "018cdfe51ae55e766af787cdc14e95e286ec20183278f73920b7cf1ae2dd7a07",
}


def _historical_topk_overlap(host: np.ndarray) -> float:
    """``FamilyReducer.gate``'s earlier top-k Jaccard, verbatim."""
    activated = host.astype(np.float64)
    activated = activated * (1 / (1 + np.exp(-np.clip(activated, -88, 88))))
    k = min(100, activated.shape[1] // 10)
    top = np.argsort(np.abs(activated), axis=1)[:, -k:]
    overlaps = []
    for a, b in zip(top[:-1], top[1:], strict=True):
        sa, sb = set(a), set(b)
        overlaps.append(len(sa & sb) / len(sa | sb) if sa | sb else 0.0)
    return float(np.mean(overlaps))


def _tied_surface(seed: int, steps: int = 48, width: int = 1536) -> np.ndarray:
    """A bf16-like gate surface: few distinct magnitudes, so ties straddle the cut."""
    rng = np.random.default_rng(seed)
    x = rng.integers(-40, 41, size=(steps, width)).astype(np.float32) / 8
    return torch.from_numpy(x).bfloat16().float().numpy()


def _boundary_ties(host: np.ndarray) -> int:
    a = host.astype(np.float64)
    a = np.abs(a * (1 / (1 + np.exp(-np.clip(a, -88, 88)))))
    k = min(100, a.shape[1] // 10)
    s = np.sort(a, axis=1)
    return int((s[:, -k] == s[:, -k - 1]).sum())


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_topk_jaccard_is_the_historical_one_on_tied_surfaces(seed):
    host = _tied_surface(seed)
    assert _boundary_ties(host) > len(host) // 2  # the case the sort order decides
    got, want = gate_topk_overlap(host.copy()), _historical_topk_overlap(host.copy())
    assert np.float64(got).tobytes() == np.float64(want).tobytes()


def test_the_reducer_collects_every_layer_s_topk_after_concurrent_work():
    from types import SimpleNamespace

    surfaces = {layer: _tied_surface(10 + layer) for layer in (0, 3, 5)}
    out = FeatureCollector("cpu")
    reducer = FamilyReducer.__new__(FamilyReducer)
    FamilyReducer.__init__(reducer, out, SimpleNamespace(sampled_layers=[]),
                           SimpleNamespace(gate_sparsity_threshold=0.01,
                                           temporal_n_windows=4, enable_stft=False))
    for layer, host in surfaces.items():
        reducer.gate(layer, torch.from_numpy(host))
    assert not any(name.endswith("_topk_overlap_mean") for name in out.values)
    for name, overlap in reducer.gate_topk.items():
        out.put(name, overlap.result())
    for layer, host in surfaces.items():
        value = out.values[f"gate_L{layer}_topk_overlap_mean"]
        assert value.item() == np.float32(_historical_topk_overlap(host)).item()
    assert reducer.gate_tie_exception_d2h_bytes == sum(h.nbytes for h in surfaces.values())


def test_every_lane_source_canonicalizes_to_its_identity_digest():
    root = Path(features_mod.__file__).parent
    current = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
               for name in IDENTITY_SOURCES}
    assert features_mod.canonical_sources(current) == IDENTITY_SOURCES


def test_every_equivalence_names_its_evidence():
    entries = json.loads(features_mod.SOURCE_EQUIVALENCE.read_text())["equivalences"]
    assert entries and all(e["evidence"] and e["evidence"] != "PENDING" and e["authority"]
                           for e in entries)


def test_an_equivalence_without_evidence_or_in_a_cycle_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "eq.json"
    monkeypatch.setattr(features_mod, "SOURCE_EQUIVALENCE", path)
    path.write_text(json.dumps({"equivalences": [
        dict(file="a.py", sha256="1", equivalent_to="0", authority="x", evidence="")]}))
    with pytest.raises(ValueError, match="no evidence"):
        features_mod.canonical_sources({"a.py": "1"})
    path.write_text(json.dumps({"equivalences": [
        dict(file="a.py", sha256="1", equivalent_to="0", authority="x", evidence="x"),
        dict(file="a.py", sha256="0", equivalent_to="1", authority="x", evidence="x")]}))
    with pytest.raises(ValueError, match="cycle"):
        features_mod.canonical_sources({"a.py": "1"})
    path.write_text(json.dumps({"equivalences": [
        dict(file="a.py", sha256="2", equivalent_to="1", authority="x", evidence="x"),
        dict(file="a.py", sha256="1", equivalent_to="0", authority="x", evidence="x")]}))
    assert features_mod.canonical_sources({"a.py": "2", "b.py": "9"}) == {"a.py": "0",
                                                                         "b.py": "9"}
