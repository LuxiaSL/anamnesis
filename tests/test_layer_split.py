"""A model held as a layer pipeline: the split is explicit, validated and part of identity.

A checkpoint too large for one GPU runs each decoder layer on one GPU by placing
contiguous layer ranges on successive devices. These tests hold the split's contract
on a CPU: it covers the model exactly once, it refuses what a load could move
(``"auto"``, gaps, overlaps), its digest names it, a model is checked against it
parameter by parameter, the arithmetic guard reads every device a model occupies, and
a one-device split is the one-device lane, byte for byte and in identity.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_fast_lane_equivalence import tiny_loaded
from anamnesis.config import ExtractionConfig, FeaturePipelineConfig
from anamnesis.extraction.fast.features import GpuFeatureLane
from anamnesis.extraction.fast.runtime import WORKSPACE_ENV
from anamnesis.extraction.fast.schema import resolve_gpu_schema
from anamnesis.extraction.layer_split import LayerSplit, model_devices

EIGHT = tuple(f"cuda:{i}" for i in range(8))


def test_a_balanced_split_covers_every_layer_once_in_order():
    split = LayerSplit.balanced(126, EIGHT)
    assert split.layer_ranges[0] == (0, 16) and split.layer_ranges[-1] == (111, 126)
    assert [end - start for start, end in split.layer_ranges] == [16] * 6 + [15] * 2
    assert [split.layer_device(i) for i in (0, 15, 16, 125)] == \
        ["cuda:0", "cuda:0", "cuda:1", "cuda:7"]
    assert LayerSplit.balanced(126, EIGHT) == split


def test_boundaries_name_where_each_later_device_starts():
    split = LayerSplit.from_boundaries(32, ("cuda:0", "cuda:1"), (10,))
    assert split.layer_ranges == ((0, 10), (10, 32))
    assert split.first == "cuda:0" and split.last == "cuda:1"


@pytest.mark.parametrize("devices,ranges,message", [
    (("auto",), ((0, 4),), "explicit"),
    (("cuda",), ((0, 4),), "explicit"),
    (("cuda:0", "cuda:0"), ((0, 2), (2, 4)), "twice"),
    (("cuda:0", "cuda:1"), ((0, 2), (3, 4)), "contiguous"),
    (("cuda:0", "cuda:1"), ((0, 3), (2, 4)), "contiguous"),
    (("cuda:0", "cuda:1"), ((0, 2), (2, 2)), "contiguous"),
    (("cuda:0", "cuda:1"), ((0, 2), (2, 3)), "end at 3"),
    (("cuda:0",), ((0, 2), (2, 4)), "one layer range per device"),
])
def test_a_split_that_could_move_or_miss_a_layer_is_refused(devices, ranges, message):
    with pytest.raises(ValueError, match=message):
        LayerSplit(num_layers=4, devices=devices, layer_ranges=ranges)


def test_the_digest_names_the_split_and_nothing_else():
    a = LayerSplit.from_boundaries(32, ("cuda:0", "cuda:1"), (16,))
    assert a.digest == LayerSplit.model_validate_json(a.model_dump_json()).digest
    assert a.digest != LayerSplit.from_boundaries(32, ("cuda:0", "cuda:1"), (10,)).digest
    assert a.digest != LayerSplit.from_boundaries(32, ("cuda:1", "cuda:0"), (16,)).digest
    assert len(a.digest) == 64


def test_the_device_map_places_embeddings_first_and_the_head_last():
    placement = LayerSplit.from_boundaries(4, ("cuda:2", "cuda:5"), (1,)).hf_device_map()
    assert placement["model.embed_tokens"] == placement["model.rotary_emb"] == "cuda:2"
    assert placement["model.layers.0"] == "cuda:2"
    assert [placement[f"model.layers.{i}"] for i in (1, 2, 3)] == ["cuda:5"] * 3
    assert placement["model.norm"] == placement["lm_head"] == "cuda:5"


def _fake_model(placement: dict[str, str]):
    params = [(name, SimpleNamespace(device=torch.device(device)))
              for name, device in placement.items()]
    return SimpleNamespace(named_parameters=lambda: iter(params),
                           parameters=lambda: iter(p for _, p in params))


def test_placement_is_checked_parameter_by_parameter():
    split = LayerSplit.from_boundaries(2, ("cuda:0", "cuda:1"), (1,))
    placed = {"model.embed_tokens.weight": "cuda:0",
              "model.layers.0.mlp.gate_proj.weight": "cuda:0",
              "model.layers.1.mlp.gate_proj.weight": "cuda:1",
              "model.norm.weight": "cuda:1", "lm_head.weight": "cuda:1"}
    split.check_placement(_fake_model(placed))
    with pytest.raises(ValueError, match="model.layers.1.mlp.gate_proj.weight"):
        split.check_placement(_fake_model({**placed,
                                           "model.layers.1.mlp.gate_proj.weight": "cuda:0"}))
    with pytest.raises(ValueError, match="belongs to no module"):
        split.check_placement(_fake_model({**placed, "stray.weight": "cuda:0"}))


def test_the_arithmetic_guard_reads_every_device_the_model_occupies(monkeypatch):
    """A model whose first parameter is on the host still computes on its GPUs, so the
    path-floor primitive refuses it outside the lane's arithmetic before any forward."""
    from anamnesis.extraction.replay.cached import replay_extract_incremental

    monkeypatch.delenv(WORKSPACE_ENV, raising=False)
    model = _fake_model({"model.embed_tokens.weight": "cpu", "lm_head.weight": "cuda:3"})
    assert model_devices(model) == [torch.device("cpu"), torch.device("cuda:3")]
    loaded = SimpleNamespace(model=model)
    with pytest.raises(ValueError, match=WORKSPACE_ENV):
        replay_extract_incremental(loaded, [1, 2, 3, 4], 1)


def _lane(**kwargs) -> GpuFeatureLane:
    extraction = ExtractionConfig(sampled_layers=[0, 1, 2], pca_layers=[0, 1],
                                  pca_components=5, early_layer_cutoff=8, late_layer_cutoff=24)
    families = FeaturePipelineConfig(
        include_core_blocks=True, enable_residual_trajectory=True, trajectory_layers=[1],
        enable_attention_flow=True, enable_gate_features=True, enable_per_head=True,
        enable_value_geometry=True, enable_qk_geometry=True, enable_kv_cka=True,
        contrastive_layers=[8, 16, 20, 24, 28])
    rng = np.random.default_rng(341)
    components = rng.normal(size=(5, 32)).astype(np.float32)
    schema = resolve_gpu_schema(3, 16, extraction, families, components)
    return GpuFeatureLane(extraction, families, list(schema.feature_names),
                          rng.normal(0, 0.01, size=(4, 64, 32)).astype(np.float32),
                          components, np.zeros(32, dtype=np.float32), device="cpu",
                          calibration_sha256="a" * 64, replay_path="full", **kwargs)


def test_a_one_device_split_is_the_one_device_lane_byte_for_byte():
    loaded = tiny_loaded()
    tokens = list(np.random.default_rng(7).integers(0, 64, size=24))
    plain = _lane()
    split = _lane(layer_split=LayerSplit.single(3, "cpu"))
    assert split.lane_id == plain.lane_id and split.identity == plain.identity
    a = plain.replay_span(loaded, tokens, 7, 24)
    b = split.replay_span(loaded, tokens, 7, 24)
    assert a.features.tobytes() == b.features.tobytes()
    assert "layer_split_sha256" not in b.metadata


def test_a_lane_that_reduces_off_the_split_is_refused():
    with pytest.raises(ValueError, match="does not hold"):
        _lane(layer_split=LayerSplit.from_boundaries(3, ("cuda:0", "cuda:1"), (1,)))
