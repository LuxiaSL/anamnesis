"""A lane's effective-dimension method: the default is the lane as it was, ``norms`` is a lane of its own.

A preset's ``eff_dim`` chooses how the fast lane computes the effective dimension. The
default, ``svd``, is the specification's arithmetic from the singular values. ``norms``
computes the same quantity from two Frobenius norms, which agrees to rounding rather
than to the byte, so a lane reading a ``norms`` preset must be a lane of its own: a
registry row names it, its identity and receipts say ``eff_dim: norms``, and its id is
never a default lane's. The default adds nothing anywhere, so every lane built without
it keeps the id it had.

Asserted here: a registry row extending a shipped preset with ``"eff_dim": "norms"``
reaches the lane, its id and its receipts, while shipped keys resolve as before; the
``norms`` id differs from every default id, structurally; the Frobenius arithmetic agrees
with the singular-value arithmetic to a relative 1e-12 on random and adversarial
matrices; every entry that reduces through the family reducer (the lane's replay, an
outside capture and the vLLM readout's restatement) uses the lane's method and differs
from the default only in the effective-dimension coordinates; and a ``norms`` lane still
agrees with the numeric anchor, which always uses singular values, at the agreement
proof's own tolerance.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from anamnesis.config import ExtractionConfig, FeaturePipelineConfig, resolve_preset
from anamnesis.config.models import MODELS_ENV
from anamnesis.extraction.fast.external import external_lane_id
from anamnesis.extraction.fast.families import eff_dim, norms_eff_dims
from anamnesis.extraction.fast.features import GpuFeatureLane
from anamnesis.extraction.fast.schema import resolve_gpu_schema
from anamnesis.extraction.feature_pipeline import compute_features_with_families_from_data
from anamnesis.extraction.replay.extract import replay_extract
from anamnesis.extraction.replay_config import native_replay_configs
from anamnesis.extraction.vllm.readout import reduce_capture
from test_fast_external import IDENTITY, external_reduction
from test_fast_lane_equivalence import CONTRASTIVE_LAYERS, tiny_loaded
from test_vllm_readout import fast_lane_reduction, make_example


@pytest.fixture(autouse=True)
def deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def relative(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float(((a - b).abs() / b.abs().clamp_min(1e-300)).max())


def adversarial_matrices() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(3)

    def randn(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float64)

    rank_one = randn(511, 1) @ randn(1, 128)
    return {
        "random 511x128": randn(511, 128),
        "random 129x8": randn(129, 8),
        "rank one": rank_one,
        "rank three": randn(200, 3) @ randn(3, 128),
        "rank deficient plus noise": rank_one + 1e-9 * randn(511, 128),
        "one direction dominant": randn(511, 128) * torch.logspace(0, -12, 128, dtype=torch.float64),
        "large scale": randn(64, 128) * 1e6,
        "small scale above the guard": randn(64, 128) * 1e-6,
        "wide 2x128": randn(2, 128),
        "wide 7x128": randn(7, 128),
        "one row": randn(1, 128),
        "two rows": randn(2, 8),
        "constant rows": torch.ones(40, 8, dtype=torch.float64) * 0.3,
    }


@pytest.mark.parametrize("name", list(adversarial_matrices()))
def test_norms_agree_with_singular_values_to_1e_12(name):
    x = adversarial_matrices()[name]
    assert relative(norms_eff_dims(x[None])[0], eff_dim(x)) <= 1e-12


def test_the_guard_is_the_same_on_both_sides_of_it():
    g = torch.Generator().manual_seed(9)
    x = torch.randn(30, 8, generator=g, dtype=torch.float64)
    for scale, zero in ((1e-8, True), (1e-5, False)):  # total about 2e-14, then 2e-8
        y = x * scale
        assert (eff_dim(y) == 0).item() is zero
        assert (norms_eff_dims(y[None])[0] == 0).item() is zero
    assert norms_eff_dims(torch.zeros(1, 9, 4, dtype=torch.float64))[0].item() == 0.0


def test_a_batch_of_matrices_is_each_matrix_s_own():
    matrices = adversarial_matrices()["random 511x128"].reshape(1, 511, 128).repeat(5, 1, 1)
    matrices = matrices * torch.arange(1, 6, dtype=torch.float64)[:, None, None]
    together = norms_eff_dims(matrices)
    for i in range(5):
        assert relative(together[i], eff_dim(matrices[i])) <= 1e-12


def norms_families(families: FeaturePipelineConfig) -> FeaturePipelineConfig:
    """``families`` as a ``norms`` preset's replay configuration carries it."""
    families = families.model_copy()
    families._fast_lane_eff_dim = "norms"
    return families


def lanes(dtype, **example):
    lane, capture, kwargs, weights = make_example(dtype, **example)
    components = np.eye(lane.pm.shape[2], dtype=np.float32)[:3]
    opted = GpuFeatureLane(lane.config, norms_families(lane.families), list(lane.names), lane.pm,
                           components, np.zeros(lane.pm.shape[2], np.float32), device="cpu",
                           calibration_sha256=lane.calibration_sha256)
    return lane, opted, capture, kwargs, weights


def preset_lane(preset_name: str) -> GpuFeatureLane:
    preset = resolve_preset(preset_name)
    extraction, families = native_replay_configs(preset)
    components = np.zeros((extraction.pca_components, preset.hidden_dim), np.float32)
    names = resolve_gpu_schema(preset.num_layers, 64, extraction, families, components).feature_names
    return GpuFeatureLane(extraction, families, list(names),
                          np.ones((preset.num_layers + 1, 80, preset.hidden_dim), np.float32),
                          components, np.zeros(preset.hidden_dim, np.float32), device="cpu",
                          calibration_sha256="d" * 64, replay_path="full")


def test_a_registry_row_names_the_norms_lane_and_shipped_keys_resolve_as_before(
        tmp_path, monkeypatch):
    shipped = {name: resolve_preset(name) for name in ("3b", "8b")}
    before = {name: preset_lane(name) for name in shipped}
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"presets": {"8b-normdim": {"extends": "8b", "eff_dim": "norms"}}}))
    monkeypatch.setenv(MODELS_ENV, str(path))
    for name, preset in shipped.items():
        assert resolve_preset(name) == preset and preset.eff_dim == "svd"
        lane = preset_lane(name)
        assert lane.lane_id == before[name].lane_id and "eff_dim" not in lane.identity
    row = resolve_preset("8b-normdim")
    assert row.eff_dim == "norms"
    unshared = {"name", "eff_dim", "stage0_dir", "run_prefixes"}  # a row never inherits the last two
    assert row.model_dump(exclude=unshared) == shipped["8b"].model_dump(
        exclude=unshared)
    _, families = native_replay_configs(row)
    assert families._fast_lane_eff_dim == "norms"
    assert families.model_dump() == native_replay_configs(shipped["8b"])[1].model_dump()
    lane = preset_lane("8b-normdim")
    assert lane.eff_dim == "norms" and lane.identity["eff_dim"] == "norms"
    assert lane.lane_id != before["8b"].lane_id


def test_the_norms_id_is_never_a_default_id():
    """Its identity holds a key no default identity has, so the two digests are of
    different documents; and its id differs from every default lane's built here."""
    lane, opted, *_ = lanes(torch.float32)
    assert "eff_dim" not in lane.identity
    assert opted.identity == {**lane.identity, "eff_dim": "norms"}
    defaults = {lane.lane_id}
    for steps in (3, 16, 70):
        for sampled in ((0, 1, 2), (1,)):
            defaults.add(make_example(torch.float32, steps=steps, sampled_layers=sampled)[0].lane_id)
            assert lanes(torch.float32, steps=steps, sampled_layers=sampled)[1].lane_id not in defaults
    defaults |= {preset_lane(name).lane_id for name in ("3b", "8b")}
    assert opted.lane_id not in defaults
    assert external_lane_id(opted, IDENTITY) != external_lane_id(lane, IDENTITY)
    # The method rides on the lane's own copy of the families; no dump carries it.
    assert opted.families.model_dump() == lane.families.model_dump()
    assert lane.families._fast_lane_eff_dim == "svd"


def test_an_unknown_method_is_refused():
    lane, *_ = lanes(torch.float32)
    families = lane.families.model_copy()
    families._fast_lane_eff_dim = "gram"
    with pytest.raises(ValueError, match="eff_dim"):
        GpuFeatureLane(lane.config, families, list(lane.names), lane.pm,
                       np.eye(lane.pm.shape[2], dtype=np.float32)[:3],
                       np.zeros(lane.pm.shape[2], np.float32), device="cpu",
                       calibration_sha256="a" * 64)


def eff_dim_coordinates(names):
    return np.array(["eff_dim" in name and not name.startswith("gate_") for name in names])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["float32", "bfloat16"])
@pytest.mark.parametrize("steps,start,sampled", [(16, 4, (0, 1, 2)), (70, 6, (1,)), (3, 4, (0, 2))],
                         ids=["all-sampled", "long-span", "short"])
def test_every_entry_reduces_with_the_lane_s_method(dtype, steps, start, sampled):
    lane, opted, capture, kwargs, weights = lanes(dtype, steps=steps, start=start,
                                                  sampled_layers=sampled)
    span = dict(start=kwargs["start"], end=kwargs["end"])
    default = fast_lane_reduction(lane, capture, weights, **span)
    replayed = fast_lane_reduction(opted, capture, weights, **span)
    outside = external_reduction(opted, capture, weights, **span)
    readout = reduce_capture(opted, capture, **kwargs)
    assert outside.features.tobytes() == replayed.features.tobytes()
    assert readout.features.tobytes() == replayed.features.tobytes()
    assert outside.metadata["lane_id"] == external_lane_id(opted, IDENTITY)
    assert replayed.metadata["lane_id"] == opted.lane_id != lane.lane_id
    assert replayed.metadata["eff_dim"] == "norms" and "eff_dim" not in default.metadata
    held = eff_dim_coordinates(replayed.feature_names)
    assert held.sum() == 3 * len(sampled)
    np.testing.assert_array_equal(replayed.features[~held], default.features[~held])
    np.testing.assert_allclose(replayed.features[held], default.features[held], rtol=1e-6, atol=0)


@pytest.mark.parametrize("continuation", [3, 17, 65])
def test_a_norms_lane_agrees_with_the_anchor_at_the_proof_s_tolerance(continuation):
    """The agreement proof's comparison, full replay path, with a ``norms`` lane."""
    loaded = tiny_loaded()
    prefix = 11
    rng = np.random.default_rng(492)
    tokens = rng.integers(0, 64, size=prefix + continuation).tolist()
    pm = rng.normal(0, 0.01, size=(4, prefix + continuation, 32)).astype(np.float32)
    components = rng.normal(size=(5, 32)).astype(np.float32)
    mean = rng.normal(0, 0.01, size=32).astype(np.float32)
    extraction = ExtractionConfig(sampled_layers=[0, 1, 2], pca_layers=[0, 1], pca_components=5,
                                  early_layer_cutoff=8, late_layer_cutoff=24)
    families = FeaturePipelineConfig(
        include_core_blocks=True, enable_residual_trajectory=True, trajectory_layers=[1],
        enable_attention_flow=True, enable_gate_features=True, enable_per_head=True,
        enable_value_geometry=True, enable_qk_geometry=True, enable_kv_cka=True,
        contrastive_layers=CONTRASTIVE_LAYERS)
    ref = compute_features_with_families_from_data(
        replay_extract(loaded, tokens, prefix, pm), extraction, families, components, mean)
    schema = resolve_gpu_schema(3, continuation - 1, extraction, families, components)
    assert schema.feature_names == tuple(ref.feature_names)
    lane = GpuFeatureLane(extraction, norms_families(families), ref.feature_names, pm,
                          components, mean, device="cpu", calibration_sha256="a" * 64,
                          replay_path="full")
    one = lane.replay_span(loaded, tokens, prefix, len(tokens))
    failures = [(name, float(ref.features[i]), float(one.features[i]))
                for i, name in enumerate(ref.feature_names)
                if not np.isclose(one.features[i], ref.features[i], atol=5e-6, rtol=3e-5)]
    assert not failures, failures
