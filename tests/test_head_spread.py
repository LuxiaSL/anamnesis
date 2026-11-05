"""The spread reading of the heads' disagreement (``head_disagreement = "spread"``).

A lane whose preset reads the spread replaces the head agreement with two pairs per layer.
The first is the heads' L2 spread around their mean row, ``mean_h |p_h - mean_g p_g|^2``, the
statistic a capture can stream (both its terms are running moments of the per-head rows). The
second is the across-head variance of their entropies. Checked here:

- the schema names the four columns in the agreement's place, and nothing else moves;
- the reference extractor's values are the definitions;
- the capture's form (sums of squares from each head's final row) is the same statistic;
- a signed row carrying ``head_spread`` reduces, per row and batched, to the features of the
  same forward's weights, within the statistics contract's tolerance;
- a row without the surface is refused naming it, the lane has its own id, and the default
  reading keeps every name and the lane id it had.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import test_vllm_readout as readout
from anamnesis.extraction.fast.engine_reducer import EngineReducer, surfaces_for
from anamnesis.extraction.fast.external import ExternalCaptureError
from anamnesis.extraction.fast.external_stats import stats_from_weights
from anamnesis.extraction.fast.schema import resolve_gpu_schema
from anamnesis.extraction.state_extractor import (
    RawGenerationData,
    _batch_entropy,
    extract_attention_and_deltas,
)
from test_engine_reducer import ENGINE, WEIGHTS, signed_example
from test_fast_external_stats import TOLERANCE, weights_reduction


@pytest.fixture(autouse=True)
def deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


@pytest.fixture()
def spread_lanes(monkeypatch):
    """Every lane the readout fixtures build reads the spread."""
    original = readout.lane_configuration

    def configured(*args, **kwargs):
        extraction, families, components = original(*args, **kwargs)
        extraction._head_disagreement = "spread"
        return extraction, families, components

    monkeypatch.setattr(readout, "lane_configuration", configured)


def test_the_schema_names_the_spread_in_the_agreements_place():
    extraction, families, components = readout.lane_configuration((0, 1, 2))
    jsd = list(resolve_gpu_schema(readout.N_LAYERS, 16, extraction, families, components).feature_names)
    extraction._head_disagreement = "spread"
    spread = list(resolve_gpu_schema(readout.N_LAYERS, 16, extraction, families, components).feature_names)
    assert not [n for n in spread if n.startswith("head_agreement_")]
    for layer in range(readout.N_LAYERS):
        i = jsd.index(f"head_agreement_mean_L{layer}")
        assert jsd[i:i + 2] == [f"head_agreement_mean_L{layer}", f"head_agreement_std_L{layer}"]
    rest = [n for n in jsd if not n.startswith("head_agreement_")]
    assert [n for n in spread if not n.startswith(("head_spread_", "head_entropy_var_"))] == rest
    assert len(spread) == len(jsd) + 2 * readout.N_LAYERS


def _raw(seed=0, steps=12, layers=3, heads=4, prompt=5):
    g = np.random.default_rng(seed)
    attentions = []
    for t in range(steps):
        a = g.random((layers, heads, prompt + t + 1)) ** 3
        attentions.append((a / a.sum(axis=-1, keepdims=True)).astype(np.float32))
    return RawGenerationData(hidden_states=[np.zeros((layers + 1, 4), np.float32) for _ in range(steps)],
                             attentions=attentions, logits=[np.zeros(5, np.float32)] * steps,
                             chosen_token_ids=np.zeros(steps, np.float32), pre_rope_keys={},
                             prompt_length=prompt)


def test_the_reference_computes_the_definitions():
    extraction, _, _ = readout.lane_configuration((0, 1, 2))
    extraction._head_disagreement = "spread"
    data = _raw()
    values, names = extract_attention_and_deltas(data, extraction)
    got = dict(zip(names, values))
    for layer in range(3):
        spreads, variances = [], []
        for a in data.attentions:  # 12 steps: every step is sampled
            p = a[layer].astype(np.float64)
            p = p / p.sum(axis=1, keepdims=True)
            spreads.append(((p - p.mean(axis=0)) ** 2).sum(axis=1).mean())
            variances.append(np.var(_batch_entropy(p)))
        np.testing.assert_allclose(got[f"head_spread_mean_L{layer}"], np.mean(spreads), rtol=1e-6)
        np.testing.assert_allclose(got[f"head_spread_std_L{layer}"], np.std(np.float32(spreads)), rtol=1e-5)
        np.testing.assert_allclose(got[f"head_entropy_var_mean_L{layer}"], np.mean(variances), rtol=1e-6)


def test_the_captures_form_is_the_same_statistic():
    """(1/H) sum_h sum_c p_hc^2 - sum_c pbar_c^2, the sums a stats pass keeps from each head's
    final row, equals the spread about the mean row."""
    lane, row, token_ids, full, _ = signed_example(steps=16, start=4)
    begin, end = row.start, len(token_ids)
    stats = stats_from_weights(full, begin, end, lane.config.sampled_layers)
    for layer, w in enumerate(full):
        w = w if w.ndim == 4 else w.unsqueeze(0)
        a = w[0, :, begin:end - 1, :end].permute(1, 0, 2).double()
        valid = torch.arange(end)[None, :] < torch.arange(begin + 1, end)[:, None]
        p = a * valid[:, None, :]
        p = p / p.sum(dim=-1, keepdim=True)
        form = (p * p).sum(dim=-1).mean(dim=1) - (p.mean(dim=1) ** 2).sum(dim=-1)
        torch.testing.assert_close(stats["head_spread"][:, layer].double(), form, rtol=1e-5, atol=1e-7)


def _spread_row(steps, start, sampled):
    lane, row, token_ids, full, capture = signed_example(steps, start, sampled)
    stats = stats_from_weights(full, row.start, len(token_ids), lane.config.sampled_layers)
    surfaces = dict(row.surfaces)
    del surfaces["mean_entropy"]
    surfaces["head_spread"] = stats["head_spread"]
    row.surfaces = surfaces
    return lane, row, token_ids, full, capture


@pytest.mark.parametrize("steps,start,sampled", [(3, 4, (0, 2)), (16, 4, (0, 2)), (40, 6, (1,))],
                         ids=["steps3", "steps16", "steps40"])
@pytest.mark.parametrize("batched", [False, True], ids=["per-row", "batched"])
def test_a_spread_row_reduces_to_the_weights_features(spread_lanes, steps, start, sampled, batched):
    from anamnesis.extraction.fast.engine_reducer import routing_points

    lane, row, token_ids, full, capture = _spread_row(steps, start, sampled)
    assert "head_spread" in surfaces_for(lane) and "mean_entropy" not in surfaces_for(lane)
    got = EngineReducer(lane_for=lambda s: lane, batched=batched, expected_weights=WEIGHTS).reduce(
        [row], token_ids=[token_ids], engine=ENGINE)[0]
    reference = weights_reduction(lane, capture, full, start=row.start, end=len(token_ids),
                                  routing=routing_points(row), committed=row.surfaces["committed"])
    names = list(reference.feature_names)
    assert got.meta["feature_names"] == names
    assert [n for n in names if n.startswith("head_spread_")] and not [n for n in names
                                                                       if n.startswith("head_agreement_")]
    np.testing.assert_allclose(got.features, reference.features, **TOLERANCE)


def test_a_spread_lane_refuses_a_row_without_its_surface(spread_lanes):
    lane, row, token_ids, _, _ = _spread_row(16, 4, (0, 2))
    del row.surfaces["head_spread"]
    with pytest.raises(ExternalCaptureError, match="head_spread"):
        EngineReducer(lane_for=lambda s: lane, batched=False).reduce(
            [row], token_ids=[token_ids], engine=ENGINE)


def test_the_spread_lane_has_its_own_id_and_the_default_keeps_its(monkeypatch):
    jsd_lane, *_ = signed_example()
    jsd_names = jsd_lane.names
    original = readout.lane_configuration

    def configured(*args, **kwargs):
        extraction, families, components = original(*args, **kwargs)
        extraction._head_disagreement = "spread"
        return extraction, families, components

    monkeypatch.setattr(readout, "lane_configuration", configured)
    spread_lane, *_ = signed_example()
    assert spread_lane.lane_id != jsd_lane.lane_id
    monkeypatch.setattr(readout, "lane_configuration", original)
    again, *_ = signed_example()
    assert again.lane_id == jsd_lane.lane_id and again.names == jsd_names


def test_an_export_carries_the_mixture_surface_its_row_was_signed_with(tmp_path, spread_lanes):
    from anamnesis.extraction.fast.replay_export import ReplayExportError, load_export, save_export

    lane, row, token_ids, _, _ = _spread_row(16, 4, (0, 2))
    header = dict(sign_id="s1", request_id="s1", token_ids=list(token_ids), prompt_length=int(row.start),
                  start=int(row.start), positions=int(row.positions), weights_sha256=WEIGHTS,
                  engine_identity_sha256="e" * 64, pins_sha256=None, lever_plan_sha256=None,
                  lane_id="engine-stats-b1-toy", replayed_bitwise=True,
                  layers=dict(effective=readout.N_LAYERS, sampled=list(lane.config.sampled_layers)),
                  route_sources=[list(p) for p in row.route_sources], final_sources=int(row.final_sources))
    save_export(tmp_path / "row", row, header)
    _, loaded = load_export(tmp_path / "row")
    assert "head_spread" in loaded.surfaces and "mean_entropy" not in loaded.surfaces
    row.surfaces["mean_entropy"] = row.surfaces["head_spread"]
    with pytest.raises(ReplayExportError, match="one of"):
        save_export(tmp_path / "both", row, header)
