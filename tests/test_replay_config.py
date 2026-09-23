"""The shared replay battery: one declaration both lanes read.

`native_replay_configs` is the battery a replay runs, and it sits above both extraction
paths on purpose — an arithmetic backend is a way of computing features, not a licence to
redefine which features exist. If the fast lane and the anchor were configured
separately, a comparison between them would be measuring two batteries.

The test is written as a duplicate: the expected configuration is spelled out field by
field beside the call, for every registered preset and both PCA settings, and the two
must agree by `model_dump`. That is deliberate redundancy — the point is that a silent
edit to the battery cannot pass, and a test that recomputed the configuration from the
same source would pass no matter what it said.

Pure configuration; no model, no device.
"""

import pytest

from anamnesis.config import ExtractionConfig, FeaturePipelineConfig, preset_names, resolve_preset
from anamnesis.extraction.replay_config import native_replay_configs


@pytest.mark.parametrize("model", list(preset_names()))
@pytest.mark.parametrize("enable_pca", [False, True])
def test_shared_native_config_matches_original_inline_contract(model, enable_pca):
    preset = resolve_preset(model)
    actual_ec, actual_fc = native_replay_configs(preset, enable_pca=enable_pca)
    expected_ec = ExtractionConfig(
        sampled_layers=preset.sampled_layers,
        pca_layers=preset.pca_layers,
        early_layer_cutoff=preset.early_layer_cutoff,
        late_layer_cutoff=preset.late_layer_cutoff,
        enable_residual_pca=enable_pca,
    )
    expected_fc = FeaturePipelineConfig(
        include_core_blocks=True,
        enable_residual_trajectory=True,
        enable_attention_flow=True,
        enable_gate_features=True,
        enable_temporal_dynamics=False,
        enable_per_head=True,
        enable_stft=True,
        enable_contrastive_projection=False,
        enable_value_geometry=True,
        enable_qk_geometry=True,
        enable_kv_cka=True,
        enable_expert_routing=True,
        trajectory_layers=preset.trajectory_layers,
        contrastive_layers=preset.contrastive_layers,
    )
    assert actual_ec.model_dump() == expected_ec.model_dump()
    assert actual_fc.model_dump() == expected_fc.model_dump()
