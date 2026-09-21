"""Shared native replay battery; arithmetic backends do not redefine features."""

from anamnesis.config import ExtractionConfig, FeaturePipelineConfig, ModelPreset


def native_replay_configs(
    preset: ModelPreset, *, enable_pca: bool = True
) -> tuple[ExtractionConfig, FeaturePipelineConfig]:
    extraction_config = ExtractionConfig(
        sampled_layers=preset.sampled_layers,
        pca_layers=preset.pca_layers,
        early_layer_cutoff=preset.early_layer_cutoff,
        late_layer_cutoff=preset.late_layer_cutoff,
        enable_residual_pca=enable_pca,
    )
    family_config = FeaturePipelineConfig(
        include_core_blocks=True,
        enable_residual_trajectory=True,
        enable_attention_flow=True,
        enable_gate_features=True,
        enable_temporal_dynamics=False,  # v3: temporal_dynamics ignored
        enable_per_head=True,  # v3: new surface
        enable_stft=True,
        enable_contrastive_projection=False,  # contrastive is a separate addon
        # vmb matrix completion pass 1 (prereg Stage A(ii), census 2026-07-12): the
        # deployed 2,713-dim v3 vector carried ZERO value/qk/cka features — the
        # families existed but were never enabled here. Floors must cover every
        # featurized cell natively (ordering rule), so the battery vector is the
        # v3 superset. Old fat_01 signatures remain the frozen 2,713 baseline.
        enable_value_geometry=True,
        enable_qk_geometry=True,
        enable_kv_cka=True,
        # MoE expert routing (vmb arm A7, M6): None-guarded — the xrt family returns empty
        # for dense models (router_dist is None), so enabling it here is a no-op everywhere
        # except DeepSeek-V2-Lite, where it adds the 60 xrt features to the battery vector.
        enable_expert_routing=True,
        trajectory_layers=preset.trajectory_layers,
        contrastive_layers=preset.contrastive_layers,
    )
    return extraction_config, family_config
