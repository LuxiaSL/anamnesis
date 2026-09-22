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
        enable_per_head=True,
        enable_stft=True,
        # A floor must cover every featurized cell natively, so the battery vector enables
        # these three families even though the frozen fat_01 signatures on disk predate them
        # and carry none of their columns. The battery vector is a superset of that baseline,
        # never a replacement for it.
        enable_value_geometry=True,
        enable_qk_geometry=True,
        enable_kv_cka=True,
        # MoE expert routing: None-guarded — the xrt family returns empty for dense models
        # (router_dist is None), so enabling it here is a no-op everywhere except the
        # DeepSeek-V2-Lite class, where it appends the xrt block to the battery vector.
        enable_expert_routing=True,
        trajectory_layers=preset.trajectory_layers,
        contrastive_layers=preset.contrastive_layers,
    )
    return extraction_config, family_config
