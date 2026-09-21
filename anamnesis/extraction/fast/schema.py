"""Resolve supported dense-Llama feature names using the canonical CPU extractor.

Only names/slices are retained. Tiny synthetic tensors preserve layer counts,
PCA ranks and short-span branches; their numerical values are never emitted as
signatures. This avoids either a duplicated naming registry or a costly real
CPU replay merely to discover the schema.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anamnesis.config import ExtractionConfig, FeaturePipelineConfig
from anamnesis.extraction.feature_pipeline import compute_features_v2_from_data
from anamnesis.extraction.state_extractor import RawGenerationData


@dataclass(frozen=True)
class GpuFeatureSchema:
    feature_names: tuple[str, ...]
    family_slices: dict[str, tuple[int, int]]


def resolve_gpu_schema(
    num_layers: int,
    n_steps: int,
    extraction: ExtractionConfig,
    families: FeaturePipelineConfig,
    pca_components: np.ndarray | dict[int, np.ndarray],
) -> GpuFeatureSchema:
    if num_layers < 1 or n_steps < 1:
        raise ValueError("positive layer/step counts required")
    if (
        families.enable_path_signature
        or families.enable_contrastive_projection
        or families.enable_temporal_dynamics
    ):
        raise ValueError(
            "schema resolver supports only the declared GPU feature families"
        )
    # Name cardinality changes for the very shortest spans (notably gate/CKA).
    # Other time-series branches retain the same fixed-width name contract.
    steps = min(n_steps, max(8, families.temporal_n_windows))
    sampled = extraction.sampled_layers
    heads = {
        layer: [np.zeros((1, 1), dtype=np.float32) for _ in range(steps)]
        for layer in sampled
    }
    gates = {
        layer: [np.zeros(1, dtype=np.float32) for _ in range(steps)]
        for layer in sampled
    }
    raw = RawGenerationData(
        hidden_states=[
            np.zeros((num_layers + 1, 1), dtype=np.float32) for _ in range(steps)
        ],
        attentions=[
            np.ones((num_layers, 1, t + 2), dtype=np.float32) / (t + 2)
            for t in range(steps)
        ],
        logits=[np.zeros(5, dtype=np.float32) for _ in range(steps)],
        chosen_token_ids=np.zeros(steps, dtype=np.float32),
        pre_rope_keys=heads,
        prompt_length=1,
        v_proj_values=heads,
        queries=heads,
        gate_activations=gates,
    )
    if isinstance(pca_components, dict):
        components = {
            layer: np.zeros((value.shape[0], 1), dtype=np.float32)
            for layer, value in pca_components.items()
        }
        means = {layer: np.zeros(1, dtype=np.float32) for layer in components}
    else:
        components = np.zeros((pca_components.shape[0], 1), dtype=np.float32)
        means = np.zeros(1, dtype=np.float32)
    result = compute_features_v2_from_data(raw, extraction, families, components, means)
    if len(result.feature_names) != len(set(result.feature_names)):
        raise ValueError("canonical schema contains duplicate names")
    return GpuFeatureSchema(tuple(result.feature_names), dict(result.block_slices))
