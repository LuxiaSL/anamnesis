"""Pluggable feature families for GPU-free feature engineering.

Each module exports functions that accept RawGenerationData and return
FeatureFamilyResult. The feature_pipeline orchestrates calling enabled
families and concatenating results.

Shared machinery:
    operators        — Reusable temporal operators (windowing, STFT)
    _helpers         — Math utilities re-exported from state_extractor

Families, by the substrate they read:
    attention_flow   — Region decomposition, recency bias, head diversity
    temporal_dynamics— Windowed attention/cache/key metrics with STFT
    per_head         — Head heterogeneity that head-averaging destroys
    residual_stream  — Trajectory features: velocity, curvature, directness
    contrastive_projection — A trained projection applied to hidden states
    path_signature   — Level-2 log-signature (iterated integrals) of a projected path
    gate_features    — SwiGLU gate sparsity, drift, effective dimension
    value_geometry   — v_proj value-vector spread, effective dimension, drift
    qk_geometry      — Pre-RoPE query geometry and query-key content alignment
    key_cka          — Cross-layer linear CKA over keys and values (basis-invariant)
    attn_res         — Cross-block attention-residual routing, and the reference
                       example of an OPTIONAL PER-ARCHITECTURE family
    expert_routing   — Mixture-of-experts router allocation

The last two read substrates only some architectures have. They follow one
pattern, documented in attn_res.py: the substrate is an optional field on
RawGenerationData, the orchestrator gates on that field's presence rather than on
a model's name, and a family with nothing to read is absent from the vector
rather than zero-filled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

F32 = NDArray[np.float32]


@dataclass
class FeatureFamilyResult:
    """Output from a pluggable feature family.

    Same contract as extract_tier*() in state_extractor.py:
    a flat feature vector paired with names.
    """

    features: F32  # flat feature vector
    feature_names: list[str]  # one name per dimension
    family_name: str  # e.g. "residual_trajectory", "attention_flow"

    def __len__(self) -> int:
        return len(self.features)

    @staticmethod
    def empty(family_name: str) -> "FeatureFamilyResult":
        """Return an empty result (e.g., when data is missing)."""
        return FeatureFamilyResult(
            features=np.array([], dtype=np.float32),
            feature_names=[],
            family_name=family_name,
        )
