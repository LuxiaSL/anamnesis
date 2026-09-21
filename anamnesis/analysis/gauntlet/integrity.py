"""Section 1: Data integrity and descriptive statistics."""

from __future__ import annotations

import numpy as np

from .signature_io import (
    ATTENTION_AND_DELTAS,
    CACHE_AND_KEYS,
    NORMS_AND_OUTPUT_STATS,
    RESIDUAL_PCA,
    AnalysisData,
)
from .schemas import (
    IntegrityResult,
    LengthByModeStats,
    LengthOverallStats,
    NanInfCount,
    BlockValueRange,
    BlockVarianceReport,
)
from .utils import get_available_blocks


def run_integrity_checks(data: AnalysisData) -> IntegrityResult:
    """Run all data integrity and descriptive checks."""
    samples_per_mode = {
        m: int(np.sum(data.mode_mask(m))) for m in data.unique_modes
    }
    samples_per_topic = {
        t: int(np.sum(data.topic_mask(t))) for t in data.unique_topics
    }
    balanced = len(set(samples_per_mode.values())) == 1

    available_blocks, _ = get_available_blocks(data)

    # Every block below comes from `get_available_blocks`, which reads the loaded
    # matrices, so each one is present by construction and is read without a guard.
    block_dims: dict[str, int] = {
        block: data.get_block(block).shape[1] for block in available_blocks
    }
    total_features = sum(
        block_dims.get(t, 0) for t in [NORMS_AND_OUTPUT_STATS, ATTENTION_AND_DELTAS, CACHE_AND_KEYS, RESIDUAL_PCA]
    )

    # NaN / Inf checks
    nan_inf: dict[str, NanInfCount] = {}
    all_clean = True
    for block in available_blocks:
        X = data.get_block(block)
        nan_count = int(np.sum(np.isnan(X)))
        inf_count = int(np.sum(np.isinf(X)))
        nan_inf[block] = NanInfCount(nan=nan_count, inf=inf_count)
        if nan_count > 0 or inf_count > 0:
            all_clean = False

    # Per-feature variance (detect constant features)
    variance_report: dict[str, BlockVarianceReport] = {}
    for block in available_blocks:
        X = data.get_block(block)
        var = X.var(axis=0)
        variance_report[block] = BlockVarianceReport(
            n_features=int(X.shape[1]),
            n_constant=int(np.sum(var < 1e-12)),
            n_near_constant=int(np.sum(var < 1e-6)),
        )

    # Generation length distribution by mode
    length_by_mode: dict[str, LengthByModeStats] | None = None
    length_overall: LengthOverallStats | None = None
    if data.generation_lengths is not None:
        length_by_mode = {}
        for mode in data.unique_modes:
            mask = data.mode_mask(mode)
            lengths = data.generation_lengths[mask]
            length_by_mode[mode] = LengthByModeStats(
                mean=float(np.mean(lengths)),
                std=float(np.std(lengths)),
                min=int(np.min(lengths)),
                max=int(np.max(lengths)),
                median=float(np.median(lengths)),
            )

        all_lengths = data.generation_lengths
        length_overall = LengthOverallStats(
            mean=float(np.mean(all_lengths)),
            std=float(np.std(all_lengths)),
            min=int(np.min(all_lengths)),
            max=int(np.max(all_lengths)),
        )

    # Feature value range summary per block
    value_ranges: dict[str, BlockValueRange] = {}
    for block in available_blocks:
        X = data.get_block(block)
        value_ranges[block] = BlockValueRange(
            global_mean=float(np.mean(X)),
            global_std=float(np.std(X)),
            global_min=float(np.min(X)),
            global_max=float(np.max(X)),
            feature_mean_range=[
                float(np.min(X.mean(axis=0))),
                float(np.max(X.mean(axis=0))),
            ],
        )

    return IntegrityResult(
        n_samples=data.n_samples,
        n_modes=len(data.unique_modes),
        n_topics=len(data.unique_topics),
        modes=data.unique_modes,
        topics=data.unique_topics,
        samples_per_mode=samples_per_mode,
        samples_per_topic=samples_per_topic,
        balanced=balanced,
        block_dims=block_dims,
        total_features=total_features,
        nan_inf=nan_inf,
        all_clean=all_clean,
        variance_report=variance_report,
        length_by_mode=length_by_mode,
        length_overall=length_overall,
        value_ranges=value_ranges,
    )
