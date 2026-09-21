"""Section 1: Data integrity and descriptive statistics."""

from __future__ import annotations

import numpy as np

from .signature_io import AnalysisData
from .schemas import (
    IntegrityResult,
    LengthByModeStats,
    LengthOverallStats,
    NanInfCount,
    TierValueRange,
    TierVarianceReport,
)
from .utils import get_available_tiers


def run_integrity_checks(data: AnalysisData) -> IntegrityResult:
    """Run all data integrity and descriptive checks."""
    samples_per_mode = {
        m: int(np.sum(data.mode_mask(m))) for m in data.unique_modes
    }
    samples_per_topic = {
        t: int(np.sum(data.topic_mask(t))) for t in data.unique_topics
    }
    balanced = len(set(samples_per_mode.values())) == 1

    available_tiers, _ = get_available_tiers(data)

    # Feature dimensions
    tier_dims: dict[str, int] = {}
    for tier in available_tiers:
        try:
            tier_dims[tier] = data.get_tier(tier).shape[1]
        except KeyError:
            tier_dims[tier] = 0
    total_features = sum(
        tier_dims.get(t, 0) for t in ["T1", "T2", "T2.5", "T3"]
    )

    # NaN / Inf checks
    nan_inf: dict[str, NanInfCount] = {}
    all_clean = True
    for tier in available_tiers:
        try:
            X = data.get_tier(tier)
            nan_count = int(np.sum(np.isnan(X)))
            inf_count = int(np.sum(np.isinf(X)))
            nan_inf[tier] = NanInfCount(nan=nan_count, inf=inf_count)
            if nan_count > 0 or inf_count > 0:
                all_clean = False
        except KeyError:
            nan_inf[tier] = NanInfCount(nan=-1, inf=-1, error="tier not found")

    # Per-feature variance (detect constant features)
    variance_report: dict[str, TierVarianceReport] = {}
    for tier in available_tiers:
        try:
            X = data.get_tier(tier)
            var = X.var(axis=0)
            variance_report[tier] = TierVarianceReport(
                n_features=int(X.shape[1]),
                n_constant=int(np.sum(var < 1e-12)),
                n_near_constant=int(np.sum(var < 1e-6)),
            )
        except KeyError:
            pass

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

    # Feature value range summary per tier
    value_ranges: dict[str, TierValueRange] = {}
    for tier in available_tiers:
        try:
            X = data.get_tier(tier)
            value_ranges[tier] = TierValueRange(
                global_mean=float(np.mean(X)),
                global_std=float(np.std(X)),
                global_min=float(np.min(X)),
                global_max=float(np.max(X)),
                feature_mean_range=[
                    float(np.min(X.mean(axis=0))),
                    float(np.max(X.mean(axis=0))),
                ],
            )
        except KeyError:
            pass

    return IntegrityResult(
        n_samples=data.n_samples,
        n_modes=len(data.unique_modes),
        n_topics=len(data.unique_topics),
        modes=data.unique_modes,
        topics=data.unique_topics,
        samples_per_mode=samples_per_mode,
        samples_per_topic=samples_per_topic,
        balanced=balanced,
        tier_dims=tier_dims,
        total_features=total_features,
        nan_inf=nan_inf,
        all_clean=all_clean,
        variance_report=variance_report,
        length_by_mode=length_by_mode,
        length_overall=length_overall,
        value_ranges=value_ranges,
    )
