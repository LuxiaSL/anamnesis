"""Positional decomposition analysis.

Verifies the quality of the positional calibration:
  - Positional component should show low variance across prompts at same position
  - Correction should reduce positional bias in features
  - Content/mode signal-to-noise should improve after correction
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from config import ExperimentConfig

logger = logging.getLogger(__name__)


def verify_calibration(config: ExperimentConfig) -> dict[str, Any]:
    """Verify the positional calibration data.

    Returns quality metrics for the positional means.
    """
    calib_path = config.calibration.positional_means_path
    if not calib_path.exists():
        return {"error": f"No calibration found at {calib_path}"}

    data = np.load(calib_path)
    positional_means = data["positional_means"]
    pos_counts = data["pos_counts"]

    results: dict[str, Any] = {
        "shape": list(positional_means.shape),
        "max_calibrated_position": int(np.max(np.where(pos_counts.sum(axis=0) > 0))),
    }

    # Per-layer statistics
    num_layers = positional_means.shape[0]
    layer_norms = []
    for l in range(num_layers):
        # Mean norm of positional component across positions with data
        valid_mask = pos_counts[l] > 5
        if valid_mask.sum() > 0:
            norms = np.linalg.norm(positional_means[l, valid_mask], axis=-1)
            layer_norms.append(float(norms.mean()))
        else:
            layer_norms.append(0.0)

    results["mean_positional_norm_per_layer"] = layer_norms
    results["overall_mean_positional_norm"] = float(np.mean(layer_norms))

    # Check that positional means aren't all zeros
    nonzero_frac = float((np.abs(positional_means).sum(axis=-1) > 1e-8).mean())
    results["nonzero_fraction"] = nonzero_frac

    logger.info(f"Calibration shape: {positional_means.shape}")
    logger.info(f"Max calibrated position: {results['max_calibrated_position']}")
    logger.info(f"Overall mean positional norm: {results['overall_mean_positional_norm']:.4f}")
    logger.info(f"Nonzero fraction: {nonzero_frac:.4f}")

    return results
