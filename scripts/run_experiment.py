#!/usr/bin/env python3
"""Run the full 135-generation experiment.

Usage:
    python scripts/run_experiment.py                  # Run all 135
    python scripts/run_experiment.py --resume-from 50 # Resume from gen 50
    python scripts/run_experiment.py --set D          # Run only Set D (positive control)
    python scripts/run_experiment.py --dry-run        # Print specs without running
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ExperimentConfig
from extraction.model_loader import load_model
from extraction.generation_runner import build_generation_specs, run_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 0 experiment")
    parser.add_argument("--resume-from", type=int, default=0, help="Resume from this generation ID")
    parser.add_argument("--set", type=str, default=None, help="Run only this prompt set (A/B/C/D)")
    parser.add_argument("--dry-run", action="store_true", help="Print specs without running")
    parser.add_argument("--no-tier3", action="store_true", help="Skip Tier 3 PCA features")
    args = parser.parse_args()

    config = ExperimentConfig()
    config.ensure_dirs()

    # Build specs
    specs = build_generation_specs(config)
    logger.info(f"Total generation specs: {len(specs)}")

    # Filter by set if requested
    if args.set:
        specs = [s for s in specs if s.prompt_set == args.set.upper()]
        logger.info(f"Filtered to Set {args.set.upper()}: {len(specs)} specs")

    if args.dry_run:
        for s in specs:
            print(f"  Gen {s.generation_id:3d}: Set {s.prompt_set} | {s.mode:12s} | {s.topic[:40]:40s} | seed={s.seed}")
        print(f"\nTotal: {len(specs)} generations")
        return

    # Load calibration data
    positional_means = None
    calib_path = config.calibration.positional_means_path
    if calib_path.exists():
        calib = np.load(calib_path)
        positional_means = calib["positional_means"]
        logger.info(f"Loaded positional means: {positional_means.shape}")
    else:
        logger.warning(f"No positional calibration found at {calib_path} — running without correction")

    # Load PCA model for Tier 3
    pca_components = None
    pca_mean = None
    if not args.no_tier3:
        pca_path = config.calibration.pca_model_path
        if pca_path.exists():
            with open(pca_path, "rb") as f:
                pca_data = pickle.load(f)
            pca_components = pca_data["components"]
            pca_mean = pca_data["mean"]
            logger.info(f"Loaded PCA model: {pca_components.shape[0]} components")
        else:
            logger.warning(f"No PCA model found at {pca_path} — Tier 3 will be empty")
            config.extraction.enable_tier3 = False
    else:
        config.extraction.enable_tier3 = False
        logger.info("Tier 3 disabled")

    # Load model
    loaded = load_model(config.model, sampled_layers=config.extraction.sampled_layers)

    # Run
    metadata = run_experiment(
        loaded=loaded,
        config=config,
        positional_means=positional_means,
        pca_components=pca_components,
        pca_mean=pca_mean,
        specs=specs,
        resume_from=args.resume_from,
    )

    logger.info(f"Done! {len(metadata)} generations completed.")

    # Cleanup
    loaded.remove_hooks()


if __name__ == "__main__":
    main()
