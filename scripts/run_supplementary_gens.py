#!/usr/bin/env python3
"""Generate supplementary data for Phase 0 closing experiments.

Produces 30 generations in 3 conditions:
  - temp_low (10):   linear mode, temperature=0.3, Set A topics 0-9
  - temp_high (10):  linear mode, temperature=0.9, Set A topics 0-9
  - prompt_swap (10): socratic system prompt + linear user directive, temp=0.7

Saves to outputs/runs/<run>/supplementary/ with gen IDs 1000-1029
to avoid collision with main Run 4 IDs (0-159).

Usage (on RunPod):
    export ANAMNESIS_RUN_NAME=run4_format_controlled
    python scripts/run_supplementary_gens.py
    python scripts/run_supplementary_gens.py --dry-run  # Print specs only
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CALIBRATION_DIR,
    ExperimentConfig,
    GenerationSpec,
    OUTPUTS_DIR,
    PROCESSING_MODES,
    PROMPTS_PATH,
)
from extraction.generation_runner import make_seed, run_single_generation, save_generation
from extraction.model_loader import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Generation ID ranges (no collision with Run 4's 0-159)
TEMP_LOW_START = 1000
TEMP_HIGH_START = 1010
PROMPT_SWAP_START = 1020


def save_generation_4digit(
    gen_id: int,
    result: Any,
    metadata: dict[str, Any],
    signatures_dir: Path,
) -> tuple[Path, Path]:
    """Save generation with 4-digit zero-padded filenames for IDs >= 1000."""
    npz_path = signatures_dir / f"gen_{gen_id:04d}.npz"
    json_path = signatures_dir / f"gen_{gen_id:04d}.json"

    save_dict: dict[str, Any] = {
        "features": result.features,
        "feature_names": np.array(result.feature_names),
    }
    if result.knnlm_baseline is not None:
        save_dict["knnlm_baseline"] = result.knnlm_baseline

    for tier_name, (start, end) in result.tier_slices.items():
        save_dict[f"features_{tier_name}"] = result.features[start:end]

    np.savez_compressed(npz_path, **save_dict)

    meta_copy = metadata.copy()
    meta_copy["tier_slices"] = {k: list(v) for k, v in metadata["tier_slices"].items()}
    with open(json_path, "w") as f:
        json.dump(meta_copy, f, indent=2, default=str)

    return npz_path, json_path


def build_supplementary_specs() -> list[tuple[GenerationSpec, str, float]]:
    """Build generation specs for all 30 supplementary generations.

    Returns:
        List of (spec, condition_name, temperature) tuples.
    """
    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)

    set_a_topics = prompts["topics"]["set_a"]  # 10 topics
    template = prompts.get("user_prompt_template", "Write about: {topic}")

    specs: list[tuple[GenerationSpec, str, float]] = []

    # Condition 1: temp_low — linear mode, temperature 0.3
    for i, topic in enumerate(set_a_topics):
        specs.append((
            GenerationSpec(
                generation_id=TEMP_LOW_START + i,
                prompt_set="S",  # Supplementary
                topic=topic,
                topic_idx=i,
                mode="linear",
                mode_idx=0,
                system_prompt=PROCESSING_MODES["linear"],
                user_prompt=template.format(topic=topic),
                seed=make_seed(300 + i, 0, prompt_set="S"),
            ),
            "temp_low",
            0.3,
        ))

    # Condition 2: temp_high — linear mode, temperature 0.9
    for i, topic in enumerate(set_a_topics):
        specs.append((
            GenerationSpec(
                generation_id=TEMP_HIGH_START + i,
                prompt_set="S",
                topic=topic,
                topic_idx=i,
                mode="linear",
                mode_idx=0,
                system_prompt=PROCESSING_MODES["linear"],
                user_prompt=template.format(topic=topic),
                seed=make_seed(400 + i, 0, prompt_set="S"),
            ),
            "temp_high",
            0.9,
        ))

    # Condition 3: prompt_swap — socratic system prompt, linear user directive
    swap_directive = (
        "Write your response as straightforward sequential exposition. "
        "Do not ask questions or use Socratic devices.\n\n"
    )
    for i, topic in enumerate(set_a_topics):
        user_prompt = swap_directive + template.format(topic=topic)
        specs.append((
            GenerationSpec(
                generation_id=PROMPT_SWAP_START + i,
                prompt_set="S",
                topic=topic,
                topic_idx=i,
                mode="socratic",  # system prompt is socratic
                mode_idx=2,
                system_prompt=PROCESSING_MODES["socratic"],
                user_prompt=user_prompt,
                seed=make_seed(500 + i, 0, prompt_set="S"),
            ),
            "prompt_swap",
            0.7,  # default temperature
        ))

    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run supplementary generations")
    parser.add_argument("--dry-run", action="store_true", help="Print specs without running")
    args = parser.parse_args()

    config = ExperimentConfig()
    supplementary_dir = OUTPUTS_DIR / "supplementary"
    supplementary_dir.mkdir(parents=True, exist_ok=True)

    specs = build_supplementary_specs()
    logger.info(f"Total supplementary specs: {len(specs)}")

    if args.dry_run:
        for spec, condition, temp in specs:
            print(
                f"  Gen {spec.generation_id:4d}: {condition:12s} | "
                f"temp={temp} | {spec.mode:10s} | {spec.topic[:40]:40s} | "
                f"seed={spec.seed}"
            )
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
        logger.warning(f"No positional calibration at {calib_path}")

    # Load PCA model
    pca_components = None
    pca_mean = None
    pca_path = config.calibration.pca_model_path
    if pca_path.exists():
        with open(pca_path, "rb") as f:
            pca_data = pickle.load(f)
        pca_components = pca_data["components"]
        pca_mean = pca_data["mean"]
        logger.info(f"Loaded PCA model: {pca_components.shape[0]} components")
    else:
        logger.warning(f"No PCA model at {pca_path} — Tier 3 will be empty")
        config.extraction.enable_tier3 = False

    # Load model
    loaded = load_model(config.model, sampled_layers=config.extraction.sampled_layers)

    # Store original temperature to restore between conditions
    original_temp = config.generation.temperature
    all_metadata: list[dict] = []
    failed_ids: list[int] = []

    t_start = time.time()

    for idx, (spec, condition, temperature) in enumerate(specs):
        try:
            # Set temperature for this condition
            config.generation.temperature = temperature

            result, metadata = run_single_generation(
                loaded=loaded,
                spec=spec,
                config=config,
                positional_means=positional_means,
                pca_components=pca_components,
                pca_mean=pca_mean,
            )

            # Add supplementary-specific metadata
            metadata["condition"] = condition
            metadata["temperature_actual"] = temperature

            # Save with 4-digit IDs
            save_generation_4digit(
                gen_id=spec.generation_id,
                result=result,
                metadata=metadata,
                signatures_dir=supplementary_dir,
            )

            all_metadata.append(metadata)

            logger.info(
                f"[{idx + 1}/{len(specs)}] Gen {spec.generation_id}: "
                f"{condition} | temp={temperature} | {spec.topic[:30]} | "
                f"{metadata['num_generated_tokens']} tokens | "
                f"{metadata['timing']['total_seconds']}s"
            )

        except Exception as e:
            logger.error(f"Generation {spec.generation_id} failed: {e}", exc_info=True)
            failed_ids.append(spec.generation_id)
            import gc
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # Restore original temperature
    config.generation.temperature = original_temp

    elapsed = time.time() - t_start

    # Save master metadata
    supp_metadata_path = OUTPUTS_DIR / "supplementary_metadata.json"
    with open(supp_metadata_path, "w") as f:
        json.dump({
            "total_generations": len(all_metadata),
            "failed_ids": failed_ids,
            "conditions": {
                "temp_low": {
                    "temperature": 0.3,
                    "mode": "linear",
                    "gen_ids": list(range(TEMP_LOW_START, TEMP_LOW_START + 10)),
                },
                "temp_high": {
                    "temperature": 0.9,
                    "mode": "linear",
                    "gen_ids": list(range(TEMP_HIGH_START, TEMP_HIGH_START + 10)),
                },
                "prompt_swap": {
                    "temperature": 0.7,
                    "system_mode": "socratic",
                    "user_directive": "linear override",
                    "gen_ids": list(range(PROMPT_SWAP_START, PROMPT_SWAP_START + 10)),
                },
            },
            "elapsed_seconds": round(elapsed, 1),
            "generations": all_metadata,
        }, f, indent=2, default=str)

    logger.info(
        f"Done! {len(all_metadata)} succeeded, {len(failed_ids)} failed. "
        f"Elapsed: {elapsed:.0f}s. Metadata: {supp_metadata_path}"
    )

    # Cleanup
    loaded.remove_hooks()

    # Print summary
    print("\n" + "=" * 60)
    print("SUPPLEMENTARY GENERATIONS COMPLETE")
    print("=" * 60)
    print(f"Total: {len(all_metadata)}/{len(specs)} succeeded")
    if failed_ids:
        print(f"Failed: {failed_ids}")
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"Output dir: {supplementary_dir}")
    print(f"Metadata: {supp_metadata_path}")


if __name__ == "__main__":
    main()
