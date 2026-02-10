#!/usr/bin/env python3
"""Positional decomposition calibration.

Runs diverse prompts through the model, collects hidden states at all positions,
and computes per-layer, per-position mean hidden states. These are subtracted
during feature extraction to remove positional encoding artifacts.

Also fits PCA on hidden states for Tier 3 features.

Outputs:
  - outputs/positional_means.npz  — positional component estimates
  - outputs/pca_model.pkl         — fitted PCA (components + mean)
"""

from __future__ import annotations

import gc
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ExperimentConfig
from extraction.model_loader import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Diverse calibration prompts — varied content to wash out content-specific signal
CALIBRATION_PROMPTS = [
    "Explain how photosynthesis works in plants.",
    "What are the main causes of the French Revolution?",
    "Describe the process of making traditional Japanese ramen.",
    "How do electric vehicles compare to gasoline cars?",
    "What is the significance of the Rosetta Stone?",
    "Explain the concept of supply and demand in economics.",
    "How does the human immune system fight infections?",
    "Describe the architecture of Gothic cathedrals.",
    "What are the principles of object-oriented programming?",
    "How do tides work and what causes them?",
    "Explain the theory of plate tectonics.",
    "What makes a good leader?",
    "How do birds navigate during migration?",
    "Describe the water cycle and its importance.",
    "What is quantum entanglement?",
    "How do vaccines work?",
    "Explain the causes and effects of inflation.",
    "What are the different types of clouds?",
    "How does a combustion engine work?",
    "Describe the life cycle of a star.",
    "What is machine learning and how does it differ from traditional programming?",
    "How do earthquakes happen?",
    "Explain the basics of music theory.",
    "What are renewable energy sources?",
    "How does the stock market work?",
    "Describe the process of fermentation.",
    "What are the effects of sleep deprivation?",
    "How do submarines work?",
    "Explain the concept of natural selection.",
    "What is the significance of pi in mathematics?",
    "How do 3D printers work?",
    "Describe the history of the internet.",
    "What causes aurora borealis?",
    "How do computers store and retrieve data?",
    "Explain the process of osmosis.",
    "What are the major types of rocks?",
    "How do airplanes fly?",
    "Describe the structure of DNA.",
    "What is cryptocurrency and how does blockchain work?",
    "How do telescopes work?",
    "Explain the greenhouse effect.",
    "What are the stages of grief?",
    "How does sonar work?",
    "Describe the Silk Road and its importance.",
    "What is dark matter?",
    "How do coral reefs form?",
    "Explain the basics of game theory.",
    "What are the layers of the atmosphere?",
    "How does a nuclear reactor work?",
    "Describe the process of cheese making.",
]


def main() -> None:
    config = ExperimentConfig()
    config.ensure_dirs()

    logger.info(f"Calibration with {len(CALIBRATION_PROMPTS)} prompts")

    # Load model without hooks (not needed for calibration)
    loaded = load_model(config.model, sampled_layers=[])
    loaded.disable_hooks()

    num_layers_plus_embed = config.model.num_layers + 1  # 29 for Llama 3.2 3B
    hidden_dim = config.model.hidden_dim
    max_positions = config.generation.max_new_tokens + 200  # generous buffer

    # Accumulators: [num_layers+1, max_positions] sums and counts
    pos_sums = np.zeros((num_layers_plus_embed, max_positions, hidden_dim), dtype=np.float64)
    pos_counts = np.zeros((num_layers_plus_embed, max_positions), dtype=np.int64)

    # Also collect hidden states for PCA fitting
    pca_samples: list[np.ndarray] = []  # list of [hidden_dim] vectors

    for prompt_idx, prompt_text in enumerate(tqdm(CALIBRATION_PROMPTS, desc="Calibration")):
        messages = [{"role": "user", "content": prompt_text}]
        result = loaded.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        )
        # apply_chat_template may return a tensor or a BatchEncoding
        if isinstance(result, torch.Tensor):
            input_ids = result
        else:
            input_ids = result["input_ids"]
        prompt_length = input_ids.shape[1]
        device = next(loaded.model.parameters()).device
        input_ids = input_ids.to(device)

        torch.manual_seed(prompt_idx)

        with torch.no_grad():
            outputs = loaded.model.generate(
                input_ids,
                max_new_tokens=config.calibration.calibration_max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                output_hidden_states=True,
                output_attentions=False,  # not needed for calibration
                return_dict_in_generate=True,
            )

        # Process hidden states
        # Prefill: outputs.hidden_states[0] — tuple of (num_layers+1) tensors
        # Each tensor: [1, prompt_len, hidden_dim]
        prefill = outputs.hidden_states[0]
        for l in range(num_layers_plus_embed):
            h = prefill[l][0].cpu().float().numpy()  # [prompt_len, hidden_dim]
            for pos in range(min(h.shape[0], max_positions)):
                pos_sums[l, pos] += h[pos].astype(np.float64)
                pos_counts[l, pos] += 1

        # Generation steps: outputs.hidden_states[1:]
        for t in range(1, len(outputs.hidden_states)):
            step = outputs.hidden_states[t]
            abs_pos = prompt_length + t - 1
            if abs_pos >= max_positions:
                break
            for l in range(min(len(step), num_layers_plus_embed)):
                h = step[l][0, -1].cpu().float().numpy()  # [hidden_dim]
                pos_sums[l, abs_pos] += h.astype(np.float64)
                pos_counts[l, abs_pos] += 1

        # Collect PCA samples: first, middle, last generated token from mid/late layers
        num_gen_steps = len(outputs.hidden_states) - 1
        if num_gen_steps > 0:
            sample_positions = [1, max(1, num_gen_steps // 2), num_gen_steps]
            for t in sample_positions:
                if t < len(outputs.hidden_states):
                    for l_idx in config.extraction.pca_layers:
                        if l_idx + 1 < len(outputs.hidden_states[t]):
                            h = outputs.hidden_states[t][l_idx + 1][0, -1].cpu().float().numpy()
                            pca_samples.append(h)

        # Cleanup
        del outputs
        del input_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ── Compute positional means ──
    # Only where we have sufficient data (count > 5)
    mask = pos_counts > 5
    positional_means = np.zeros_like(pos_sums, dtype=np.float32)
    for l in range(num_layers_plus_embed):
        for pos in range(max_positions):
            if mask[l, pos]:
                positional_means[l, pos] = (pos_sums[l, pos] / pos_counts[l, pos]).astype(np.float32)

    # Save
    calib_path = config.calibration.positional_means_path
    np.savez_compressed(
        calib_path,
        positional_means=positional_means,
        pos_counts=pos_counts,
    )
    logger.info(f"Positional means saved to {calib_path}")
    logger.info(f"  Shape: {positional_means.shape}")
    logger.info(f"  Max calibrated position: {np.max(np.where(pos_counts.sum(axis=0) > 0))}")

    # ── Fit PCA ──
    if pca_samples:
        from sklearn.decomposition import PCA

        pca_matrix = np.stack(pca_samples).astype(np.float64)
        logger.info(f"Fitting PCA on {pca_matrix.shape[0]} samples of dim {pca_matrix.shape[1]}")

        n_components = min(config.extraction.pca_components, pca_matrix.shape[0], pca_matrix.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(pca_matrix)

        pca_data = {
            "components": pca.components_.astype(np.float32),
            "mean": pca.mean_.astype(np.float32),
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }

        pca_path = config.calibration.pca_model_path
        with open(pca_path, "wb") as f:
            pickle.dump(pca_data, f)

        logger.info(f"PCA model saved to {pca_path}")
        logger.info(f"  Components: {pca.components_.shape}")
        logger.info(f"  Explained variance (top 10): {pca.explained_variance_ratio_[:10]}")
        logger.info(f"  Total explained: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        logger.warning("No PCA samples collected — Tier 3 features will be empty")

    logger.info("Calibration complete!")


if __name__ == "__main__":
    main()
