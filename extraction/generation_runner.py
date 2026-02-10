"""Orchestrates: prompt formatting → seed → generate → extract → save → cleanup.

This module ties together model_loader and state_extractor. It handles:
  - Building chat-formatted prompts via tokenizer.apply_chat_template()
  - Setting per-generation random seeds
  - Running model.generate() with correct flags
  - Converting torch outputs to numpy for state_extractor
  - Saving .npz + .json per generation
  - GPU memory cleanup between generations
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm import tqdm

from config import (
    ExperimentConfig,
    GenerationSpec,
    PROCESSING_MODES,
    PROMPTS_PATH,
)
from extraction.model_loader import LoadedModel
from extraction.state_extractor import (
    ExtractionResult,
    RawGenerationData,
    extract_all_features,
)

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]


def make_seed(topic_idx: int, mode_idx: int, rep_idx: int = 0) -> int:
    """Deterministic seed from generation coordinates."""
    raw = f"{topic_idx}_{mode_idx}_{rep_idx}"
    return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)


def format_prompt(
    loaded: LoadedModel,
    system_prompt: str,
    user_prompt: str,
) -> tuple[torch.Tensor, int]:
    """Format prompt using chat template, return (input_ids, prompt_length).

    Returns:
        input_ids: [1, seq_len] tensor on model device
        prompt_length: number of tokens in the formatted prompt
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    result = loaded.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # apply_chat_template may return a tensor directly or a BatchEncoding
    if isinstance(result, torch.Tensor):
        input_ids = result
    else:
        input_ids = result["input_ids"]
    prompt_length = input_ids.shape[1]
    device = next(loaded.model.parameters()).device
    input_ids = input_ids.to(device)
    return input_ids, prompt_length


def run_single_generation(
    loaded: LoadedModel,
    spec: GenerationSpec,
    config: ExperimentConfig,
    positional_means: F32 | None = None,
    pca_components: F32 | None = None,
    pca_mean: F32 | None = None,
) -> tuple[ExtractionResult, dict[str, Any]]:
    """Run a single generation and extract features.

    Returns:
        (extraction_result, metadata_dict)
    """
    t_start = time.time()

    # Set seed
    torch.manual_seed(spec.seed)
    torch.cuda.manual_seed_all(spec.seed)
    np.random.seed(spec.seed % (2**32))

    # Format prompt
    input_ids, prompt_length = format_prompt(
        loaded, spec.system_prompt, spec.user_prompt,
    )

    # Clear hook state from previous generation
    loaded.clear_hook_state()
    loaded.enable_hooks()

    # Generate
    gen_config = config.generation
    with torch.no_grad():
        outputs = loaded.model.generate(
            input_ids,
            max_new_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            top_p=gen_config.top_p,
            do_sample=gen_config.do_sample,
            output_hidden_states=gen_config.output_hidden_states,
            output_attentions=gen_config.output_attentions,
            output_logits=gen_config.output_logits,
            return_dict_in_generate=gen_config.return_dict_in_generate,
        )

    t_generate = time.time()

    # Convert outputs to numpy for state_extractor
    raw_data = _convert_outputs_to_raw(
        outputs=outputs,
        prompt_length=prompt_length,
        hook_state=loaded.hook_state,
        sampled_layers=config.extraction.sampled_layers,
        positional_means=positional_means,
    )

    # Extract features
    result = extract_all_features(
        raw_data,
        config.extraction,
        pca_components=pca_components,
        pca_mean=pca_mean,
    )

    t_extract = time.time()

    # Decode generated text
    generated_ids = outputs.sequences[0, prompt_length:]
    generated_text = loaded.tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Build metadata
    metadata = {
        "generation_id": spec.generation_id,
        "prompt_set": spec.prompt_set,
        "topic": spec.topic,
        "topic_idx": spec.topic_idx,
        "mode": spec.mode,
        "mode_idx": spec.mode_idx,
        "system_prompt": spec.system_prompt,
        "user_prompt": spec.user_prompt,
        "seed": spec.seed,
        "repetition": spec.repetition,
        "generated_text": generated_text,
        "num_generated_tokens": len(generated_ids),
        "prompt_length": prompt_length,
        "num_features": len(result.features),
        "tier_slices": result.tier_slices,
        "timing": {
            "generation_seconds": round(t_generate - t_start, 2),
            "extraction_seconds": round(t_extract - t_generate, 2),
            "total_seconds": round(t_extract - t_start, 2),
        },
    }

    # Cleanup GPU memory — must delete in caller scope, not via helper
    del outputs
    del input_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return result, metadata


def _convert_outputs_to_raw(
    outputs: Any,
    prompt_length: int,
    hook_state: Any,
    sampled_layers: list[int],
    positional_means: F32 | None,
) -> RawGenerationData:
    """Convert torch generation outputs to numpy arrays for state_extractor.

    Key indexing notes:
      - outputs.hidden_states[t][l]: t=generation step, l=layer (0=embedding, 1..N=transformer)
      - outputs.hidden_states[0] is the prefill step (shape [1, prompt_len, hidden_dim])
      - Generation steps start at index 1, each [1, 1, hidden_dim]
      - outputs.attentions[t][l]: [1, num_heads, 1, current_seq_len] at generation steps
      - outputs.logits[t]: [1, vocab_size] at each generation step (0-indexed, no prefill)
    """
    # Number of generation steps (exclude prefill)
    num_gen_steps = len(outputs.hidden_states) - 1

    # Hidden states: for each generation step, stack all layers
    hidden_states_list: list[F32] = []
    for t in range(1, len(outputs.hidden_states)):
        # outputs.hidden_states[t] is a tuple of (num_layers+1) tensors
        # Each tensor: [1, 1, hidden_dim] for generation steps
        layers = []
        for l_tensor in outputs.hidden_states[t]:
            # Take the single token position: [1, 1, hidden_dim] → [hidden_dim]
            layers.append(l_tensor[0, -1].cpu().float().numpy())
        hidden_states_list.append(np.stack(layers))  # [num_layers+1, hidden_dim]

    # Attentions: for each generation step
    attentions_list: list[F32] = []
    for t in range(len(outputs.attentions)):
        # outputs.attentions[t] is a tuple of num_layers tensors
        # Each: [1, num_heads, 1, current_seq_len] → [num_heads, current_seq_len]
        layers = []
        for l_tensor in outputs.attentions[t]:
            layers.append(l_tensor[0, :, -1, :].cpu().float().numpy())
        attentions_list.append(np.stack(layers))  # [num_layers, num_heads, seq_len]

    # Logits: outputs.logits is a tuple of T tensors, each [1, vocab_size]
    logits_list: list[F32] = []
    for t in range(len(outputs.logits)):
        logits_list.append(outputs.logits[t][0].cpu().float().numpy())

    # Chosen token IDs (the generated sequence, excluding prompt)
    chosen_ids = outputs.sequences[0, prompt_length:].cpu().numpy().astype(np.float32)

    # Pre-RoPE keys from hooks
    pre_rope_keys: dict[int, list[F32]] = {}
    for l_idx in sampled_layers:
        gen_keys = hook_state.get_generation_keys(l_idx)
        if gen_keys:
            pre_rope_keys[l_idx] = [
                k[0].numpy().astype(np.float32)  # [num_kv_heads, 1, head_dim] → [num_kv_heads, head_dim]
                for k in gen_keys
            ]
            # Squeeze the seq_len=1 dimension
            pre_rope_keys[l_idx] = [
                k[:, 0, :] if k.ndim == 3 else k
                for k in pre_rope_keys[l_idx]
            ]

    return RawGenerationData(
        hidden_states=hidden_states_list,
        attentions=attentions_list,
        logits=logits_list,
        chosen_token_ids=chosen_ids,
        pre_rope_keys=pre_rope_keys,
        prompt_length=prompt_length,
        positional_means=positional_means,
    )


def _cleanup_gpu(outputs: Any, input_ids: torch.Tensor) -> None:
    """Release GPU memory after a generation."""
    del outputs
    del input_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def save_generation(
    gen_id: int,
    result: ExtractionResult,
    metadata: dict[str, Any],
    signatures_dir: Path,
) -> tuple[Path, Path]:
    """Save extraction result and metadata to disk.

    Returns (npz_path, json_path).
    """
    npz_path = signatures_dir / f"gen_{gen_id:03d}.npz"
    json_path = signatures_dir / f"gen_{gen_id:03d}.json"

    # Save features
    save_dict: dict[str, Any] = {
        "features": result.features,
        "feature_names": np.array(result.feature_names),
    }
    if result.knnlm_baseline is not None:
        save_dict["knnlm_baseline"] = result.knnlm_baseline

    # Save per-tier slices as separate arrays for convenience
    for tier_name, (start, end) in result.tier_slices.items():
        save_dict[f"features_{tier_name}"] = result.features[start:end]

    np.savez_compressed(npz_path, **save_dict)

    # Save metadata
    # Convert tuple slices to lists for JSON serialization
    meta_copy = metadata.copy()
    meta_copy["tier_slices"] = {k: list(v) for k, v in metadata["tier_slices"].items()}
    with open(json_path, "w") as f:
        json.dump(meta_copy, f, indent=2, default=str)

    return npz_path, json_path


def build_generation_specs(config: ExperimentConfig) -> list[GenerationSpec]:
    """Build all 135 generation specs from prompt_sets.json."""
    with open(config.prompts_path) as f:
        prompts = json.load(f)

    specs: list[GenerationSpec] = []
    gen_id = 0
    modes = list(PROCESSING_MODES.keys())
    template = prompts.get("user_prompt_template", "Write about: {topic}")

    # Set A: 10 topics × 5 modes = 50
    for topic_idx, topic in enumerate(prompts["topics"]["set_a"]):
        for mode_idx, mode in enumerate(modes):
            specs.append(GenerationSpec(
                generation_id=gen_id,
                prompt_set="A",
                topic=topic,
                topic_idx=topic_idx,
                mode=mode,
                mode_idx=mode_idx,
                system_prompt=PROCESSING_MODES[mode],
                user_prompt=template.format(topic=topic),
                seed=make_seed(topic_idx, mode_idx),
            ))
            gen_id += 1

    # Set B: 5 topics × 5 modes = 25
    for b_idx, topic in enumerate(prompts["topics"]["set_b"]):
        topic_idx = 10 + b_idx  # continue numbering after Set A
        for mode_idx, mode in enumerate(modes):
            specs.append(GenerationSpec(
                generation_id=gen_id,
                prompt_set="B",
                topic=topic,
                topic_idx=topic_idx,
                mode=mode,
                mode_idx=mode_idx,
                system_prompt=PROCESSING_MODES[mode],
                user_prompt=template.format(topic=topic),
                seed=make_seed(topic_idx, mode_idx),
            ))
            gen_id += 1

    # Set C: 5 pairs × 10 reps = 50
    for pair in prompts["noise_floor_pairs"]:
        topic_idx = pair["topic_idx"]
        topic = prompts["topics"]["set_a"][topic_idx]
        mode = pair["mode"]
        mode_idx = modes.index(mode)
        for rep in range(pair["repetitions"]):
            specs.append(GenerationSpec(
                generation_id=gen_id,
                prompt_set="C",
                topic=topic,
                topic_idx=topic_idx,
                mode=mode,
                mode_idx=mode_idx,
                system_prompt=PROCESSING_MODES[mode],
                user_prompt=template.format(topic=topic),
                seed=make_seed(topic_idx, mode_idx, rep),
                repetition=rep,
            ))
            gen_id += 1

    # Set D: 5 knows + 5 doesn't know = 10
    d_mode = prompts["positive_control"]["mode"]
    d_mode_idx = modes.index(d_mode)
    for q_idx, question in enumerate(prompts["positive_control"]["knows"]):
        specs.append(GenerationSpec(
            generation_id=gen_id,
            prompt_set="D",
            topic=f"knows_{q_idx}",
            topic_idx=100 + q_idx,  # distinct namespace
            mode=d_mode,
            mode_idx=d_mode_idx,
            system_prompt=PROCESSING_MODES[d_mode],
            user_prompt=question,
            seed=make_seed(100 + q_idx, d_mode_idx),
        ))
        gen_id += 1

    for q_idx, question in enumerate(prompts["positive_control"]["doesnt_know"]):
        specs.append(GenerationSpec(
            generation_id=gen_id,
            prompt_set="D",
            topic=f"doesnt_know_{q_idx}",
            topic_idx=200 + q_idx,
            mode=d_mode,
            mode_idx=d_mode_idx,
            system_prompt=PROCESSING_MODES[d_mode],
            user_prompt=question,
            seed=make_seed(200 + q_idx, d_mode_idx),
        ))
        gen_id += 1

    return specs


def run_experiment(
    loaded: LoadedModel,
    config: ExperimentConfig,
    positional_means: F32 | None = None,
    pca_components: F32 | None = None,
    pca_mean: F32 | None = None,
    specs: list[GenerationSpec] | None = None,
    resume_from: int = 0,
) -> list[dict[str, Any]]:
    """Run the full experiment (or a subset of specs).

    Args:
        loaded: Loaded model with hooks.
        config: Experiment configuration.
        positional_means: Pre-computed positional means for correction.
        pca_components: Pre-fitted PCA components for Tier 3.
        pca_mean: PCA mean vector.
        specs: Generation specs to run. If None, builds from prompt_sets.json.
        resume_from: Skip generations with ID < this value.

    Returns:
        List of metadata dicts for all completed generations.
    """
    config.ensure_dirs()

    if specs is None:
        specs = build_generation_specs(config)

    all_metadata: list[dict[str, Any]] = []
    failed_ids: list[int] = []

    # Filter for resume
    specs_to_run = [s for s in specs if s.generation_id >= resume_from]
    logger.info(f"Running {len(specs_to_run)} generations (total specs: {len(specs)})")

    for spec in tqdm(specs_to_run, desc="Generating"):
        try:
            result, metadata = run_single_generation(
                loaded=loaded,
                spec=spec,
                config=config,
                positional_means=positional_means,
                pca_components=pca_components,
                pca_mean=pca_mean,
            )

            save_generation(
                gen_id=spec.generation_id,
                result=result,
                metadata=metadata,
                signatures_dir=config.signatures_dir,
            )

            all_metadata.append(metadata)

            if spec.generation_id % 10 == 0:
                logger.info(
                    f"Gen {spec.generation_id}: {spec.prompt_set}/{spec.mode}/{spec.topic[:30]} "
                    f"— {metadata['num_generated_tokens']} tokens, "
                    f"{metadata['timing']['total_seconds']}s, "
                    f"{metadata['num_features']} features"
                )

        except Exception as e:
            logger.error(f"Generation {spec.generation_id} failed: {e}", exc_info=True)
            failed_ids.append(spec.generation_id)
            # Best-effort GPU cleanup on failure
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue

    # Save master metadata index
    metadata_path = config.metadata_path
    with open(metadata_path, "w") as f:
        json.dump({
            "total_generations": len(all_metadata),
            "failed_ids": failed_ids,
            "model": config.model.model_dump(),
            "generation_config": config.generation.model_dump(),
            "extraction_config": config.extraction.model_dump(),
            "generations": all_metadata,
        }, f, indent=2, default=str)

    logger.info(
        f"Experiment complete: {len(all_metadata)} succeeded, "
        f"{len(failed_ids)} failed. Metadata saved to {metadata_path}"
    )

    return all_metadata
