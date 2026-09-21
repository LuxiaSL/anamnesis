"""Calibrate a model: what its states do because of position, and what is left.

Every corrected feature this instrument computes rests on two model-specific
artifacts, and this is where a new model gets them:

* ``positional_means.npz`` — the mean state at each position of each layer, over a
  fixed set of content-diverse prompts. Subtracting it is what removes the part of
  a state that is a function of *where a token is* rather than of how it was
  processed. Without it, a position effect reads as a signature.
* ``pca_model.pkl`` — a basis for the residual stream at the sampled layers, fitted
  over the same prompts.

The PCA is fitted on **positionally corrected** states, at each layer separately.
Fitting on raw states and applying the result to corrected ones is a
fit-and-apply mismatch: the basis then describes a distribution the projection
never sees, and the components it spends on position are wasted. ``--pooled``
asks for the single basis over uncorrected states instead, which is the shape the
earliest banked signatures were computed under and is kept for reproducing them.

One invocation does both artifacts for a new model. Existing positional means are
reused rather than refitted, because the correction a banked signature was
computed under must not move underneath it — a basis can be refitted against the
same means and compared, which is the whole point of being able to refit one.
``--refit-means`` overwrites them, and is the one flag here that invalidates
everything downstream of the directory it writes into.

The prompt set is fixed and lives in this module. It is not a corpus, it is a
ruler: fifty prompts spread across science, history, craft, economics and
mechanics, whose only job is to wash content out of the average so that what
remains is position. Changing it changes every number downstream of it, so it
changes only with a reason recorded beside the bank it produced.
"""

from __future__ import annotations

import argparse
import gc
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from anamnesis.config import MODEL_PRESETS, resolve_preset

logger = logging.getLogger(__name__)

POSITION_COUNT_FLOOR = 5
"""A position whose mean is an average of this many states or fewer is left at
zero: a mean over one or two prompts is not a mean, it is one of the states."""

CALIBRATION_PROMPTS: tuple[str, ...] = (
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
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    p.add_argument("--model-path", default=None, help="Local checkpoint; default: the preset's id")
    p.add_argument(
        "--out-dir", type=Path, default=None, help="Default: the preset's calibration directory"
    )
    p.add_argument("--num-prompts", type=int, default=None, help="Use only the first N prompts")
    p.add_argument("--max-new-tokens", type=int, default=None, help="Default: the preset's budget")
    p.add_argument("--n-components", type=int, default=None, help="Default: the extraction default")
    p.add_argument(
        "--pooled",
        action="store_true",
        help="Fit one basis over uncorrected states, the shape the earliest banks used",
    )
    p.add_argument(
        "--pca-name",
        default=None,
        help="Filename for the basis; default pca_model.pkl for a pooled fit and "
             "pca_model_corrected.pkl for the per-layer corrected one",
    )
    p.add_argument(
        "--refit-means",
        action="store_true",
        help="Recompute the positional means even when they exist, which invalidates every "
             "signature already computed against them",
    )
    p.add_argument("--dry-run", action="store_true", help="Print the configuration and stop")
    return p


def resolve_paths(args: argparse.Namespace) -> tuple[Any, Path, Path]:
    """The preset row, the positional-means path, and the basis path."""
    from anamnesis.config import ExperimentConfig

    preset = resolve_preset(args.model)
    config = ExperimentConfig.from_preset(preset)
    out_dir = args.out_dir or config.calibration.positional_means_path.parent
    default_name = "pca_model.pkl" if args.pooled else "pca_model_corrected.pkl"
    return preset, Path(out_dir) / "positional_means.npz", Path(out_dir) / (
        args.pca_name or default_name
    )


def _generate_states(loaded: Any, preset: Any, prompts: tuple[str, ...], max_new_tokens: int):
    """Yield ``(prompt_length, hidden_states)`` for each calibration prompt.

    Seeded by prompt index so a calibration is reproducible, and a checkpoint with
    no chat template takes the bare prompt — a base model's calibration must match
    the bare prompts its generations will use.
    """
    import torch

    device = next(loaded.model.parameters()).device
    for index, text in enumerate(prompts):
        if loaded.tokenizer.chat_template is None:
            result = loaded.tokenizer(text, return_tensors="pt")["input_ids"]
        else:
            result = loaded.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
        input_ids = (result if torch.is_tensor(result) else result["input_ids"]).to(device)
        torch.manual_seed(index)
        with torch.no_grad():
            out = loaded.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=preset.temperature,
                top_p=0.9,
                do_sample=True,
                eos_token_id=preset.eos_token_ids,
                output_hidden_states=True,
                output_attentions=False,
                return_dict_in_generate=True,
            )
        yield int(input_ids.shape[1]), out.hidden_states
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if (index + 1) % 10 == 0:
            logger.info(f"calibration {index + 1}/{len(prompts)}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args(argv)
    preset, means_path, pca_path = resolve_paths(args)
    prompts = (
        CALIBRATION_PROMPTS
        if args.num_prompts is None
        else CALIBRATION_PROMPTS[: args.num_prompts]
    )
    max_new_tokens = args.max_new_tokens or 512

    if args.dry_run:
        print(f"model: {args.model_path or preset.model_id}")
        print(f"  layers: {preset.num_layers}   hidden: {preset.hidden_dim}")
        print(f"  attention heads: {preset.num_attention_heads}   kv heads: {preset.num_kv_heads}")
        print(f"  dtype: {preset.torch_dtype}   temperature: {preset.temperature}")
        print(f"  sampled layers: {list(preset.sampled_layers)}")
        print(f"  pca layers: {list(preset.pca_layers)}")
        print(f"  stop tokens: {list(preset.eos_token_ids)}")
        print(f"  prompts: {len(prompts)}   tokens each: {max_new_tokens}")
        print(f"  position floor: a mean over more than {POSITION_COUNT_FLOOR} states")
        print(f"  basis fit: {'pooled, uncorrected' if args.pooled else 'per layer, corrected'}")
        reuse = means_path.exists() and not args.refit_means
        print(f"  positional means -> {'reused from' if reuse else 'written to'} {means_path}")
        print(f"  basis -> {pca_path}")
        return

    import torch
    from sklearn.decomposition import PCA

    from anamnesis.config import ExtractionConfig, ModelConfig
    from anamnesis.extraction.calibration import load_positional_means
    from anamnesis.extraction.model_loader import load_model

    means_path.parent.mkdir(parents=True, exist_ok=True)
    n_components = args.n_components or ExtractionConfig.from_preset(preset).pca_components
    pca_layers = list(preset.pca_layers)

    existing_means = (
        None if args.refit_means or not means_path.exists()
        else load_positional_means(means_path.parent)
    )
    if existing_means is not None:
        logger.info(
            f"reusing the positional means at {means_path}: a correction a bank was computed "
            "under does not move; pass --refit-means to overwrite it"
        )

    config = ModelConfig.from_preset(preset, model_id=args.model_path or preset.model_id)
    logger.info(f"calibrating {config.model_id} over {len(prompts)} prompts")
    loaded = load_model(config, sampled_layers=[])
    loaded.disable_hooks()  # calibration reads hidden states from the forward, not from hooks

    depth_plus_embedding = preset.num_layers + 1
    max_positions = max_new_tokens + 200
    position_sums = np.zeros(
        (depth_plus_embedding, max_positions, preset.hidden_dim), dtype=np.float64
    )
    position_counts = np.zeros((depth_plus_embedding, max_positions), dtype=np.int64)
    # A sample is kept with the position it came from, because correcting it needs
    # that position and which correction to apply is not known until the means are.
    samples: list[tuple[int, int, np.ndarray]] = []

    for prompt_length, hidden_states in _generate_states(
        loaded, preset, prompts, max_new_tokens
    ):
        if existing_means is None:
            prefill = hidden_states[0]
            for layer in range(min(len(prefill), depth_plus_embedding)):
                states = prefill[layer][0].cpu().float().numpy()
                for position in range(min(states.shape[0], max_positions)):
                    position_sums[layer, position] += states[position].astype(np.float64)
                    position_counts[layer, position] += 1
            for step in range(1, len(hidden_states)):
                absolute = prompt_length + step - 1
                if absolute >= max_positions:
                    break
                states = hidden_states[step]
                for layer in range(min(len(states), depth_plus_embedding)):
                    vector = states[layer][0, -1].cpu().float().numpy()
                    position_sums[layer, absolute] += vector.astype(np.float64)
                    position_counts[layer, absolute] += 1

        n_steps = len(hidden_states) - 1
        if n_steps <= 0:
            continue
        midpoint = max(1, n_steps // 2)
        # A pooled fit keeps the three sample points as they fall, including when a
        # short generation makes two of them the same step; the per-layer fit takes
        # the distinct ones. Both are the shape their banked artifacts were fitted
        # under, so neither is normalised into the other.
        steps = [1, midpoint, n_steps] if args.pooled else sorted({1, midpoint, n_steps})
        for step in steps:
            if step >= len(hidden_states):
                continue
            absolute = prompt_length + step - 1
            for layer in pca_layers:
                if layer + 1 >= len(hidden_states[step]):
                    continue
                samples.append((
                    layer,
                    absolute,
                    hidden_states[step][layer + 1][0, -1].cpu().float().numpy(),
                ))

    means = existing_means
    if means is None:
        covered = position_counts > POSITION_COUNT_FLOOR
        safe_counts = np.where(covered, position_counts, 1)
        means = np.where(
            covered[:, :, np.newaxis],
            (position_sums / safe_counts[:, :, np.newaxis]).astype(np.float32),
            0.0,
        ).astype(np.float32)
        np.savez_compressed(means_path, positional_means=means, pos_counts=position_counts)
        furthest = (
            int(np.max(np.where(position_counts.sum(axis=0) > 0)))
            if position_counts.sum() > 0
            else 0
        )
        logger.info(f"positional means {means.shape} -> {means_path} (furthest position {furthest})")

    basis: dict[str, Any] | dict[int, dict[str, Any]]
    if args.pooled:
        if not samples:
            raise SystemExit("no basis samples were collected; the residual-PCA features would be empty")
        matrix = np.stack([vector for _, _, vector in samples]).astype(np.float64)
        fitted = PCA(n_components=min(n_components, *matrix.shape)).fit(matrix)
        basis = {
            "components": fitted.components_.astype(np.float32),
            "mean": fitted.mean_.astype(np.float32),
            "explained_variance_ratio": fitted.explained_variance_ratio_,
        }
        logger.info(
            f"pooled basis {fitted.components_.shape} over {matrix.shape[0]} samples, "
            f"explaining {fitted.explained_variance_ratio_.sum():.3f}"
        )
    else:
        basis = {}
        corrected: dict[int, list[np.ndarray]] = {layer: [] for layer in pca_layers}
        for layer, absolute, vector in samples:
            if absolute < means.shape[1]:
                corrected[layer].append(vector - means[layer + 1, absolute])
        for layer in pca_layers:
            if not corrected[layer]:
                raise SystemExit(f"no corrected samples at layer {layer}; the fit would be empty")
            matrix = np.stack(corrected[layer]).astype(np.float64)
            fitted = PCA(n_components=min(n_components, *matrix.shape)).fit(matrix)
            basis[int(layer)] = {
                "components": fitted.components_.astype(np.float32),
                "mean": fitted.mean_.astype(np.float32),
                "explained_variance_ratio": fitted.explained_variance_ratio_,
            }
            logger.info(
                f"layer {layer}: {matrix.shape[0]} corrected samples -> "
                f"{fitted.components_.shape}, explaining "
                f"{fitted.explained_variance_ratio_.sum():.3f}"
            )

    with open(pca_path, "wb") as f:
        pickle.dump(basis, f)
    logger.info(f"basis -> {pca_path}")

    loaded.remove_hooks()
    del loaded
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("calibration complete")


if __name__ == "__main__":
    main()
