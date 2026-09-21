"""Extract a run in one process: generate, collect, featurise, save.

The whole instrument in one pass, and the shortest path to a signature. A model is
loaded with its hooks, a spec list is built from a mode set and the topic sets, and
each generation is sampled and featurised as it is produced. Resume is per
generation: a killed run re-run under the same name continues.

``--modes run4`` is the five format-controlled modes, ``--modes mixed`` the
eight-mode set. ``--include-prompt-swap`` adds the confound condition — mode A's
system prompt under a directive that forces mode B's execution — which is how a
signature that tracked the instruction rather than the execution would be caught.
``--save-raw`` banks the per-token tensors beside the vectors, which is what makes
``run_recompute.py`` possible afterwards and is worth the disk on any run whose
features are not final.

For a corpus large enough to want more than one device, generate and replay
separately instead: ``run_gen_tokens.py`` banks the token ids and
``run_replay.py`` featurises them, which also means a later change to the capture
surface costs a replay rather than a re-generation.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

from anamnesis.config import MODEL_PRESETS

logger = logging.getLogger(__name__)

MODE_SETS = ("run4", "mixed")
SMOKE_MAX_NEW_TOKENS = 100
"""Token budget for a smoke pass — enough to exercise every hook and short enough
to finish while someone watches."""


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    p.add_argument(
        "--modes", choices=list(MODE_SETS), default="run4",
        help="run4: the five format-controlled modes; mixed: the eight-mode set",
    )
    p.add_argument("--n-samples", type=int, default=20, help="Generations per mode")
    p.add_argument("--run-name", required=True, help="Names the run directory under the outputs root")
    p.add_argument("--save-raw", action="store_true", help="Bank per-token tensors beside the vectors")
    p.add_argument(
        "--include-prompt-swap", action="store_true", help="Add the prompt-swap confound condition"
    )
    p.add_argument("--no-pca", action="store_true", help="Skip the residual-PCA features")
    p.add_argument("--smoke-test", action="store_true", help="One sample per mode, short generations")
    p.add_argument("--dry-run", action="store_true", help="Print the pass and stop")
    return p


def mode_prompts(mode_set: str) -> dict[str, str]:
    """The system prompts one mode set names, in its own order.

    A mode label in banked data means the exact prompt text here, so the set is
    read from the modes package rather than restated.
    """
    from anamnesis.modes.extended_modes import EXTENDED_MODES
    from anamnesis.modes.run4_modes import RUN4_MODES

    if mode_set == "run4":
        return dict(RUN4_MODES)
    if mode_set == "mixed":
        return dict(EXTENDED_MODES)
    raise ValueError(f"unknown mode set {mode_set!r}; choose one of {MODE_SETS}")


def build_config(args: argparse.Namespace) -> Any:
    """The pass, derived from one preset row so its sections cannot disagree."""
    from anamnesis.config import ExperimentConfig

    n_samples = 1 if args.smoke_test else args.n_samples
    return ExperimentConfig.from_preset(
        args.model,
        run_name=args.run_name,
        extraction_overrides={
            "enable_residual_pca": not args.no_pca,
            "save_raw_tensors": args.save_raw,
        },
        generation_overrides=(
            {"max_new_tokens": SMOKE_MAX_NEW_TOKENS} if args.smoke_test else {}
        ),
    ), n_samples


def build_specs(args: argparse.Namespace, config: Any, n_samples: int) -> list[Any]:
    """Every generation this pass will run, standard conditions then swaps."""
    from anamnesis.extraction.generation_runner import (
        build_generation_specs,
        build_prompt_swap_specs,
        trim_per_mode,
    )

    import json

    prompts = mode_prompts(args.modes)
    prompt_set = f"{args.model.upper()}_{args.modes}"
    sets = json.loads(config.prompts_path.read_text())["topics"]
    n_topics = len(sets["set_a"]) + len(sets["set_b"])
    # Repetitions cover whole topic sets; the per-mode count is then trimmed to
    # exactly what was asked for, so every mode covers the same leading topics.
    reps = max(1, -(-n_samples // max(1, n_topics)))
    specs = trim_per_mode(
        build_generation_specs(
            config, mode_dict=prompts, num_reps=reps, prompt_set=prompt_set
        ),
        n_samples,
    )
    if args.include_prompt_swap:
        specs += build_prompt_swap_specs(config, prompt_set=prompt_set)
    return specs


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args(argv)
    config, n_samples = build_config(args)
    specs = build_specs(args, config, n_samples)

    counts: dict[str, int] = {}
    for spec in specs:
        counts[spec.mode] = counts.get(spec.mode, 0) + 1
    print(f"Extraction run: {args.run_name}")
    print(f"  model: {config.model.model_id}")
    print(f"  modes: {args.modes} ({len(counts)} conditions)")
    for mode in sorted(counts):
        print(f"    {mode}: {counts[mode]}")
    print(f"  total: {len(specs)} generations")
    print(f"  raw tensors: {args.save_raw}")
    print(f"  output: {config.outputs_dir}")
    if args.dry_run:
        return

    from anamnesis.extraction.calibration import load_calibration
    from anamnesis.extraction.generation_runner import run_experiment
    from anamnesis.extraction.model_loader import load_model

    config.ensure_dirs()
    positional_means, pca_components, pca_mean = load_calibration(
        config.calibration.positional_means_path.parent, enable_pca=not args.no_pca
    )
    loaded = load_model(
        config.model,
        sampled_layers=config.extraction.sampled_layers,
        # Gate capture is what a raw bank is for; a pass that is not banking raw
        # has nothing to do with the gates it would collect.
        register_gate_hooks=args.save_raw,
    )
    depth = len(loaded.model.model.layers)
    if depth != config.model.num_layers:
        raise SystemExit(
            f"the checkpoint has {depth} decoder layers but the preset says "
            f"{config.model.num_layers}; the layer plan would index the wrong sites"
        )
    try:
        metadata = run_experiment(
            loaded=loaded,
            config=config,
            positional_means=positional_means,
            pca_components=pca_components,
            pca_mean=pca_mean,
            specs=specs,
            save_raw=args.save_raw,
        )
    finally:
        loaded.remove_hooks()
    logger.info(f"extraction complete: {len(metadata)} generations under {config.outputs_dir}")


if __name__ == "__main__":
    main()
