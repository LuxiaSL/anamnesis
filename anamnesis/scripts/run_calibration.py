"""Calibrate a model: measure its positional means and fit its residual basis.

One invocation does both artifacts for a new model, at the decode policy the
model's preset row names. The fitting itself, the prompt set that serves as the
ruler, and why each artifact has the shape it does are all in
:mod:`anamnesis.extraction.calibration_fit`; this command resolves a model name to
paths, loads the checkpoint, and writes what the fit returns.

Two refusals guard artifacts that others already rest on, because either one moving
silently invalidates every signature computed against it:

* existing positional means are reused rather than measured again, and
  ``--refit-means`` is what overwrites them;
* an existing basis is not overwritten, and ``--refit-basis`` is what replaces it.
  A directory's basis is also the one every consumer of that directory projects
  onto, so a fit of the other shape landing on it changes features without
  changing any argument.
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

from anamnesis.config import MODEL_PRESETS, GenerationConfig, ModelPreset, resolve_preset
from anamnesis.extraction import calibration_fit
from anamnesis.extraction.calibration import PCA_MODEL_NAME, POSITIONAL_MEANS_NAME

logger = logging.getLogger(__name__)


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
        help=f"Filename for the basis; default {PCA_MODEL_NAME}, which is the name every "
             f"consumer of a calibration directory reads",
    )
    p.add_argument(
        "--refit-means",
        action="store_true",
        help="Recompute the positional means even when they exist, which invalidates every "
             "signature already computed against them",
    )
    p.add_argument(
        "--refit-basis",
        action="store_true",
        help="Replace an existing basis at the target filename",
    )
    p.add_argument("--dry-run", action="store_true", help="Print the configuration and stop")
    return p


def resolve_paths(args: argparse.Namespace) -> tuple[ModelPreset, Path, Path]:
    """The preset row, the positional-means path, and the basis path."""
    from anamnesis.config import ExperimentConfig

    preset = resolve_preset(args.model)
    config = ExperimentConfig.from_preset(preset)
    out_dir = Path(args.out_dir or config.calibration.positional_means_path.parent)
    return (
        preset,
        out_dir / POSITIONAL_MEANS_NAME,
        out_dir / (args.pca_name or PCA_MODEL_NAME),
    )


def describe(
    args: argparse.Namespace,
    preset: ModelPreset,
    settings: GenerationConfig,
    prompts: tuple[str, ...],
    means_path: Path,
    basis_path: Path,
) -> None:
    """Print what this invocation would do, for ``--dry-run``."""
    print(f"model: {args.model_path or preset.model_id}")
    print(f"  layers: {preset.num_layers}   hidden: {preset.hidden_dim}")
    print(f"  attention heads: {preset.num_attention_heads}   kv heads: {preset.num_kv_heads}")
    print(f"  dtype: {preset.torch_dtype}")
    print(f"  sampled layers: {list(preset.sampled_layers)}")
    print(f"  pca layers: {list(preset.pca_layers)}")
    print(f"  stop tokens: {list(settings.eos_token_ids)}")
    print(
        f"  decode: temperature {settings.temperature}   top_p {settings.top_p}   "
        f"tokens {settings.max_new_tokens}"
    )
    print(f"  prompts: {len(prompts)}")
    print(
        f"  position floor: a mean over more than "
        f"{calibration_fit.POSITION_COUNT_FLOOR} states"
    )
    print(f"  basis fit: {'pooled, uncorrected' if args.pooled else 'per layer, corrected'}")
    reuse = means_path.exists() and not args.refit_means
    print(f"  positional means -> {'reused from' if reuse else 'written to'} {means_path}")
    held = basis_path.exists() and not args.refit_basis
    print(f"  basis -> {basis_path}{'  (already there; --refit-basis replaces it)' if held else ''}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args(argv)
    preset, means_path, basis_path = resolve_paths(args)
    prompts = (
        calibration_fit.CALIBRATION_PROMPTS
        if args.num_prompts is None
        else calibration_fit.CALIBRATION_PROMPTS[: args.num_prompts]
    )
    settings = calibration_fit.generation_settings(
        preset, max_new_tokens=args.max_new_tokens
    )

    if args.dry_run:
        describe(args, preset, settings, prompts, means_path, basis_path)
        return

    if basis_path.exists() and not args.refit_basis:
        raise SystemExit(
            f"a basis is already at {basis_path}, and every pass over this directory "
            f"projects onto it; pass --refit-basis to replace it, or --pca-name to write "
            f"this fit beside it"
        )

    import torch

    from anamnesis.config import ExtractionConfig, ModelConfig
    from anamnesis.extraction.model_loader import load_model

    n_components = args.n_components or ExtractionConfig.from_preset(preset).pca_components
    existing_means = calibration_fit.read_existing_means(means_path, args.refit_means)

    config = ModelConfig.from_preset(preset, model_id=args.model_path or preset.model_id)
    logger.info(f"calibrating {config.model_id} over {len(prompts)} prompts")
    loaded = load_model(config, sampled_layers=[])
    loaded.disable_hooks()  # calibration reads hidden states from the forward, not from hooks
    try:
        fit = calibration_fit.fit_calibration(
            calibration_fit.generate_prompt_states(loaded, prompts, settings),
            preset=preset,
            settings=settings,
            n_components=n_components,
            pooled=args.pooled,
            existing_means=existing_means,
        )
    except calibration_fit.CalibrationFitError as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        loaded.remove_hooks()
        del loaded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    calibration_fit.write_calibration(fit, means_path, basis_path)
    logger.info("calibration complete")


if __name__ == "__main__":
    main()
