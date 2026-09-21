"""Fit the contrastive projection a model's signatures will be computed through.

One fit per model, banked as a calibration artifact: hidden states sampled from banked
raw tensors at the preset's projection layers, a small network trained on triplets with
whole generations held out, and the best weights written as numpy arrays so inference
needs no torch.

The positional means are an argument rather than a later step. A projection fitted on
raw states and applied to positionally corrected ones is a fit-and-apply mismatch, so
whichever the features will use is what the fit sees.

The artifact is verified before the command exits: it is loaded back through the
feature family that will apply it, and one row is projected. A file that cannot be read
by its consumer is not a calibration artifact.

    python -m anamnesis.scripts.train_contrastive_projection --model 8b \\
        --raw-dir outputs/runs/8b_fat_01/raw_tensors \\
        --output-path outputs/calibration/llama31_8b/contrastive_projection.npz
    python -m anamnesis.scripts.train_contrastive_projection --model 8b \\
        --raw-dir outputs/runs/8b_fat_01/raw_tensors \\
        --output-path outputs/calibration/llama31_8b/contrastive_projection.npz \\
        --positional-means outputs/calibration/llama31_8b/positional_means.npz
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from anamnesis.config import MODEL_PRESETS, resolve_preset

logger = logging.getLogger(__name__)

SIGNATURES_DIRNAME = "signatures"
"""Where the mode labels are read from when no metadata directory is named: the
signatures beside the raw tensors, which is the layout a run writes."""


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_contrastive_projection.py", description=__doc__.splitlines()[0]
    )
    p.add_argument("--model", choices=list(MODEL_PRESETS), required=True)
    p.add_argument("--raw-dir", type=Path, required=True, help="Directory of banked raw tensors")
    p.add_argument(
        "--metadata-dir", type=Path, default=None,
        help=f"Directory of per-generation metadata (default: ../{SIGNATURES_DIRNAME})",
    )
    p.add_argument("--output-path", type=Path, required=True, help="Where the .npz is written")
    p.add_argument(
        "--positional-means", type=Path, default=None,
        help="Positional means the states are corrected by before fitting",
    )
    p.add_argument("--temporal-samples", type=int, default=5)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--bottleneck-dim", type=int, default=32)
    p.add_argument("--n-epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args(argv)
    preset = resolve_preset(args.model)
    layers = list(preset.contrastive_layers)
    metadata_dir = args.metadata_dir or (args.raw_dir.parent / SIGNATURES_DIRNAME)
    logger.info(f"model {preset.name}: projection layers {layers}, metadata from {metadata_dir}")

    from anamnesis.analysis.contrastive_mlp import load_hidden_state_samples, train_projection

    X, y, groups = load_hidden_state_samples(
        args.raw_dir,
        metadata_dir,
        layers,
        temporal_samples=args.temporal_samples,
        positional_means_path=args.positional_means,
    )
    weights = train_projection(
        X, y, groups,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        n_epochs=args.n_epochs,
        seed=args.seed,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_path, **weights)
    logger.info(
        f"projection -> {args.output_path} "
        f"({args.output_path.stat().st_size / 1024:.1f} KB)"
    )

    from anamnesis.extraction.feature_families.contrastive_projection import (
        ContrastiveProjectionInference,
    )

    inference = ContrastiveProjectionInference.load(args.output_path)
    embedding = inference.project(X[0])
    logger.info(
        f"verified through the feature family: {X.shape[1]} in, {len(embedding)} out, "
        f"norm {np.linalg.norm(embedding):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
