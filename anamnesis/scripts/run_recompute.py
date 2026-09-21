"""Recompute signatures from banked raw tensors, on a CPU.

The point of banking raw tensors is that a feature vector is not the last word: a
family can be added, or a calibration basis refit, and the whole corpus can be
re-featurised without a device and without re-generating a token. This is that
loop — the offline half of the instrument, and the one that makes the capture
surface an experimental variable rather than a commitment.

Raw tensors are banked with the positional means already subtracted, so the means
are supplied back here for the families that need them uncorrected. ``--pca-model``
points at a basis other than the calibration directory's own, which is how a refit
basis is evaluated against the same tensors the deployed one was applied to.

The arithmetic belongs to :mod:`anamnesis.extraction.feature_pipeline` and stays
there. This command chooses a model's layer plan and a family set and hands them
over; it computes nothing itself, which is what keeps a recomputed vector
comparable with a banked one.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from anamnesis.config import MODEL_PRESETS, ExtractionConfig, FeaturePipelineConfig, resolve_preset

logger = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    p.add_argument("--run-dir", type=Path, required=True, help="Run directory holding the tensors")
    p.add_argument("--calib-dir", type=Path, required=True)
    p.add_argument(
        "--pca-model",
        type=Path,
        default=None,
        help="A basis other than the calibration directory's pca_model.pkl",
    )
    p.add_argument("--raw-subdir", default="raw_tensors_v3")
    p.add_argument("--out-subdir", default="signatures_v3_c5", help="Where the new vectors land")
    p.add_argument(
        "--metadata-subdir",
        default="signatures_v3",
        help="Signatures whose per-generation metadata the new ones inherit",
    )
    p.add_argument("--workers", type=int, default=48, help="Processes over the tensor files")
    return p


def configs(model: str) -> tuple[ExtractionConfig, FeaturePipelineConfig]:
    """The layer plan and the family set this recompute runs under.

    The family set is the one the banked vectors were computed with: the baseline
    surfaces, the residual trajectory, attention flow, gate features, per-head
    heterogeneity and the spectral operator, with windowed temporal decomposition
    and the learned contrastive projection off. Naming it here rather than reusing
    the replay surface's set is deliberate — the replay surface grew families that
    the banked vectors this command reproduces do not contain, and a recompute
    that silently widened its vector would not be a recompute.
    """
    preset = resolve_preset(model)
    return (
        ExtractionConfig.from_preset(preset, enable_residual_pca=True),
        FeaturePipelineConfig.from_preset(
            preset,
            include_core_blocks=True,
            enable_residual_trajectory=True,
            enable_attention_flow=True,
            enable_gate_features=True,
            enable_temporal_dynamics=False,
            enable_per_head=True,
            enable_stft=True,
            enable_contrastive_projection=False,
        ),
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args(argv)

    from anamnesis.extraction.calibration import load_positional_means
    from anamnesis.extraction.feature_pipeline import _load_pca_model, recompute_all_features

    extraction, families = configs(args.model)
    positional_means = load_positional_means(args.calib_dir)
    if positional_means is None:
        raise SystemExit(
            f"recompute needs the positional means the tensors were banked against; "
            f"none at {args.calib_dir}"
        )
    pca_path = args.pca_model or (args.calib_dir / "pca_model.pkl")
    components, mean = _load_pca_model(pca_path)
    logger.info(
        f"PCA: {pca_path.name} ({'per-layer' if isinstance(components, dict) else 'pooled'})"
    )

    recompute_all_features(
        raw_dir=args.run_dir / args.raw_subdir,
        output_dir=args.run_dir / args.out_subdir,
        config=extraction,
        pca_components=components,
        pca_mean=mean,
        metadata_dir=args.run_dir / args.metadata_subdir,
        family_config=families,
        n_workers=args.workers,
        positional_means=positional_means,
    )
    logger.info(f"recompute done -> {args.run_dir / args.out_subdir}")


if __name__ == "__main__":
    main()
