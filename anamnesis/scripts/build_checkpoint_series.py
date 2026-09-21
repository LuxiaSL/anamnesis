"""Enumerate a training run's adapter checkpoints into a replay series.

Writes the series document ``run_replay_multickpt.py`` consumes: one row per checkpoint,
in step order with ``final`` last, each pointing at a per-step run directory under the
cohort root so the series and the signatures it produces share one layout.

It refuses a directory whose checkpoints carry no adapter configuration. Those are
full-weight checkpoints, which cannot be swapped into a loaded base — they are replayed one
model load per checkpoint, and the refusal says which command does that.

    python -m anamnesis.scripts.build_checkpoint_series \\
        --ckpt-dir /scratch/partc/cell4/checkpoints/qwen_cat_dpo_r16_s0 \\
        --arm cat_dpo_r16_s0 \\
        --run-root outputs/runs/vmb_a6cohort_qwen \\
        --out cells_cat_dpo.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_checkpoint_series.py", description=__doc__.splitlines()[0]
    )
    p.add_argument(
        "--ckpt-dir", type=Path, required=True,
        help="Training directory holding checkpoint-* subdirectories",
    )
    p.add_argument("--arm", required=True, help="Cohort subdirectory label for this series")
    p.add_argument(
        "--run-root", type=Path, required=True,
        help="Cohort run root each checkpoint's signatures are written under",
    )
    p.add_argument("--out", type=Path, required=True, help="Where the series document is written")
    p.add_argument(
        "--no-final", action="store_true",
        help="Leave out the `final` directory, keeping the numbered steps only",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parser().parse_args(argv)

    from anamnesis.extraction.replay.checkpoint_series import series_from_adapter_dir

    series = series_from_adapter_dir(
        args.ckpt_dir,
        arm=args.arm,
        run_root=args.run_root,
        include_final=not args.no_final,
    )
    series.write(args.out)
    for checkpoint in series.checkpoints:
        logger.info(f"  {checkpoint.label:32s} -> {checkpoint.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
