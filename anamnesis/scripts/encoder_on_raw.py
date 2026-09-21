"""Read a matched contrast three ways: hand features, raw linear, raw encoder.

When a hand-built feature set shows nothing, there are two possibilities and they have
opposite consequences: nothing is there, or the projection dropped it. This puts a linear
and a nonlinear readout on the *raw* state of the same generations, on the identical
split, so the two can be told apart.

The arms must be matched: the same continuations replayed with and without the
intervention, so the token sequences are identical and a classifier's success is a read
of the computation rather than of the content. Only generations present in both arms
enter.

The pass is post-hoc and light — the cost was banking the raw captures — so it runs on a
CPU or one small card.

    python -m anamnesis.scripts.encoder_on_raw --model 3b \\
        --positive-raw /dev/shm/steered_raw --positive-run outputs/runs/steered \\
        --negative-raw /dev/shm/unsteered_raw --negative-run outputs/runs/unsteered \\
        --source-metadata outputs/battery/vmb_stage0_3b/metadata.json \\
        --out-dir outputs/battery/arms/A5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

RECEIPT_STEM = "encoder_on_raw"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="encoder_on_raw.py", description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="Label for the receipt; the pass loads no model")
    p.add_argument("--positive-raw", type=Path, required=True, help="Raw captures of the steered arm")
    p.add_argument("--positive-run", type=Path, required=True, help="That arm's run directory")
    p.add_argument("--negative-raw", type=Path, required=True, help="Raw captures of the control arm")
    p.add_argument("--negative-run", type=Path, required=True, help="That arm's run directory")
    p.add_argument(
        "--source-metadata", type=Path, required=True,
        help="metadata.json of the run the continuations came from, for topics and lengths",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu", help="Device the fold reduction and readouts run on")
    p.add_argument(
        "--surfaces", default="residual,attention",
        help="Raw surfaces to concatenate; residual is fast, attention is slow to resample",
    )
    p.add_argument("--sig-subdir", default="signatures_v3")
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--deep-epochs", type=int, default=800)
    p.add_argument(
        "--intervention", default="",
        help="What the positive arm carried, recorded in the receipt",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args(argv)

    from anamnesis.analysis.audit_lib import gen_metadata_by_id
    from anamnesis.analysis.encoder_ladder import Arm, run_ladder

    surfaces = tuple(s.strip() for s in args.surfaces.split(",") if s.strip())
    if not surfaces:
        raise SystemExit("--surfaces named no surface")
    device = args.device
    if device != "cpu":
        import torch

        if not torch.cuda.is_available():
            logger.warning(f"{device} is not available; the ladder runs on the CPU")
            device = "cpu"

    source_metadata = gen_metadata_by_id(args.source_metadata)
    logger.info(f"source metadata: {len(source_metadata)} generations")

    result = run_ladder(
        Arm(label="steered", raw_dir=args.positive_raw, run_dir=args.positive_run, y=1),
        Arm(label="control", raw_dir=args.negative_raw, run_dir=args.negative_run, y=0),
        source_metadata,
        device=device,
        surfaces=surfaces,
        signatures_subdir=args.sig_subdir,
        n_seeds=args.n_seeds,
        deep_epochs=args.deep_epochs,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{RECEIPT_STEM}_{args.model}.json"
    path.write_text(
        json.dumps(
            {"model": args.model, "intervention": args.intervention, **result.model_dump()},
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )
    logger.info(f"ladder -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
