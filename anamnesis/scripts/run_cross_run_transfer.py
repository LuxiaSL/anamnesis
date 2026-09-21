"""Transfer a learned projection between two runs' mode vocabularies.

Train the contrastive projection on one run, embed the other, and ask which of the
training run's modes each test mode lands nearest — scored against the pre-registered
pairing, in both directions, over several seeds. The LDA direction test runs beside it
and asks the same question linearly, which is what makes the
directions-versus-manifolds dissociation visible when it is there.

The two runs must have been extracted by the same pipeline under the same calibration:
one projection reads both, so a feature-width mismatch is refused rather than trimmed.

    python -m anamnesis.scripts.run_cross_run_transfer
    python -m anamnesis.scripts.run_cross_run_transfer --n-seeds 10
    python -m anamnesis.scripts.run_cross_run_transfer \\
        --train-run run_8b_baseline --test-run run_8b_r2_equivalent
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_RUN = "run_8b_baseline"
DEFAULT_TEST_RUN = "run_8b_r2_equivalent"
DEFAULT_OUTPUT_RUN = "8b_cross_run_transfer"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_cross_run_transfer.py", description=__doc__.splitlines()[0]
    )
    p.add_argument(
        "--train-run", default=DEFAULT_TRAIN_RUN,
        help="Run directory under the outputs root whose modes the projection is trained on",
    )
    p.add_argument(
        "--test-run", default=DEFAULT_TEST_RUN,
        help="Run directory whose modes are embedded into that projection",
    )
    p.add_argument(
        "--output-run", default=DEFAULT_OUTPUT_RUN,
        help="Subdirectory of the analysis outputs root the result is written to",
    )
    p.add_argument("--outputs-base", type=Path, default=None, help="Outputs root, overriding the configured one")
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--bottleneck-dim", type=int, default=32)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument(
        "--feature-key", default="features",
        help="Signature npz key holding the vector (default: the baseline blocks)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parser().parse_args(argv)

    from anamnesis.analysis.cross_run import cross_run_transfer, headline
    from anamnesis.config import outputs_root

    base = args.outputs_base or outputs_root()
    results = cross_run_transfer(
        train_run=args.train_run,
        test_run=args.test_run,
        feature_key=args.feature_key,
        n_seeds=args.n_seeds,
        bottleneck_dim=args.bottleneck_dim,
        n_epochs=args.n_epochs,
        outputs_base=base,
    )
    output_dir = base / "analysis" / args.output_run
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"results -> {path}")
    for line in headline(results):
        logger.info(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
