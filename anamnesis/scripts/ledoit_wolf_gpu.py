"""Qualify the fast shrinkage-covariance path on this box, against scikit-learn.

A whitened steering vector is ``Sigma^-1 delta``, and at residual width the covariance is
estimated with Ledoit-Wolf shrinkage. The fast path computes the same closed form with the
arithmetic rearranged, which makes it a port rather than a second estimator — and a port
is either exact on a box or it is not, so this measures it rather than asserting it.

Three numbers, and the third is the one that matters: the shrinkage factor, the relative
Frobenius error of Sigma, and the cosine between the two whitened directions. A vector is
built out of that direction, so a cosine below the bar fails the check whatever the other
two say.

Run it once per box before building vectors through the fast path, and after any change to
the accelerator stack.

    python -m anamnesis.scripts.ledoit_wolf_gpu
    python -m anamnesis.scripts.ledoit_wolf_gpu --n 3000 --d 512 --device cuda
"""

from __future__ import annotations

import argparse
import logging

from anamnesis.steering.covariance import DEFAULT_DEVICE, check_against_reference

logger = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ledoit_wolf_gpu.py", description=__doc__.splitlines()[0])
    p.add_argument(
        "--n", type=int, default=3000,
        help="Samples in the check; small enough that the reference finishes",
    )
    p.add_argument("--d", type=int, default=512, help="Features in the check")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device", default=DEFAULT_DEVICE,
        help=f"Device the fast path runs on (default: {DEFAULT_DEVICE})",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parser().parse_args(argv)
    result = check_against_reference(
        n_samples=args.n, n_features=args.d, seed=args.seed, device=args.device
    )
    logger.info(result.reading())
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
