"""Write a synthetic signature bank, so the reading side runs before a model exists.

The numbers are drawn, not measured, and mean nothing about any model;
:mod:`anamnesis.synthetic_bank` states what the construction does and does not put in
them. This is the shortest path from a fresh checkout to a gauntlet that reports
something.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from anamnesis.config import outputs_root
from anamnesis.synthetic_bank import SyntheticBankSpec, write_synthetic_bank

logger = logging.getLogger(__name__)

DEFAULT_RUN = "synthetic_demo"
SIGNATURES_DIRNAME = "signatures"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        default=DEFAULT_RUN,
        help="run name to write under the outputs root (default: %(default)s)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        help="write here instead, as the signatures directory itself",
    )
    parser.add_argument("--topics", type=int, help="distinct topics, shared across modes")
    parser.add_argument("--repetitions", type=int, help="generations per mode-topic pair")
    parser.add_argument(
        "--modes", help="comma-separated mode labels; the five hard modes by default"
    )
    parser.add_argument("--seed", type=int, help="the same seed writes the same bank")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)

    overrides: dict[str, object] = {}
    if args.topics is not None:
        overrides["topics"] = args.topics
    if args.repetitions is not None:
        overrides["repetitions"] = args.repetitions
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.modes:
        overrides["modes"] = tuple(m.strip() for m in args.modes.split(",") if m.strip())

    spec = SyntheticBankSpec(**overrides)
    destination = args.dest or (outputs_root() / "runs" / args.run / SIGNATURES_DIRNAME)

    bank = write_synthetic_bank(destination, spec)
    logger.info(
        "wrote %d synthetic generations, %d features wide, over %d modes and %d topics",
        bank.generations,
        bank.width,
        len(bank.modes),
        bank.topics,
    )
    logger.info("bank: %s", bank.directory)
    logger.info("lane: %s", bank.lane_id)
    logger.info(
        "every block carries the same separating strength per column, so a reading of "
        "this bank still ranks blocks — by their widths and by the seed. That ranking "
        "is a property of the fixture and says nothing about any substrate."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
