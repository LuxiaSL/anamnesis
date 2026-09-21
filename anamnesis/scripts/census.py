"""Take the sub-perceptual census over every banked arm record.

A row is a class member when its internals rung materially exceeds both of the rungs
that do not read internals. This reads the banked arm records, computes each row's gap,
applies the declared bars, and writes the census as JSON and as a table — with the class
object, which is the census's actual subject: which rows are members, per model.

The judge-defense gate runs before anything is written. A blind judge's *failure* may
not be the evidence that makes a row a member, and a row whose judge gap is large enough
to quote carries its hardening status or the census refuses to exist.

    python -m anamnesis.scripts.census --arms-root outputs/battery/arms \\
        --out-dir outputs/battery/census
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="census.py", description=__doc__.splitlines()[0])
    p.add_argument(
        "--arms-root", type=Path, required=True,
        help="Directory holding one subdirectory per arm record",
    )
    p.add_argument("--out-dir", type=Path, required=True, help="Where the census is written")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args(argv)

    from anamnesis.analysis.battery.census import MEMBER, class_object, run_census

    rows = run_census(args.arms_root, args.out_dir)
    members = class_object(rows)
    logger.info(f"{len(rows)} rows, {sum(1 for r in rows if r.status == MEMBER)} members")
    for model, entry in members.items():
        logger.info(
            f"  {model}: {len(entry['members'])} member(s), "
            f"{len(entry['borderline'])} borderline"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
