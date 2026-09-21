"""Lay a dose ladder's texts side by side, matched by prompt, as a readable document.

The eyeball channel beside the numbers. For each of a handful of prompts it prints the
same generation across a vector's dose ladder and against its controls at a matched dose,
so one question can be answered by reading: does the vector induce the **mode** while a
random direction of the same magnitude merely degrades?

Matching is by generation id, which is the prompt. Seeds differ per cell, so two cells'
texts for one id are the same prompt written twice — a style-and-mode comparison, not a
token-level control, and the document says so where it is read.

The ladder stops at the strongest dose that was collected, which sits below the
mode-induction peak. Read the trend, not the ceiling.

    python -m anamnesis.scripts.qual_extract --model qwen-7b \\
        --run-dir outputs/battery/vmb_a5_qwen_7b --site 18 \\
        --gen-ids 1 41 81 121 161 --out outputs/battery/qual_qwen.md
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from anamnesis.steering.readouts import DOSE_LADDER, QUALITATIVE_CHARS

logger = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qual_extract.py", description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="Which model's cells these are, for the heading")
    p.add_argument("--run-dir", type=Path, required=True, help="Directory holding the cell directories")
    p.add_argument("--site", type=int, required=True, help="Injection site the cells were built at")
    p.add_argument(
        "--gen-ids", type=int, nargs="+", required=True,
        help="Generations to show; a handful, since each is a block of six cells",
    )
    p.add_argument("--vector", default="V3", help="The vector whose ladder is read")
    p.add_argument(
        "--doses", type=float, nargs="+", default=list(DOSE_LADDER),
        help=f"Doses of the ladder (default: {list(DOSE_LADDER)})",
    )
    p.add_argument("--control-dose", type=float, default=0.3, help="Dose the controls are read at")
    p.add_argument("--chars", type=int, default=QUALITATIVE_CHARS, help="Characters shown per text")
    p.add_argument("--out", type=Path, required=True, help="Where the document is written")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parser().parse_args(argv)

    from anamnesis.steering.readouts import cell_ladder, qualitative_markdown

    if not args.run_dir.is_dir():
        raise SystemExit(f"run directory not found: {args.run_dir}")
    ladder = cell_ladder(
        args.site, vector=args.vector, doses=args.doses, control_dose=args.control_dose
    )
    document = qualitative_markdown(
        args.run_dir,
        model=args.model,
        site=args.site,
        gen_ids=args.gen_ids,
        ladder=ladder,
        chars=args.chars,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(document, encoding="utf-8")
    logger.info(
        f"{args.out}: {len(args.gen_ids)} prompts x {len(ladder)} cells "
        f"(site L{args.site}, vector {args.vector})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
