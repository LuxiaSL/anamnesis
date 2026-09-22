"""Run the prompt-swap confound test over one or more banked runs.

For each swap pair, a binary classifier is trained on the two pure modes and asked
where the swap generations land: on the mode whose system prompt they were given, or
on the mode whose execution they were asked for. The answer is a count, block by block,
and pooled with a direction named at the 1.5:1 bar.

A run needs swap generations in its bank; a run without them is named and skipped
rather than reported as an ambiguous result.

    python -m anamnesis.scripts.run_binary_prompt_swap --run 8b_v2
    python -m anamnesis.scripts.run_binary_prompt_swap --run 8b_v2 3b_v2
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from anamnesis.config import UnknownRunError, outputs_root, resolve_run, run_names

logger = logging.getLogger(__name__)

RESULTS_NAME = "binary_prompt_swap_results.json"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_binary_prompt_swap.py", description=__doc__.splitlines()[0]
    )
    p.add_argument(
        "--run", nargs="+", required=True,
        help=f"Registry run names. Known: {list(run_names())}",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where the results document is written (default: the analysis outputs root)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args(argv)

    from anamnesis.analysis.prompt_swap import run_binary_prompt_swap, swap_report

    results = {}
    for name in args.run:
        try:
            resolved = resolve_run(name)
        except UnknownRunError as exc:
            logger.error(f"{exc}")
            continue
        missing = resolved.missing_dirs()
        if missing:
            logger.error(f"{name}: missing {[str(m) for m in missing]}")
            continue
        logger.info(f"=== prompt-swap test: {name} ===")
        try:
            results[name] = run_binary_prompt_swap(
                run_name=name,
                signature_dir=resolved.signature_dir,
            )
        except (ValueError, FileNotFoundError) as exc:
            logger.error(f"{name}: not tested ({exc})")

    if not results:
        raise SystemExit("no run was testable — see the errors above")

    output_dir = args.output_dir or (outputs_root() / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / RESULTS_NAME
    path.write_text(json.dumps(swap_report(results), indent=2), encoding="utf-8")
    logger.info(f"results -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
