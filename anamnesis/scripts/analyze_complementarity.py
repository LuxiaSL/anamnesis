"""Read several banked gauntlet results and report what sits between them.

Seven readings that only exist across runs and across families: cross-run
consistency, pairwise resolution by difficulty, the complementarity of blocks' hard-pair
profiles, feature importance grouped by family and sub-family, each block's hardest
confusion, whether the registered accuracy ordering of the core blocks holds, and what
the engineered families add over the unions of the core blocks.

Nothing here runs a classifier. It re-reads results that were expensive to produce,
which is the whole reason the cross-run questions are cheap to ask.

The mode-subset passes are read whenever they are present, and named when they are
not: they are what make an engineered corpus comparable to a five-mode baseline, so
there is nothing to gain from a flag that asks for them conditionally.

    python -m anamnesis.scripts.analyze_complementarity
    python -m anamnesis.scripts.analyze_complementarity --analysis-dir outputs/analysis
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REPORT_NAME = "complementarity_report.json"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="analyze_complementarity.py", description=__doc__.splitlines()[0]
    )
    p.add_argument(
        "--analysis-dir", type=Path, default=None,
        help="Directory holding one subdirectory per run (default: the analysis outputs root)",
    )
    p.add_argument("--output", type=Path, default=None, help="Where the report is written")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parser().parse_args(argv)

    from anamnesis.analysis.complementarity import complementarity_report, load_report_inputs
    from anamnesis.config import outputs_root

    analysis_dir = args.analysis_dir or (outputs_root() / "analysis")
    results = load_report_inputs(analysis_dir, include_subsets=True)
    if not results:
        raise SystemExit(
            f"no banked gauntlet results under {analysis_dir} — run run_gauntlet.py first"
        )
    logger.info(f"{len(results)} runs loaded")
    report = complementarity_report(results)

    path = args.output or (analysis_dir / REPORT_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info(f"report -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
