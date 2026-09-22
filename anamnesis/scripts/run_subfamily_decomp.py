"""Cut each feature family into its sub-families and score every part.

Within a family, which signal carries the classification? The cut is made from the
family's own feature names, a random forest is run on each part under stratified
cross-validation, and the whole family's accuracy sits beside the parts as the
comparison.

``--modes`` narrows the corpus and changes where the result is written, because a
decomposition over five modes and one over eight are not the same table.

    python -m anamnesis.scripts.run_subfamily_decomp --run 8b_v2
    python -m anamnesis.scripts.run_subfamily_decomp --run 8b_v2 \\
        --modes linear,socratic,contrastive,dialectical,analogical
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from anamnesis.config import UnknownRunError, outputs_root, resolve_run, run_names

logger = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_subfamily_decomp.py", description=__doc__.splitlines()[0]
    )
    p.add_argument(
        "--run", nargs="+", required=True,
        help=f"Registry run names. Known: {list(run_names())}",
    )
    p.add_argument("--modes", default=None, help="Comma-separated modes to include")
    p.add_argument("--output-dir", type=Path, default=None, help="Where the document is written")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args(argv)
    mode_filter = [m.strip() for m in args.modes.split(",")] if args.modes else None

    from anamnesis.analysis.gauntlet.signature_io import load_run4
    from anamnesis.analysis.subfamily import (
        decompose_run,
        decomposition_document,
        default_output_path,
    )

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
        logger.info(f"=== sub-family decomposition: {name} ===")
        if mode_filter:
            logger.info(f"  mode filter: {mode_filter}")
        data = load_run4(
            signature_dir=resolved.signature_dir,
            core_only=True,
            addon_dirs=list(resolved.addon_dirs) or None,
            mode_filter=mode_filter,
        )
        logger.info(f"  {data.n_samples} samples over {len(data.unique_modes)} modes")
        results[name] = decompose_run(data)

    if not results:
        raise SystemExit("no run was decomposable — see the errors above")

    path = default_output_path(
        args.output_dir or (outputs_root() / "analysis"), mode_filter=mode_filter
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decomposition_document(results), indent=2), encoding="utf-8")
    logger.info(f"results -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
