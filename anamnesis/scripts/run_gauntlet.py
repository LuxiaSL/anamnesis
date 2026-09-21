"""Run the analysis gauntlet over one run's signatures.

Eleven standing analyses over one corpus, because a claim about signatures is
usually a claim about several of them agreeing: an accuracy means one thing beside a
clean orthogonality result and another beside a length-only baseline reaching the
same number.

A run is named from the registry, or its signature directory is given directly.
``--resume`` picks up from the checkpoint the previous pass wrote, which is what
makes a long gauntlet survivable; ``--skip`` drops sections by number; ``--modes``
narrows the corpus to a mode subset and writes beside the full result rather than
over it, because a pass over five modes and a pass over eight are different
measurements.

This command was ``run_unified_analysis`` in the extraction repository, and there is
no alias: the port map is the bridge for a reader holding the old name.

    python -m anamnesis.scripts.run_gauntlet --run 8b_v2
    python -m anamnesis.scripts.run_gauntlet --run 8b_v2 --resume --skip 8 9
    python -m anamnesis.scripts.run_gauntlet --run 8b_v2 \\
        --modes linear,socratic,contrastive,dialectical,analogical
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from anamnesis.config import UnknownRunError, outputs_root, resolve_run, run_names

logger = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_gauntlet.py", description=__doc__.splitlines()[0])
    p.add_argument(
        "--run", required=True,
        help=f"Registry name, or any label when --sig-dir is given. Known: {list(run_names())}",
    )
    p.add_argument("--sig-dir", type=Path, default=None, help="Signature directory, overriding the registry")
    p.add_argument("--output-dir", type=Path, default=None, help="Where results.json is written")
    p.add_argument(
        "--all-reps", action="store_true",
        help="Use every repetition rather than one per topic-mode pair",
    )
    p.add_argument("--skip", type=int, nargs="+", default=[], help="Section numbers to skip (1-11)")
    p.add_argument(
        "--addon-dirs", type=Path, nargs="+", default=None,
        help="Extra signature directories to merge, for split feature sets",
    )
    p.add_argument("--resume", action="store_true", help="Skip sections that already have results")
    p.add_argument("--modes", default=None, help="Comma-separated modes to include")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args(argv)

    signature_dir = args.sig_dir
    addon_dirs = list(args.addon_dirs) if args.addon_dirs else None
    if signature_dir is None:
        try:
            resolved = resolve_run(args.run)
        except UnknownRunError as exc:
            raise SystemExit(
                f"{exc}. Name a registry run, or give --sig-dir for a corpus outside it."
            ) from exc
        signature_dir = resolved.signature_dir
        if addon_dirs is None and resolved.addon_dirs:
            addon_dirs = list(resolved.addon_dirs)
    if not signature_dir.is_dir():
        raise SystemExit(f"signature directory not found: {signature_dir}")

    mode_filter = [m.strip() for m in args.modes.split(",")] if args.modes else None
    output_dir = args.output_dir
    if mode_filter and output_dir is None:
        output_dir = outputs_root() / "analysis" / f"{args.run}_{len(mode_filter)}way"
        logger.info(f"mode filter {mode_filter} -> {output_dir}")

    from anamnesis.analysis.gauntlet import run_full_analysis

    run_full_analysis(
        signature_dir=signature_dir,
        run_name=args.run,
        output_dir=output_dir,
        core_only=not args.all_reps,
        skip_sections=set(args.skip) if args.skip else None,
        resume=args.resume,
        addon_dirs=addon_dirs,
        mode_filter=mode_filter,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
