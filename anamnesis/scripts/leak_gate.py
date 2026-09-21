"""Gate a feature set: does it carry the signal, or the topic it was measured on?

Names one cell per condition, selects feature sets by name, and reports for each the
topic-grouped accuracy, the naive accuracy it is not allowed to be quoted as, the
permutation null band it has to clear, and a topic-decode readout that says how much
content the set carries on its own.

A set that clears its null is a leak-safe carrier. A set whose naive accuracy is well
above its grouped one while the grouped one sits inside the null is leak-dominated. A
set at the null either way is inert. The three are different statements and the
command prints which one it measured.

The cells are the caller's: the arm protocols that built them stay in the frozen
record, and what this needs is a directory of signatures and its metadata.

    python -m anamnesis.scripts.leak_gate \\
        --cell linear=outputs/runs/pure_linear --cell socratic=outputs/runs/pure_socratic \\
        --feature-set routing_cka=prefix:xrt_cka_ --expect 6 \\
        --feature-set all_cka=contains:cka \\
        --subdir signatures_v3_x2 --out outputs/battery/arms/A1/leak_gate.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PREFIX = "prefix:"
CONTAINS = "contains:"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="leak_gate.py", description=__doc__.splitlines()[0])
    p.add_argument(
        "--cell", action="append", required=True, metavar="LABEL=RUN_DIR",
        help="A condition's run directory; repeat once per condition, at least twice",
    )
    p.add_argument(
        "--feature-set", action="append", required=True, metavar=f"NAME={PREFIX}TEXT",
        help=f"A named feature set, selected by {PREFIX}<text> or {CONTAINS}<text>",
    )
    p.add_argument(
        "--expect", type=int, default=None,
        help="Refuse unless the FIRST feature set has exactly this many features",
    )
    p.add_argument("--subdir", default="signatures_v3", help="Signature subdirectory of each cell")
    p.add_argument("--nperm", type=int, default=1000, help="Label permutations for the null band")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True, help="Where the gate document is written")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args(argv)

    from anamnesis.analysis.leak_gate import (
        SignatureCell,
        load_corpus,
        run_leak_gate,
        select_features,
    )

    cells = []
    for spec in args.cell:
        label, _, directory = spec.partition("=")
        if not directory:
            raise SystemExit(f"--cell takes LABEL=RUN_DIR; got {spec!r}")
        cells.append(
            SignatureCell(label=label, run_dir=Path(directory), signatures_subdir=args.subdir)
        )
    corpus = load_corpus(cells)
    logger.info(
        f"{len(corpus.labels)} cells, {len(corpus.y)} rows, {corpus.n_topics} topics, "
        f"{len(corpus.feature_names)} features"
    )

    feature_sets: dict[str, list[int]] = {}
    for spec in args.feature_set:
        name, _, selector = spec.partition("=")
        if selector.startswith(PREFIX):
            columns = select_features(corpus.feature_names, prefix=selector[len(PREFIX):])
        elif selector.startswith(CONTAINS):
            columns = select_features(corpus.feature_names, contains=selector[len(CONTAINS):])
        else:
            raise SystemExit(
                f"--feature-set takes NAME={PREFIX}<text> or NAME={CONTAINS}<text>; got {spec!r}"
            )
        if not columns:
            raise SystemExit(f"feature set {name!r} selected no feature of this corpus")
        feature_sets[name] = columns

    first = next(iter(feature_sets))
    if args.expect is not None and len(feature_sets[first]) != args.expect:
        raise SystemExit(
            f"feature set {first!r} has {len(feature_sets[first])} features, expected "
            f"{args.expect}: {[corpus.feature_names[i] for i in feature_sets[first]]}"
        )

    document = run_leak_gate(
        cells, feature_sets, nperm=args.nperm, seed=args.seed, corpus=corpus
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=1), encoding="utf-8")
    logger.info(f"gate -> {args.out}")
    return 0 if document["quotable"][first] else 1


if __name__ == "__main__":
    raise SystemExit(main())
