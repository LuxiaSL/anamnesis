"""Turn a banked trajectory bank into a path-signature design matrix and its null.

The projection already happened: a ``paths_*.npz`` holds each generation's trajectory in
the banked coordinates. This computes the signature features over them, at one layer, one
rank and one level, and writes the matrix with its feature names and the indices of the
paths that entered — a path too short for the level asked for is dropped and counted, not
zero-filled.

``--null-seeds`` adds the increment-permutation null on the same real paths, which is
what a level-2 number is read against: shuffling a path's increments preserves its
endpoint and its level-1 signature exactly while destroying the order the level-2 terms
are about.

    python -m anamnesis.scripts.pathsig_features --bank outputs/paths/paths_8b_L16.npz \\
        --layer 16 --k 8 --level 2 --out outputs/pathsig/e1_L16_k8.npz
    python -m anamnesis.scripts.pathsig_features --bank <bank> --layer 16 --k 8 --level 2 \\
        --null-seeds 1 2 3 --out <file>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pathsig_features.py", description=__doc__.splitlines()[0])
    p.add_argument("--bank", type=Path, required=True, help="Banked paths npz")
    p.add_argument(
        "--variant", choices=("pc", "nopc"), default="pc",
        help="Projected trajectories or unprojected ones, as the bank holds them",
    )
    p.add_argument("--layer", type=int, required=True, help="The site the trajectories come from")
    p.add_argument("--k", type=int, required=True, help="Coordinates to use, up to the banked rank")
    p.add_argument("--level", type=int, choices=(1, 2), required=True)
    p.add_argument(
        "--no-time-augment", action="store_true",
        help="Drop the time coordinate, which the level-2 terms read reparametrization through",
    )
    p.add_argument(
        "--null-seeds", type=int, nargs="+", default=None,
        help="Increment-permutation null seeds; at least three when given",
    )
    p.add_argument("--out", type=Path, required=True, help="Where the design matrix is written")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args(argv)

    from anamnesis.extraction.path_banks import PathBank, null_matrices, signature_matrix

    bank = PathBank.load(args.bank, variant=args.variant)
    logger.info(f"{bank.n} paths, banked rank {bank.k_max}, variant {bank.variant}")
    design = signature_matrix(
        bank,
        layer=args.layer,
        k=args.k,
        level=args.level,
        time_augment=not args.no_time_augment,
    )
    logger.info(
        f"design matrix {design.X.shape} over {design.n_kept} paths "
        f"({design.n_dropped} too short for level {args.level})"
    )

    arrays: dict[str, np.ndarray] = {
        "X": design.X,
        "feature_names": np.array(design.names, dtype=object),
        "kept": np.asarray(design.kept, dtype=np.int64),
        "gen_ids": np.asarray(bank.gen_ids)[np.asarray(design.kept, dtype=np.int64)],
    }
    if args.null_seeds:
        nulls = null_matrices(
            bank,
            layer=args.layer,
            k=args.k,
            level=args.level,
            seeds=tuple(args.null_seeds),
            time_augment=not args.no_time_augment,
        )
        for seed, null in zip(args.null_seeds, nulls):
            arrays[f"X_null_{seed}"] = null.X
            arrays[f"kept_null_{seed}"] = np.asarray(null.kept, dtype=np.int64)
        logger.info(f"null: {len(nulls)} shuffles of the same paths")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    logger.info(f"features -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
