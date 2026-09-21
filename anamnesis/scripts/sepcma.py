"""Probe whether a direction search is worth its budget, before spending it.

A search for a direction in the residual stream evaluates a whole generation pass per
candidate, so the budget is the experiment's cost. This asks the cheapest useful question
about it first: on an **idealized** version of the same landscape — noiseless, unimodal,
the same dimension, the same number of evaluations — does the optimizer climb?

A planted unit direction and the ``|cos|`` objective are that landscape. A best alignment
several times chance says the budget is adequate and a flat result says it is not, at
which point the answer is a different budget or a different parameterization rather than
a run that will fail slowly.

``--sphere`` runs the convex sanity check instead: an optimizer that cannot reach the
origin of a sphere has an arithmetic problem, not a landscape problem, and that is worth
separating before a flat alignment is read as a statement about the budget.

    python -m anamnesis.scripts.sepcma --dim 3072
    python -m anamnesis.scripts.sepcma --dim 256 --dim 3072 --budget-multiple 10
    python -m anamnesis.scripts.sepcma --sphere --dim 64
"""

from __future__ import annotations

import argparse
import logging

from anamnesis.optimize import BUDGET_MULTIPLE, minimize_sphere, probe_budget

logger = logging.getLogger(__name__)

SPHERE_TARGET = 1e-8
"""Where the convex check is called solved. It is a property of the arithmetic, not of a
landscape, so it is fixed rather than an argument."""


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sepcma.py", description=__doc__.splitlines()[0])
    p.add_argument(
        "--dim", type=int, action="append", required=True,
        help="Dimension to probe; repeat to probe several",
    )
    p.add_argument(
        "--budget-multiple", type=int, default=BUDGET_MULTIPLE,
        help=f"Evaluations per dimension (default: {BUDGET_MULTIPLE})",
    )
    p.add_argument("--seed", type=int, default=11)
    p.add_argument(
        "--sphere", action="store_true",
        help="Run the convex sanity check instead of the planted-direction probe",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parser().parse_args(argv)

    failures = 0
    for dim in args.dim:
        if args.sphere:
            best, evals = minimize_sphere(dim, seed=args.seed, target=SPHERE_TARGET)
            solved = best < SPHERE_TARGET
            logger.info(
                f"sphere d={dim}: best {best:.2e} after {evals} evals -> "
                f"{'OK' if solved else 'FAILED'}"
            )
            failures += 0 if solved else 1
            continue
        probe = probe_budget(dim, seed=args.seed, budget_multiple=args.budget_multiple)
        logger.info(probe.reading())
        failures += 0 if probe.climbed else 1
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
