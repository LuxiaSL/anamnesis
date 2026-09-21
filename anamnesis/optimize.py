"""Black-box search in high dimensions: separable CMA-ES, and the probe for it.

Some questions about an intervention are not gradient questions. "Which direction
in the residual stream maximizes a selectivity score" has an objective that is a
whole generation pass plus a judgement — no derivative, expensive to evaluate, and
thousands of dimensions wide. That is an evolution strategy's problem.

This is **separable** CMA-ES: the covariance model is diagonal, so the state is
``O(d)`` rather than ``O(d^2)`` and an iteration costs a vector operation instead of
an eigendecomposition. At the widths here a full covariance is not merely slow, it
is unlearnable from the number of evaluations a budget allows, and the separable
variant's learning-rate speedup of ``(d+2)/3`` is what makes the diagonal model
adapt at all inside one.

It is implemented rather than imported for three reasons that are all about
dependence: the optimizer is thirty lines of documented arithmetic, it is
deterministic under a seed, and it fits on one screen where a reader can check it
against the paper. The convention is **maximization** — the fitness a caller passes
is a score, and higher is better.

**The budget probe is the point of the landscapes below.** Before spending real
evaluations, a caller can ask whether the search can solve an *idealized* version of
the same problem — noiseless, unimodal, same dimension, same budget. On a planted
direction at ``10 * d`` evaluations, a climb well above chance alignment says the
budget is worth spending and a flat result says it is not. That is the cheapest
possible thing to learn about a search, and it is the reason this module has a
command of its own.

Nothing here knows about signatures or steering. It is arithmetic, and it is at the
package root for that reason: a caller with a fitness function and a dimension is
its whole audience.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

F64 = NDArray[np.float64]

VARIANCE_FLOOR = 1e-20
"""A diagonal variance is never allowed below this: a coordinate at exactly zero
variance is a coordinate the search can never move again."""

BUDGET_MULTIPLE = 10
"""Evaluations per dimension the probe spends by default, which is the budget the
real searches are planned around."""

CLIMB_FACTOR = 5.0
"""How far above chance alignment the planted-direction probe has to reach before
the budget counts as adequate."""


class SepCMA:
    """Separable CMA-ES: diagonal covariance, with the sep-CMA learning-rate speedup.

    Use it as ask-and-tell, which is what lets the objective be anything at all —
    including a pass that runs on another machine:

        opt = SepCMA(dim=3072, seed=1)
        while budget_remains:
            candidates = opt.ask()          # (lambda, dim)
            opt.tell(candidates, scores)    # higher score is better

    The state is the mean, the step size, the diagonal variances and the two
    evolution paths; :meth:`state` and :meth:`restore` carry all of it including the
    generator, so a search can be checkpointed and resumed without changing what it
    would have done.
    """

    def __init__(
        self,
        dim: int,
        seed: int,
        sigma0: float = 1.0,
        x0: NDArray[Any] | None = None,
        lam: int | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = int(dim)
        self.rng = np.random.default_rng(seed)
        self.lam = int(lam) if lam else 4 + int(3 * np.log(self.dim))
        if self.lam < 2:
            raise ValueError(f"population must be at least 2, got {self.lam}")
        self.mu = self.lam // 2
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = weights / weights.sum()
        self.mueff = 1.0 / float(self.w @ self.w)

        d, mueff = self.dim, self.mueff
        self.cs = (mueff + 2) / (d + mueff + 5)
        self.ds = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (d + 1)) - 1) + self.cs
        self.cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        rank_one = 2 / ((d + 1.3) ** 2 + mueff)
        rank_mu = min(1 - rank_one, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        speedup = (d + 2) / 3.0
        self.c1 = min(1.0, rank_one * speedup)
        self.cmu = min(1.0 - self.c1, rank_mu * speedup)
        self.chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d * d))

        self.mean = np.zeros(d) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
        self.sigma = float(sigma0)
        self.C = np.ones(d)
        self.ps = np.zeros(d)
        self.pc = np.zeros(d)
        self.gen = 0
        self.evals = 0
        self._last_z: F64 | None = None

    def ask(self) -> F64:
        """The next generation of candidates, ``(lambda, dim)``."""
        self._last_z = self.rng.standard_normal((self.lam, self.dim))
        return self.mean + self.sigma * self._last_z * np.sqrt(self.C)

    def tell(self, X: NDArray[Any], fit: NDArray[Any]) -> None:
        """Update the search from the candidates and their scores; higher is better.

        Raises
        ------
        ValueError
            When ``tell`` does not follow an ``ask`` with that ask's candidates. The
            update is expressed in the standard normal draws behind the candidates,
            so scoring a different population would update the search with steps it
            never took.
        """
        if self._last_z is None or np.shape(X) != (self.lam, self.dim):
            raise ValueError("tell() must follow ask() with the same candidates")
        ranked = np.argsort(-np.asarray(fit))[: self.mu]
        Z = self._last_z[ranked]
        Y = Z * np.sqrt(self.C)
        z_weighted = self.w @ Z
        y_weighted = self.w @ Y

        self.mean = self.mean + self.sigma * y_weighted
        self.ps = (1 - self.cs) * self.ps + np.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * z_weighted
        ps_norm = float(np.linalg.norm(self.ps))
        stalled = ps_norm / np.sqrt(
            1 - (1 - self.cs) ** (2 * (self.gen + 1))
        ) / self.chiN < 1.4 + 2 / (self.dim + 1)
        self.pc = (1 - self.cc) * self.pc + (
            np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_weighted if stalled else 0.0
        )
        correction = (1 - float(stalled)) * self.cc * (2 - self.cc)
        self.C = (
            (1 - self.c1 - self.cmu) * self.C
            + self.c1 * (self.pc ** 2 + correction * self.C)
            + self.cmu * (self.w @ (Y ** 2))
        )
        self.C = np.maximum(self.C, VARIANCE_FLOOR)
        self.sigma = self.sigma * float(
            np.exp((self.cs / self.ds) * (ps_norm / self.chiN - 1))
        )
        self.gen += 1
        self.evals += self.lam
        self._last_z = None

    def state(self) -> dict[str, Any]:
        """Everything a resume needs, generator included."""
        return {
            "mean": self.mean,
            "sigma": self.sigma,
            "C": self.C,
            "ps": self.ps,
            "pc": self.pc,
            "gen": self.gen,
            "evals": self.evals,
            "rng_state": self.rng.bit_generator.state,
        }

    def restore(self, state: dict[str, Any]) -> None:
        """Resume from a banked state, so the search continues rather than restarts."""
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.sigma = float(state["sigma"])
        self.C = np.asarray(state["C"], dtype=np.float64)
        self.ps = np.asarray(state["ps"], dtype=np.float64)
        self.pc = np.asarray(state["pc"], dtype=np.float64)
        self.gen, self.evals = int(state["gen"]), int(state["evals"])
        self.rng.bit_generator.state = state["rng_state"]


def sphere_fitness(X: NDArray[Any]) -> F64:
    """Negative squared norm: the convex sanity landscape, maximized at the origin."""
    return -np.asarray((np.asarray(X) ** 2).sum(axis=1), dtype=np.float64)


def alignment_fitness(X: NDArray[Any], direction: NDArray[Any]) -> F64:
    """``|cos|`` against a planted unit direction: scale-free and unimodal on the sphere.

    This is the shape a real direction search has — what is wanted is an
    orientation, not a magnitude — which is why it is the landscape the budget probe
    reports on.
    """
    rows = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=1)
    return np.abs(rows @ np.asarray(direction, dtype=np.float64)) / np.maximum(norms, 1e-12)


def chance_alignment(dim: int) -> float:
    """The alignment a random direction reaches in ``dim`` dimensions: ``1/sqrt(d)``."""
    return 1.0 / float(np.sqrt(dim))


class BudgetProbe(BaseModel):
    """What a search achieved on the idealized landscape, and whether that is a climb."""

    model_config = ConfigDict(extra="forbid")

    dim: int = Field(gt=0)
    budget: int = Field(gt=0)
    evals: int = Field(gt=0)
    best_alignment: float
    chance: float
    ratio: float
    climbed: bool

    def reading(self) -> str:
        """One line: the alignment reached, against chance, with the verdict."""
        verdict = "budget is adequate" if self.climbed else "budget is NOT adequate"
        return (
            f"d={self.dim} at {self.evals}/{self.budget} evals: best alignment "
            f"{self.best_alignment:.3f} against chance {self.chance:.3f} "
            f"(x{self.ratio:.0f}) -> {verdict}"
        )


def probe_budget(
    dim: int,
    *,
    seed: int = 11,
    budget_multiple: int = BUDGET_MULTIPLE,
    sigma0: float = 1.0,
    climb_factor: float = CLIMB_FACTOR,
) -> BudgetProbe:
    """Can the search find a planted direction in ``dim`` dimensions on this budget?

    The landscape is the noiseless, unimodal version of a direction search: if the
    optimizer cannot climb here, no budget-matched search on the real objective will
    either, and that is worth knowing before any evaluation is spent.
    """
    rng = np.random.default_rng(0)
    direction = rng.standard_normal(dim)
    direction /= np.linalg.norm(direction)
    optimizer = SepCMA(dim=dim, seed=seed, sigma0=sigma0)
    budget = budget_multiple * dim
    best = 0.0
    while optimizer.evals < budget:
        candidates = optimizer.ask()
        scores = alignment_fitness(candidates, direction)
        optimizer.tell(candidates, scores)
        best = max(best, float(scores.max()))
    chance = chance_alignment(dim)
    return BudgetProbe(
        dim=dim,
        budget=budget,
        evals=optimizer.evals,
        best_alignment=best,
        chance=chance,
        ratio=best / chance,
        climbed=best > climb_factor * chance,
    )


def minimize_sphere(
    dim: int, *, seed: int = 7, x0_value: float = 3.0, max_generations: int = 800,
    target: float = 1e-8,
) -> tuple[float, int]:
    """Drive the search to the origin from a displaced start; return ``(best, evals)``.

    The convex sanity check: an optimizer that cannot reach the origin of a sphere
    has an arithmetic error, not a landscape problem, and this separates the two
    before a planted-direction result is read.
    """
    optimizer = SepCMA(dim=dim, seed=seed, sigma0=1.0, x0=np.full(dim, x0_value))
    best = float("inf")
    for _ in range(max_generations):
        candidates = optimizer.ask()
        values = (candidates ** 2).sum(axis=1)
        optimizer.tell(candidates, -values)
        best = min(best, float(values.min()))
        if best < target:
            break
    return best, optimizer.evals


def search(
    dim: int,
    fitness: Callable[[NDArray[Any]], NDArray[Any]],
    *,
    budget: int,
    seed: int = 1,
    sigma0: float = 1.0,
    x0: NDArray[Any] | None = None,
) -> tuple[F64, float, SepCMA]:
    """Maximize ``fitness`` over ``budget`` evaluations; return the best row and score.

    The loop is here so a caller with a batch-scoring function does not write it
    again — and it is the only thing here that is a loop rather than a law, which is
    why the optimizer stays usable ask-and-tell for callers whose evaluations happen
    somewhere else entirely.
    """
    optimizer = SepCMA(dim=dim, seed=seed, sigma0=sigma0, x0=x0)
    best_row: F64 | None = None
    best_score = -np.inf
    while optimizer.evals < budget:
        candidates = optimizer.ask()
        scores = np.asarray(fitness(candidates), dtype=np.float64)
        if scores.shape != (optimizer.lam,):
            raise ValueError(
                f"fitness returned {scores.shape}, expected one score per candidate "
                f"({optimizer.lam},)"
            )
        optimizer.tell(candidates, scores)
        top = int(scores.argmax())
        if float(scores[top]) > best_score:
            best_score = float(scores[top])
            best_row = np.asarray(candidates[top], dtype=np.float64).copy()
    if best_row is None:
        raise ValueError(f"budget {budget} is smaller than one generation ({optimizer.lam})")
    return best_row, best_score, optimizer


__all__ = [
    "BUDGET_MULTIPLE",
    "BudgetProbe",
    "CLIMB_FACTOR",
    "SepCMA",
    "alignment_fitness",
    "chance_alignment",
    "minimize_sphere",
    "probe_budget",
    "search",
    "sphere_fitness",
]
