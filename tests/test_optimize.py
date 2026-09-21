"""Separable CMA-ES: that it climbs, that it resumes exactly, that it refuses misuse.

An optimizer is easy to write and easy to get subtly wrong, and a subtly wrong one still
returns numbers. So the tests are the two landscapes the module documents plus the
properties a caller depends on:

  * the **sphere** is the convex sanity check: from a displaced start the search reaches
    the origin, which says the covariance and step-size updates are arithmetic rather
    than noise;
  * the **planted direction** is the shape a real search has, and the probe's whole
    purpose is the verdict — a climb well above chance alignment at a fixed budget;
  * chance alignment is ``1/sqrt(d)``, which is what makes the ratio meaningful;
  * ``tell`` must follow its own ``ask``: the update is expressed in the draws behind the
    candidates, so scoring a different population would update the search with steps it
    never took, and that is a refusal rather than a wrong answer;
  * ``state``/``restore`` carry the generator, so a resumed search continues the run it
    was rather than starting a statistically similar one.

CPU only; no model, no device, and the dimensions are small so the suite stays fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from anamnesis.optimize import (
    BUDGET_MULTIPLE,
    SepCMA,
    alignment_fitness,
    chance_alignment,
    minimize_sphere,
    probe_budget,
    search,
    sphere_fitness,
)


def test_the_search_reaches_the_origin_of_a_sphere() -> None:
    best, evals = minimize_sphere(32, seed=7, x0_value=3.0, max_generations=800)
    assert best < 1e-8, f"the convex check did not converge: best {best:.3e}"
    assert evals > 0


def test_a_planted_direction_is_found_well_above_chance() -> None:
    probe = probe_budget(64, seed=11)
    assert probe.budget == BUDGET_MULTIPLE * 64
    assert probe.evals >= probe.budget
    assert probe.chance == pytest.approx(chance_alignment(64))
    assert probe.ratio > 5.0
    assert probe.climbed
    assert "budget is adequate" in probe.reading()


def test_the_probe_reports_a_failure_rather_than_hiding_it() -> None:
    """One generation of evaluations cannot climb, and the verdict says so."""
    probe = probe_budget(64, seed=3, budget_multiple=1, climb_factor=100.0)
    assert not probe.climbed
    assert "NOT adequate" in probe.reading()


def test_the_two_landscapes_are_what_they_claim() -> None:
    rows = np.array([[0.0, 0.0], [3.0, 4.0]])
    assert sphere_fitness(rows).tolist() == [0.0, -25.0]
    direction = np.array([1.0, 0.0])
    aligned = alignment_fitness(np.array([[2.0, 0.0], [0.0, 2.0]]), direction)
    assert aligned[0] == pytest.approx(1.0), "a parallel row is perfectly aligned"
    assert aligned[1] == pytest.approx(0.0), "a perpendicular row is not"
    assert alignment_fitness(rows * 10, direction).tolist() == pytest.approx(
        alignment_fitness(rows, direction).tolist()
    ), "the objective is scale-free, which is what makes it a direction search"


def test_telling_the_optimizer_about_a_population_it_did_not_ask_for_is_refused() -> None:
    optimizer = SepCMA(dim=8, seed=1)
    with pytest.raises(ValueError, match="must follow ask"):
        optimizer.tell(np.zeros((optimizer.lam, 8)), np.zeros(optimizer.lam))
    candidates = optimizer.ask()
    with pytest.raises(ValueError, match="must follow ask"):
        optimizer.tell(candidates[:-1], np.zeros(optimizer.lam - 1))
    optimizer.tell(candidates, sphere_fitness(candidates))
    with pytest.raises(ValueError, match="must follow ask"):
        optimizer.tell(candidates, sphere_fitness(candidates))


def test_a_degenerate_construction_is_refused() -> None:
    with pytest.raises(ValueError, match="dim must be positive"):
        SepCMA(dim=0, seed=1)
    with pytest.raises(ValueError, match="population"):
        SepCMA(dim=4, seed=1, lam=1)


def test_a_restored_search_continues_the_run_it_was() -> None:
    first = SepCMA(dim=6, seed=5)
    for _ in range(3):
        candidates = first.ask()
        first.tell(candidates, sphere_fitness(candidates))
    snapshot = first.state()

    resumed = SepCMA(dim=6, seed=999)
    resumed.restore(snapshot)
    assert resumed.gen == first.gen and resumed.evals == first.evals
    assert np.allclose(resumed.mean, first.mean) and resumed.sigma == first.sigma

    original = first.ask()
    continued = resumed.ask()
    assert np.allclose(original, continued), "the generator travels with the state"


def test_the_search_loop_returns_the_best_row_it_evaluated() -> None:
    direction = np.zeros(128)
    direction[3] = 1.0
    best_row, best_score, optimizer = search(
        128, lambda X: alignment_fitness(X, direction), budget=128 * 20, seed=2
    )
    assert best_score > 5 * chance_alignment(128)
    assert best_row.shape == (128,)
    assert optimizer.evals >= 128 * 20
    assert float(alignment_fitness(best_row.reshape(1, -1), direction)[0]) == pytest.approx(
        best_score
    ), "the row returned is the row that scored"


def test_a_fitness_of_the_wrong_shape_is_refused() -> None:
    with pytest.raises(ValueError, match="one score per candidate"):
        search(4, lambda X: np.zeros(len(X) + 1), budget=100, seed=1)
    with pytest.raises(ValueError, match="smaller than one generation"):
        search(4, sphere_fitness, budget=0, seed=1)
