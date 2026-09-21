"""Section 9's Mantel test, at test scale.

``run_semantic`` is not called here: it embeds the corpus, which wants
sentence-transformers and a download. ``_mantel_test`` needs neither — it is two
distance matrices, a correlation and a permutation null — so the property that
matters is tested directly.

That property is the p-value's convention. A Mantel test is a permutation test,
so its p is the add-one statistic from ``anamnesis.analysis.battery.stats`` like
every other permutation p in the package: ``(hits + 1) / (n + 1)``, never zero and
never finer than ``1 / (n_permutations + 1)``. The section is checked against that
function rather than against itself, by reproducing the null it drew.

CPU only; no banked data, no model, no GPU, no optional dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

from anamnesis.analysis.battery.stats import permutation_pvalue, permutation_resolution
from anamnesis.analysis.gauntlet.schemas import MantelResult
from anamnesis.analysis.gauntlet.semantic import _mantel_test


def distance_matrix(rng: np.random.Generator, n: int, dims: int = 4) -> np.ndarray:
    """Euclidean distances between ``n`` points drawn independently."""
    X = rng.standard_normal((n, dims))
    return np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)


def reproduce_null(
    D_compute: np.ndarray, D_semantic: np.ndarray, n_permutations: int, seed: int,
) -> np.ndarray:
    """The null ``_mantel_test`` draws, rebuilt from the same seed.

    The permutation relabels the objects — the same order applied to rows and to
    columns — rather than shuffling the pair distances, so every draw is still a
    distance matrix over the same objects.
    """
    n = D_compute.shape[0]
    idx = np.triu_indices(n, k=1)
    x = D_compute[idx]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        draws.append(float(np.corrcoef(x, D_semantic[np.ix_(perm, perm)][idx])[0, 1]))
    return np.array(draws)


def test_the_mantel_p_value_is_the_one_the_battery_computes() -> None:
    """The section reports what ``permutation_pvalue`` returns for its own null.

    The null is reproducible from the seed, so it can be rebuilt here and handed to
    the shared statistic directly. Agreement is the receipt that section 9 owns no
    second copy of the arithmetic.
    """
    rng = np.random.default_rng(11)
    D_compute = distance_matrix(rng, 24)
    D_semantic = distance_matrix(rng, 24)
    n_permutations = 20

    result = _mantel_test(
        D_compute, D_semantic, n_permutations=n_permutations, seed=42,
    )
    null = reproduce_null(D_compute, D_semantic, n_permutations, seed=42)

    assert isinstance(result, MantelResult)
    assert result.null_mean == pytest.approx(float(np.mean(null))), (
        "the reproduced null matches the one the section drew"
    )
    assert result.null_std == pytest.approx(float(np.std(null)))
    assert result.p_value == pytest.approx(permutation_pvalue(result.r, null))


def test_the_mantel_p_value_carries_the_add_one_correction() -> None:
    """The reported p is (hits+1)/(N+1), not the hit rate with a floor under it.

    Two independently drawn point clouds put the observed correlation in the middle
    of its own null, which is the band the two conventions disagree over:
    ``(hits + 1) / (N + 1)`` lands on the 1/(N+1) lattice, while ``hits / N``
    clamped from below at 1/(N+1) lands off it and lower.
    """
    rng = np.random.default_rng(2)
    D_compute = distance_matrix(rng, 24)
    D_semantic = distance_matrix(rng, 24)
    n_permutations = 20

    result = _mantel_test(
        D_compute, D_semantic, n_permutations=n_permutations, seed=42,
    )

    lattice_position = result.p_value * (n_permutations + 1)
    assert lattice_position == pytest.approx(round(lattice_position), abs=1e-9), (
        "an add-one p-value is a whole number of 1/(N+1) steps"
    )
    hits = round(lattice_position) - 1
    assert 1 <= hits <= n_permutations - 1, (
        "unrelated matrices put the observation inside its own null; got "
        f"{hits} of {n_permutations}"
    )
    assert result.p_value == pytest.approx((hits + 1) / (n_permutations + 1))
    assert result.p_value > max(hits / n_permutations, permutation_resolution(
        n_permutations
    )), "the hit rate with a floor under it is the anti-conservative reading"


def test_a_matrix_against_itself_reports_the_finest_p_available() -> None:
    """Perfect correlation is the no-hit case: p is 1/(N+1), and not zero.

    Nothing a relabelling can produce beats r = 1, so this is the strongest claim
    twenty permutations can make — and it is still 0.048, not 0.
    """
    rng = np.random.default_rng(5)
    D = distance_matrix(rng, 24)
    n_permutations = 20

    result = _mantel_test(D, D, n_permutations=n_permutations, seed=42)

    assert result.r == pytest.approx(1.0)
    assert result.p_value == pytest.approx(permutation_resolution(n_permutations))
    assert result.p_value > 0.0
