"""The fast shrinkage covariance, against the estimator it is a port of.

The claim is exactness, not approximation, so the test is the claim: on anisotropic
correlated data the fast path's shrinkage factor, its Sigma and — the one that matters —
the whitened direction it implies all agree with scikit-learn's. The check runs on a CPU,
which is the point: a port that can only be verified on an accelerator cannot be verified
in CI.

Also pinned: the shrinkage bounds (a factor in ``[0, 1]``), the two degenerate cases the
reference handles specially, and that the solve degrades to least squares on a singular
Sigma rather than raising — a singular covariance is a fact about the sample, and the
minimum-norm direction is the answer that fact permits.

CPU only; no accelerator, no model.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.covariance import LedoitWolf

from anamnesis.steering.covariance import (
    DIRECTION_TOLERANCE,
    check_against_reference,
    eigh,
    shrinkage_covariance,
    solve,
)


def anisotropic(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mixing = rng.standard_normal((d, d)).astype(np.float32) / np.sqrt(d)
    scales = rng.uniform(0.5, 4.0, size=d).astype(np.float32)
    return (rng.standard_normal((n, d)).astype(np.float32) @ mixing) * scales


def test_the_fast_path_agrees_with_the_reference_on_a_cpu() -> None:
    result = check_against_reference(n_samples=400, n_features=64, seed=1)
    assert result.passed, result.reading()
    assert result.whitened_direction_cosine > DIRECTION_TOLERANCE
    assert "PASS" in result.reading()


def test_the_shrinkage_factor_is_the_references_own_number() -> None:
    X = anisotropic(300, 48, seed=2)
    reference = LedoitWolf(assume_centered=False).fit(X.astype(np.float64))
    sigma, shrinkage = shrinkage_covariance(X)
    assert shrinkage == pytest.approx(float(reference.shrinkage_), abs=1e-6)
    assert 0.0 <= shrinkage <= 1.0
    assert np.allclose(sigma, reference.covariance_, rtol=1e-4, atol=1e-6)
    assert np.allclose(sigma, sigma.T), "a covariance estimate is symmetric"


def test_already_centred_data_may_say_so() -> None:
    X = anisotropic(200, 32, seed=3)
    centred = X - X.mean(axis=0, keepdims=True)
    told, _ = shrinkage_covariance(centred, assume_centered=True)
    inferred, _ = shrinkage_covariance(centred, assume_centered=False)
    assert np.allclose(told, inferred, rtol=1e-4, atol=1e-6)


def test_a_singular_covariance_degrades_rather_than_raising() -> None:
    d = 8
    sigma = np.zeros((d, d))
    sigma[0, 0] = 1.0
    b = np.ones(d)
    direction = solve(sigma, b)
    assert direction.shape == (d,)
    assert np.all(np.isfinite(direction))


def test_the_eigendecomposition_comes_back_ascending_and_reconstructs() -> None:
    X = anisotropic(150, 16, seed=4)
    sigma, _ = shrinkage_covariance(X)
    values, vectors = eigh(sigma)
    assert np.all(np.diff(values) >= -1e-9), "eigenvalues ascend"
    assert np.all(values > 0), "a shrunk covariance is positive definite"
    assert np.allclose(vectors @ np.diag(values) @ vectors.T, sigma, atol=1e-8)


def test_a_failing_check_is_reported_as_a_refusal_to_build_vectors() -> None:
    """The reading is what a caller acts on, so the failing wording is pinned too."""
    result = check_against_reference(n_samples=200, n_features=32, seed=5)
    failed = result.model_copy(update={"passed": False})
    assert "do not build vectors" in failed.reading()
