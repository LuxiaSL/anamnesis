"""Shrinkage covariance at width, on whichever device is at hand.

A whitened steering vector is ``Sigma^-1 delta``, so building one at a residual
width of several thousand means estimating a covariance of that width from a
sample that is not much larger. The Ledoit-Wolf estimator is what makes that
well-posed: it shrinks the empirical covariance toward a scaled identity by a
factor derived from the data, in closed form, with no cross-validation.

Closed form is why this port is exact rather than approximate — and it is a port
rather than a second estimator. The estimator of record is
:func:`anamnesis.steering.vectors.ledoit_wolf_covariance`, which calls
scikit-learn's; :func:`shrinkage_covariance` here computes the same number by
rearranging the same arithmetic, and carries the different name so a reader can
never be unsure which of the two they are holding.

**The two rearrangements**, with ``Xc`` the centred data ``(n, d)`` and
``S = Xc^T Xc / n``:

* the Frobenius term is ``sum((Xc^T Xc)**2) / n**2``, which is ``||S||_F^2`` — so it
  reads off ``S``, which is already formed, instead of forming a second matrix;
* the shrinkage numerator is ``sum(X2^T @ X2)`` with ``X2 = Xc**2``, and expanding
  the double sum gives ``sum_k (row sum of X2[k])^2`` — an ``O(n d)`` reduction in
  place of an ``O(n d^2)`` matmul. This is the win that matters.

What remains at ``O(n d^2)`` is ``S`` itself: one matmul, which is the operation an
accelerator exists for.

**Precision is chosen, not inherited.** The matmul runs in float32, which is more
than the bf16-derived states carry, and everything numerically delicate — the
shrinkage arithmetic, the solve, the eigendecomposition — runs in float64. TF32 is
switched off around the matmul: it would quietly drop the accumulation to ten
mantissa bits, and this is a covariance estimate that a vector will be built from.

The estimator is checked against scikit-learn rather than trusted, on a sample
small enough for the reference to finish. That check is what qualifies the path on a
box, which is why it is a function here and a command in ``scripts/``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

F64 = NDArray[np.float64]

DEFAULT_DEVICE = "cpu"
"""The device a caller gets when it names none. The arithmetic is device-agnostic —
an accelerator makes it fast, and the same call on a CPU is what makes the agreement
check runnable anywhere."""

SHRINKAGE_TOLERANCE = 1e-4
COVARIANCE_TOLERANCE = 1e-4
DIRECTION_TOLERANCE = 0.9999
"""Agreement bars for the check against scikit-learn: the shrinkage factor, the
relative Frobenius error of Sigma, and the cosine between the two whitened
directions. The third is the one that matters for a steering vector, because it is
the quantity a vector is built out of."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a hard dependency here
        raise ImportError("shrinkage covariance at width needs torch") from exc
    return torch


def shrinkage_covariance(
    X: NDArray[Any], device: str = DEFAULT_DEVICE, *, assume_centered: bool = False
) -> tuple[F64, float]:
    """Ledoit-Wolf shrunk covariance and its shrinkage factor.

    Mirrors ``sklearn.covariance.ledoit_wolf``, including the ``beta = min(beta, delta)``
    clamp and the rule that a zero numerator means no shrinkage at all — the two
    places a reimplementation drifts from the reference without failing.
    """
    torch = _torch()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        n_samples, n_features = X.shape
        rows = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=device)
        if not assume_centered:
            rows = rows - rows.mean(dim=0, keepdim=True)

        empirical = (rows.T @ rows) / n_samples

        squared = rows * rows
        diagonal = squared.sum(dim=0) / n_samples
        trace = diagonal.sum()
        mu = trace / n_features

        frobenius = (empirical.double() ** 2).sum()
        row_sums = squared.sum(dim=1).double()
        numerator_raw = (row_sums * row_sums).sum()

        numerator = (1.0 / (n_features * n_samples)) * (numerator_raw / n_samples - frobenius)
        denominator = (
            frobenius
            - 2.0 * mu.double() * trace.double()
            + n_features * (mu.double() ** 2)
        ) / n_features
        clamped = torch.minimum(numerator, denominator)
        shrinkage = 0.0 if float(clamped) == 0.0 else float(clamped / denominator)

        sigma = empirical.double() * (1.0 - shrinkage)
        sigma.diagonal().add_(shrinkage * mu.double())
        out = sigma.cpu().numpy()
        del rows, squared, empirical, sigma
        if device != "cpu":
            torch.cuda.empty_cache()
        return out, shrinkage
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def solve(sigma: NDArray[Any], b: NDArray[Any], device: str = DEFAULT_DEVICE) -> F64:
    """``Sigma^-1 b`` in float64, degrading to least squares on a singular Sigma.

    A singular covariance is a fact about the sample, not a crash: the least-squares
    solution is the minimum-norm direction consistent with it, and the fall back is
    logged so a vector built through it is traceable.
    """
    torch = _torch()
    A = torch.as_tensor(sigma, dtype=torch.float64, device=device)
    y = torch.as_tensor(np.ascontiguousarray(b), dtype=torch.float64, device=device)
    try:
        solution = torch.linalg.solve(A, y)
    except RuntimeError as exc:
        logger.warning(f"solve failed ({exc}); falling back to least squares")
        solution = torch.linalg.lstsq(A, y.unsqueeze(-1)).solution.squeeze(-1)
    return solution.cpu().numpy()


def eigh(sigma: NDArray[Any], device: str = DEFAULT_DEVICE) -> tuple[F64, F64]:
    """Symmetric eigendecomposition in float64: eigenvalues ascending, then vectors."""
    torch = _torch()
    A = torch.as_tensor(sigma, dtype=torch.float64, device=device)
    values, vectors = torch.linalg.eigh(A)
    return values.cpu().numpy(), vectors.cpu().numpy()


class AgreementCheck(BaseModel):
    """How far this estimator sits from the reference, and whether that passes."""

    model_config = ConfigDict(extra="forbid")

    n_samples: int = Field(gt=0)
    n_features: int = Field(gt=0)
    device: str
    shrinkage_reference: float
    shrinkage_here: float
    shrinkage_abs_diff: float
    sigma_relative_frobenius_error: float
    whitened_direction_cosine: float
    passed: bool

    def reading(self) -> str:
        """The one line a caller acts on."""
        verdict = "PASS" if self.passed else "FAIL — do not build vectors through this path"
        return (
            f"n={self.n_samples} d={self.n_features} on {self.device}: "
            f"shrinkage {self.shrinkage_here:.8f} vs {self.shrinkage_reference:.8f} "
            f"(|diff| {self.shrinkage_abs_diff:.2e}), "
            f"Sigma relative Frobenius error {self.sigma_relative_frobenius_error:.3e}, "
            f"cos(Sigma^-1 b) {self.whitened_direction_cosine:.8f} -> {verdict}"
        )


def check_against_reference(
    *, n_samples: int = 3000, n_features: int = 512, seed: int = 0, device: str = DEFAULT_DEVICE
) -> AgreementCheck:
    """Measure this path against scikit-learn on synthetic anisotropic data.

    The data is correlated and anisotropic on purpose: an isotropic sample makes the
    shrinkage term nearly degenerate, so an estimator could agree on it while
    disagreeing on anything real.
    """
    from sklearn.covariance import LedoitWolf

    rng = np.random.default_rng(seed)
    mixing = rng.standard_normal((n_features, n_features)).astype(np.float32) / np.sqrt(n_features)
    scales = rng.uniform(0.5, 4.0, size=n_features).astype(np.float32)
    X = (rng.standard_normal((n_samples, n_features)).astype(np.float32) @ mixing) * scales

    reference = LedoitWolf(assume_centered=False).fit(X.astype(np.float64))
    sigma, shrinkage = shrinkage_covariance(X, device=device)

    shrinkage_diff = abs(reference.shrinkage_ - shrinkage)
    relative_error = float(
        np.linalg.norm(sigma - reference.covariance_)
        / max(float(np.linalg.norm(reference.covariance_)), 1e-12)
    )
    b = rng.standard_normal(n_features)
    reference_direction = np.linalg.solve(reference.covariance_, b)
    here_direction = solve(sigma, b, device=device)
    cosine = float(
        reference_direction
        @ here_direction
        / (np.linalg.norm(reference_direction) * np.linalg.norm(here_direction))
    )
    return AgreementCheck(
        n_samples=n_samples,
        n_features=n_features,
        device=device,
        shrinkage_reference=float(reference.shrinkage_),
        shrinkage_here=float(shrinkage),
        shrinkage_abs_diff=float(shrinkage_diff),
        sigma_relative_frobenius_error=relative_error,
        whitened_direction_cosine=cosine,
        passed=bool(
            shrinkage_diff < SHRINKAGE_TOLERANCE
            and relative_error < COVARIANCE_TOLERANCE
            and cosine > DIRECTION_TOLERANCE
        ),
    )


__all__ = [
    "AgreementCheck",
    "COVARIANCE_TOLERANCE",
    "DEFAULT_DEVICE",
    "DIRECTION_TOLERANCE",
    "SHRINKAGE_TOLERANCE",
    "check_against_reference",
    "eigh",
    "shrinkage_covariance",
    "solve",
]
