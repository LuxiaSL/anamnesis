"""The hybrid basis: directions of variance across states, then within groups, orthonormal.

States are drawn with planted structure: each context has a mean in a subspace U, and its
group of siblings varies around that mean in a subspace V that the across part can't see.
The fit must find U in its across part and V in its within part, with orthonormal rows. The
across part must be the per-layer fit's own components byte for byte. The within part must
hold at least as much within-group variance as Gram-Schmidt of an independent within-group
fit, on a case built so the two differ. Two halves split by context must agree on every
planted direction.
"""

from __future__ import annotations

import numpy as np
import pytest

from anamnesis.extraction.calibration_fit import (
    CalibrationFitError,
    _fit_one,
    _group_contrasts,
    _orthonormal_rows,
    fit_hybrid_basis,
    hybrid_reproducibility,
    principal_cosines,
)

WIDTH = 24


def planted(seed: int, contexts: int = 120, siblings: int = 6, across=3, within=2, overlap: float = 0.0):
    """States ``[contexts * siblings, WIDTH]``, their group labels and the planted subspaces.

    Context means vary along ``U`` (``across`` directions, large). Siblings vary around their
    context's mean along ``V`` (``within`` directions), which is orthogonal to ``U`` unless
    ``overlap`` tilts it into ``U``."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((WIDTH, WIDTH)))
    u, v = q[:, :across].T, q[:, across:across + within].T
    if overlap:
        v = (1 - overlap) * v + overlap * u[:within]
        v /= np.linalg.norm(v, axis=1, keepdims=True)
    centres = (rng.standard_normal((contexts, across)) * np.array([6.0, 5.0, 4.0][:across])) @ u
    noise = rng.standard_normal((contexts, siblings, WIDTH)) * 0.05
    wiggle = (rng.standard_normal((contexts, siblings, within)) * np.array([2.0, 1.5][:within])) @ v
    states = (centres[:, None, :] + wiggle + noise).reshape(-1, WIDTH)
    groups = np.repeat(np.arange(contexts), siblings)
    return states, groups, u, v


def test_the_parts_find_the_planted_subspaces_with_orthonormal_rows():
    states, groups, u, v = planted(0)
    basis = fit_hybrid_basis(states, groups, 3, 2)
    c = basis["components"].astype(np.float64)
    assert c.shape == (5, WIDTH) and basis["parts"] == {"across": 3, "within": 2}
    assert np.allclose(c @ c.T, np.eye(5), atol=1e-6)
    assert principal_cosines(c[:3], u).min() > 0.999
    assert principal_cosines(c[3:], v).min() > 0.999
    assert basis["within_variance_ratio"][3:].sum() > 0.95


def test_the_across_part_is_the_per_layer_fit():
    states, groups, _, _ = planted(1)
    basis = fit_hybrid_basis(states, groups, 3, 2)
    alone = _fit_one(states.astype(np.float64), 3)
    assert basis["components"][:3].tobytes() == alone["components"].tobytes()
    assert basis["mean"].tobytes() == alone["mean"].tobytes()


def _gram_schmidt_variant(states, groups, n_across, n_within):
    """The other construction: an independent within-group PCA, then Gram-Schmidt against the
    across part."""
    a = _fit_one(states.astype(np.float64), n_across)["components"].astype(np.float64)
    contrasts = _group_contrasts(states.astype(np.float64), groups)
    _, _, vt = np.linalg.svd(contrasts, full_matrices=False)
    w = vt[:n_within] - (vt[:n_within] @ a.T) @ a
    return _orthonormal_rows(w), contrasts


def test_the_within_part_holds_no_less_than_gram_schmidt_and_more_when_they_differ():
    # Where the two agree: as many within-group directions as the part keeps, outside the across span.
    states, groups, _, _ = planted(3)
    basis = fit_hybrid_basis(states, groups, 3, 2)
    gs, contrasts = _gram_schmidt_variant(states, groups, 3, 2)
    ours = ((contrasts @ basis["components"][3:].astype(np.float64).T) ** 2).sum()
    assert ours >= ((contrasts @ gs.T) ** 2).sum() * (1 - 1e-6)
    # Where they differ: four within-group directions, the two strongest mostly inside the across
    # span. An independent within-group fit spends its two directions on those, and Gram-Schmidt
    # keeps their small remainders; the fit here takes the two strongest directions outside it.
    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.standard_normal((WIDTH, WIDTH)))
    u, out = q[:, :3].T, q[:, 3:5].T
    inside = 0.95 * u[:2] + 0.312 * q[:, 5:7].T
    within = np.vstack([inside, out])
    centres = (rng.standard_normal((150, 3)) * 6.0) @ u
    wiggle = (rng.standard_normal((150, 6, 4)) * np.array([3.0, 3.0, 1.0, 1.0])) @ within
    states = (centres[:, None, :] + wiggle + rng.standard_normal((150, 6, WIDTH)) * 0.02).reshape(-1, WIDTH)
    groups = np.repeat(np.arange(150), 6)
    basis = fit_hybrid_basis(states, groups, 3, 2)
    gs, contrasts = _gram_schmidt_variant(states, groups, 3, 2)
    ours = ((contrasts @ basis["components"][3:].astype(np.float64).T) ** 2).sum()
    theirs = ((contrasts @ gs.T) ** 2).sum()
    assert ours > 1.2 * theirs


def test_halves_split_by_context_agree_on_every_planted_direction():
    states, groups, _, _ = planted(4, contexts=200)
    halves = (groups % 2) == 1
    rep = hybrid_reproducibility(states, groups, halves, 3, 2)
    assert rep["across"]["determined"] == 3 and rep["within"]["determined"] == 2
    assert rep["whole"]["determined"] == 5
    with pytest.raises(CalibrationFitError, match="both halves"):
        hybrid_reproducibility(states, groups, np.arange(len(groups)) % 2 == 1, 3, 2)


def test_refusals():
    states, groups, _, _ = planted(5, contexts=10)
    with pytest.raises(CalibrationFitError, match="one member"):
        fit_hybrid_basis(states[:7], np.arange(7), 2, 1)
    with pytest.raises(CalibrationFitError, match="components from"):
        fit_hybrid_basis(states, groups, WIDTH, 1)
    with pytest.raises(CalibrationFitError, match="group labels"):
        fit_hybrid_basis(states, groups[:-1], 2, 1)
    with pytest.raises(CalibrationFitError, match="negative"):
        fit_hybrid_basis(states, groups, -1, 1)
    zero_across = fit_hybrid_basis(states, groups, 0, 2)
    assert zero_across["components"].shape == (2, WIDTH)
