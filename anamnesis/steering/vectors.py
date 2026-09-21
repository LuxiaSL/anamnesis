"""Constructing a steering vector, and the conventions that decide what it means.

Six constructions live here, and they were six scripts because each arrived with
an experiment. Collected, they are one vocabulary:

**Mean difference** (CAA). ``unit(mean(A) - mean(B))`` over mean residuals of two
conditions. Two laws, and they are not the same law: :func:`mean_difference` is
the difference of the two condition means, while :func:`paired_mean_difference`
averages the *per-pair* difference over pairs matched on topic. They coincide
only when the pairing is complete and balanced, and the paired form is what the
banked contrastive vectors are.

**Whitened mean difference** (LDA). ``unit(Σ⁻¹Δ)``. A contrast can be plainly
detectable while the raw mean difference is nearly orthogonal to the direction
that separates it, and then mean-difference steering has almost no purchase.
Which Σ is not a detail: :func:`pooled_within_class_covariance` estimates it from
both conditions' centred rows, :func:`target_covariance` from the injection
target's tokens alone — the convention for a model-difference install lever,
where the distribution being written into is the metric.

**Band projection.** ``unit(P[lo:hi] v)``, ``P`` projecting onto covariance
eigenvectors ranked ``lo..hi`` in *descending* eigenvalue order — the easy thing
to get backwards, so :class:`Spectrum` sorts explicitly. **Orthogonalization**
(``unit(v - (v·u)u)``) separates a candidate's own content from a component it
shares with a known direction.

**Matched nulls.** A random unit vector is the null for an isotropic claim and
nothing else; a band-confined vector is read against nulls drawn *in its band*
(:func:`random_band_vector`), because a pooled null mixes supports.

**Dose.** The magnitude unit is the model's own median residual norm at the
injection site, so ``alpha_frac`` means the same fraction of typical state across
sites, and every absolute alpha is reconstructible from the stamp.

**The sweep law.** A site is chosen by *held-out* Cohen's d along the mean
difference: average the samples of each prompt first, split over prompts, fit the
direction on the training half, read d on the held-out half. Averaging per prompt
is what stops repeated samples of one prompt from counting as independent points;
splitting over prompts is what makes the number honest rather than in-sample.

**All of it is per-model.** The basis, the covariance, the dose currency and the
site are four properties of one checkpoint, and a vector built here carries no
licence to be injected into another model.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]

#: Eigen-index band, descending eigenvalue order, shared by the band projection
#: and its matched nulls. The top of the spectrum is where the model's state
#: already lives and the tail is off-manifold; the band is the region where a
#: direction is both expressible and not simply the dominant variance.
BAND: tuple[int, int] = (16, 256)

#: Seed for the isotropic random controls banked beside a construction.
RANDOM_SEED: int = 20260713

#: Seed for band-confined matched nulls.
NULL_SEED: int = 20260729

#: The two file names a vector bank is written under. On-disk keys are frozen:
#: banked runs are read by these names.
BANK_NPZ = "a5_vectors.npz"
BANK_STAMPS = "a5_vectors_stamps.json"

#: Date substituted into a chat template when one asks for it. Pinned to a
#: literal so a contrastive-prompt construction is reproducible from text: a
#: template that interpolates today's date makes the prompt, and therefore the
#: vector, a function of when it was built.
CHAT_TEMPLATE_DATE = "12 Jul 2026"

#: The formality contrast: the system-prompt pair the banked V1 vectors were
#: built from. A label in banked data means this exact text.
FORMAL_SYSTEM_PROMPT = (
    "You are an extremely formal assistant. Respond with maximal formality: "
    "precise, ceremonious, professional register; no contractions, no "
    "colloquialisms, no humor."
)
INFORMAL_SYSTEM_PROMPT = (
    "You are a super casual assistant. Keep it loose and chatty — use slang, "
    "contractions, and casual asides, like you're texting a friend."
)

#: The user-prompt templates the formality contrast is measured over, one pair of
#: generations per (topic, template).
CONTRAST_TEMPLATES: tuple[str, ...] = ("Write about {topic}.", "Explain {topic} to a beginner.")


# ── Elementary geometry ───────────────────────────────────────────────────────
def unit(v: NDArray[Any]) -> F64:
    """``v`` scaled to unit length in float64. Raises on a zero vector."""
    v64 = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(v64))
    if norm <= 0.0:
        raise ValueError("cannot unit-normalize a zero vector")
    return v64 / norm


def cosine(a: NDArray[Any], b: NDArray[Any]) -> float:
    """Cosine between two vectors, without assuming either is normalized."""
    a64, b64 = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a64) * np.linalg.norm(b64))
    if denom <= 0.0:
        raise ValueError("cannot take a cosine against a zero vector")
    return float(a64 @ b64 / denom)


# ── The covariance eigenbasis ─────────────────────────────────────────────────
@dataclass(frozen=True)
class Spectrum:
    """A residual covariance's eigendecomposition, held in descending order.

    The order is the invariant. A banked eigendecomposition may be stored
    ascending — that is what :func:`numpy.linalg.eigh` returns — and the
    difference between reading the top of a spectrum and reading its tail is a
    sort. Every constructor here sorts explicitly, so a ``Spectrum`` in hand is
    descending whatever its source was. ``ridge`` regularizes the inverse:
    eigenvalues reach machine zero in a space wider than the sample count, and a
    bare inverse there reports the conditioning rather than the vector.
    """

    evals: F64
    evecs: F64
    ridge: float = 0.0

    def __post_init__(self) -> None:
        if self.evecs.ndim != 2 or self.evals.ndim != 1:
            raise ValueError(f"expected [d] evals and [d, d] evecs, got {self.evals.shape} / {self.evecs.shape}")
        if self.evecs.shape[1] != self.evals.shape[0]:
            raise ValueError(f"evecs {self.evecs.shape} does not match {self.evals.shape[0]} eigenvalues")

    @property
    def d(self) -> int:
        return int(self.evals.shape[0])

    @staticmethod
    def from_arrays(evals: NDArray[Any], evecs: NDArray[Any], ridge: float = 0.0) -> Spectrum:
        """Sort an eigenpair into descending order and clip negative eigenvalues."""
        ev = np.asarray(evals, dtype=np.float64)
        U = np.asarray(evecs, dtype=np.float64)
        order = np.argsort(-ev)
        return Spectrum(evals=np.clip(ev[order], 0.0, None), evecs=U[:, order], ridge=float(ridge))

    @staticmethod
    def from_covariance(sigma: NDArray[Any], ridge_rel: float = 0.0) -> Spectrum:
        """Eigendecompose a covariance matrix; ``ridge = ridge_rel × mean(eigenvalue)``."""
        evals, evecs = np.linalg.eigh(np.asarray(sigma, dtype=np.float64))
        evals = np.clip(evals, 0.0, None)
        return Spectrum.from_arrays(evals, evecs, ridge=ridge_rel * float(evals.mean()))

    @staticmethod
    def from_rows(rows: NDArray[Any], ridge_rel: float = 0.0) -> Spectrum:
        """Covariance of ``[n, d]`` observations, then its eigendecomposition.

        The estimator is the centred sample covariance with ``n - 1`` in the
        denominator, which is what a residual-covariance screen is built on.
        """
        X = np.asarray(rows, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"expected [n, d] rows, got shape {X.shape}")
        centred = X - X.mean(axis=0)
        sigma = (centred.T @ centred) / max(len(centred) - 1, 1)
        return Spectrum.from_covariance(sigma, ridge_rel=ridge_rel)

    @staticmethod
    def from_npz(path: Path | str, evals_key: str = "evals", evecs_key: str = "evecs",
                 ridge_key: str = "ridge") -> Spectrum:
        """Read a banked eigendecomposition, sorting it descending on the way in."""
        z = np.load(Path(path))
        for key in (evals_key, evecs_key):
            if key not in z:
                raise KeyError(f"{path} missing key {key!r} (has {list(z.keys())})")
        ridge = float(z[ridge_key]) if ridge_key in z else 0.0
        return Spectrum.from_arrays(z[evals_key], z[evecs_key], ridge=ridge)

    def energy_profile(self, v: NDArray[Any]) -> F64:
        """Squared projection of a unit ``v`` onto each eigendirection; sums to 1."""
        return (self.evecs.T @ unit(v)) ** 2

    def mahalanobis(self, v: NDArray[Any]) -> float:
        """``vᵀ(Σ + ridge·I)⁻¹v`` for unit ``v``: how far into the low-variance tail it points."""
        coeff = self.evecs.T @ unit(v)
        return float((coeff**2 / (self.evals + self.ridge)).sum())

    def band_basis(self, band: tuple[int, int] = BAND) -> F64:
        """The ``[d, hi-lo]`` eigenvector block ranked ``lo..hi``, descending."""
        lo, hi = _validated_band(band, self.d)
        return self.evecs[:, lo:hi]

    def mass_profile(self, v: NDArray[Any], band: tuple[int, int] = BAND) -> dict[str, float]:
        """Fraction of ``v``'s energy above, inside and below a band."""
        lo, hi = _validated_band(band, self.d)
        m = self.energy_profile(v)
        return {
            f"top{lo}": float(m[:lo].sum()),
            f"band{lo}_{hi}": float(m[lo:hi].sum()),
            f"tail{hi}plus": float(m[hi:].sum()),
        }


def _validated_band(band: tuple[int, int], d: int) -> tuple[int, int]:
    lo, hi = int(band[0]), int(band[1])
    if not 0 <= lo < hi:
        raise ValueError(f"band must satisfy 0 <= lo < hi, got {band}")
    if lo >= d:
        raise ValueError(f"band {band} starts past the spectrum's {d} dimensions")
    return lo, min(hi, d)


# ── Mean difference (CAA) ─────────────────────────────────────────────────────
def mean_difference(pos: NDArray[Any], neg: NDArray[Any]) -> F64:
    """``mean(pos) - mean(neg)`` over ``[n, d]`` per-generation means, unnormalized.

    The norm is a reported quantity — a tiny one is the tell that a construction
    lives in the low-variance tail — so the displacement is kept raw.
    """
    A = np.asarray(pos, dtype=np.float64)
    B = np.asarray(neg, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError(f"expected [n, d] and [m, d] with equal d, got {A.shape} / {B.shape}")
    if len(A) == 0 or len(B) == 0:
        raise ValueError("mean difference needs at least one observation per condition")
    return A.mean(axis=0) - B.mean(axis=0)


def paired_mean_difference(pairs: Sequence[tuple[NDArray[Any], NDArray[Any]]]) -> F64:
    """Mean of the per-pair differences, unnormalized — the banked construction.

    Each element is one matched pair's ``(positive, negative)`` mean residual, so
    a pair that could not be formed is absent rather than averaged in on one side.
    """
    if not pairs:
        raise ValueError("paired mean difference needs at least one pair")
    diffs = [np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64) for a, b in pairs]
    return np.mean(np.stack(diffs), axis=0)


def per_prompt_average(rows: NDArray[Any], prompt_ids: Sequence[str]) -> dict[str, F64]:
    """Average the repeated samples of each prompt — the displacement convention.

    The unit of a contrast is the prompt, not the sample: averaging first keeps M
    samples of one prompt from acting like M independent observations downstream,
    the sweep's split included.
    """
    X = np.asarray(rows, dtype=np.float64)
    if len(X) != len(prompt_ids):
        raise ValueError(f"{len(X)} rows against {len(prompt_ids)} prompt ids")
    grouped: dict[str, list[NDArray[Any]]] = defaultdict(list)
    for row, pid in zip(X, prompt_ids):
        grouped[str(pid)].append(row)
    return {pid: np.mean(np.stack(vals), axis=0) for pid, vals in grouped.items()}


def pair_on_prompts(pos: Mapping[str, F64], neg: Mapping[str, F64],
                    min_shared: int = 8) -> tuple[F64, F64, list[str]]:
    """Align two per-prompt maps on their shared prompts, in sorted order.

    ``min_shared`` refuses a pairing too thin to carry a split rather than
    reporting a number over three prompts.
    """
    shared = sorted(set(pos) & set(neg))
    if len(shared) < min_shared:
        raise ValueError(f"only {len(shared)} shared prompts, fewer than the {min_shared} required")
    return (np.stack([pos[p] for p in shared]), np.stack([neg[p] for p in shared]), shared)


# ── Whitened mean difference (LDA) ────────────────────────────────────────────
def ledoit_wolf_covariance(rows: NDArray[Any], shrink_scale: float | None = None) -> tuple[F64, float]:
    """Ledoit–Wolf shrinkage covariance of ``[n, d]`` rows, and the shrinkage used.

    Shrinkage is not optional at these widths: the residual stream is wider than
    any sample of it, so the empirical covariance is singular and ``Σ⁻¹Δ``
    without shrinkage is noise amplification. ``shrink_scale`` rescales the
    automatic λ and rebuilds Σ at that value through the convex form
    ``Σ(λ) = (1-λ)·S + λ·(tr S / d)·I``, clipped to ``[0, 1]``, so the whitened
    direction's dependence on λ can be measured rather than assumed; ``None`` is
    the automatic estimate.
    """
    from sklearn.covariance import LedoitWolf

    X = np.asarray(rows, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected [n, d] rows, got shape {X.shape}")
    lw = LedoitWolf().fit(X)
    sigma = np.asarray(lw.covariance_, dtype=np.float64)
    shrinkage = float(lw.shrinkage_)
    if shrink_scale is not None:
        empirical = np.cov(X, rowvar=False, bias=True)
        lam = float(np.clip(shrinkage * float(shrink_scale), 0.0, 1.0))
        mu = float(np.trace(empirical)) / empirical.shape[0]
        sigma = (1.0 - lam) * empirical + lam * mu * np.eye(empirical.shape[0])
        shrinkage = lam
    return sigma, shrinkage


def pooled_within_class_rows(pos: NDArray[Any], neg: NDArray[Any]) -> F64:
    """Both conditions' rows centred on their own means — the within-class scatter.

    The estimand is the variation *inside* a condition; stacking uncentred rows
    puts the between-condition difference into the metric meant to divide it out.
    """
    A = np.asarray(pos, dtype=np.float64)
    B = np.asarray(neg, dtype=np.float64)
    return np.vstack([A - A.mean(axis=0), B - B.mean(axis=0)])


def pooled_within_class_covariance(pos: NDArray[Any], neg: NDArray[Any],
                                   shrink_scale: float | None = None) -> tuple[F64, float]:
    """Σ from both conditions' centred rows — the two-class discriminant metric."""
    return ledoit_wolf_covariance(pooled_within_class_rows(pos, neg), shrink_scale=shrink_scale)


def target_covariance(target_rows: NDArray[Any], shrink_scale: float | None = None) -> tuple[F64, float]:
    """Σ from the injection target's own tokens — the install-lever metric.

    When a direction is built from a difference between two models but injected
    into one of them, the model being written into is what has to absorb it, so
    its covariance is the metric. Pooling the other model's rows in whitens
    against a space that never occurs at steering time.
    """
    return ledoit_wolf_covariance(target_rows, shrink_scale=shrink_scale)


def whitened_direction(delta: NDArray[Any], sigma: NDArray[Any]) -> F64:
    """``Σ⁻¹Δ``, unnormalized: the discriminative direction behind a mean difference."""
    d64 = np.asarray(delta, dtype=np.float64)
    return np.linalg.solve(np.asarray(sigma, dtype=np.float64), d64)


def whitening_diagnostics(pos: NDArray[Any], neg: NDArray[Any], sigma: NDArray[Any],
                          shrinkage: float) -> dict[str, float | int]:
    """What the whitening did, and whether the raw mean difference was already it.

    ``cos_delta_whitened`` is the residual-space test: low means the mean
    difference misses the discriminative direction, and mean-difference steering
    acts weak at a site where the contrast is plainly detectable.
    ``mahalanobis_d`` is ``sqrt(ΔᵀΣ⁻¹Δ)``, the whitened separation that picks the
    layer; ``raw_caa_cohend_proxy`` is ``‖Δ‖`` over the within-class rms beside it.
    """
    A = np.asarray(pos, dtype=np.float64)
    B = np.asarray(neg, dtype=np.float64)
    delta = mean_difference(A, B)
    w = whitened_direction(delta, sigma)
    scatter = 0.5 * (
        np.sqrt((((A - A.mean(axis=0)) ** 2).sum(axis=1)).mean())
        + np.sqrt((((B - B.mean(axis=0)) ** 2).sum(axis=1)).mean())
    )
    return {
        "cos_delta_whitened": abs(cosine(delta, w)),
        "mahalanobis_d": float(np.sqrt(max(float(delta @ w), 0.0))),
        "raw_caa_cohend_proxy": float(np.linalg.norm(delta) / scatter) if scatter > 0 else 0.0,
        "lw_shrinkage": float(shrinkage),
        "delta_norm": float(np.linalg.norm(delta)),
        "n_pos": int(len(A)),
        "n_neg": int(len(B)),
    }


# ── Band projection ───────────────────────────────────────────────────────────
def band_pass(v: NDArray[Any], spectrum: Spectrum, band: tuple[int, int] = BAND) -> F64:
    """``unit(P[lo:hi] v)``: the part of ``v`` inside a descending-eigenvalue band.

    Raises when ``v`` has no band component: the vector that would come back is a
    normalized rounding error.
    """
    v64 = np.asarray(v, dtype=np.float64)
    U = spectrum.band_basis(band)
    if U.shape[0] != v64.shape[0]:
        raise ValueError(f"spectrum is {U.shape[0]}-dimensional, vector is {v64.shape[0]}")
    projected = U @ (U.T @ v64)
    if np.linalg.norm(projected) <= 1e-12:
        raise ValueError("vector has no component inside the band — no member can be constructed")
    return unit(projected)


def band_pass_anatomy(v: NDArray[Any], spectrum: Spectrum, band: tuple[int, int] = BAND,
                      comparators: Mapping[str, NDArray[Any]] | None = None) -> dict[str, Any]:
    """A band member beside what it was built from, so the member can be read.

    ``band_mass_of_raw`` is the fraction of the source the band retained. A small
    one says the member is an amplified residue rather than a genuine
    band-confined direction — an artifact of renormalizing almost nothing.
    """
    v64 = np.asarray(v, dtype=np.float64)
    member = band_pass(v64, spectrum, band)
    U = spectrum.band_basis(band)
    projected = U @ (U.T @ v64)
    return {
        "band": list(band),
        "band_mass_of_raw": float(np.linalg.norm(projected) / np.linalg.norm(v64)),
        "mass_profiles": {
            "raw": spectrum.mass_profile(v64, band),
            "member": spectrum.mass_profile(member, band),
        },
        "cos": {name: cosine(member, c) for name, c in (comparators or {}).items()},
    }


# ── Orthogonalization ─────────────────────────────────────────────────────────
def orthogonalize(v: NDArray[Any], reference: NDArray[Any], min_residual: float = 1e-6) -> F64:
    """``unit(v - (v·u)u)``: ``v`` with its component along ``reference`` removed.

    Both inputs are unit-normalized first, so the projection coefficient is the
    cosine. Raises below ``min_residual``: a vector parallel to the reference has
    no independent direction, and normalizing what is left manufactures one out
    of numerical noise.
    """
    v_unit = unit(v)
    u_unit = unit(reference)
    residual = v_unit - (v_unit @ u_unit) * u_unit
    frac = float(np.linalg.norm(residual))
    if frac <= min_residual:
        raise ValueError(f"vector is parallel to the reference (residual fraction {frac:.3g})")
    return residual / frac


def orthogonalization_anatomy(v: NDArray[Any], reference: NDArray[Any],
                              max_residual_cosine: float = 1e-8) -> tuple[F64, dict[str, float]]:
    """Orthogonalize, then check the orthogonality actually holds.

    ``residual_norm_fraction`` is how much of the source survives — whether the
    result is a real independent component or the last few percent of one. The
    cosine check is hard: a construction whose defining property fails
    numerically raises rather than banking.
    """
    v_unit = unit(v)
    u_unit = unit(reference)
    perp = orthogonalize(v_unit, u_unit)
    residual_cos = cosine(perp, u_unit)
    if abs(residual_cos) > max_residual_cosine:
        raise ValueError(f"orthogonalization failed: cosine to the reference is {residual_cos:.3g}")
    return perp, {
        "cos_source_reference": cosine(v_unit, u_unit),
        "residual_norm_fraction": float(np.linalg.norm(v_unit - (v_unit @ u_unit) * u_unit)),
        "cos_perp_reference": residual_cos,
        "cos_perp_source": cosine(perp, v_unit),
    }


# ── Matched nulls ─────────────────────────────────────────────────────────────
def random_unit_vectors(dim: int, count: int = 3, seed: int = RANDOM_SEED) -> dict[str, F32]:
    """``count`` seeded isotropic unit vectors, keyed ``R1..Rn``.

    The null for a direction with no spectral confinement. A band- or
    tail-confined vector needs :func:`random_band_vector` instead: an isotropic
    draw is not its matched control.
    """
    rng = np.random.default_rng(seed)
    return {f"R{i}": unit(rng.standard_normal(int(dim))).astype(np.float32) for i in range(1, count + 1)}


def random_band_vector(spectrum: Spectrum, band: tuple[int, int] = BAND,
                       rng: np.random.Generator | None = None) -> F64:
    """A unit vector drawn inside a band: the matched null for a band member.

    Support is the thing being matched, because a vector's deformation cost is a
    function of where in the spectrum it points and a pooled null mixes supports.
    """
    generator = rng if rng is not None else np.random.default_rng(NULL_SEED)
    U = spectrum.band_basis(band)
    return unit(U @ generator.standard_normal(U.shape[1]))


# ── Dose ──────────────────────────────────────────────────────────────────────
def dose_alpha(alpha_frac: float, median_residual_norm: float) -> float:
    """Absolute injection magnitude from a fraction of typical state.

    A unit vector says nothing about how hard to push. The currency is the
    model's own median residual norm at the site, so ``alpha_frac`` is comparable
    across sites while the absolute alpha stays reconstructible from the stamp.
    """
    if median_residual_norm <= 0.0:
        raise ValueError(f"median residual norm must be positive, got {median_residual_norm}")
    return float(alpha_frac) * float(median_residual_norm)


def median_row_norm(rows: NDArray[Any]) -> float:
    """Median L2 norm over ``[n, d]`` observations — the dose currency, pooled.

    Pooled over positions rather than averaged per generation: the quantity is
    the typical norm of a state being written into, and a median of
    per-generation medians weights a short generation like a long one.
    """
    X = np.asarray(rows, dtype=np.float64)
    if X.ndim != 2 or len(X) == 0:
        raise ValueError(f"expected non-empty [n, d] rows, got shape {X.shape}")
    return float(np.median(np.linalg.norm(X, axis=1)))


# ── The sweep law ─────────────────────────────────────────────────────────────
def heldout_cohens_d(
    a_fit: NDArray[Any], a_eval: NDArray[Any], b_fit: NDArray[Any], b_eval: NDArray[Any],
    *, min_direction_norm: float, sd_floor: float | None,
) -> float:
    """Cohen's d along a direction fitted on one split and read on another.

    The direction is the mean difference of the *fitting* rows; d is measured on
    the *evaluation* rows. Fitting and reading on the same rows reports
    in-sample optimism, which at these widths is large enough to invent a peak.

    Two conventions differ where the statistic degenerates, and both are
    available rather than silently merged: a numeric ``sd_floor`` divides by
    ``max(sd, sd_floor)`` so a zero-variance projection returns a large value,
    while ``None`` returns ``0.0`` there. Below ``min_direction_norm`` the fitted
    direction counts as absent and the result is ``0.0``.
    """
    a_fit64 = np.asarray(a_fit, dtype=np.float64)
    b_fit64 = np.asarray(b_fit, dtype=np.float64)
    direction = a_fit64.mean(axis=0) - b_fit64.mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm < min_direction_norm:
        return 0.0
    direction = direction / norm
    pa = np.asarray(a_eval, dtype=np.float64) @ direction
    pb = np.asarray(b_eval, dtype=np.float64) @ direction
    pooled = float(np.sqrt(0.5 * (pa.var(ddof=1) + pb.var(ddof=1))))
    if sd_floor is None:
        return float((pa.mean() - pb.mean()) / pooled) if pooled > 0 else 0.0
    return float((pa.mean() - pb.mean()) / max(pooled, sd_floor))


def half_split_sweep(
    a: NDArray[Any], b: NDArray[Any], k_splits: int = 50, rng: np.random.Generator | None = None,
) -> tuple[F64, F64]:
    """Per-layer held-out Cohen's d over K random half-splits, and its spread.

    Inputs are ``[n_prompts, n_layers, d]`` — per-prompt averaged already, which
    is what makes a split over the first axis a split over prompts. Returns the
    mean and standard deviation of d at each layer across the splits; the peak
    layer is the site, and the spread says whether the peak is a peak.
    """
    A = np.asarray(a, dtype=np.float64)
    B = np.asarray(b, dtype=np.float64)
    if A.ndim != 3 or B.ndim != 3 or A.shape[1:] != B.shape[1:]:
        raise ValueError(f"expected [n, L, d] with matching L and d, got {A.shape} / {B.shape}")
    generator = rng if rng is not None else np.random.default_rng(17)
    ds = np.zeros((int(k_splits), A.shape[1]), dtype=np.float64)
    for k in range(int(k_splits)):
        ia, ib = generator.permutation(len(A)), generator.permutation(len(B))
        ha, hb = len(A) // 2, len(B) // 2
        a_tr, a_te, b_tr, b_te = A[ia[:ha]], A[ia[ha:]], B[ib[:hb]], B[ib[hb:]]
        for s in range(A.shape[1]):
            ds[k, s] = heldout_cohens_d(a_tr[:, s], a_te[:, s], b_tr[:, s], b_te[:, s],
                                        min_direction_norm=1e-8, sd_floor=1e-8)
    return ds.mean(axis=0), ds.std(axis=0)


# ── Capture: residual means and norms over a model's own forwards ─────────────
def chat_input_ids(tokenizer: Any, user: str, system: str | None) -> Any:
    """Chat-template token ids for one turn, with the template's date pinned."""
    messages: list[dict[str, str]] = [{"role": "user", "content": user}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    result = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", date_string=CHAT_TEMPLATE_DATE
    )
    return result if hasattr(result, "shape") else result["input_ids"]


def mean_residual_at_sites(model: Any, ids: Any, prompt_length: int, sites: Sequence[int]) -> dict[int, F64]:
    """Mean residual over generated positions at each site, one forward pass.

    A site is the *input* to that decoder layer, ``hidden_states[site]`` — index
    0 is the embedding output, so the index and the layer number line up for an
    input and are off by one for an output.
    """
    import torch

    with torch.no_grad():
        out = model(
            ids.to(next(model.parameters()).device),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    result: dict[int, F64] = {}
    for site in sites:
        h = out.hidden_states[site][0, int(prompt_length):]
        result[int(site)] = h.float().mean(dim=0).cpu().numpy().astype(np.float64)
    return result


def replay_entries(manifest_path: Path | str) -> dict[str, dict[str, Any]]:
    """The ``entries`` map of a replay manifest, keyed by generation id as a string."""
    return json.loads(Path(manifest_path).read_text())["entries"]


def capture_mean_residuals(
    model: Any, entries: Mapping[str, Mapping[str, Any]], sites: Sequence[int],
    limit: int | None = None, min_generated: int = 1, log_every: int = 50,
) -> dict[int, F64]:
    """``{site: [n_gens, d]}`` per-generation mean residuals over banked spans.

    Teacher-forced over banked text, so the capture is reproducible from the
    manifest and does not depend on sampling.
    """
    import torch

    keys = sorted(entries, key=lambda k: int(k))
    if limit is not None:
        keys = keys[:limit]
    device = next(model.parameters()).device
    accumulated: dict[int, list[F64]] = {int(s): [] for s in sites}
    for i, key in enumerate(keys):
        entry = entries[key]
        ids = torch.tensor([entry["input_ids"]], dtype=torch.long, device=device)
        prompt_length = int(entry["prompt_length"])
        if ids.shape[1] - prompt_length < min_generated:
            continue
        means = mean_residual_at_sites(model, ids, prompt_length, sites)
        for site, value in means.items():
            accumulated[site].append(value)
        if log_every and (i + 1) % log_every == 0:
            logger.info(f"captured {i + 1}/{len(keys)} generations")
    for site, rows in accumulated.items():
        if not rows:
            raise ValueError(f"no usable generations for site {site}")
    return {site: np.stack(rows) for site, rows in accumulated.items()}


def capture_median_residual_norms(model: Any, entries: Mapping[str, Mapping[str, Any]],
                                  sites: Sequence[int], n_gens: int = 20) -> dict[str, float]:
    """``{"L<site>": median ‖h‖}`` over generated positions — the dose currency.

    Generations are an even stride through the banked ids rather than the first
    ``n_gens``, so the norm is read across the protocol's classes instead of
    whichever class happens to be banked first.
    """
    import torch

    all_ids = sorted(int(k) for k in entries)
    stride = max(1, len(all_ids) // max(n_gens, 1))
    chosen = all_ids[::stride][:n_gens]
    device = next(model.parameters()).device
    per_site: dict[int, list[float]] = {int(s): [] for s in sites}
    for gen_id in chosen:
        entry = entries[str(gen_id)]
        ids = torch.tensor([entry["input_ids"]], dtype=torch.long, device=device)
        prompt_length = int(entry["prompt_length"])
        with torch.no_grad():
            out = model(ids, use_cache=False, output_hidden_states=True, return_dict=True)
        for site in per_site:
            h = out.hidden_states[site][0, prompt_length:]
            per_site[site].extend(h.float().norm(dim=-1).cpu().numpy().tolist())
    norms = {f"L{site}": float(np.median(values)) for site, values in per_site.items()}
    logger.info(f"median residual norms {norms} over {len(chosen)} generations")
    return norms


# ── The two banked constructions ──────────────────────────────────────────────
def _vectors_from_pairs(pairs: Mapping[int, list[tuple[F64, F64]]], key_prefix: str,
                        stamp: Mapping[str, Any]) -> tuple[dict[str, F32], dict[str, dict[str, Any]]]:
    """Paired differences per site into unit vectors and their stamps."""
    vectors: dict[str, F32] = {}
    stamps: dict[str, dict[str, Any]] = {}
    for site, site_pairs in pairs.items():
        raw = paired_mean_difference(site_pairs)
        key = f"{key_prefix}_L{site}"
        vectors[key] = unit(raw).astype(np.float32)
        stamps[key] = dict(stamp, raw_norm=float(np.linalg.norm(raw)))
    return vectors, stamps


def build_contrastive_prompt_vectors(
    model: Any, tokenizer: Any, topics: Sequence[str], sites: Sequence[int], *,
    system_pair: tuple[str, str] = (FORMAL_SYSTEM_PROMPT, INFORMAL_SYSTEM_PROMPT),
    templates: Sequence[str] = CONTRAST_TEMPLATES, max_new_tokens: int = 160,
    temperature: float = 0.7, top_p: float = 0.9, eos_token_ids: Sequence[int] = (),
    pad_token_id: int | None = None, key_prefix: str = "V1", trait: str = "formality",
    min_generated: int = 8,
) -> tuple[dict[str, F32], dict[str, dict[str, Any]]]:
    """Contrast two system prompts over the same user prompts, one vector per site.

    Each ``(topic, template)`` yields one seeded generation under each system
    prompt and contributes the difference of their mean residuals. A pair whose
    generation is too short to have a span is dropped whole: keeping one side
    would put an unmatched condition mean into a paired average.
    """
    import torch

    device = next(model.parameters()).device
    pairs: dict[int, list[tuple[F64, F64]]] = {int(s): [] for s in sites}
    n_pairs = 0
    for topic_index, topic in enumerate(topics):
        for template_index, template in enumerate(templates):
            user = template.format(topic=topic)
            seed = 100000 + topic_index * 10 + template_index
            pair_means: dict[str, dict[int, F64]] | None = {}
            for condition, system_prompt in zip(("positive", "negative"), system_pair):
                ids = chat_input_ids(tokenizer, user, system_prompt).to(device)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                with torch.no_grad():
                    sequence = model.generate(
                        ids,
                        attention_mask=torch.ones_like(ids),
                        max_new_tokens=int(max_new_tokens),
                        do_sample=True,
                        temperature=float(temperature),
                        top_p=float(top_p),
                        eos_token_id=list(eos_token_ids) or None,
                        pad_token_id=pad_token_id,
                    )
                if sequence.shape[1] - ids.shape[1] < min_generated:
                    logger.warning(
                        f"{key_prefix} {condition} topic {topic_index} template {template_index}: "
                        "generation too short, dropping the pair"
                    )
                    pair_means = None
                    break
                assert pair_means is not None
                pair_means[condition] = mean_residual_at_sites(model, sequence, int(ids.shape[1]), sites)
            if pair_means is None:
                continue
            for site in pairs:
                pairs[site].append((pair_means["positive"][site], pair_means["negative"][site]))
            n_pairs += 1

    logger.info(f"{key_prefix} built from {n_pairs} pairs")
    return _vectors_from_pairs(
        pairs, key_prefix, {"trait": trait, "route": "contrastive-prompt", "n_pairs": n_pairs}
    )


def build_replay_contrast_vectors(
    model: Any, corpora: Mapping[str, tuple[Path, Path]], pair: tuple[str, str],
    sites: Sequence[int], *, per_topic: int = 2, key_prefix: str = "V3",
    trait: str = "mode-dir0", min_generated: int = 8,
) -> tuple[dict[str, F32], dict[str, dict[str, Any]]]:
    """Contrast two banked corpora on shared topics, one vector per site.

    ``corpora`` maps each label in ``pair`` to its ``(replay_manifest,
    metadata)`` paths. Pairing is by topic, because the contrast is how the same
    subject was processed and an unpaired difference of corpus means carries
    whatever the topics differed in. Teacher-forced replay throughout, so the
    construction is reproducible from the manifests.
    """
    import torch

    manifests: dict[str, dict[str, Any]] = {}
    by_topic: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for label in pair:
        manifest_path, metadata_path = corpora[label]
        manifests[label] = replay_entries(manifest_path)
        document = json.loads(Path(metadata_path).read_text())
        generations = document["generations"] if "generations" in document else document
        grouped: dict[int, list[dict[str, Any]]] = {}
        for generation in generations:
            grouped.setdefault(int(generation["topic_idx"]), []).append(generation)
        by_topic[label] = grouped

    shared_topics = sorted(set(by_topic[pair[0]]) & set(by_topic[pair[1]]))
    pairs: dict[int, list[tuple[F64, F64]]] = {int(s): [] for s in sites}
    n_topics = 0
    for topic in shared_topics:
        means: dict[str, dict[int, F64]] = {}
        for label in pair:
            generations = sorted(by_topic[label][topic], key=lambda g: int(g["generation_id"]))[:per_topic]
            per_generation: list[dict[int, F64]] = []
            for generation in generations:
                entry = manifests[label].get(str(generation["generation_id"]))
                if entry is None or (len(entry["input_ids"]) - entry["prompt_length"]) < min_generated:
                    continue
                ids = torch.tensor([entry["input_ids"]], dtype=torch.long)
                per_generation.append(
                    mean_residual_at_sites(model, ids, int(entry["prompt_length"]), sites)
                )
            if not per_generation:
                break
            means[label] = {
                site: np.mean([m[site] for m in per_generation], axis=0) for site in pairs
            }
        if len(means) != len(pair):
            continue
        for site in pairs:
            pairs[site].append((means[pair[0]][site], means[pair[1]][site]))
        n_topics += 1

    logger.info(f"{key_prefix} built from {n_topics} same-topic pairs ({pair[0]} - {pair[1]})")
    return _vectors_from_pairs(pairs, key_prefix, {
        "trait": trait, "route": "activation-contrast", "pair": list(pair),
        "n_topics": n_topics, "per_topic": int(per_topic),
    })


def build_whitened_vectors(
    pos: Mapping[int, NDArray[Any]], neg: Mapping[int, NDArray[Any]], *,
    shrink_scale: float | None = None, key_prefix: str = "V3w", raw_key_prefix: str = "V3raw",
) -> tuple[dict[str, F32], dict[str, dict[str, Any]]]:
    """Whitened and raw mean-difference vectors per site, from captured means.

    Both come from the same capture on purpose: the raw direction is the control
    for whether whitening moved anything, and a control computed from a different
    capture would confound the two.
    """
    vectors: dict[str, F32] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for site in sorted(pos):
        A = np.asarray(pos[site], dtype=np.float64)
        B = np.asarray(neg[site], dtype=np.float64)
        sigma, shrinkage = pooled_within_class_covariance(A, B, shrink_scale=shrink_scale)
        delta = mean_difference(A, B)
        vectors[f"{key_prefix}_L{site}"] = unit(whitened_direction(delta, sigma)).astype(np.float32)
        vectors[f"{raw_key_prefix}_L{site}"] = unit(delta).astype(np.float32)
        diagnostics[f"L{site}"] = dict(
            whitening_diagnostics(A, B, sigma, shrinkage), shrink_scale=shrink_scale
        )
    return vectors, diagnostics


def build_install_vectors(
    donor_means: Mapping[int, NDArray[Any]], base_means: Mapping[int, NDArray[Any]],
    target_tokens: Mapping[int, NDArray[Any]], *, band: tuple[int, int] = BAND,
    seed: int = NULL_SEED, key_prefix: str = "Install",
) -> tuple[dict[str, F64], dict[str, dict[str, Any]]]:
    """The model-difference install lever: raw, whitened, band null and dose per site.

    ``Δ`` is the difference of two models' means over the same text; ``Σ`` and the
    dose currency both come from ``target_tokens``, the per-token rows of the
    model being injected into. Those are different banks when the contrast is
    content-controlled, and conflating them whitens against a distribution that
    never occurs at steering time.
    """
    rng = np.random.default_rng(seed)
    vectors: dict[str, F64] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for site in sorted(donor_means):
        delta = mean_difference(np.asarray(donor_means[site]), np.asarray(base_means[site]))
        tokens = np.asarray(target_tokens[site], dtype=np.float64)
        sigma, shrinkage = target_covariance(tokens)
        raw = unit(delta)
        whitened = unit(whitened_direction(delta, sigma))
        spectrum = Spectrum.from_covariance(sigma)
        vectors[f"{key_prefix}_L{site}"] = raw
        vectors[f"{key_prefix}W_L{site}"] = whitened
        vectors[f"Rband_L{site}"] = random_band_vector(spectrum, band, rng)
        median_norm = median_row_norm(tokens)
        vectors[f"median_norm_L{site}"] = np.asarray(median_norm, dtype=np.float64)
        diagnostics[f"L{site}"] = {
            "delta_norm": float(np.linalg.norm(delta)),
            "cos_raw_whitened": float(raw @ whitened),
            "lw_shrinkage": float(shrinkage),
            "target_median_token_norm": median_norm,
            "n_target_tokens": int(len(tokens)),
            "band": list(band),
        }
    return vectors, diagnostics


# ── Banking ───────────────────────────────────────────────────────────────────
def save_vector_bank(out_dir: Path | str, vectors: Mapping[str, NDArray[Any]],
                     stamps: Mapping[str, Any]) -> tuple[Path, Path]:
    """Write a vector bank and its stamps under the frozen on-disk names.

    The stamps are what make the arrays usable later: the site list, the median
    residual norms every absolute alpha derives from, and what each key was built
    from. An array without them is a direction nobody can dose.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    npz_path = directory / BANK_NPZ
    stamps_path = directory / BANK_STAMPS
    np.savez(npz_path, **{k: np.asarray(v) for k, v in vectors.items()})
    stamps_path.write_text(json.dumps(dict(stamps), indent=2))
    logger.info(f"banked {len(vectors)} vectors to {npz_path}")
    return npz_path, stamps_path


def load_vector_bank(bank_dir: Path | str) -> tuple[dict[str, F32], dict[str, Any]]:
    """Read a vector bank and its stamps from a directory."""
    directory = Path(bank_dir)
    npz_path, stamps_path = directory / BANK_NPZ, directory / BANK_STAMPS
    if not npz_path.is_file():
        raise FileNotFoundError(f"no vector bank at {npz_path}")
    with np.load(npz_path) as bank:
        vectors = {key: np.asarray(bank[key], dtype=np.float32) for key in bank.files}
    stamps = json.loads(stamps_path.read_text()) if stamps_path.is_file() else {}
    return vectors, stamps


def load_vector(bank: Mapping[str, NDArray[Any]], key: str, source: str = "bank") -> F32:
    """One unit vector out of a bank, warning when it is not actually unit.

    A bank is written unit-normalized, so a norm away from one means a dose
    computed against that assumption is wrong by the same factor.
    """
    if key not in bank:
        raise KeyError(f"vector key {key!r} not in {source} (has {sorted(bank)})")
    v = np.asarray(bank[key], dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if not 0.99 < norm < 1.01:
        logger.warning(f"vector {key} in {source} has norm {norm:.4f} where unit is expected")
    return v


def site_of_key(key: str, default_site: int) -> int:
    """The injection site a bank key names, or ``default_site`` when it names none.

    Keys carry their site as a ``_L<n>`` suffix. A key without one is
    site-independent — an isotropic control is the same vector wherever it is
    injected — and is dosed and read at the site it is compared at.
    """
    if "_L" in key:
        tail = key.rsplit("_L", 1)[1]
        if tail.isdigit():
            return int(tail)
    return int(default_site)
