"""Section 11 schemas: curvature, geodesics and tangent structure.

The finest geometric cut the gauntlet takes: tangent-space angles between modes
and the variance each explains, geodesic-versus-euclidean distortion, curvature
across neighbourhood scales, and persistent-homology Betti numbers.
"""

from __future__ import annotations

from pydantic import BaseModel

from anamnesis.analysis.gauntlet.schemas.base import _FORBID


class TangentAngles(BaseModel):
    """Principal angles between two mode subspaces."""

    model_config = _FORBID

    mean_angle_deg: float
    max_angle_deg: float
    min_angle_deg: float
    angles_deg: list[float]


class ModeVarianceExplained(BaseModel):
    """PCA explained-variance-ratio summary for one mode."""

    model_config = _FORBID

    explained_variance_ratio: list[float]
    cumulative_5: float
    cumulative_10: float


class TangentSpaceResult(BaseModel):
    """Local tangent-space alignment + per-mode variance explained.

    Returns an error stub on PCA failure; otherwise populates all three
    success fields.
    """

    model_config = _FORBID

    pairwise_angles: dict[str, TangentAngles] | None = None
    mode_variance_explained: dict[str, ModeVarianceExplained] | None = None
    n_components: int | None = None
    error: str | None = None


class GeodesicOverall(BaseModel):
    """Overall geodesic-vs-Euclidean distortion stats."""

    model_config = _FORBID

    mean_distortion: float
    std_distortion: float
    max_distortion: float
    median_distortion: float


class GeodesicPerMode(BaseModel):
    """Per-mode geodesic distortion (within-mode pairs only)."""

    model_config = _FORBID

    mean: float
    std: float


class GeodesicDistortionResult(BaseModel):
    """Isomap geodesic / Euclidean distortion diagnostics."""

    model_config = _FORBID

    overall: GeodesicOverall | None = None
    per_mode: dict[str, GeodesicPerMode] | None = None
    within_mode_mean: float | None = None
    between_mode_mean: float | None = None
    isomap_n_neighbors: int | None = None
    reconstruction_error: float | None = None
    error: str | None = None


class CurvatureScaleEntry(BaseModel):
    """Local-PCA reconstruction-error proxy for curvature at one scale."""

    model_config = _FORBID

    mean_curvature: float
    std_curvature: float
    per_mode: dict[str, float]


class CurvatureResult(BaseModel):
    """Scale-dependent curvature proxies.

    ``per_scale`` keys are stringified integers matching ``scales``.
    """

    model_config = _FORBID

    scales: list[int] | None = None
    per_scale: dict[str, CurvatureScaleEntry] | None = None
    error: str | None = None


class BettiNumberEntry(BaseModel):
    """Persistent-homology summary for one dimension (H0/H1/H2)."""

    model_config = _FORBID

    n_features: int
    n_finite: int
    mean_lifetime: float
    max_lifetime: float
    median_lifetime: float


class PersistentHomologyResult(BaseModel):
    """Persistent homology via ripser (or error stub)."""

    model_config = _FORBID

    betti_numbers: dict[str, BettiNumberEntry] | None = None
    n_subsampled: int | None = None
    error: str | None = None


class ManifoldGeometryResult(BaseModel):
    """Section 11 result: tangent space, geodesic distortion,
    curvature proxies, and persistent homology.

    Every field is optional so a section that could not run — the union it
    reads is not in this corpus — round-trips as a stub carrying its reason.
    """

    model_config = _FORBID

    tangent_space: TangentSpaceResult | None = None
    geodesic_distortion: GeodesicDistortionResult | None = None
    curvature: CurvatureResult | None = None
    persistent_homology: PersistentHomologyResult | None = None
    error: str | None = None
