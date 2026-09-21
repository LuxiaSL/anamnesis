"""Section 6 schemas: mode-cloud topology and hyperbolicity.

Per-metric nearest-pair and outgroup-ratio summaries, and the Gromov
delta-hyperbolicity read in both euclidean and correlation distance.
"""

from __future__ import annotations

from pydantic import BaseModel

from anamnesis.analysis.gauntlet.schemas.base import _FORBID


class TopologyMetricSummary(BaseModel):
    """Nearest/farthest centroid-pair summary for one distance metric."""

    model_config = _FORBID

    nearest_pair: str
    nearest_dist: float
    farthest_pair: str
    farthest_dist: float
    analogical_outgroup_ratio: float


class GromovDeltaResult(BaseModel):
    """Gromov delta-hyperbolicity diagnostics (Euclidean)."""

    model_config = _FORBID

    delta_max: float
    delta_relative: float
    delta_mean: float
    delta_median: float
    diameter: float
    n_quadruples: int


class TopologyResult(BaseModel):
    """Section 6 result: centroid distances, hierarchical clustering,
    topology summary, and delta-hyperbolicity.

    ``hierarchical_clustering`` values are Newick-string trees on success
    or ``"ERROR: ..."`` strings on failure.

    Every field is optional so a section that could not run — the union it
    reads is not in this corpus — round-trips as a stub carrying its reason.
    """

    model_config = _FORBID

    block: str | None = None
    euclidean_centroid_distances: dict[str, float] | None = None
    cosine_centroid_distances: dict[str, float] | None = None
    manhattan_centroid_distances: dict[str, float] | None = None
    hierarchical_clustering: dict[str, str] | None = None
    topology_summary: dict[str, TopologyMetricSummary] | None = None
    gromov_delta_euclidean: GromovDeltaResult | None = None
    error: str | None = None
