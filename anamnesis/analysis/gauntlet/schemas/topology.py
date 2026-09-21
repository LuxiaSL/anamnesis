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
    """

    model_config = _FORBID

    block: str
    euclidean_centroid_distances: dict[str, float]
    cosine_centroid_distances: dict[str, float]
    manhattan_centroid_distances: dict[str, float]
    hierarchical_clustering: dict[str, str]
    topology_summary: dict[str, TopologyMetricSummary]
    gromov_delta_euclidean: GromovDeltaResult
