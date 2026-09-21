"""The top-level composite: one run, every section.

Run metadata plus the eleven typed section results, each Optional because
``--skip`` can leave any of them unpopulated. ``run_full_analysis`` accumulates
into a plain dict and validates it into this model on the way out, so the JSON
on disk and the returned object describe the same run.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from anamnesis.analysis.gauntlet.schemas.base import _FORBID
from anamnesis.analysis.gauntlet.schemas.ccgp import CCGPResult
from anamnesis.analysis.gauntlet.schemas.classification import ClassificationResult
from anamnesis.analysis.gauntlet.schemas.clustering import ClusteringResult
from anamnesis.analysis.gauntlet.schemas.contrastive import ContrastiveResult
from anamnesis.analysis.gauntlet.schemas.integrity import IntegrityResult
from anamnesis.analysis.gauntlet.schemas.intrinsic_dimension import IntrinsicDimensionResult
from anamnesis.analysis.gauntlet.schemas.manifold_geometry import ManifoldGeometryResult
from anamnesis.analysis.gauntlet.schemas.scorecard import ScorecardResult
from anamnesis.analysis.gauntlet.schemas.semantic import SemanticResult
from anamnesis.analysis.gauntlet.schemas.legacy_bin_readout import LegacyBinReadoutResult
from anamnesis.analysis.gauntlet.schemas.topology import TopologyResult


class AnalysisResults(BaseModel):
    """Top-level ``run_full_analysis`` output.

    Wraps run metadata (``run_name``, ``timestamp``, ``core_only``,
    ``last_updated``, ``n_samples``, ``section_times``) and all 11 typed
    section results. Sections are Optional because ``--skip`` can leave
    any of them unpopulated. Unknown keys are forbidden so drift is
    caught eagerly.

    The JSON wire format is identical to the pre-typed layout: the
    orchestrator accumulates into a dict and serializes via
    ``clean_for_json`` (which handles BaseModel values), so
    ``results.json`` on disk stays structurally unchanged.
    """

    model_config = _FORBID

    # Run metadata
    run_name: str
    timestamp: str
    core_only: bool
    last_updated: str
    n_samples: int
    section_times: dict[str, float] = Field(default_factory=dict)

    # Section results (11 in the current pipeline)
    integrity: IntegrityResult | None = None
    classification: ClassificationResult | None = None
    legacy_bin_readout: LegacyBinReadoutResult | None = None
    intrinsic_dimension: IntrinsicDimensionResult | None = None
    ccgp: CCGPResult | None = None
    topology: TopologyResult | None = None
    clustering: ClusteringResult | None = None
    contrastive: ContrastiveResult | None = None
    semantic: SemanticResult | None = None
    scorecard: ScorecardResult | None = None
    manifold_geometry: ManifoldGeometryResult | None = None
