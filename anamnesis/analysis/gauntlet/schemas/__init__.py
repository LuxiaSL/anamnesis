"""Typed result schemas, one module per gauntlet section.

Each ``run_<section>()`` in the gauntlet returns the ``*Result`` model named
after it, and ``AnalysisResults`` is the composite one run produces. The JSON
written under ``outputs/analysis/<run>/results.json`` is the model's own wire
shape: ``clean_for_json`` stays in the write path for NaN/Inf scrubbing and
numpy coercion, and nothing here reshapes a banked file's structure.

``compat`` is the read side of a rename: it maps an older file's field names onto
the current ones so a banked document still loads, and it is the one place to look
when following a citation from an older file.

The split is by section because that is how the results are produced, consumed
and skipped — a section's schema, its runner and its ``--skip`` number move
together. ``base`` holds the one shared model config and the rules that follow
from it. This package re-exports every model, so a caller that wants several
sections' types names this package rather than chasing eleven modules, while a
caller that wants one section can import that module alone.
"""

from __future__ import annotations

from anamnesis.analysis.gauntlet.schemas.compat import (
    FIELD_RENAMES,
    SECTION_RENAMES,
    migrate_banked_results,
)
from anamnesis.analysis.gauntlet.schemas.ccgp import (
    CCGPDichotomy,
    CCGPResult,
    CCGPSummary,
    CCGPVariant,
)
from anamnesis.analysis.gauntlet.schemas.classification import (
    ClassificationResult,
    ClassifierAccuracyResult,
    ClassifierWithConfusionResult,
    CVStabilityResult,
    LengthOnlyResult,
    PerModeLengthStats,
    PermutationTestResult,
    BlockClassificationResult,
    TopicHeldoutResult,
)
from anamnesis.analysis.gauntlet.schemas.clustering import (
    ClusteringResult,
    EmbeddingResult,
    PerModeSilhouetteStats,
    BlockSilhouette,
)
from anamnesis.analysis.gauntlet.schemas.contrastive import (
    CapacitySweepEntry,
    ContrastiveAblationEntry,
    ContrastivePairwiseEntry,
    ContrastiveResult,
    ContrastiveSuperAdditivity,
    ContrastiveBlockAblation,
    ContrastiveBlockResult,
    LinearBaselineEntry,
)
from anamnesis.analysis.gauntlet.schemas.integrity import (
    IntegrityResult,
    LengthByModeStats,
    LengthOverallStats,
    NanInfCount,
    BlockValueRange,
    BlockVarianceReport,
)
from anamnesis.analysis.gauntlet.schemas.intrinsic_dimension import (
    BootstrapStats,
    GlobalBlockIDResult,
    GRIDEResult,
    IntrinsicDimensionResult,
    PerModeIDResult,
    BlockConvergenceResult,
)
from anamnesis.analysis.gauntlet.schemas.manifold_geometry import (
    BettiNumberEntry,
    CurvatureResult,
    CurvatureScaleEntry,
    GeodesicDistortionResult,
    GeodesicOverall,
    GeodesicPerMode,
    ManifoldGeometryResult,
    ModeVarianceExplained,
    PersistentHomologyResult,
    TangentAngles,
    TangentSpaceResult,
)
from anamnesis.analysis.gauntlet.schemas.results import AnalysisResults
from anamnesis.analysis.gauntlet.schemas.scorecard import (
    ScorecardPrediction,
    ScorecardResult,
    ScorecardSummary,
)
from anamnesis.analysis.gauntlet.schemas.semantic import (
    ClassificationScore,
    ContrastiveProjectionComparisonEntry,
    ContrastiveProjectionComparisonResult,
    JaccardStats,
    MantelResult,
    PerModeSurfaceVsCompute,
    PerModeSurfaceVsComputeResult,
    PerBlockSemanticResult,
    PromptSwapConfoundResult,
    PromptSwapBlockResult,
    RetrievalFeatureSet,
    RetrievalResult,
    SemanticClassifierBundle,
    SemanticResult,
    ShuffleControlsResult,
    TextToComputeR2,
)
from anamnesis.analysis.gauntlet.schemas.legacy_bin_readout import (
    CohensDPerTopicResult,
    CrossGroupAblation,
    FeatureImportanceEntry,
    LeaveOneOutEntry,
    PairwiseBlockCombo,
    PerTopicEffectSize,
    StdVsMeanResult,
    LegacyBinReadoutResult,
    BlockRankingEntry,
    TripleBlockCombo,
)
from anamnesis.analysis.gauntlet.schemas.topology import (
    GromovDeltaResult,
    TopologyMetricSummary,
    TopologyResult,
)

__all__ = [
    "AnalysisResults",
    "BettiNumberEntry",
    "BootstrapStats",
    "CCGPDichotomy",
    "CCGPResult",
    "CCGPSummary",
    "CCGPVariant",
    "CVStabilityResult",
    "CapacitySweepEntry",
    "ClassificationResult",
    "ClassificationScore",
    "ClassifierAccuracyResult",
    "ClassifierWithConfusionResult",
    "ClusteringResult",
    "CohensDPerTopicResult",
    "ContrastiveAblationEntry",
    "ContrastivePairwiseEntry",
    "ContrastiveProjectionComparisonEntry",
    "ContrastiveProjectionComparisonResult",
    "ContrastiveResult",
    "ContrastiveSuperAdditivity",
    "ContrastiveBlockAblation",
    "ContrastiveBlockResult",
    "CrossGroupAblation",
    "CurvatureResult",
    "CurvatureScaleEntry",
    "EmbeddingResult",
    "FeatureImportanceEntry",
    "GRIDEResult",
    "GeodesicDistortionResult",
    "GeodesicOverall",
    "GeodesicPerMode",
    "GlobalBlockIDResult",
    "GromovDeltaResult",
    "IntegrityResult",
    "IntrinsicDimensionResult",
    "FIELD_RENAMES",
    "JaccardStats",
    "LeaveOneOutEntry",
    "LengthByModeStats",
    "LengthOnlyResult",
    "LengthOverallStats",
    "LinearBaselineEntry",
    "ManifoldGeometryResult",
    "MantelResult",
    "ModeVarianceExplained",
    "NanInfCount",
    "PairwiseBlockCombo",
    "PerModeIDResult",
    "PerModeLengthStats",
    "PerModeSilhouetteStats",
    "PerModeSurfaceVsCompute",
    "PerModeSurfaceVsComputeResult",
    "PerBlockSemanticResult",
    "PerTopicEffectSize",
    "PermutationTestResult",
    "PersistentHomologyResult",
    "PromptSwapConfoundResult",
    "PromptSwapBlockResult",
    "RetrievalFeatureSet",
    "RetrievalResult",
    "SECTION_RENAMES",
    "ScorecardPrediction",
    "ScorecardResult",
    "ScorecardSummary",
    "SemanticClassifierBundle",
    "SemanticResult",
    "ShuffleControlsResult",
    "StdVsMeanResult",
    "TangentAngles",
    "TangentSpaceResult",
    "TextToComputeR2",
    "LegacyBinReadoutResult",
    "BlockClassificationResult",
    "BlockConvergenceResult",
    "BlockRankingEntry",
    "BlockSilhouette",
    "BlockValueRange",
    "BlockVarianceReport",
    "TopicHeldoutResult",
    "TopologyMetricSummary",
    "TopologyResult",
    "TripleBlockCombo",
    "migrate_banked_results",
]
