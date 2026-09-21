"""Typed result schemas, one module per gauntlet section.

Each ``run_<section>()`` in the gauntlet returns the ``*Result`` model named
after it, and ``AnalysisResults`` is the composite one run produces. The JSON
written under ``outputs/analysis/<run>/results.json`` is the model's own wire
shape: ``clean_for_json`` stays in the write path for NaN/Inf scrubbing and
numpy coercion, and nothing here reshapes a banked file's structure.

The split is by section because that is how the results are produced, consumed
and skipped — a section's schema, its runner and its ``--skip`` number move
together. ``base`` holds the one shared model config and the rules that follow
from it. This package re-exports every model, so a caller that wants several
sections' types names this package rather than chasing eleven modules, while a
caller that wants one section can import that module alone.
"""

from __future__ import annotations

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
    TierClassificationResult,
    TopicHeldoutResult,
)
from anamnesis.analysis.gauntlet.schemas.clustering import (
    ClusteringResult,
    EmbeddingResult,
    PerModeSilhouetteStats,
    TierSilhouette,
)
from anamnesis.analysis.gauntlet.schemas.contrastive import (
    CapacitySweepEntry,
    ContrastiveAblationEntry,
    ContrastivePairwiseEntry,
    ContrastiveResult,
    ContrastiveSuperAdditivity,
    ContrastiveTierAblation,
    ContrastiveTierResult,
    LinearBaselineEntry,
)
from anamnesis.analysis.gauntlet.schemas.integrity import (
    IntegrityResult,
    LengthByModeStats,
    LengthOverallStats,
    NanInfCount,
    TierValueRange,
    TierVarianceReport,
)
from anamnesis.analysis.gauntlet.schemas.intrinsic_dimension import (
    BootstrapStats,
    GlobalTierIDResult,
    GRIDEResult,
    IntrinsicDimensionResult,
    PerModeIDResult,
    TierConvergenceResult,
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
    PerTierSemanticResult,
    PromptSwapConfoundResult,
    PromptSwapTierResult,
    RetrievalFeatureSet,
    RetrievalResult,
    SemanticClassifierBundle,
    SemanticResult,
    ShuffleControlsResult,
    TextToComputeR2,
)
from anamnesis.analysis.gauntlet.schemas.tier_ablation import (
    CohensDPerTopicResult,
    CrossGroupAblation,
    FeatureImportanceEntry,
    LeaveOneOutEntry,
    PairwiseTierCombo,
    PerTopicEffectSize,
    StdVsMeanResult,
    TierAblationResult,
    TierRankingEntry,
    TripleTierCombo,
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
    "ContrastiveTierAblation",
    "ContrastiveTierResult",
    "CrossGroupAblation",
    "CurvatureResult",
    "CurvatureScaleEntry",
    "EmbeddingResult",
    "FeatureImportanceEntry",
    "GRIDEResult",
    "GeodesicDistortionResult",
    "GeodesicOverall",
    "GeodesicPerMode",
    "GlobalTierIDResult",
    "GromovDeltaResult",
    "IntegrityResult",
    "IntrinsicDimensionResult",
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
    "PairwiseTierCombo",
    "PerModeIDResult",
    "PerModeLengthStats",
    "PerModeSilhouetteStats",
    "PerModeSurfaceVsCompute",
    "PerModeSurfaceVsComputeResult",
    "PerTierSemanticResult",
    "PerTopicEffectSize",
    "PermutationTestResult",
    "PersistentHomologyResult",
    "PromptSwapConfoundResult",
    "PromptSwapTierResult",
    "RetrievalFeatureSet",
    "RetrievalResult",
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
    "TierAblationResult",
    "TierClassificationResult",
    "TierConvergenceResult",
    "TierRankingEntry",
    "TierSilhouette",
    "TierValueRange",
    "TierVarianceReport",
    "TopicHeldoutResult",
    "TopologyMetricSummary",
    "TopologyResult",
    "TripleTierCombo",
]
