"""Configuration: per-model facts, one pass's settings, and the run registry.

Three concerns, three modules, one import surface:

* :mod:`anamnesis.config.models` — the preset registry and the loader config.
  ``ModelConfig.from_preset("llama31_8b")`` is the bridge from a model's name to
  its architecture. The rows are data, in ``models.json`` beside the module, and
  ``ANAMNESIS_MODELS`` names further files of the same shape.
* :mod:`anamnesis.config.experiment` — what one extraction pass is.
  ``ExperimentConfig.from_preset("8b")`` derives the model, decode, extraction
  and calibration configs together, so they cannot disagree.
* :mod:`anamnesis.config.runs` — named runs, resolved against the data roots in
  :mod:`anamnesis.config.paths`.

Nothing here imports the model prompts from :mod:`anamnesis.modes`, and nothing
here imports torch. Configuration is readable on a machine with no GPU and no
model weights, which is what lets analysis, tests and the numeric anchor share
one description of a run.
"""

from __future__ import annotations

from anamnesis.config.experiment import (
    CalibrationConfig,
    ExperimentConfig,
    ExtractionConfig,
    FeaturePipelineConfig,
    GenerationConfig,
    GenerationSpec,
    ProcessingMode,
)
from anamnesis.config.models import (
    ATTENTION_WITHOUT_WEIGHTS,
    EAGER_ATTENTION,
    MODELS_ENV,
    MODELS_FILE,
    AttentionKind,
    ModelConfig,
    ModelPreset,
    ModelRegistry,
    ModelRegistryError,
    ModelRegistryFile,
    UnknownPresetError,
    layer_counts_by_run_prefix,
    load_registry,
    preset_names,
    presets_with_floors,
    registry_paths,
    resolve_preset,
)
from anamnesis.config.paths import (
    DATA_ROOTS,
    DEFAULT_PROMPT_SET,
    DEFAULT_RUN_NAME,
    DataRoot,
    PathResolutionError,
    data_root,
    legacy_data_root,
    legacy_prompts_path,
    outputs_root,
    package_root,
    prompts_dir,
    prompts_path,
    resolve_data_path,
    resolve_prompts_path,
    run_name,
    run_outputs_dir,
    user_data_root,
)
from anamnesis.config.runs import (
    RUNS_FILE,
    ResolvedRun,
    RunsRegistryError,
    RunSpec,
    UnknownRunError,
    get_run,
    load_runs,
    resolve_run,
    run_names,
)

__all__ = [
    "ATTENTION_WITHOUT_WEIGHTS",
    "AttentionKind",
    "CalibrationConfig",
    "DATA_ROOTS",
    "DEFAULT_PROMPT_SET",
    "DEFAULT_RUN_NAME",
    "DataRoot",
    "EAGER_ATTENTION",
    "ExperimentConfig",
    "ExtractionConfig",
    "FeaturePipelineConfig",
    "GenerationConfig",
    "GenerationSpec",
    "MODELS_ENV",
    "MODELS_FILE",
    "ModelConfig",
    "ModelPreset",
    "ModelRegistry",
    "ModelRegistryError",
    "ModelRegistryFile",
    "PathResolutionError",
    "ProcessingMode",
    "RUNS_FILE",
    "ResolvedRun",
    "RunSpec",
    "RunsRegistryError",
    "UnknownPresetError",
    "UnknownRunError",
    "data_root",
    "get_run",
    "layer_counts_by_run_prefix",
    "legacy_data_root",
    "legacy_prompts_path",
    "load_registry",
    "load_runs",
    "outputs_root",
    "package_root",
    "preset_names",
    "presets_with_floors",
    "prompts_dir",
    "prompts_path",
    "registry_paths",
    "resolve_data_path",
    "resolve_preset",
    "resolve_prompts_path",
    "resolve_run",
    "run_name",
    "run_names",
    "run_outputs_dir",
    "user_data_root",
]
