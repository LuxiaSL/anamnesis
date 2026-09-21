"""What one extraction pass is: how to decode, what to read, where it lands.

The classes here split along the boundary a run actually has:

:class:`GenerationConfig`
    the decode policy handed to the model's generation loop;
:class:`ExtractionConfig`
    which features the extractor computes and over what windows;
:class:`FeaturePipelineConfig`
    which engineered families run over the saved raw tensors;
:class:`CalibrationConfig`
    the two artifacts positional decomposition needs;
:class:`ExperimentConfig`
    all of the above plus the output layout, stamped with the preset it came
    from.

Every field that varies between models is required, and every class carries a
``from_preset`` constructor that fills those fields from one registry row. A
caller therefore cannot leave a per-model field at a value that belongs to some
other model, which is the failure mode a default would invite: a 3B pass
decoding at the 8B temperature reads as a clean result, not as an error.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from anamnesis.config.models import ModelConfig, ModelPreset, resolve_preset
from anamnesis.config.paths import resolve_prompts_path, run_outputs_dir

ProcessingMode = Literal[
    "linear",
    "analogical",
    "socratic",
    "contrastive",
    "dialectical",
]
"""The five format-controlled modes of the core protocol.

A generation's ``mode`` field is a plain string rather than this literal, because
the extended and prompt-swap sets name modes outside it. The literal is the
narrow type for code that handles the core five and nothing else; the mode
prompts themselves live in :mod:`anamnesis.modes`.
"""


class GenerationConfig(BaseModel):
    """The decode policy, and the outputs extraction cannot run without.

    The four output flags are what make a generation observable: hidden states,
    attention weights and logits are the substrates features are read from, and a
    dictionary return is what carries them back. Turning one off removes a source
    from the feature vector rather than raising, so they default on.
    """

    model_config = ConfigDict(extra="forbid")

    max_new_tokens: int = Field(gt=0, description="Token budget per generation")
    temperature: float = Field(gt=0.0, description="Decode temperature")
    top_p: float = Field(gt=0.0, le=1.0, description="Nucleus mass")
    do_sample: bool = Field(default=True, description="Sample rather than take the argmax")

    eos_token_ids: list[int] = Field(
        min_length=1, description="Token IDs that end a generation, including end-of-turn variants"
    )

    output_hidden_states: bool = Field(default=True, description="Return per-layer hidden states")
    output_attentions: bool = Field(default=True, description="Return attention weights")
    output_logits: bool = Field(default=True, description="Return per-step logits")
    return_dict_in_generate: bool = Field(default=True, description="Return a structured result")

    @classmethod
    def from_preset(cls, preset: str | ModelPreset, **overrides: Any) -> GenerationConfig:
        """The decode policy a preset's checkpoint was characterised at."""
        row = resolve_preset(preset)
        fields: dict[str, Any] = {
            "max_new_tokens": row.max_new_tokens,
            "temperature": row.temperature,
            "top_p": row.top_p,
            "eos_token_ids": list(row.eos_token_ids),
        }
        fields.update(overrides)
        return cls(**fields)


class ExtractionConfig(BaseModel):
    """Which features the extractor computes, and over what windows.

    The layer fields are required: a layer plan belongs to a model's depth, and
    the same index means a different fraction of the network in each one.
    """

    model_config = ConfigDict(extra="forbid")

    sampled_layers: list[int] = Field(
        min_length=1, description="Layers keys, attention and spectral features are read at"
    )
    pca_layers: list[int] = Field(
        min_length=1, description="Layers the residual-stream PCA projection is applied at"
    )
    pca_components: int = Field(default=50, gt=0, description="Retained PCA components")
    pca_temporal_samples: int = Field(default=5, gt=0, description="Positions the PCA is sampled at")

    trajectory_points: int = Field(default=5, gt=0, description="Positions a trajectory is sampled at")

    spectral_subsample_step: int = Field(
        default=10, gt=0, description="Stride for spectral subsampling"
    )

    epoch_window_size: int = Field(default=50, gt=0, description="Window for cache-epoch detection")
    epoch_stride: int = Field(default=25, gt=0, description="Stride between epoch windows")

    surprise_window: int = Field(default=20, gt=0, description="Window for Bayesian surprise")
    surprise_threshold_sigma: float = Field(
        default=1.5, gt=0.0, description="Surprise threshold in standard deviations"
    )

    knnlm_pca_components: int = Field(
        default=100, gt=0, description="Components for the kNN-LM baseline projection"
    )

    enable_tier1: bool = Field(default=True, description="Compute the output-source baseline block")
    enable_tier2: bool = Field(default=True, description="Compute the attention-dynamics baseline block")
    enable_tier2_5: bool = Field(default=True, description="Compute the cache-geometry baseline block")
    enable_tier3: bool = Field(default=True, description="Compute the residual-PCA baseline block")
    enable_knnlm_baseline: bool = Field(default=True, description="Compute the kNN-LM baseline block")

    early_layer_cutoff: int = Field(
        ge=0, description="At or below this a layer counts as early for cross-layer agreement"
    )
    late_layer_cutoff: int = Field(
        ge=0, description="At or above this a layer counts as late for cross-layer agreement"
    )

    save_raw_tensors: bool = Field(
        default=False, description="Save raw per-token tensors beside the feature vectors"
    )
    raw_logits_top_k: int = Field(
        default=50, gt=0, description="Top logits kept per step when saving raw tensors"
    )
    raw_hidden_dtype: str = Field(
        default="float16", description="Dtype saved hidden states and attention are cast to"
    )

    @model_validator(mode="after")
    def _bands_ordered(self) -> Self:
        if self.early_layer_cutoff > self.late_layer_cutoff:
            raise ValueError(
                f"early_layer_cutoff {self.early_layer_cutoff} is above "
                f"late_layer_cutoff {self.late_layer_cutoff}"
            )
        return self

    @classmethod
    def from_preset(cls, preset: str | ModelPreset, **overrides: Any) -> ExtractionConfig:
        """The layer plan a preset declares, with the rest of the defaults."""
        row = resolve_preset(preset)
        fields: dict[str, Any] = {
            "sampled_layers": list(row.sampled_layers),
            "pca_layers": list(row.pca_layers),
            "early_layer_cutoff": row.early_layer_cutoff,
            "late_layer_cutoff": row.late_layer_cutoff,
        }
        fields.update(overrides)
        config = cls(**fields)
        config.check_depth(row)
        return config

    def check_depth(self, preset: str | ModelPreset) -> None:
        """Raise when a layer index falls outside a model's depth.

        Layer plans arrive from the registry, from a command line and from a
        resumed run, so the check is available on its own rather than only inside
        the constructor.
        """
        row = resolve_preset(preset)
        depth = range(row.num_layers)
        for name in ("sampled_layers", "pca_layers"):
            outside = [layer for layer in getattr(self, name) if layer not in depth]
            if outside:
                raise ValueError(
                    f"{name} names layer(s) {outside} outside {row.name}'s depth "
                    f"0..{row.num_layers - 1}"
                )
        for name in ("early_layer_cutoff", "late_layer_cutoff"):
            if getattr(self, name) not in depth:
                raise ValueError(
                    f"{name}={getattr(self, name)} is outside {row.name}'s depth "
                    f"0..{row.num_layers - 1}"
                )


class FeaturePipelineConfig(BaseModel):
    """Which engineered families run over the saved raw tensors.

    Each family is a self-contained extractor over banked tensors, so a family is
    enabled independently of the rest and of the baseline blocks. The layer
    fields are required for the same reason as in :class:`ExtractionConfig`.
    """

    model_config = ConfigDict(extra="forbid")

    include_baseline_tiers: bool = Field(
        default=True, description="Include the baseline blocks from the numeric anchor"
    )

    enable_residual_trajectory: bool = Field(
        default=False, description="Extract residual trajectory features: velocity, curvature, directness"
    )
    trajectory_layers: list[int] = Field(
        min_length=1, description="Layers residual trajectories are traced at"
    )

    enable_contrastive_projection: bool = Field(
        default=False, description="Apply a trained contrastive projection to hidden states"
    )
    contrastive_model_path: Path | None = Field(
        default=None, description="Trained contrastive projection to apply"
    )
    contrastive_layers: list[int] = Field(
        min_length=1, description="Layers the contrastive projection reads"
    )
    contrastive_temporal_samples: int = Field(
        default=5, gt=0, description="Positions the contrastive projection is sampled at"
    )

    enable_attention_flow: bool = Field(
        default=False, description="Extract attention flow: region decomposition, recency, head diversity"
    )

    enable_gate_features: bool = Field(default=False, description="Extract gate activation features")
    gate_sparsity_threshold: float = Field(
        default=0.01, gt=0.0, description="Activation magnitude above which a gate counts as active"
    )

    enable_temporal_dynamics: bool = Field(
        default=False, description="Extract windowed temporal decompositions of the core metrics"
    )

    enable_per_head: bool = Field(
        default=False, description="Extract per-head attention and key heterogeneity"
    )

    enable_value_geometry: bool = Field(
        default=False,
        description="Extract value-vector geometry: spread, effective dimension, drift, novelty",
    )

    enable_qk_geometry: bool = Field(
        default=False,
        description="Extract pre-RoPE query geometry and query-key content alignment",
    )

    enable_kv_cka: bool = Field(
        default=False,
        description="Extract cross-layer linear CKA for keys and values, which is basis-invariant",
    )

    enable_attn_res: bool = Field(
        default=False,
        description="Extract cross-block attention-residual routing: anchor against recency, concentration",
    )

    enable_expert_routing: bool = Field(
        default=False,
        description=(
            "Extract expert-routing features on a mixture-of-experts checkpoint; "
            "a dense checkpoint supplies no router distribution and the family returns nothing"
        ),
    )
    expert_routing_top_k: int = Field(
        default=6, gt=0, description="Experts selected per token, which sets coverage and load"
    )

    enable_path_signature: bool = Field(
        default=False,
        description=(
            "Extract log-signature features — net displacement and Levy areas — of the "
            "projected, time-augmented residual path"
        ),
    )
    path_signature_layers: list[int] = Field(
        default_factory=list, description="Sites the path signature is taken at, one per model"
    )
    path_signature_basis_path: Path | None = Field(
        default=None,
        description="Banked calibration PCA supplying the projection basis; required when the family runs",
    )
    path_signature_k: int = Field(
        default=4, gt=0, description="Projection rank; the path has one more dimension when time-augmented"
    )
    path_signature_time_augment: bool = Field(
        default=True, description="Append normalised position before integrating, which keeps pacing"
    )
    path_signature_level: int = Field(
        default=2, ge=1, description="Log-signature truncation level: 1 is displacement, 2 adds Levy areas"
    )
    path_signature_basis_label: str = Field(
        default="pcaA", description="Basis token embedded in every path-signature feature name"
    )
    path_signature_permute_seed: int | None = Field(
        default=None,
        description=(
            "Permute the increments with this seed and re-cumulate, which is the family's null: "
            "level-1 columns are invariant and level-2 columns die, and names are unchanged so a "
            "null bank stays column-comparable"
        ),
    )

    temporal_n_windows: int = Field(default=4, gt=0, description="Windows for windowed statistics")
    enable_stft: bool = Field(
        default=True, description="Include short-time Fourier features in the temporal operators"
    )
    stft_nperseg: int = Field(
        default=64, gt=0, description="Short-time Fourier window length, sized for short generations"
    )

    @field_validator("path_signature_layers")
    @classmethod
    def _sites_unique(cls, layers: list[int]) -> list[int]:
        if len(set(layers)) != len(layers):
            raise ValueError(f"path_signature_layers repeats a site: {layers}")
        return layers

    @model_validator(mode="after")
    def _path_signature_supplied(self) -> Self:
        if not self.enable_path_signature:
            return self
        if not self.path_signature_layers:
            raise ValueError(
                "enable_path_signature is set but path_signature_layers is empty; "
                "the site is per-model and the family has no default"
            )
        if self.path_signature_basis_path is None:
            raise ValueError(
                "enable_path_signature is set but path_signature_basis_path is unset; "
                "the family projects onto a supplied basis and never fits one"
            )
        return self

    @classmethod
    def from_preset(cls, preset: str | ModelPreset, **overrides: Any) -> FeaturePipelineConfig:
        """The per-model layer fields a preset declares, with the families off.

        Which families a pass runs is a property of the pass, not of the model,
        so every ``enable_*`` flag keeps its default and the caller names the ones
        it wants.
        """
        row = resolve_preset(preset)
        fields: dict[str, Any] = {
            "trajectory_layers": list(row.trajectory_layers),
            "contrastive_layers": list(row.contrastive_layers),
        }
        fields.update(overrides)
        return cls(**fields)


class CalibrationConfig(BaseModel):
    """The two artifacts positional decomposition reads.

    Positional means subtract the position-driven part of a state; the PCA model
    projects what is left. Both belong to one model, so both paths are required.
    """

    model_config = ConfigDict(extra="forbid")

    num_calibration_prompts: int = Field(
        default=50, gt=0, description="Prompts the calibration pass generates over"
    )
    calibration_max_tokens: int = Field(
        default=512, gt=0, description="Token budget per calibration generation"
    )
    positional_means_path: Path = Field(description="Per-position means to subtract")
    pca_model_path: Path = Field(description="Fitted residual-stream PCA")

    POSITIONAL_MEANS_NAME: ClassVar[str] = "positional_means.npz"
    PCA_MODEL_NAME: ClassVar[str] = "pca_model.pkl"

    @classmethod
    def from_preset(cls, preset: str | ModelPreset, **overrides: Any) -> CalibrationConfig:
        """The calibration artifacts of a preset, at the directory it declares."""
        row = resolve_preset(preset)
        directory = row.resolved_calibration_dir()
        fields: dict[str, Any] = {
            "positional_means_path": directory / cls.POSITIONAL_MEANS_NAME,
            "pca_model_path": directory / cls.PCA_MODEL_NAME,
        }
        fields.update(overrides)
        return cls(**fields)

    @classmethod
    def in_directory(cls, directory: Path, **overrides: Any) -> CalibrationConfig:
        """The calibration artifacts at an explicit directory."""
        fields: dict[str, Any] = {
            "positional_means_path": directory / cls.POSITIONAL_MEANS_NAME,
            "pca_model_path": directory / cls.PCA_MODEL_NAME,
        }
        fields.update(overrides)
        return cls(**fields)

    def artifact_paths(self) -> tuple[Path, Path]:
        """Both artifacts, in the order a calibration pass writes them."""
        return (self.positional_means_path, self.pca_model_path)


class GenerationSpec(BaseModel):
    """One generation, fully determined before the model is called.

    ``mode`` is a string rather than :data:`ProcessingMode` so that the extended
    and prompt-swap sets fit the same spec: a swap generation's mode names the
    pair, not one of the core five.
    """

    model_config = ConfigDict(extra="forbid")

    generation_id: int = Field(ge=0, description="Index within the run")
    prompt_set: str = Field(description="Prompt set the topic was drawn from")
    topic: str = Field(description="Topic text")
    topic_idx: int = Field(ge=0, description="Topic index within its set")
    mode: str = Field(description="Mode name, from whichever mode set the run uses")
    mode_idx: int = Field(ge=0, description="Mode index within its set")
    system_prompt: str = Field(description="System prompt as sent")
    user_prompt: str = Field(description="User prompt as sent")
    seed: int = Field(description="Seed the sampler is set to")
    repetition: int = Field(default=0, ge=0, description="Repetition index for this topic and mode")


class ExperimentConfig(BaseModel):
    """One extraction pass: the model, the policy, the features, the layout.

    Build it with :meth:`from_preset`, which is the single place a model's facts
    become a runnable configuration. The output paths default to the run
    directory under the outputs root, so a caller that only names a model and a
    run gets the layout every reader of this data expects.
    """

    model_config = ConfigDict(extra="forbid")

    preset_name: str = Field(description="The preset this pass was configured from")
    model: ModelConfig
    generation: GenerationConfig
    extraction: ExtractionConfig
    calibration: CalibrationConfig

    outputs_dir: Path = Field(description="Run directory everything below is written under")
    signatures_dir: Path = Field(description="Feature vectors, one file per generation")
    figures_dir: Path = Field(description="Figures produced from this run")
    prompts_path: Path = Field(description="Prompt sets the topics are read from")
    metadata_path: Path = Field(description="Run metadata, generations under the generations key")
    results_path: Path = Field(description="Analysis results for this run")

    @classmethod
    def from_preset(
        cls,
        preset: str | ModelPreset,
        *,
        run_name: str | None = None,
        outputs_dir: Path | None = None,
        prompts_path: Path | None = None,
        model_overrides: Mapping[str, Any] | None = None,
        generation_overrides: Mapping[str, Any] | None = None,
        extraction_overrides: Mapping[str, Any] | None = None,
        calibration_overrides: Mapping[str, Any] | None = None,
    ) -> ExperimentConfig:
        """Everything one pass needs, derived from one registry row.

        ``run_name`` names the directory under the outputs root; ``outputs_dir``
        overrides that choice outright. The four override mappings reach the
        section configs, so a caller can shorten a smoke run
        (``generation_overrides={"max_new_tokens": 100}``) without restating the
        model.

        Raises
        ------
        UnknownPresetError
            When the preset name is not in the registry.
        pydantic.ValidationError
            When an override names a field a section config does not have.
        """
        row = resolve_preset(preset)
        run_dir = outputs_dir if outputs_dir is not None else run_outputs_dir(run_name)
        return cls(
            preset_name=row.name,
            model=ModelConfig.from_preset(row, **dict(model_overrides or {})),
            generation=GenerationConfig.from_preset(row, **dict(generation_overrides or {})),
            extraction=ExtractionConfig.from_preset(row, **dict(extraction_overrides or {})),
            calibration=CalibrationConfig.from_preset(row, **dict(calibration_overrides or {})),
            outputs_dir=run_dir,
            signatures_dir=run_dir / "signatures",
            figures_dir=run_dir / "figures",
            prompts_path=prompts_path if prompts_path is not None else resolve_prompts_path(),
            metadata_path=run_dir / "metadata.json",
            results_path=run_dir / "results.json",
        )

    def ensure_dirs(self) -> None:
        """Create the directories this pass writes into, including calibration's."""
        for directory in (self.outputs_dir, self.signatures_dir, self.figures_dir):
            directory.mkdir(parents=True, exist_ok=True)
        for artifact in self.calibration.artifact_paths():
            artifact.parent.mkdir(parents=True, exist_ok=True)
