"""Pydantic configuration models for the Phase 0 experiment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent

# Run versioning: each experiment run gets its own directory under outputs/runs/.
# Calibration data (model-specific, not run-specific) lives in outputs/calibration/.
# Set via ANAMNESIS_RUN_NAME env var, or defaults to the current run.
RUN_NAME: str = os.environ.get("ANAMNESIS_RUN_NAME", "run3_process_modes")

OUTPUTS_BASE = PROJECT_ROOT / "outputs"
CALIBRATION_DIR = OUTPUTS_BASE / "calibration"
OUTPUTS_DIR = OUTPUTS_BASE / "runs" / RUN_NAME
SIGNATURES_DIR = OUTPUTS_DIR / "signatures"
FIGURES_DIR = OUTPUTS_DIR / "figures"
PROMPTS_PATH = PROJECT_ROOT / "prompts" / "prompt_sets.json"


# ── Model ──────────────────────────────────────────────────────────────────────

class ModelConfig(BaseModel):
    """Configuration for the target model."""

    model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    torch_dtype: str = "float16"
    attn_implementation: str = "eager"  # required — flash/sdpa don't return attn weights
    device_map: str = "auto"

    # Architecture constants (Llama 3.2 3B Instruct)
    num_layers: int = 28
    hidden_dim: int = 3072
    num_attention_heads: int = 24
    num_kv_heads: int = 8  # GQA — fewer KV heads than query heads
    head_dim: int = 128
    vocab_size: int = 128256


# ── Generation ─────────────────────────────────────────────────────────────────

class GenerationConfig(BaseModel):
    """Parameters for model.generate()."""

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

    # Llama 3.2 Instruct stop tokens: <|eot_id|> (end-of-turn) + <|end_of_text|>
    eos_token_ids: list[int] = Field(
        default=[128001, 128009],
        description="Token IDs that signal end of generation",
    )

    # These MUST be True for extraction
    output_hidden_states: bool = True
    output_attentions: bool = True
    output_logits: bool = True  # spec omits this — required for Tier 1 logit features
    return_dict_in_generate: bool = True


# ── Extraction ─────────────────────────────────────────────────────────────────

class ExtractionConfig(BaseModel):
    """Controls which features to extract and how."""

    # Layer subsampling — denser at 60-80% mark per Pochinkov finding
    sampled_layers: list[int] = Field(
        default=[0, 7, 14, 18, 21, 24, 27],
        description="Layers to extract KV cache and spectral features from",
    )

    # Tier 3 PCA layers — prioritize planning-relevant zone
    pca_layers: list[int] = Field(
        default=[7, 14, 18, 21, 24],
        description="Layers for residual stream PCA projection",
    )
    pca_components: int = 50  # reduced to 20 during analysis if needed
    pca_temporal_samples: int = 5  # positions per generation: first, T//4, T//2, 3T//4, last

    # Temporal sampling for trajectories
    trajectory_points: int = 5  # [0, T//4, T//2, 3T//4, T-1]

    # Spectral features
    spectral_subsample_step: int = 10  # every Nth generation step for spectral matrix

    # KV cache epoch detection
    epoch_window_size: int = 50
    epoch_stride: int = 25

    # Bayesian surprise
    surprise_window: int = 20
    surprise_threshold_sigma: float = 1.5

    # kNN-LM baseline
    knnlm_pca_components: int = 100  # reduce 3072 → 100 for comparison

    # Enable/disable tiers
    enable_tier1: bool = True
    enable_tier2: bool = True
    enable_tier2_5: bool = True
    enable_tier3: bool = True
    enable_knnlm_baseline: bool = True


# ── Calibration ────────────────────────────────────────────────────────────────

class CalibrationConfig(BaseModel):
    """Settings for positional decomposition calibration.

    Calibration data is model-specific (not run-specific) and shared across runs.
    """

    num_calibration_prompts: int = 50
    calibration_max_tokens: int = 512
    positional_means_path: Path = Field(
        default=CALIBRATION_DIR / "positional_means.npz",
    )
    pca_model_path: Path = Field(
        default=CALIBRATION_DIR / "pca_model.pkl",
    )


# ── Experiment ─────────────────────────────────────────────────────────────────

ProcessingMode = Literal[
    "linear",
    "analogical",
    "socratic",
    "contrastive",
    "dialectical",
]

# Shared format constraint appended to every mode prompt.
# Controls for the format confound identified in Run 3: modes that differ in
# reasoning strategy should NOT also differ in visual formatting.
_FORMAT_CONSTRAINT = (
    " Write in flowing paragraphs. Do not use bullet points, numbered lists, "
    "headers, or any visual formatting structure."
)

PROCESSING_MODES: dict[ProcessingMode, str] = {
    "linear": (
        "Present your ideas in a clear sequence, each building on the last. "
        "Move forward without backtracking or reconsidering previous points. "
        "Lay out the topic step by step from beginning to end."
        + _FORMAT_CONSTRAINT
    ),
    "analogical": (
        "Explain this primarily through extended analogies and parallels to "
        "other domains. For each key concept, find a comparison from everyday "
        "life or another field that illuminates it. Build understanding "
        "through these connections."
        + _FORMAT_CONSTRAINT
    ),
    "socratic": (
        "Develop your exploration through a sequence of questions and "
        "provisional answers. Pose a question, offer a tentative answer, "
        "then use that answer to generate the next question. Let the chain "
        "of inquiry drive the explanation forward."
        + _FORMAT_CONSTRAINT
    ),
    "contrastive": (
        "Explore this by comparing and contrasting multiple perspectives or "
        "approaches. For each major point, present at least two different "
        "viewpoints and evaluate their relative strengths and weaknesses."
        + _FORMAT_CONSTRAINT
    ),
    "dialectical": (
        "Begin by proposing a clear position on the topic. Then challenge "
        "that position with the strongest counterarguments you can find. "
        "Work toward a revised understanding that accounts for both the "
        "original position and its critiques."
        + _FORMAT_CONSTRAINT
    ),
}

MODE_INDEX: dict[ProcessingMode, int] = {
    "linear": 0,
    "analogical": 1,
    "socratic": 2,
    "contrastive": 3,
    "dialectical": 4,
}


class PromptSetConfig(BaseModel):
    """Identifies a prompt set within the experiment."""

    set_name: Literal["A", "B", "C", "D"]


class GenerationSpec(BaseModel):
    """Full specification for a single generation run."""

    generation_id: int
    prompt_set: str
    topic: str
    topic_idx: int
    mode: ProcessingMode
    mode_idx: int
    system_prompt: str
    user_prompt: str
    seed: int
    repetition: int = 0  # >0 only for Set C


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)

    outputs_dir: Path = OUTPUTS_DIR
    signatures_dir: Path = SIGNATURES_DIR
    figures_dir: Path = FIGURES_DIR
    prompts_path: Path = PROMPTS_PATH
    metadata_path: Path = Field(default=OUTPUTS_DIR / "metadata.json")
    results_path: Path = Field(default=OUTPUTS_DIR / "results.json")

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.signatures_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
