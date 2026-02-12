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
    "structured",
    "associative",
    "deliberative",
    "compressed",
    "pedagogical",
]

PROCESSING_MODES: dict[ProcessingMode, str] = {
    "structured": (
        "Use numbered sections, headers, and bullet points. Present one idea "
        "per paragraph in logical order. Define terms before using them. "
        "Build each point on the previous one. End with a clear summary."
    ),
    "associative": (
        "Write in a stream of consciousness. Jump between ideas mid-sentence. "
        "Use fragments, dashes, ellipses. Follow tangents wherever they lead. "
        "Connect distant concepts through metaphor and analogy. Do not use "
        "headers, numbered lists, or any organizing structure."
    ),
    "deliberative": (
        "Think through this out loud. Consider a possibility, then poke holes "
        "in it. Weigh alternatives explicitly: 'on one hand... but then...' "
        "Show the messy middle of reasoning — false starts, corrections, "
        "revised conclusions. Arrive at your answer through visible elimination."
    ),
    "compressed": (
        "Maximum information density. No filler words, no elaboration, no "
        "examples unless essential. Short sentences. Telegram style. Every "
        "word must earn its place. If you can cut a word without losing "
        "meaning, cut it."
    ),
    "pedagogical": (
        "Teach this to a curious beginner. Start with intuition before "
        "formalism. Use concrete examples and everyday analogies. Ask "
        "rhetorical questions to guide understanding. Check comprehension: "
        "'Does that make sense? Here's why...' Build from simple to complex."
    ),
}

MODE_INDEX: dict[ProcessingMode, int] = {
    "structured": 0,
    "associative": 1,
    "deliberative": 2,
    "compressed": 3,
    "pedagogical": 4,
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
