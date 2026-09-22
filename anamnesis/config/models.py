"""Per-model facts, and the one bridge from a preset to a usable configuration.

A **preset** is everything that varies between models: the checkpoint, its
dtype, its architecture, the layers extraction samples, its native decode
parameters, its end-of-generation tokens, and where its calibration artifacts
live. :data:`MODEL_PRESETS` is the registry of those facts, and it is the only
place they are written down.

A **config** is what a caller hands to a loader or an extractor. Every config in
this package is built from a preset by a ``from_preset`` constructor — never by
copying fields across by hand. That is the whole point of the split: a script
that names a model gets an architecture, a layer plan, a decode policy and a
calibration path that agree with each other, because one function produced all
four from one row of data.

Two constraints are enforced here rather than left to a reader:

* ``attn_implementation`` must return attention weights. Fused attention
  kernels do not, and extraction that silently loses them produces feature
  vectors with the attention source missing. The validator rejects the kernels
  known to drop the weights.
* The query-head count must be an integer multiple of the key/value-head count.
  Under grouped-query attention the two differ, attention weights index by query
  head while the cache indexes by key/value head, and per-head analysis is wrong
  whenever the group size is not exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from anamnesis.config.paths import DataRoot, resolve_data_path

AttentionKind = Literal["local", "global"]
"""Whether a layer attends over a sliding window or the full context."""

ATTENTION_WITHOUT_WEIGHTS: frozenset[str] = frozenset(
    {"sdpa", "flash_attention_2", "flash_attention_3", "flex_attention"}
)
"""Attention kernels that return no attention weights to a hook or an output."""

EAGER_ATTENTION = "eager"
"""The attention implementation extraction requires."""


class UnknownPresetError(KeyError):
    """A preset name that the registry does not hold."""


class ModelPreset(BaseModel):
    """The per-model facts every configuration is derived from.

    Layer fields are tuples because the registry is process-global: a consumer
    that received a preset cannot reshape another consumer's layer plan.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    name: str = Field(description="Registry key, and the provenance stamp on configs built from it")
    model_id: str = Field(description="Checkpoint identifier as the model hub spells it")
    torch_dtype: str = Field(description="The checkpoint's native dtype")

    num_layers: int = Field(gt=0, description="Decoder layer count")
    hidden_dim: int = Field(gt=0, description="Residual stream width")
    num_attention_heads: int = Field(gt=0, description="Query heads")
    num_kv_heads: int = Field(gt=0, description="Key/value heads; below the query count under GQA")
    head_dim: int = Field(gt=0, description="Per-head width of the attention value path")

    sampled_layers: tuple[int, ...] = Field(
        description="Layers extraction reads keys, attention and spectral features at"
    )
    pca_layers: tuple[int, ...] = Field(description="Layers the residual-stream PCA projects")
    trajectory_layers: tuple[int, ...] = Field(description="Layers residual trajectories are traced at")
    contrastive_layers: tuple[int, ...] = Field(description="Layers the contrastive projection reads")
    early_layer_cutoff: int = Field(ge=0, description="At or below this a layer counts as early")
    late_layer_cutoff: int = Field(ge=0, description="At or above this a layer counts as late")

    temperature: float = Field(gt=0.0, description="The checkpoint's native decode temperature")
    top_p: float = Field(gt=0.0, le=1.0, description="The checkpoint's native nucleus mass")
    max_new_tokens: int = Field(gt=0, description="Generation length of the extraction protocol")
    eos_token_ids: tuple[int, ...] = Field(
        description="Every token that ends a generation, including end-of-turn variants"
    )

    calibration_root: DataRoot = Field(description="Which data root the calibration directory hangs from")
    calibration_dir: str = Field(description="Calibration directory, relative to its root")

    attention_layer_types: dict[int, AttentionKind] | None = Field(
        default=None,
        description=(
            "Per-sampled-layer attention kind for interleaved-attention architectures; "
            "None means every layer attends over the full context"
        ),
    )

    @field_validator("sampled_layers", "pca_layers", "trajectory_layers", "contrastive_layers")
    @classmethod
    def _strictly_increasing(cls, layers: tuple[int, ...], info: Any) -> tuple[int, ...]:
        if not layers:
            raise ValueError(f"{info.field_name} is empty; a layer plan needs at least one layer")
        if list(layers) != sorted(set(layers)):
            raise ValueError(
                f"{info.field_name} must be strictly increasing and free of repeats, got {layers}"
            )
        return layers

    @field_validator("eos_token_ids")
    @classmethod
    def _eos_present(cls, ids: tuple[int, ...]) -> tuple[int, ...]:
        if not ids:
            raise ValueError(
                "eos_token_ids is empty; generation would run to the token budget on every sample"
            )
        if len(set(ids)) != len(ids):
            raise ValueError(f"eos_token_ids repeats a token: {ids}")
        return ids

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if self.num_attention_heads % self.num_kv_heads:
            raise ValueError(
                f"{self.name}: {self.num_attention_heads} query heads do not divide into "
                f"{self.num_kv_heads} key/value heads; the GQA group size must be exact"
            )
        depth = range(self.num_layers)
        for field in ("sampled_layers", "pca_layers", "trajectory_layers", "contrastive_layers"):
            for layer in getattr(self, field):
                if layer not in depth:
                    raise ValueError(
                        f"{self.name}: {field} names layer {layer}, outside 0..{self.num_layers - 1}"
                    )
        for field in ("early_layer_cutoff", "late_layer_cutoff"):
            if getattr(self, field) not in depth:
                raise ValueError(
                    f"{self.name}: {field}={getattr(self, field)} is outside 0..{self.num_layers - 1}"
                )
        if self.early_layer_cutoff > self.late_layer_cutoff:
            raise ValueError(
                f"{self.name}: early_layer_cutoff {self.early_layer_cutoff} is above "
                f"late_layer_cutoff {self.late_layer_cutoff}"
            )
        if self.attention_layer_types is not None:
            missing = sorted(set(self.sampled_layers) - set(self.attention_layer_types))
            if missing:
                raise ValueError(
                    f"{self.name}: attention_layer_types omits sampled layers {missing}; "
                    "an interleaved architecture must declare the kind of every sampled layer"
                )
        return self

    @property
    def kv_group_size(self) -> int:
        """Query heads per key/value head: 1 under multi-head attention."""
        return self.num_attention_heads // self.num_kv_heads

    @property
    def is_grouped_query(self) -> bool:
        """Whether query heads outnumber key/value heads."""
        return self.kv_group_size > 1

    def resolved_calibration_dir(self) -> Path:
        """The calibration directory as a location on this machine."""
        return resolve_data_path(self.calibration_root, self.calibration_dir)

    def attention_kind(self, layer: int) -> AttentionKind:
        """The attention kind of one layer: ``global`` unless declared local."""
        if self.attention_layer_types is None:
            return "global"
        return self.attention_layer_types.get(layer, "global")

    def global_layers(self) -> tuple[int, ...]:
        """The sampled layers that attend over the full context.

        Local sliding-window layers are structurally recency-dominated, so
        cross-model comparisons of the attention source use global layers; local
        attention cells stay per-model exploratory.
        """
        return tuple(layer for layer in self.sampled_layers if self.attention_kind(layer) == "global")


MODEL_PRESETS: dict[str, ModelPreset] = {
    "8b": ModelPreset(
        name="8b",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype="bfloat16",
        num_layers=32,
        hidden_dim=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        head_dim=128,
        # Proportional depth sampling at [0%, 25%, 50%, 63%, 75%, 88%, 97%], denser
        # through 60-80% where mode signal concentrates.
        sampled_layers=(0, 8, 16, 20, 24, 28, 31),
        pca_layers=(8, 16, 20, 24, 28),
        trajectory_layers=(8, 16, 20, 24, 28),
        contrastive_layers=(8, 16, 20, 24, 28),
        early_layer_cutoff=8,
        late_layer_cutoff=24,
        temperature=0.6,
        top_p=0.9,
        max_new_tokens=512,
        # 128001 end-of-text, 128008 end-of-message, 128009 end-of-turn.
        eos_token_ids=(128001, 128008, 128009),
        calibration_root="outputs",
        calibration_dir="calibration/llama31_8b",
    ),
    "3b": ModelPreset(
        name="3b",
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        torch_dtype="float16",
        num_layers=28,
        hidden_dim=3072,
        num_attention_heads=24,
        num_kv_heads=8,
        head_dim=128,
        sampled_layers=(0, 7, 14, 18, 21, 24, 27),
        pca_layers=(7, 14, 18, 21, 24),
        trajectory_layers=(7, 14, 18, 21, 24),
        contrastive_layers=(7, 14, 18, 21, 24),
        early_layer_cutoff=7,
        late_layer_cutoff=21,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=512,
        # 128001 end-of-text, 128009 end-of-turn; this checkpoint has no
        # end-of-message token.
        eos_token_ids=(128001, 128009),
        # The 3B corpus and its calibration live in the Phase-0 tree, reached
        # through the legacy-data root.
        calibration_root="legacy",
        calibration_dir="outputs/calibration",
    ),
    "olmo2-7b": ModelPreset(
        # A base checkpoint with no chat template: bare prompts only, and no
        # system prompt. Full multi-head attention, so the grouped-query caveat
        # about head indexing does not apply. Query and key RMSNorm sit between
        # the projections and RoPE, so k_proj hooks capture pre-norm pre-RoPE
        # keys: position-free holds, but the substrate is not a Llama
        # checkpoint's post-projection keys.
        name="olmo2-7b",
        model_id="allenai/OLMo-2-1124-7B",
        torch_dtype="bfloat16",
        num_layers=32,
        hidden_dim=4096,
        num_attention_heads=32,
        num_kv_heads=32,
        head_dim=128,
        sampled_layers=(0, 8, 16, 20, 24, 28, 31),
        pca_layers=(8, 16, 20, 24, 28),
        trajectory_layers=(8, 16, 20, 24, 28),
        contrastive_layers=(8, 16, 20, 24, 28),
        early_layer_cutoff=8,
        late_layer_cutoff=24,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=512,
        eos_token_ids=(100257,),
        calibration_root="outputs",
        calibration_dir="calibration/olmo2_7b",
    ),
    "gemma3-27b": ModelPreset(
        # A multimodal wrapper class, so decoder layers resolve through the
        # loader's layer-finding helper rather than a fixed attribute path.
        # Attention interleaves five local sliding-window layers to one global
        # layer, global at every sixth: the sampled layers prefer global ones,
        # with layer 0 kept local as the early-band anchor.
        name="gemma3-27b",
        model_id="google/gemma-3-27b-it",
        torch_dtype="bfloat16",
        num_layers=62,
        hidden_dim=5376,
        num_attention_heads=32,
        num_kv_heads=16,
        head_dim=128,
        sampled_layers=(0, 11, 23, 35, 41, 53, 59),
        pca_layers=(11, 23, 35, 41, 53),
        trajectory_layers=(11, 23, 35, 41, 53),
        contrastive_layers=(11, 23, 35, 41, 53),
        early_layer_cutoff=15,
        late_layer_cutoff=46,
        # The model card's native sampling.
        temperature=1.0,
        top_p=0.95,
        max_new_tokens=512,
        eos_token_ids=(1, 106),
        calibration_root="outputs",
        calibration_dir="calibration/gemma3_27b",
        attention_layer_types={
            0: "local",
            11: "global",
            23: "global",
            35: "global",
            41: "global",
            53: "global",
            59: "global",
        },
    ),
    "qwen-7b": ModelPreset(
        name="qwen-7b",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        torch_dtype="bfloat16",
        num_layers=28,
        hidden_dim=3584,
        num_attention_heads=28,
        num_kv_heads=4,
        head_dim=128,
        sampled_layers=(0, 7, 14, 18, 21, 24, 27),
        pca_layers=(7, 14, 18, 21, 24),
        trajectory_layers=(7, 14, 18, 21, 24),
        contrastive_layers=(7, 14, 18, 21, 24),
        early_layer_cutoff=7,
        late_layer_cutoff=21,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=512,
        eos_token_ids=(151643, 151645),
        calibration_root="outputs",
        calibration_dir="calibration/qwen25_7b",
    ),
    "dsv2-lite": ModelPreset(
        # A mixture-of-experts checkpoint: two shared and sixty-four routed
        # experts, greedy top-6, layer 0 a dense MLP and layers 1-26 routed.
        # Load it with trust_remote_code=False — the bundled remote code puts
        # gate projections per expert, which the routing hooks do not target.
        # Attention is latent: the keys source is the position-free part of the
        # fused key/value projection, sliced in
        # `anamnesis/extraction/model_loader.py`. With no separate key or value
        # projection module, values and queries are not part of this
        # checkpoint's feature surface.
        name="dsv2-lite",
        model_id="deepseek-ai/DeepSeek-V2-Lite-Chat",
        torch_dtype="bfloat16",
        num_layers=27,
        hidden_dim=2048,
        # The config's key/value head count, which the attention weights follow;
        # the keys-source capture is the 512-wide latent, not these heads.
        num_attention_heads=16,
        num_kv_heads=16,
        # The value head width. Query and key heads are 192 wide: 128 content
        # dimensions and 64 positional ones.
        head_dim=128,
        # Proportional depth, mid-heavy. Layer 0 is dense, so it yields no
        # expert-routing features.
        sampled_layers=(0, 5, 11, 15, 18, 22, 26),
        pca_layers=(5, 11, 15, 18, 22),
        trajectory_layers=(5, 11, 15, 18, 22),
        contrastive_layers=(5, 11, 15, 18, 22),
        early_layer_cutoff=7,
        late_layer_cutoff=20,
        # The checkpoint's own generation_config.json, which is colder than
        # every other preset here.
        temperature=0.3,
        top_p=0.95,
        max_new_tokens=512,
        # A single end-of-sentence token; this checkpoint has no separate
        # end-of-turn token.
        eos_token_ids=(100001,),
        calibration_root="outputs",
        calibration_dir="calibration/dsv2_lite",
    ),
}

PRESET_ALIASES: dict[str, str] = {
    "llama31_8b": "8b",
    "llama32_3b": "3b",
    "llama-3.1-8b": "8b",
    "llama-3.2-3b": "3b",
    "olmo2_7b": "olmo2-7b",
    "gemma3_27b": "gemma3-27b",
    "qwen25_7b": "qwen-7b",
    "qwen2.5-7b": "qwen-7b",
    "dsv2_lite": "dsv2-lite",
}
"""Longer spellings of registry keys, including the names calibration directories use."""


def preset_names() -> tuple[str, ...]:
    """Every registry key, in registry order."""
    return tuple(MODEL_PRESETS)


def resolve_preset(preset: str | ModelPreset) -> ModelPreset:
    """The preset a name denotes, or the preset itself when one is given.

    Lookup tries the registry key, then :data:`PRESET_ALIASES`, then a
    punctuation-insensitive form of both, so ``dsv2-lite``, ``dsv2_lite`` and
    ``DSV2_Lite`` all reach one row.

    Raises
    ------
    UnknownPresetError
        Naming every key and alias the registry holds.
    """
    if isinstance(preset, ModelPreset):
        return preset
    if preset in MODEL_PRESETS:
        return MODEL_PRESETS[preset]
    if preset in PRESET_ALIASES:
        return MODEL_PRESETS[PRESET_ALIASES[preset]]

    def normalise(text: str) -> str:
        return text.strip().lower().replace("_", "-").replace(".", "")

    wanted = normalise(preset)
    for key in MODEL_PRESETS:
        if normalise(key) == wanted:
            return MODEL_PRESETS[key]
    for alias, key in PRESET_ALIASES.items():
        if normalise(alias) == wanted:
            return MODEL_PRESETS[key]
    raise UnknownPresetError(
        f"unknown model preset {preset!r}; "
        f"presets: {', '.join(sorted(MODEL_PRESETS))}; "
        f"aliases: {', '.join(sorted(PRESET_ALIASES))}"
    )


class ModelConfig(BaseModel):
    """What a loader needs to put one checkpoint on a device with hooks on it.

    Build it with :meth:`from_preset`. The architecture fields have no defaults,
    so a configuration cannot silently describe a model other than the one being
    loaded.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str = Field(description="Checkpoint identifier, or a local path to one")
    torch_dtype: str = Field(description="Dtype to load the weights in")
    attn_implementation: str = Field(
        default=EAGER_ATTENTION,
        description="Attention kernel; extraction requires one that returns attention weights",
    )
    device_map: str = Field(default="auto", description="Device placement policy")

    num_layers: int = Field(gt=0, description="Decoder layer count")
    hidden_dim: int = Field(gt=0, description="Residual stream width")
    num_attention_heads: int = Field(gt=0, description="Query heads")
    num_kv_heads: int = Field(gt=0, description="Key/value heads")
    head_dim: int = Field(gt=0, description="Per-head width of the attention value path")

    preset_name: str | None = Field(
        default=None, description="The preset this was built from, carried as provenance"
    )

    @field_validator("attn_implementation")
    @classmethod
    def _returns_attention_weights(cls, implementation: str) -> str:
        if implementation in ATTENTION_WITHOUT_WEIGHTS:
            raise ValueError(
                f"attn_implementation={implementation!r} returns no attention weights, "
                f"so extraction would drop the attention source; use {EAGER_ATTENTION!r}"
            )
        return implementation

    @model_validator(mode="after")
    def _heads_divide(self) -> Self:
        if self.num_attention_heads % self.num_kv_heads:
            raise ValueError(
                f"{self.num_attention_heads} query heads do not divide into "
                f"{self.num_kv_heads} key/value heads; the GQA group size must be exact"
            )
        return self

    @property
    def kv_group_size(self) -> int:
        """Query heads per key/value head: 1 under multi-head attention."""
        return self.num_attention_heads // self.num_kv_heads

    @classmethod
    def from_preset(cls, preset: str | ModelPreset, **overrides: Any) -> ModelConfig:
        """The loader configuration a preset implies.

        ``overrides`` set fields of this class, which is how a caller points at a
        local checkpoint directory (``model_id=...``) or pins placement
        (``device_map=...``) without restating the architecture.

        Raises
        ------
        UnknownPresetError
            When the preset name is not in the registry.
        pydantic.ValidationError
            When an override names a field this class does not have, or gives a
            value the constraints reject.
        """
        row = resolve_preset(preset)
        fields: dict[str, Any] = {
            "model_id": row.model_id,
            "torch_dtype": row.torch_dtype,
            "attn_implementation": EAGER_ATTENTION,
            "num_layers": row.num_layers,
            "hidden_dim": row.hidden_dim,
            "num_attention_heads": row.num_attention_heads,
            "num_kv_heads": row.num_kv_heads,
            "head_dim": row.head_dim,
            "preset_name": row.name,
        }
        fields.update(overrides)
        return cls(**fields)
