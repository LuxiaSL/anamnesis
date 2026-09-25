"""Per-model facts, and the one bridge from a preset to a usable configuration.

A **preset** is everything that varies between models: the checkpoint, its
dtype, its architecture, the layers extraction samples, its native decode
parameters, its end-of-generation tokens, and where its calibration artifacts
live. The registry of those facts is :data:`MODELS_FILE`, a JSON file beside this
module, because adding a model is adding a row and a row is not a code change.

A **config** is what a caller hands to a loader or an extractor. Every config in
this package is built from a preset by a ``from_preset`` constructor — never by
copying fields across by hand. That is the whole point of the split: a script
that names a model gets an architecture, a layer plan, a decode policy and a
calibration path that agree with each other, because one function produced all
four from one row of data.

Extending the registry
----------------------

:data:`MODELS_ENV` names further registry files, separated the way ``PATH`` is,
and every one of them is merged over the shipped file. So a model this package
never heard of is onboarded by writing a row:

.. code-block:: json

    {"presets": {"my-model": {"name": "my-model", "model_id": "...", "...": "..."}}}

Merging is **additive and refuses collisions.** A row cannot redefine a shipped
preset, alias or depth, because a banked corpus's label means the shipped row and
silently reshaping it would change what every stored vector was produced under.
A variant of a shipped model is a new key, and the refusal names the key and the
two files it came from. A file named in the environment that cannot be read is an
error rather than a fall-through: a registry that quietly ignored it would resolve
the shipped name and run the wrong model.

Three constraints are enforced here rather than left to a reader:

* ``attn_implementation`` must return attention weights. Fused attention
  kernels do not, and extraction that silently loses them produces feature
  vectors with the attention source missing. The validator rejects the kernels
  known to drop the weights.
* The query-head count must be an integer multiple of the key/value-head count.
  Under grouped-query attention the two differ, attention weights index by query
  head while the cache indexes by key/value head, and per-head analysis is wrong
  whenever the group size is not exact.
* A run-name prefix resolves to one model. Prefixes come from registry keys,
  aliases, each row's ``run_prefixes`` and the depth-only rows, and two models
  claiming one prefix is refused — otherwise :func:`layer_counts_by_run_prefix`
  would hand a corpus another model's depth.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from anamnesis.config.paths import DataRoot, resolve_data_path

AttentionKind = Literal["local", "global"]
"""Whether a layer attends over a sliding window or the full context."""

ATTENTION_WITHOUT_WEIGHTS: frozenset[str] = frozenset(
    {"sdpa", "flash_attention_2", "flash_attention_3", "flex_attention"}
)
"""Attention kernels that return no attention weights to a hook or an output."""

EAGER_ATTENTION = "eager"
"""The attention implementation extraction requires."""

MODELS_FILE: Path = Path(__file__).resolve().parent / "models.json"
"""The registry file this package ships."""

MODELS_ENV = "ANAMNESIS_MODELS"
"""Environment variable naming further registry files, separated like ``PATH``."""


class ModelRegistryError(RuntimeError):
    """A registry file is missing, unreadable, or not what this module expects."""


class UnknownPresetError(KeyError):
    """A preset name that the registry does not hold."""


class ModelPreset(BaseModel):
    """The per-model facts every configuration is derived from.

    Layer fields are tuples because a registry is process-global: a consumer
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
    pca_components_by_layer: dict[int, int] | None = Field(
        default=None,
        description=(
            "Components the per-layer residual basis keeps at each PCA layer, keyed by every "
            "layer in pca_layers. A calibration fits each layer to its own count, and the "
            "extraction projects each layer onto the rows its basis holds, so the count can "
            "follow what each layer's samples determine. None keeps the extraction's uniform "
            "count at every layer"
        ),
    )
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

    stage0_dir: str | None = Field(
        default=None,
        description=(
            "Directory under the outputs root's battery tree holding this model's floors; "
            "spelled as it exists on disk, which is not always the registry key. "
            "None until the model has floors"
        ),
    )

    run_prefixes: tuple[str, ...] = Field(
        default=(),
        description=(
            "Further spellings a run directory of this model may begin with, beyond the "
            "registry key and the aliases pointing at it"
        ),
    )

    notes: str = Field(
        default="",
        description="What a reader of this row needs to know about the checkpoint itself",
    )

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

    @field_validator("run_prefixes")
    @classmethod
    def _prefixes_are_usable(cls, prefixes: tuple[str, ...]) -> tuple[str, ...]:
        for prefix in prefixes:
            if not prefix.strip() or prefix != prefix.strip():
                raise ValueError(f"run_prefixes holds {prefix!r}; a prefix is non-empty and untrimmed")
        if len(set(prefixes)) != len(prefixes):
            raise ValueError(f"run_prefixes repeats a spelling: {prefixes}")
        return prefixes

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
        if self.pca_components_by_layer is not None:
            if set(self.pca_components_by_layer) != set(self.pca_layers):
                raise ValueError(
                    f"{self.name}: pca_components_by_layer names layers "
                    f"{sorted(self.pca_components_by_layer)}, and pca_layers is "
                    f"{list(self.pca_layers)}; a count is given for every PCA layer or none"
                )
            if any(count <= 0 for count in self.pca_components_by_layer.values()):
                raise ValueError(
                    f"{self.name}: pca_components_by_layer holds a count below one: "
                    f"{self.pca_components_by_layer}"
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


class ModelRegistryFile(BaseModel):
    """The shape of one registry file: rows, alternative spellings, and depths.

    ``run_depths`` is the one fact a model can be in the registry for alone: a
    run-name prefix and the layer count its depth bands are fractions of. A corpus
    whose model this package cannot load still needs that denominator, and nothing
    else about the model is knowable from the corpus.
    """

    model_config = ConfigDict(extra="forbid")

    description: str = Field(default="", description="What this file is, for a reader who opens it")
    presets: dict[str, ModelPreset] = Field(default_factory=dict)
    aliases: dict[str, str] = Field(
        default_factory=dict, description="Alternative spelling to the registry key it means"
    )
    run_depths: dict[str, int] = Field(
        default_factory=dict,
        description="Run-name prefix to layer count, for models with no preset row",
    )

    @field_validator("run_depths")
    @classmethod
    def _depths_are_positive(cls, depths: dict[str, int]) -> dict[str, int]:
        for prefix, count in depths.items():
            if count <= 0:
                raise ValueError(f"run_depths[{prefix!r}]={count}; a layer count is positive")
        return depths


class ModelRegistry(BaseModel):
    """Every registry file, merged, with the prefix map the depth bands read."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    presets: dict[str, ModelPreset]
    aliases: dict[str, str]
    run_depths: dict[str, int]
    sources: tuple[Path, ...] = Field(description="The files this was read from, in merge order")

    def names(self) -> tuple[str, ...]:
        """Every registry key, in merge order."""
        return tuple(self.presets)

    def resolve(self, preset: str) -> ModelPreset:
        """The preset a name denotes.

        Lookup tries the registry key, then the aliases, then a
        punctuation-insensitive form of both, so ``dsv2-lite``, ``dsv2_lite`` and
        ``DSV2_Lite`` all reach one row.

        Raises
        ------
        UnknownPresetError
            Naming every key and alias the registry holds, and the files it read.
        """
        if preset in self.presets:
            return self.presets[preset]
        if preset in self.aliases:
            return self.presets[self.aliases[preset]]
        wanted = _normalise(preset)
        for key in self.presets:
            if _normalise(key) == wanted:
                return self.presets[key]
        for alias, key in self.aliases.items():
            if _normalise(alias) == wanted:
                return self.presets[key]
        raise UnknownPresetError(
            f"unknown model preset {preset!r}; "
            f"presets: {', '.join(sorted(self.presets))}; "
            f"aliases: {', '.join(sorted(self.aliases))}; "
            f"read from: {', '.join(str(path) for path in self.sources)}; "
            f"add a row and name its file in {MODELS_ENV}"
        )

    def layer_counts_by_run_prefix(self) -> dict[str, int]:
        """Every run-name prefix this registry answers for, and its layer count.

        A prefix is a registry key, an alias, a row's own ``run_prefixes``, or a
        depth-only row. Aliases are in because an alias names the same model, so a
        run directory spelled with one has that model's depth.
        """
        counts: dict[str, int] = {}
        for key, row in self.presets.items():
            for prefix in (key, *row.run_prefixes):
                counts[prefix] = row.num_layers
        for alias, key in self.aliases.items():
            counts[alias] = self.presets[key].num_layers
        counts.update(self.run_depths)
        return counts


def _normalise(text: str) -> str:
    """A spelling with case, underscores and dots removed, for tolerant lookup."""
    return text.strip().lower().replace("_", "-").replace(".", "")


def registry_paths() -> tuple[Path, ...]:
    """The shipped registry file, then every file named in :data:`MODELS_ENV`.

    The environment value is read at call time rather than at import, so a process
    can point at another registry without reloading the package.
    """
    extra = os.environ.get(MODELS_ENV, "")
    paths = [MODELS_FILE]
    for entry in extra.split(os.pathsep):
        text = entry.strip()
        if text:
            paths.append(Path(text).expanduser())
    return tuple(paths)


def _read_file(path: Path) -> ModelRegistryFile:
    """One registry file, parsed and validated.

    Raises
    ------
    ModelRegistryError
        When the file is absent, is not readable as JSON, does not match the
        registry shape, or holds a row whose ``name`` disagrees with its key.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ModelRegistryError(f"model registry unreadable: {path} ({exc})") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelRegistryError(f"{path}: invalid JSON at line {exc.lineno} ({exc.msg})") from exc
    try:
        parsed = ModelRegistryFile.model_validate(payload)
    except ValidationError as exc:
        raise ModelRegistryError(f"{path}: not a model registry ({exc})") from exc
    for key, row in parsed.presets.items():
        if row.name != key:
            raise ModelRegistryError(
                f"{path}: preset {key!r} carries name {row.name!r}; the key and the name are one thing"
            )
    return parsed


def _merge(files: list[tuple[Path, ModelRegistryFile]]) -> ModelRegistry:
    """Merge registry files in order, refusing any name two of them claim.

    Raises
    ------
    ModelRegistryError
        On a collision, naming the key and both files; on an alias pointing at a
        key no file defines, or shadowing a key; on a prefix two models claim.
    """
    presets: dict[str, ModelPreset] = {}
    aliases: dict[str, str] = {}
    depths: dict[str, int] = {}
    owner: dict[tuple[str, str], Path] = {}

    def claim(kind: str, key: str, path: Path) -> None:
        held = owner.get((kind, key))
        if held is not None:
            raise ModelRegistryError(
                f"{path}: {kind} {key!r} is already defined in {held}; a registry file adds "
                f"names and never redefines them, because banked data means the row already here"
            )
        owner[(kind, key)] = path

    for path, parsed in files:
        for key, row in parsed.presets.items():
            claim("preset", key, path)
            presets[key] = row
        for alias, target in parsed.aliases.items():
            claim("alias", alias, path)
            aliases[alias] = target
        for prefix, count in parsed.run_depths.items():
            claim("run depth", prefix, path)
            depths[prefix] = count

    for alias, target in aliases.items():
        if target not in presets:
            raise ModelRegistryError(
                f"alias {alias!r} points at {target!r}, which no registry file defines; "
                f"presets: {', '.join(sorted(presets))}"
            )
        if alias in presets:
            raise ModelRegistryError(
                f"{alias!r} is both a preset key and an alias; one spelling names one thing"
            )

    prefix_owner: dict[str, str] = {}
    for key, row in presets.items():
        for prefix in (key, *row.run_prefixes):
            held = prefix_owner.setdefault(prefix, key)
            if held != key:
                raise ModelRegistryError(
                    f"run prefix {prefix!r} is claimed by both {held!r} and {key!r}; "
                    "a run directory's prefix has to resolve to one model's depth"
                )
    for alias, target in aliases.items():
        held = prefix_owner.setdefault(alias, target)
        if held != target:
            raise ModelRegistryError(
                f"run prefix {alias!r} is claimed by both {held!r} and {target!r}; "
                "a run directory's prefix has to resolve to one model's depth"
            )
    for prefix in depths:
        if prefix in prefix_owner:
            raise ModelRegistryError(
                f"run depth {prefix!r} restates the layer count of preset {prefix_owner[prefix]!r}; "
                "a model's depth has one home, which is its preset row"
            )

    return ModelRegistry(
        presets=presets, aliases=aliases, run_depths=depths, sources=tuple(p for p, _ in files)
    )


_CACHE: dict[tuple[tuple[str, int, int], ...], ModelRegistry] = {}


def load_registry(paths: tuple[Path, ...] | None = None) -> ModelRegistry:
    """Every registry file, merged and validated.

    The result is cached against each file's path, size and modification time, so
    repeated lookups do not re-read the files while an edit to one of them is seen.

    Raises
    ------
    ModelRegistryError
        When a file is missing or malformed, or two files claim one name.
    """
    sources = registry_paths() if paths is None else tuple(paths)
    stamps: list[tuple[str, int, int]] = []
    for path in sources:
        try:
            stat = path.stat()
        except OSError as exc:
            raise ModelRegistryError(f"model registry unreadable: {path} ({exc})") from exc
        stamps.append((str(path), stat.st_size, stat.st_mtime_ns))
    key = tuple(stamps)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    registry = _merge([(path, _read_file(path)) for path in sources])
    _CACHE[key] = registry
    return registry


def preset_names() -> tuple[str, ...]:
    """Every registry key, in merge order."""
    return load_registry().names()


def resolve_preset(preset: str | ModelPreset) -> ModelPreset:
    """The preset a name denotes, or the preset itself when one is given.

    Raises
    ------
    UnknownPresetError
        Naming every key and alias the registry holds.
    ModelRegistryError
        When a registry file cannot be read.
    """
    if isinstance(preset, ModelPreset):
        return preset
    return load_registry().resolve(preset)


def layer_counts_by_run_prefix() -> dict[str, int]:
    """Every run-name prefix the registry answers for, and its layer count."""
    return load_registry().layer_counts_by_run_prefix()


def presets_with_floors() -> dict[str, ModelPreset]:
    """The rows that declare a ``stage0_dir``, keyed by registry key."""
    return {key: row for key, row in load_registry().presets.items() if row.stage0_dir}


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
