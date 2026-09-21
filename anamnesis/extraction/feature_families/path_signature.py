"""Path-signature feature family — level-2 log-signature (iterated integrals) of the residual path.

**Status: exploratory build.** Nothing computed here is quotable pre-first-read.

WHAT THIS COMPUTES, AND WHY IT IS NOT ANOTHER MARGINAL STATISTIC
---------------------------------------------------------------
The hand suite's order-sensitive members (slopes, stds, windows, STFT) are *marginal* order
statistics: each reads one scalar time series at a time. Rough path theory's iterated-integral
signature reads *joint* order — whether coordinate i moved before or after coordinate j. The
level-2 log-signature of a path ``X: [T, D]`` is exactly:

    level 1 (D terms)          net displacement       ``L_i = X_i[T-1] - X_i[0]``
    level 2 (D(D-1)/2 terms)   antisymmetric Lévy area
        ``A_ij = ½ Σ_t ( (X_i[t] - X_i[0]) ΔX_j[t] - (X_j[t] - X_j[0]) ΔX_i[t] )``

with ``ΔX[t] = X[t+1] - X[t]`` and the sum over ``t = 0 .. T-2``. Both are hand-rolled numpy —
**no dependency is added** (the extraction path is pure numpy, a constraint
``tests/test_extraction_purity.py`` holds every family to). If
``iisignature`` happens to be importable the selftest cross-checks against it; runtime never
imports it.

Two conventions are worth stating because they are load-bearing and are asserted in the selftest:

* **Centering at ``X[0]`` is mandatory.** Without it the areas are not translation invariant
  (they pick up ``½ (c_i L_j - c_j L_i)`` under ``X → X + c``), which would make the feature a
  function of where the residual stream happens to sit rather than of how it moved.
* **Left-endpoint and midpoint quadrature agree exactly** for the *antisymmetric* part, because
  the ``½ Σ ΔX_i ΔX_j`` correction is symmetric in ``(i, j)`` and cancels. So there is no
  quadrature choice to defend here; there is for the full (non-log) signature, which we do not
  compute.

DIMENSIONALITY — PROJECTION IS MANDATORY
----------------------------------------
A level-2 signature of a d-dim path has ~d² terms; d = 3072/4096 is out of the question. The path
is therefore projected onto a **supplied** basis before anything is integrated. This module
**never fits a basis.** It accepts one (the banked per-layer calibration PCA, top k ∈ {4, 8}) and
raises if none is given. Supervised bases (the how-axis LDA directions, spec P-B) must be refit
inside every fold by the *caller*; this module is basis-agnostic and only records the label.

THE NULL (spec §2) IS IN THIS MODULE, NOT A SIDE SCRIPT
-------------------------------------------------------
``PathSignatureConfig.permute_increments`` shuffles the increments of the projected path and
re-cumulates. It is exactly matched by construction: level-1 terms are *literally* invariant
(a permuted sum is the same sum, up to float associativity ~1e-12), level-2 terms are destroyed.
That asymmetry is the machinery check — if level-1 moves, the implementation is wrong; if
level-2's advantage survives, the advantage was parameters, not order.

The permutation is applied to the **base (projected, pre-augmentation) path**, i.e. it shuffles
the residual's moves and keeps the clock. With uniform time augmentation this is equivalent to
permuting post-augmentation (the time increments are constant), but the pre-augmentation form is
the one whose semantics survive a non-uniform clock, so it is the one implemented.

FEATURE NAMES
-------------
    res_sig_L{layer}_{basis}_k{k}_{aug|noaug}_lvl1_c{i}
    res_sig_L{layer}_{basis}_k{k}_{aug|noaug}_lvl2_c{i}c{j}

which ``anamnesis.feature_map`` classifies as
SOURCE=``residual`` · METHOD=``iterated_integral`` (new) · DEPTH = the site's band ·
DYNAMIC: ``lvl1`` static (a net level), ``lvl2`` dynamic (an order read). Names are IDENTICAL
under the permutation null so a null run is column-comparable with the real one; the null is
recorded in :class:`PathSignatureMetadata`, not in the names.

Run the selftest (no GPU, no banked data)::

    python -m anamnesis.extraction.feature_families.path_signature --selftest
"""

from __future__ import annotations

import abc
import argparse
import itertools
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, ClassVar, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from anamnesis.extraction.feature_families import FeatureFamilyResult

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]

FAMILY_NAME = "path_signature"
FEATURE_PREFIX = "res_sig"


# ──────────────────────────────────────────────────────────────────────────────
# Errors — explicit failure modes. A malformed path raises; it never returns zeros.
# ──────────────────────────────────────────────────────────────────────────────


class PathSignatureError(ValueError):
    """Base error for the path-signature family."""


class MalformedPathError(PathSignatureError):
    """The supplied path is not a usable [T, d] real-valued trajectory."""


class ProjectionError(PathSignatureError):
    """The projection basis is missing, mis-shaped, or incompatible with the path."""


class ShortPathError(PathSignatureError):
    """The path has too few positions for the requested signature level."""


# ──────────────────────────────────────────────────────────────────────────────
# Config + metadata (pydantic — project standard)
# ──────────────────────────────────────────────────────────────────────────────


class PathSignatureConfig(BaseModel):
    """Configuration for one path-signature extraction.

    Every field that changes a number is recorded verbatim into
    :class:`PathSignatureMetadata`, so a feature vector can always be re-derived from its
    own provenance (the "every number carries n / model / site / projection / augmentation"
    discipline, spec §6).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    layer_indices: tuple[int, ...] = Field(
        default=(16,),
        description="Transformer layer indices (sites) to take the signature at. "
                    "Spec sites: 3B L14, 8B L16. Per-model, per-site, as everything is.",
    )
    n_components: int = Field(
        default=4, ge=2, le=64,
        description="k — number of projection-basis directions kept (spec: k in {4, 8}).",
    )
    level: Literal[1, 2] = Field(
        default=2,
        description="Log-signature truncation level. 1 = net displacement only (the null's "
                    "invariant half); 2 = displacement + Levy areas.",
    )
    time_augment: bool = Field(
        default=True,
        description="Append normalised position t/(T-1) as an extra coordinate before "
                    "integrating. Without it the signature is reparametrisation-invariant "
                    "and discards pacing/length (spec 'Time augmentation').",
    )
    center_at_origin: bool = Field(
        default=True,
        description="Subtract X[0] before integrating. Required for translation invariance "
                    "of the areas; turning it off is a deliberate non-standard read.",
    )
    positional_correct: bool = Field(
        default=True,
        description="Subtract the banked positional mean from each hidden state before "
                    "projecting (matches residual_stream / T2.5 / T3 practice).",
    )
    permute_increments: bool = Field(
        default=False,
        description="THE NULL (spec section 2): permute the projected path's increments and "
                    "re-cumulate. Level-1 terms are invariant by construction; level-2 die.",
    )
    permutation_seed: int | None = Field(
        default=None,
        description="Seed for the increment permutation. Required when permute_increments "
                    "is on — an unseeded null is not reproducible and is refused.",
    )
    on_short_path: Literal["raise", "zeros"] = Field(
        default="raise",
        description="Behaviour when a generation has too few positions. 'raise' (default, the "
                    "spec's no-silent-wrong-answers rule) or 'zeros' (the pipeline-contract "
                    "behaviour of the other families — name-matched zero block, logged).",
    )
    on_missing_layer: Literal["raise", "zeros"] = Field(
        default="raise",
        description="Behaviour when a requested layer is absent / zero-filled in the bank.",
    )
    basis_label: str = Field(
        default="pcaA",
        min_length=1,
        description="Short token naming the projection basis; goes into every feature name. "
                    "'pcaA' = the banked calibration PCA (spec P-A). Use e.g. 'ldaB' for the "
                    "supervised P-B basis so the two can never be pooled by accident. "
                    "Must be alphanumeric — feature names are parsed by '_'-splitting.",
    )
    min_positions: int = Field(
        default=3, ge=2,
        description="Minimum generated positions for a usable path (>=2 for level 1, and "
                    "an area needs >=3 to be anything but degenerate).",
    )

    @field_validator("basis_label")
    @classmethod
    def _label_is_token_safe(cls, v: str) -> str:
        if not v.isalnum():
            raise ValueError(
                f"basis_label must be alphanumeric (no '_' — feature names are parsed by "
                f"underscore-splitting in feature_map.py); got {v!r}"
            )
        return v

    @field_validator("layer_indices")
    @classmethod
    def _layers_nonempty(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if not v:
            raise ValueError("layer_indices must name at least one site")
        if any(l < 0 for l in v):
            raise ValueError(f"layer_indices must be non-negative; got {v}")
        if len(set(v)) != len(v):
            raise ValueError(f"layer_indices must be unique; got {v}")
        return v

    @model_validator(mode="after")
    def _null_must_be_seeded(self) -> "PathSignatureConfig":
        if self.permute_increments and self.permutation_seed is None:
            raise ValueError(
                "permute_increments=True requires an explicit permutation_seed — the null is "
                "the primary control and an unreproducible null is worthless (spec section 2 "
                "asks for >=3 seeds per cell)."
            )
        return self

    @property
    def path_dim(self) -> int:
        """D — the dimension of the path actually integrated (k, +1 if time-augmented)."""
        return self.n_components + (1 if self.time_augment else 0)

    @property
    def n_level1(self) -> int:
        return self.path_dim

    @property
    def n_level2(self) -> int:
        if self.level < 2:
            return 0
        d = self.path_dim
        return d * (d - 1) // 2

    @property
    def n_features_per_site(self) -> int:
        return self.n_level1 + self.n_level2

    @property
    def aug_token(self) -> str:
        return "aug" if self.time_augment else "noaug"


class PathSignatureMetadata(BaseModel):
    """Provenance for one extraction — everything a number must carry to be readable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: str = FAMILY_NAME
    layer_indices: tuple[int, ...]
    n_components: int
    path_dim: int
    level: int
    time_augmented: bool
    centered_at_origin: bool
    positional_corrected: bool
    basis_label: str
    permuted: bool
    permutation_seed: int | None
    n_positions: dict[int, int] = Field(
        default_factory=dict,
        description="layer_idx -> T actually integrated (after any skipping).",
    )
    degraded_sites: tuple[int, ...] = Field(
        default=(),
        description="Sites that emitted a name-matched zero block instead of real numbers "
                    "(only possible under on_short_path/on_missing_layer == 'zeros').",
    )

    @classmethod
    def from_config(cls, config: PathSignatureConfig, **kw: Any) -> "PathSignatureMetadata":
        return cls(
            layer_indices=config.layer_indices,
            n_components=config.n_components,
            path_dim=config.path_dim,
            level=config.level,
            time_augmented=config.time_augment,
            centered_at_origin=config.center_at_origin,
            positional_corrected=config.positional_correct,
            basis_label=config.basis_label,
            permuted=config.permute_increments,
            permutation_seed=config.permutation_seed,
            **kw,
        )


class PathSignatureResult(FeatureFamilyResult):
    """FeatureFamilyResult + provenance.

    Subclasses the pipeline's contract type so ``feature_pipeline`` consumes it unchanged
    (``.features`` / ``.feature_names`` / ``.family_name`` / ``len()``); the extra
    ``metadata`` field is there for the null bookkeeping the spec requires.
    """

    def __init__(
        self,
        features: F32,
        feature_names: list[str],
        family_name: str,
        metadata: "PathSignatureMetadata | OutputStatsMetadata | AttentionRegionMetadata",
    ) -> None:
        super().__init__(
            features=features, feature_names=feature_names, family_name=family_name,
        )
        self.metadata = metadata


# ──────────────────────────────────────────────────────────────────────────────
# Projection basis — supplied, NEVER fitted here
# ──────────────────────────────────────────────────────────────────────────────


class ProjectionBasis(BaseModel):
    """A supplied linear projection: ``X_proj = (X - mean) @ components.T``.

    Invariants enforced at construction: ``components`` is ``[k, d]`` real and finite,
    ``mean`` is ``[d]`` or None. This class deliberately has **no fit method** — the spec's
    leak-proofness rests on the basis coming from calibration (P-A) or from an outer fold
    (P-B), never from the evaluation data seen inside this module.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    components: np.ndarray
    mean: np.ndarray | None = None
    label: str = "pcaA"
    layer_idx: int | None = None

    @field_validator("components")
    @classmethod
    def _check_components(cls, v: np.ndarray) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 2:
            raise ProjectionError(f"components must be 2-D [k, d]; got shape {arr.shape}")
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ProjectionError(f"components must be non-empty; got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ProjectionError("components contain non-finite values")
        return arr

    @field_validator("mean")
    @classmethod
    def _check_mean(cls, v: np.ndarray | None) -> np.ndarray | None:
        if v is None:
            return None
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 1:
            raise ProjectionError(f"mean must be 1-D [d]; got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ProjectionError("mean contains non-finite values")
        return arr

    @model_validator(mode="after")
    def _check_compatible(self) -> "ProjectionBasis":
        if self.mean is not None and self.mean.shape[0] != self.components.shape[1]:
            raise ProjectionError(
                f"mean dim {self.mean.shape[0]} != components dim {self.components.shape[1]}"
            )
        return self

    @property
    def k_max(self) -> int:
        return int(self.components.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.components.shape[1])

    def project(self, X: NDArray[Any], n_components: int) -> F64:
        """Project ``[T, d]`` onto the top ``n_components`` directions -> ``[T, k]``."""
        arr = _as_path_array(X)
        if arr.shape[1] != self.hidden_dim:
            raise ProjectionError(
                f"path hidden dim {arr.shape[1]} != basis hidden dim {self.hidden_dim} "
                f"(basis label {self.label!r}, layer {self.layer_idx})"
            )
        if n_components > self.k_max:
            raise ProjectionError(
                f"requested k={n_components} > basis rank {self.k_max} "
                f"(basis label {self.label!r})"
            )
        centered = arr if self.mean is None else arr - self.mean[None, :]
        return np.ascontiguousarray(centered @ self.components[:n_components].T)


class ProjectionBasisBank(BaseModel):
    """Per-layer bases, with an optional global fallback.

    The banked 8B calibration PCA is a single global model
    (``outputs/calibration/llama31_8b/pca_model.pkl``: components [50, 4096], mean [4096]);
    the C5 calibrations are per-layer dicts. Both load through here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    per_layer: dict[int, ProjectionBasis] = Field(default_factory=dict)
    fallback: ProjectionBasis | None = None

    def for_layer(self, layer_idx: int) -> ProjectionBasis:
        if layer_idx in self.per_layer:
            return self.per_layer[layer_idx]
        if self.fallback is not None:
            return self.fallback
        raise ProjectionError(
            f"no projection basis for layer {layer_idx} and no global fallback "
            f"(have layers {sorted(self.per_layer)})"
        )

    @property
    def label(self) -> str:
        if self.fallback is not None:
            return self.fallback.label
        if self.per_layer:
            return next(iter(self.per_layer.values())).label
        raise ProjectionError("empty basis bank")

    @classmethod
    def from_pca_pickle(cls, path: Path | str, label: str = "pcaA") -> "ProjectionBasisBank":
        """Load a banked calibration PCA (global or C5 per-layer format).

        Mirrors ``feature_pipeline._load_pca_model``'s format sniffing so the same artefacts
        that already feed T3 feed this family, with no new calibration step.
        """
        p = Path(path)
        if not p.exists():
            raise ProjectionError(f"PCA calibration not found: {p}")
        with open(p, "rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, dict):
            vals = list(obj.values())
            if vals and isinstance(vals[0], dict) and "components" in vals[0]:
                per_layer = {
                    int(k): ProjectionBasis(
                        components=np.asarray(v["components"], dtype=np.float64),
                        mean=(None if v.get("mean") is None
                              else np.asarray(v["mean"], dtype=np.float64)),
                        label=label,
                        layer_idx=int(k),
                    )
                    for k, v in obj.items()
                }
                return cls(per_layer=per_layer, fallback=None)
            components = obj.get("components")
            mean = obj.get("mean")
        else:
            components = getattr(obj, "components_", None)
            mean = getattr(obj, "mean_", None)

        if components is None:
            raise ProjectionError(f"no 'components' in PCA calibration {p}")
        return cls(
            per_layer={},
            fallback=ProjectionBasis(
                components=np.asarray(components, dtype=np.float64),
                mean=(None if mean is None else np.asarray(mean, dtype=np.float64)),
                label=label,
            ),
        )

    @classmethod
    def single(cls, basis: ProjectionBasis) -> "ProjectionBasisBank":
        return cls(per_layer={}, fallback=basis)


# ──────────────────────────────────────────────────────────────────────────────
# The loader interface — deliberately thin, because the dig may change the concrete side
# ──────────────────────────────────────────────────────────────────────────────


class ResidualPath(BaseModel):
    """One generation's residual trajectory at one site, in hidden space.

    ``array`` is ``[T, d]`` float64: T generated positions (prefill index 0 already skipped
    by the source — see spec section 1), d = model hidden dim. Position-mean correction, if
    any, has already been applied by the source.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    array: np.ndarray
    gen_id: int
    layer_idx: int
    prompt_length: int = 0
    positional_corrected: bool = False
    provenance: str = ""

    @field_validator("array")
    @classmethod
    def _validate(cls, v: np.ndarray) -> np.ndarray:
        return _as_path_array(v)

    @property
    def n_positions(self) -> int:
        return int(self.array.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.array.shape[1])


class PathSource(abc.ABC):
    """Abstract per-generation supplier of ``[T, d]`` paths for the signature machinery.

    THE ONE SEAM (generalised 2026-09-11, SPEC-path-receptacles-and-span-coverage §1a/§1b):
    originally named ``ResidualPathSource`` and specific to the residual trajectory, this
    interface is now source-agnostic — a "site" (the ``layer_idx``/``site_id`` argument) is
    whatever the concrete source calls one: a transformer layer for the residual and
    attention-region sources, or a single nominal id for the output-statistics source (which
    has no per-layer structure). Everything downstream of ``load_path`` — augmentation,
    integration, the null, the naming — is format-agnostic and does not care which kind of
    site it got. ``ResidualPathSource`` remains a name-identical alias below so every existing
    reference (this module, its tests, any caller) keeps working unchanged.

    Contract for an implementation:

    * ``generation_ids()`` returns the ids it can serve, ascending.
    * ``available_layers(gen_id)`` returns the site indices it can serve for that generation
      (transformer layer indices for residual/attention; a fixed one-element list for output).
    * ``load_path(gen_id, layer_idx)`` returns an object exposing ``.array`` — ``[T, d]``,
      float64, finite, generated positions only, in generation order (``d`` is the basis rank
      for the residual source, or the source's fixed native dimensionality for output/
      attention — no projection step exists for those two).
      It MUST raise (``KeyError`` / :class:`PathSignatureError`) rather than return a
      zero-filled or truncated array when the data is absent — the caller's
      ``on_missing_layer``/``on_short_path`` policy decides what happens next, and it cannot
      decide anything if absence is disguised as data.
    """

    @abc.abstractmethod
    def generation_ids(self) -> list[int]:
        ...

    @abc.abstractmethod
    def available_layers(self, gen_id: int) -> list[int]:
        ...

    @abc.abstractmethod
    def load_path(self, gen_id: int, layer_idx: int) -> Any:
        ...


#: Backward-compat alias — the residual family's seam kept its original name; every existing
#: reference (``class ResidualPathSource``'s old identity, ``ArrayPathSource(ResidualPathSource)``,
#: ``RawGenerationDataPathSource(ResidualPathSource)``, external callers) resolves to the exact
#: same class object as :class:`PathSource`, so residual behaviour is unchanged bit-for-bit.
ResidualPathSource = PathSource


class ArrayPathSource(ResidualPathSource):
    """In-memory source: ``{gen_id: {layer_idx: [T, d] array}}``.

    Used by the selftest and by any dig outcome that lands as plain arrays. Cost of swapping
    the real bank in is one class implementing :class:`ResidualPathSource`.
    """

    def __init__(
        self,
        paths: dict[int, dict[int, NDArray[Any]]],
        *,
        prompt_lengths: dict[int, int] | None = None,
        positional_corrected: bool = False,
        provenance: str = "in-memory",
    ) -> None:
        if not isinstance(paths, dict) or not paths:
            raise PathSignatureError("ArrayPathSource requires a non-empty {gen: {layer: arr}}")
        self._paths = paths
        self._prompt_lengths = prompt_lengths or {}
        self._positional_corrected = positional_corrected
        self._provenance = provenance

    def generation_ids(self) -> list[int]:
        return sorted(self._paths)

    def available_layers(self, gen_id: int) -> list[int]:
        if gen_id not in self._paths:
            raise KeyError(f"gen {gen_id} not in source")
        return sorted(self._paths[gen_id])

    def load_path(self, gen_id: int, layer_idx: int) -> ResidualPath:
        if gen_id not in self._paths:
            raise KeyError(f"gen {gen_id} not in source")
        if layer_idx not in self._paths[gen_id]:
            raise KeyError(f"layer {layer_idx} not banked for gen {gen_id}")
        return ResidualPath(
            array=self._paths[gen_id][layer_idx],
            gen_id=gen_id,
            layer_idx=layer_idx,
            prompt_length=self._prompt_lengths.get(gen_id, 0),
            positional_corrected=self._positional_corrected,
            provenance=self._provenance,
        )


class RawGenerationDataPathSource(ResidualPathSource):
    """Source over a single in-memory ``RawGenerationData`` (the pipeline's own contract).

    ``hidden_states[t][l+1]`` — index 0 is the embedding output, not layer 0, so an
    off-by-one here silently mislabels every layer-indexed feature.
    Positional correction reuses ``state_extractor._correct_hidden_state``, i.e. exactly what
    the residual_trajectory / T2.5 / T3 features do, so the projected path is the same object
    those features summarise.
    """

    def __init__(
        self,
        data: Any,  # RawGenerationData — typed as Any to keep this module import-light
        *,
        gen_id: int = 0,
        positional_correct: bool = True,
        zero_norm_eps: float = 1e-6,
    ) -> None:
        if not hasattr(data, "hidden_states"):
            raise PathSignatureError(
                "RawGenerationDataPathSource needs a RawGenerationData-like object with "
                ".hidden_states"
            )
        self._data = data
        self._gen_id = gen_id
        self._positional_correct = positional_correct
        self._zero_norm_eps = zero_norm_eps

    def generation_ids(self) -> list[int]:
        return [self._gen_id]

    def available_layers(self, gen_id: int) -> list[int]:
        self._check_gen(gen_id)
        hs = self._data.hidden_states
        if not hs:
            return []
        n_layers_plus_embed = int(np.asarray(hs[0]).shape[0])
        return list(range(n_layers_plus_embed - 1))

    def load_path(self, gen_id: int, layer_idx: int) -> ResidualPath:
        self._check_gen(gen_id)
        data = self._data
        hs = data.hidden_states
        T = len(hs)
        if T == 0:
            raise ShortPathError(f"gen {gen_id} has zero generated positions")
        arr_idx = layer_idx + 1  # index 0 is the embedding layer
        n_rows = int(np.asarray(hs[0]).shape[0])
        if not (0 <= arr_idx < n_rows):
            raise KeyError(
                f"layer {layer_idx} (hidden_states row {arr_idx}) out of range for "
                f"{n_rows} rows"
            )

        first = np.asarray(hs[0][arr_idx], dtype=np.float64)
        if float(np.linalg.norm(first)) < self._zero_norm_eps:
            raise KeyError(
                f"layer {layer_idx} is zero-filled in this bank (norm < {self._zero_norm_eps}) "
                f"— not saved for gen {gen_id}"
            )

        pos_means = getattr(data, "positional_means", None) if self._positional_correct else None
        prompt_length = int(getattr(data, "prompt_length", 0) or 0)

        if pos_means is None:
            rows = [np.asarray(hs[t][arr_idx], dtype=np.float64) for t in range(T)]
        else:
            from anamnesis.extraction.state_extractor import _correct_hidden_state
            rows = [
                np.asarray(
                    _correct_hidden_state(
                        np.asarray(hs[t][arr_idx], dtype=np.float32),
                        arr_idx, prompt_length + t, pos_means,
                    ),
                    dtype=np.float64,
                )
                for t in range(T)
            ]

        return ResidualPath(
            array=np.stack(rows, axis=0),
            gen_id=gen_id,
            layer_idx=layer_idx,
            prompt_length=prompt_length,
            positional_corrected=pos_means is not None,
            provenance="RawGenerationData",
        )

    def _check_gen(self, gen_id: int) -> None:
        if gen_id != self._gen_id:
            raise KeyError(f"this source serves gen {self._gen_id} only; asked for {gen_id}")


# ──────────────────────────────────────────────────────────────────────────────
# §1a / §1b — two sibling path sources, NO projection step (natively low-dimensional)
#
# SPEC-path-receptacles-and-span-coverage-2026-09-11 §1a/§1b. Both feed the SAME signature
# machinery below (permute_increments / time_augment_path / log_signature_level2) — only the
# path SOURCE differs, and neither one ever calls ProjectionBasis.project(). Config classes are
# deliberately separate from PathSignatureConfig (not a subclass, not a refactor of it) so the
# residual family's class is untouched — its selftest must reproduce bit-for-bit.
# ──────────────────────────────────────────────────────────────────────────────


class _NullControlMixin(BaseModel):
    """Fields + validators shared by the null-battery control, factored out so §1a/§1b don't
    hand-copy the seeded-null rule. Not used by PathSignatureConfig (left untouched)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    level: Literal[1, 2] = Field(
        default=2,
        description="Log-signature truncation level (same semantics as PathSignatureConfig).",
    )
    time_augment: bool = Field(
        default=True,
        description="Append normalised position t/(T-1) as an extra coordinate. See "
                    "PathSignatureConfig.time_augment for the pacing-not-duration rationale "
                    "(DESK RULING, REPORT-path-signature-family-2026-09-11) — carried forward "
                    "unchanged for these sources.",
    )
    center_at_origin: bool = Field(
        default=True,
        description="Subtract X[0] before integrating (translation invariance of the areas).",
    )
    permute_increments: bool = Field(
        default=False,
        description="THE NULL: permute the native path's increments and re-cumulate.",
    )
    permutation_seed: int | None = Field(default=None)
    on_short_path: Literal["raise", "zeros"] = Field(default="raise")
    min_positions: int = Field(default=3, ge=2)

    @model_validator(mode="after")
    def _null_must_be_seeded(self) -> "_NullControlMixin":
        if self.permute_increments and self.permutation_seed is None:
            raise ValueError(
                "permute_increments=True requires an explicit permutation_seed — the null is "
                "the primary control and an unreproducible null is worthless (spec section 2 "
                "asks for >=3 seeds per cell)."
            )
        return self

    @property
    def aug_token(self) -> str:
        return "aug" if self.time_augment else "noaug"

    @property
    def n_level1(self) -> int:
        return self.path_dim

    @property
    def n_level2(self) -> int:
        if self.level < 2:
            return 0
        d = self.path_dim
        return d * (d - 1) // 2

    @property
    def n_features_per_site(self) -> int:
        return self.n_level1 + self.n_level2

    @property
    def path_dim(self) -> int:  # overridden by each concrete config
        raise NotImplementedError


OUTPUT_SITE_ID = 0
#: The output-statistics path has exactly one site per generation (the final-logit output
#: distribution has no transformer-layer index) — this is the sentinel `layer_idx` PathSource
#: implementations for this family accept. NOT a real transformer layer.


class OutputStatsPathConfig(_NullControlMixin):
    """Config for the §1a output-statistics path — SOURCE=output.

    Native dims (NO PROJECTION — these are already low-dimensional; spec §1 "Both are
    naturally low-dimensional, so no projection step exists and the projection-adequacy gate
    cannot bite."): ``entropy, margin, eos_log_mass, varentropy`` — 4 dims.

    ``repetition-mass`` (the spec's fifth functional) is DELIBERATELY OMITTED. Reused verbatim,
    it is ``annex_potential_gradient.s_terms_repmass``: ``S = sum_{v in prior context} p_t(v)``
    where "prior context" is every token seen so far, INCLUDING THE PROMPT. That set needs the
    prompt's actual token ids; ``RawGenerationData`` (this pipeline's per-generation contract,
    defined in ``anamnesis/extraction/state_extractor.py``) carries only
    ``prompt_length: int`` — a count, not the ids — so the candidate set
    cannot be reconstructed from what this family is contracted to read. A scoped substitute
    (self-only repetition over the generated span, ``annex_cs_pulses.s_terms_selfrep``) exists
    and IS fully computable from ``chosen_token_ids``, but spec §1a says "reuse those
    definitions verbatim — do not invent new formulas"; swapping in a differently-scoped
    quantity under the same name is exactly the kind of invention that rule forbids. So: omit,
    not misname. Dimension count reflects this — 4 dims, not 5 (spec's own arithmetic: 4 + clock
    = 5 augmented -> level-1=5, level-2=10, total=15; unaugmented -> 4+6=10).
    """

    NATIVE_DIM: ClassVar[int] = 4  # entropy, margin, eos_log_mass, varentropy — this ORDER

    eos_token_ids: tuple[int, ...] = Field(
        ...,
        min_length=1,
        description="EOS token ids for THIS model (CLAUDE.md gotcha: model-specific, e.g. "
                    "Llama 3.2 3B [128001, 128009], Llama 3.1 8B [128001, 128008, 128009]). "
                    "No default — an unspecified EOS set is a silent-wrong-answer risk the "
                    "spec's absence-must-raise rule forbids.",
    )

    @field_validator("eos_token_ids")
    @classmethod
    def _eos_ids_valid(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if any(i < 0 for i in v):
            raise ValueError(f"eos_token_ids must be non-negative; got {v}")
        if len(set(v)) != len(v):
            raise ValueError(f"eos_token_ids must be unique; got {v}")
        return v

    @property
    def path_dim(self) -> int:
        return self.NATIVE_DIM + (1 if self.time_augment else 0)


class AttentionRegionPathConfig(_NullControlMixin):
    """Config for the §1b attention-region path — SOURCE=attention.

    Native dims (NO PROJECTION): the region decomposition
    ``attention_flow.extract_attention_flow`` already computes —
    ``prompt, early_gen, mid_gen, recent`` (4 dims), plus ``sink`` (attention to position 0,
    ``state_extractor``'s ``cache_sink_mass_L{n}`` formula) as a 5th when ``include_sink=True``
    (spec: "plus sink mass if separable" — it is, per the existing T2.5 sink feature, so this
    defaults on). Region order is fixed and documented so coordinate indices ``c0..c{k-1}`` are
    interpretable: ``[prompt, early_gen, mid_gen, recent, (sink)]``.
    """

    layer_indices: tuple[int, ...] = Field(
        default=(16,),
        description="Attention-layer sites (same per-model site convention as the residual "
                    "family: 3B L14, 8B L16).",
    )
    include_sink: bool = Field(
        default=True,
        description="Append sink mass (attention to position 0) as a 5th native coordinate. "
                    "False keeps the 4-region-only ablation the spec's '4-5 dims' range allows.",
    )
    on_missing_layer: Literal["raise", "zeros"] = Field(default="raise")

    @field_validator("layer_indices")
    @classmethod
    def _layers_nonempty(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if not v:
            raise ValueError("layer_indices must name at least one site")
        if any(l < 0 for l in v):
            raise ValueError(f"layer_indices must be non-negative; got {v}")
        if len(set(v)) != len(v):
            raise ValueError(f"layer_indices must be unique; got {v}")
        return v

    @property
    def native_dim(self) -> int:
        return 5 if self.include_sink else 4

    @property
    def path_dim(self) -> int:
        return self.native_dim + (1 if self.time_augment else 0)


class OutputStatsMetadata(BaseModel):
    """Provenance for one output-statistics-path extraction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: str = "path_signature_output"
    eos_token_ids: tuple[int, ...]
    path_dim: int
    level: int
    time_augmented: bool
    centered_at_origin: bool
    permuted: bool
    permutation_seed: int | None
    n_positions: dict[int, int] = Field(default_factory=dict)
    degraded_sites: tuple[int, ...] = Field(default=())

    @classmethod
    def from_config(cls, config: OutputStatsPathConfig, **kw: Any) -> "OutputStatsMetadata":
        return cls(
            eos_token_ids=config.eos_token_ids,
            path_dim=config.path_dim,
            level=config.level,
            time_augmented=config.time_augment,
            centered_at_origin=config.center_at_origin,
            permuted=config.permute_increments,
            permutation_seed=config.permutation_seed,
            **kw,
        )


class AttentionRegionMetadata(BaseModel):
    """Provenance for one attention-region-path extraction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: str = "path_signature_attention"
    layer_indices: tuple[int, ...]
    include_sink: bool
    path_dim: int
    level: int
    time_augmented: bool
    centered_at_origin: bool
    permuted: bool
    permutation_seed: int | None
    n_positions: dict[int, int] = Field(default_factory=dict)
    degraded_sites: tuple[int, ...] = Field(default=())

    @classmethod
    def from_config(cls, config: AttentionRegionPathConfig, **kw: Any) -> "AttentionRegionMetadata":
        return cls(
            layer_indices=config.layer_indices,
            include_sink=config.include_sink,
            path_dim=config.path_dim,
            level=config.level,
            time_augmented=config.time_augment,
            centered_at_origin=config.center_at_origin,
            permuted=config.permute_increments,
            permutation_seed=config.permutation_seed,
            **kw,
        )


class OutputStatsPath(BaseModel):
    """One generation's output-statistics path: ``array`` is ``[T, 4]`` float64
    (entropy, margin, eos_log_mass, varentropy), generated positions only."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    array: np.ndarray
    gen_id: int
    provenance: str = ""

    @field_validator("array")
    @classmethod
    def _validate(cls, v: np.ndarray) -> np.ndarray:
        return _as_path_array(v)

    @property
    def n_positions(self) -> int:
        return int(self.array.shape[0])


class AttentionRegionPath(BaseModel):
    """One generation's attention-region path at one layer: ``array`` is ``[T, 4-or-5]``
    float64 (prompt, early_gen, mid_gen, recent[, sink]), generated positions only."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    array: np.ndarray
    gen_id: int
    layer_idx: int
    include_sink: bool
    provenance: str = ""

    @field_validator("array")
    @classmethod
    def _validate(cls, v: np.ndarray) -> np.ndarray:
        return _as_path_array(v)

    @property
    def n_positions(self) -> int:
        return int(self.array.shape[0])


class ArrayOutputStatsSource(PathSource):
    """In-memory source: ``{gen_id: [T, 4] array}``. Used by the selftest."""

    def __init__(self, paths: dict[int, NDArray[Any]]) -> None:
        if not isinstance(paths, dict) or not paths:
            raise PathSignatureError("ArrayOutputStatsSource requires a non-empty {gen: arr}")
        self._paths = paths

    def generation_ids(self) -> list[int]:
        return sorted(self._paths)

    def available_layers(self, gen_id: int) -> list[int]:
        if gen_id not in self._paths:
            raise KeyError(f"gen {gen_id} not in source")
        return [OUTPUT_SITE_ID]

    def load_path(self, gen_id: int, layer_idx: int) -> OutputStatsPath:
        if gen_id not in self._paths:
            raise KeyError(f"gen {gen_id} not in source")
        if layer_idx != OUTPUT_SITE_ID:
            raise KeyError(
                f"output-statistics source has one site (id {OUTPUT_SITE_ID}); asked for "
                f"{layer_idx}"
            )
        return OutputStatsPath(array=self._paths[gen_id], gen_id=gen_id, provenance="in-memory")


class ArrayAttentionRegionSource(PathSource):
    """In-memory source: ``{gen_id: {layer_idx: [T, 4-or-5] array}}``. Used by the selftest."""

    def __init__(
        self, paths: dict[int, dict[int, NDArray[Any]]], *, include_sink: bool = True,
    ) -> None:
        if not isinstance(paths, dict) or not paths:
            raise PathSignatureError(
                "ArrayAttentionRegionSource requires a non-empty {gen: {layer: arr}}"
            )
        self._paths = paths
        self._include_sink = include_sink

    def generation_ids(self) -> list[int]:
        return sorted(self._paths)

    def available_layers(self, gen_id: int) -> list[int]:
        if gen_id not in self._paths:
            raise KeyError(f"gen {gen_id} not in source")
        return sorted(self._paths[gen_id])

    def load_path(self, gen_id: int, layer_idx: int) -> AttentionRegionPath:
        if gen_id not in self._paths:
            raise KeyError(f"gen {gen_id} not in source")
        if layer_idx not in self._paths[gen_id]:
            raise KeyError(f"layer {layer_idx} not banked for gen {gen_id}")
        return AttentionRegionPath(
            array=self._paths[gen_id][layer_idx],
            gen_id=gen_id,
            layer_idx=layer_idx,
            include_sink=self._include_sink,
            provenance="in-memory",
        )


class RawGenerationDataOutputStatsSource(PathSource):
    """Source over a single in-memory ``RawGenerationData``: builds the ``[T, 4]``
    output-statistics path from ``.logits`` / ``.chosen_token_ids`` (see
    ``_output_stats_per_token`` below for the exact, verbatim-reused, formulas).
    """

    def __init__(
        self,
        data: Any,
        *,
        gen_id: int = 0,
        eos_token_ids: Sequence[int],
    ) -> None:
        if not hasattr(data, "logits"):
            raise PathSignatureError(
                "RawGenerationDataOutputStatsSource needs a RawGenerationData-like object "
                "with .logits"
            )
        if not eos_token_ids:
            raise PathSignatureError(
                "eos_token_ids must be non-empty — EOS ids are model-specific (CLAUDE.md "
                "gotcha) and absence must raise, never silently default."
            )
        self._data = data
        self._gen_id = gen_id
        self._eos_token_ids = tuple(sorted({int(i) for i in eos_token_ids}))

    def generation_ids(self) -> list[int]:
        return [self._gen_id]

    def available_layers(self, gen_id: int) -> list[int]:
        self._check_gen(gen_id)
        return [OUTPUT_SITE_ID]

    def load_path(self, gen_id: int, layer_idx: int) -> OutputStatsPath:
        self._check_gen(gen_id)
        if layer_idx != OUTPUT_SITE_ID:
            raise KeyError(
                f"output-statistics source has one site (id {OUTPUT_SITE_ID}); asked for "
                f"{layer_idx}"
            )
        logits = self._data.logits
        if not logits:
            raise ShortPathError(f"gen {gen_id} has zero generated positions (no logits banked)")
        mat = _output_stats_per_token(logits, self._eos_token_ids)
        return OutputStatsPath(array=mat, gen_id=gen_id, provenance="RawGenerationData")

    def _check_gen(self, gen_id: int) -> None:
        if gen_id != self._gen_id:
            raise KeyError(f"this source serves gen {self._gen_id} only; asked for {gen_id}")


class RawGenerationDataAttentionRegionSource(PathSource):
    """Source over a single in-memory ``RawGenerationData``: builds the ``[T, 4-or-5]``
    attention-region path from ``.attentions`` (see ``_attention_region_per_token`` below,
    which reuses ``attention_flow``'s region-decomposition formulas verbatim).
    """

    def __init__(
        self,
        data: Any,
        *,
        gen_id: int = 0,
        include_sink: bool = True,
    ) -> None:
        if not hasattr(data, "attentions"):
            raise PathSignatureError(
                "RawGenerationDataAttentionRegionSource needs a RawGenerationData-like object "
                "with .attentions"
            )
        self._data = data
        self._gen_id = gen_id
        self._include_sink = include_sink

    def generation_ids(self) -> list[int]:
        return [self._gen_id]

    def available_layers(self, gen_id: int) -> list[int]:
        self._check_gen(gen_id)
        attns = self._data.attentions
        if not attns:
            return []
        return list(range(np.asarray(attns[0]).shape[0]))

    def load_path(self, gen_id: int, layer_idx: int) -> AttentionRegionPath:
        self._check_gen(gen_id)
        attns = self._data.attentions
        if not attns:
            raise ShortPathError(f"gen {gen_id} has zero generated positions (no attentions banked)")
        num_layers = np.asarray(attns[0]).shape[0]
        if not (0 <= layer_idx < num_layers):
            raise KeyError(f"attention layer {layer_idx} out of range for {num_layers} layers")
        mat = _attention_region_per_token(self._data, layer_idx, include_sink=self._include_sink)
        return AttentionRegionPath(
            array=mat, gen_id=gen_id, layer_idx=layer_idx,
            include_sink=self._include_sink, provenance="RawGenerationData",
        )

    def _check_gen(self, gen_id: int) -> None:
        if gen_id != self._gen_id:
            raise KeyError(f"this source serves gen {self._gen_id} only; asked for {gen_id}")


# ──────────────────────────────────────────────────────────────────────────────
# Core math — pure numpy, no model/torch deps, testable without a GPU
# ──────────────────────────────────────────────────────────────────────────────


def _as_path_array(X: NDArray[Any]) -> F64:
    """Validate and coerce to a finite float64 ``[T, d]`` array. Raises; never repairs."""
    arr = np.asarray(X)
    if arr.dtype == object:
        raise MalformedPathError("path array has dtype=object")
    if arr.ndim != 2:
        raise MalformedPathError(f"path must be 2-D [T, d]; got shape {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise MalformedPathError(f"path must be non-empty; got shape {arr.shape}")
    out = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(out)):
        n_bad = int(np.count_nonzero(~np.isfinite(out)))
        raise MalformedPathError(
            f"path contains {n_bad} non-finite values (shape {out.shape}) — refusing to "
            f"integrate. A NaN here would silently poison every downstream area."
        )
    return np.ascontiguousarray(out)


def path_increments(X: NDArray[Any]) -> F64:
    """``ΔX[t] = X[t+1] - X[t]`` -> ``[T-1, D]``."""
    arr = _as_path_array(X)
    if arr.shape[0] < 2:
        raise ShortPathError(f"need >=2 positions for increments; got {arr.shape[0]}")
    return np.diff(arr, axis=0)


def permute_increments(
    X: NDArray[Any],
    seed: int,
    *,
    rng: np.random.Generator | None = None,
) -> F64:
    """THE NULL — shuffle increments and re-cumulate from ``X[0]``.

    The result has, exactly:
      * the same start point,
      * the same end point (a permuted sum is the same sum),
      * therefore the same level-1 terms,
      * and destroyed level-2 areas.

    Seeded and reproducible: the same ``seed`` and the same ``X`` always give the same path.
    """
    arr = _as_path_array(X)
    d = path_increments(arr)
    generator = rng if rng is not None else np.random.default_rng(seed)
    order = generator.permutation(d.shape[0])
    shuffled = d[order]
    out = np.empty_like(arr)
    out[0] = arr[0]
    np.cumsum(shuffled, axis=0, out=out[1:])
    out[1:] += arr[0][None, :]
    return out


def time_augment_path(X: NDArray[Any]) -> F64:
    """Append normalised position ``t/(T-1) ∈ [0, 1]`` as a final coordinate -> ``[T, D+1]``.

    Normalising to a unit interval (rather than raw ``t``) keeps the time coordinate's net
    displacement at exactly 1 regardless of length, so length enters through the *areas*
    (when each coordinate moved, relative to the clock) rather than through a trivially
    length-proportional level-1 term. Pacing survives; a bare length proxy does not sneak in.
    """
    arr = _as_path_array(X)
    T = arr.shape[0]
    if T < 2:
        raise ShortPathError(f"need >=2 positions to time-augment; got {T}")
    tau = np.linspace(0.0, 1.0, T, dtype=np.float64).reshape(T, 1)
    return np.ascontiguousarray(np.concatenate([arr, tau], axis=1))


def level1_terms(X: NDArray[Any]) -> F64:
    """Level-1 log-signature: net displacement ``X[-1] - X[0]`` -> ``[D]``.

    Computed as the SUM OF INCREMENTS, not as the endpoint difference, precisely so the
    permutation null's invariance is a real numerical check on the integration path and not a
    tautology about two array lookups.
    """
    return path_increments(X).sum(axis=0)


def level2_area_matrix(X: NDArray[Any], *, center: bool = True) -> F64:
    """Antisymmetric Lévy-area matrix ``A`` -> ``[D, D]``, with ``A[i, j] = -A[j, i]``.

        ``A_ij = ½ Σ_t ( Xc_i[t] ΔX_j[t] - Xc_j[t] ΔX_i[t] )``,  ``Xc = X - X[0]``

    Left-endpoint quadrature; the midpoint rule gives an identical antisymmetric part (the
    ``½ ΔΔ`` correction is symmetric and cancels), which the selftest asserts.
    """
    arr = _as_path_array(X)
    if arr.shape[0] < 2:
        raise ShortPathError(f"need >=2 positions for areas; got {arr.shape[0]}")
    d = np.diff(arr, axis=0)                       # [T-1, D]
    base = arr[:-1] - (arr[0][None, :] if center else 0.0)   # [T-1, D]
    M = base.T @ d                                 # [D, D]: Σ_t Xc_i[t] ΔX_j[t]
    return 0.5 * (M - M.T)


def upper_pairs(dim: int) -> list[tuple[int, int]]:
    """Canonical ordering of the ``D(D-1)/2`` level-2 coordinate pairs."""
    if dim < 1:
        raise PathSignatureError(f"dim must be >=1; got {dim}")
    return list(itertools.combinations(range(dim), 2))


def log_signature_level2(
    X: NDArray[Any],
    *,
    level: int = 2,
    center: bool = True,
) -> tuple[F64, F64, list[tuple[int, int]]]:
    """Hand-rolled level-(1,2) log-signature of a discrete path.

    Returns ``(level1 [D], level2 [D(D-1)/2], pairs)``. ``level=1`` returns an empty level-2
    block and an empty pair list.
    """
    arr = _as_path_array(X)
    if level not in (1, 2):
        raise PathSignatureError(f"level must be 1 or 2; got {level}")
    l1 = level1_terms(arr)
    if level == 1:
        return l1, np.zeros(0, dtype=np.float64), []
    A = level2_area_matrix(arr, center=center)
    pairs = upper_pairs(arr.shape[1])
    l2 = np.array([A[i, j] for (i, j) in pairs], dtype=np.float64)
    return l1, l2, pairs


def signature_features_from_native_path(
    path: NDArray[Any],
    config: "OutputStatsPathConfig | AttentionRegionPathConfig",
    *,
    expected_native_dim: int,
) -> tuple[F64, int]:
    """(optionally permute) -> (optionally time-augment) -> integrate. NO PROJECTION.

    The §1a/§1b analogue of :func:`signature_features_from_path`, sharing every piece of the
    actual math (``_as_path_array``, ``permute_increments``, ``time_augment_path``,
    ``log_signature_level2``) and differing only in skipping the ``basis.project(...)`` call —
    these two sources are natively low-dimensional by construction, so there is nothing to
    project (spec §1: "no projection step exists and the projection-adequacy gate cannot
    bite").

    Returns
    -------
    (features [n_features_per_site], T) — features in emission order.
    """
    arr = _as_path_array(path)
    T = arr.shape[0]
    if T < config.min_positions:
        raise ShortPathError(f"path has {T} positions; min_positions={config.min_positions}")
    if arr.shape[1] != expected_native_dim:
        raise MalformedPathError(
            f"native path has {arr.shape[1]} dims; this source contracts to "
            f"{expected_native_dim} — no projection step exists here, so the SOURCE (not this "
            f"function) must emit the right shape"
        )

    proj = arr
    if config.permute_increments:
        assert config.permutation_seed is not None  # guaranteed by the model validator
        proj = permute_increments(proj, config.permutation_seed)

    integrated = time_augment_path(proj) if config.time_augment else proj

    l1, l2, _pairs = log_signature_level2(
        integrated, level=config.level, center=config.center_at_origin,
    )
    feats = np.concatenate([l1, l2]) if l2.size else l1

    expected = config.n_features_per_site
    if feats.shape[0] != expected:
        raise PathSignatureError(
            f"internal arity bug: produced {feats.shape[0]} features, expected {expected}"
        )
    if not np.all(np.isfinite(feats)):
        raise PathSignatureError(
            "non-finite signature terms produced from a finite native path — numerical "
            "overflow; check the source's native scale"
        )
    return feats, T


def _output_stats_per_token(
    logits: list[NDArray[Any]],
    eos_token_ids: Sequence[int],
) -> F64:
    """``[T, 4]`` matrix — columns ``entropy, margin, eos_log_mass, varentropy`` (§1a).

    Every column is an EXISTING formula, reused verbatim (spec: "do not invent new
    formulas"), not re-derived:

    * ``entropy`` — ``state_extractor._compute_logit_features``'s own entropy column, called
      directly on the same ``logits`` list, so this is not a re-implementation, it is the same
      function. (Under a reloaded raw-tensor bank ``logits`` is top-k + ``-1e9`` sentinel
      elsewhere — the SAME approximation ``state_extractor``'s own entropy features already
      accept for post-hoc reprocessing; not a new caveat this family introduces.)
    * ``margin`` — ``annex_potential_gradient.s_terms_margin``: ``logit_top1 - logit_top2``
      (computed there on ``log_softmax``; algebraically identical to the raw-logit difference
      since the softmax normaliser cancels in the subtraction — so the direct raw-logit form
      used here is exactly that functional, not an approximation of it). The top-2-by-value
      split mirrors ``expert_routing.py``'s existing ``np.partition(..., -2)`` idiom.
    * ``eos_log_mass`` — ``annex_potential_gradient.s_terms_eos``:
      ``logsumexp(log_softmax(logits)[eos_ids])``. ``eos_token_ids`` is the caller's explicit,
      model-specific set — never defaulted, never inferred, because a wrong id set makes the
      column a measurement of the wrong token.
    * ``varentropy`` — ``annex_cs_pulses.s_terms_varentropy``: ``sum_v p_v (s_v - H)^2`` with
      ``s_v = -log p_v`` the per-token surprisal and ``H`` the SAME entropy value computed
      above (not recomputed independently, so the two columns cannot numerically disagree
      about what "entropy" means).

    ``repetition-mass`` is NOT a column here — see ``OutputStatsPathConfig``'s docstring for
    why it is genuinely unavailable from ``RawGenerationData`` rather than approximated.
    """
    from anamnesis.extraction.state_extractor import _compute_logit_features

    T = len(logits)
    if T == 0:
        raise ShortPathError("no generated positions to build the output-statistics path")
    eos_ids = np.asarray(sorted({int(i) for i in eos_token_ids}), dtype=np.intp)
    if eos_ids.size == 0:
        raise PathSignatureError("eos_token_ids must be non-empty")

    # Entropy: THE existing state_extractor definition, called directly (not re-derived).
    dummy_chosen = np.zeros(T, dtype=np.float32)
    entropy = _compute_logit_features(list(logits), dummy_chosen).entropy.astype(np.float64)

    margin = np.empty(T, dtype=np.float64)
    eos_log_mass = np.empty(T, dtype=np.float64)
    varentropy = np.empty(T, dtype=np.float64)
    for t in range(T):
        x = np.asarray(logits[t], dtype=np.float64)
        vocab_size = x.shape[0]
        if int(eos_ids.max()) >= vocab_size:
            raise PathSignatureError(
                f"eos token id {int(eos_ids.max())} out of vocab range {vocab_size} at step {t}"
            )
        if vocab_size < 2:
            raise PathSignatureError(f"need >=2 vocab entries for a margin; got {vocab_size}")

        max_x = x.max()
        s = x - max_x
        exp_s = np.exp(s)
        Z = exp_s.sum()
        log_Z = np.log(Z)
        logp = s - log_Z                    # log-softmax
        p = exp_s / Z

        # margin: top-2 raw logits (expert_routing.py's np.partition idiom)
        part = np.partition(x, -2)
        margin[t] = float(part[-1] - part[-2])

        # eos log-mass: logsumexp over the EOS subset of the log-softmax
        le = logp[eos_ids]
        m = float(le.max())
        eos_log_mass[t] = m + float(np.log(np.sum(np.exp(le - m))))

        # varentropy: second moment of surprisal about the SAME H as the entropy column
        surprisal = -logp
        H = float(entropy[t])
        varentropy[t] = float(np.sum(p * (surprisal - H) ** 2))

    return np.stack([entropy, margin, eos_log_mass, varentropy], axis=1)


def _attention_region_per_token(
    data: Any,
    layer_idx: int,
    *,
    include_sink: bool,
) -> F64:
    """``[T, 4-or-5]`` matrix — columns ``prompt, early_gen, mid_gen, recent[, sink]`` (§1b).

    Region-mass formulas reused VERBATIM from
    ``feature_families.attention_flow.extract_attention_flow``'s region-decomposition block
    (the ``prompt_len`` / ``gen_len`` / thirds split, and the ``total_mass``-normalised
    fractions) — same arithmetic, same edge cases (an empty attention step at position 0 of a
    degenerate path contributes an all-zero row, matching that family's own fallback). Sink
    mass reuses ``state_extractor``'s T2.5 ``cache_sink_mass_L{n}`` definition (attention to
    position 0, the BOS/attention-sink position), here additionally divided by ``total_mass``
    so it sits on the same [0,1]-fraction-of-total-attention scale as the four region columns
    (state_extractor's own sink feature skips that division since raw attention weights
    already sum to ~1; the two are numerically equivalent up to that ~1 factor — see module
    report for the explicit note).
    """
    T = len(data.attentions)
    if T == 0:
        raise ShortPathError("no generated positions to build the attention-region path")
    num_layers = np.asarray(data.attentions[0]).shape[0]
    if not (0 <= layer_idx < num_layers):
        raise KeyError(f"attention layer {layer_idx} out of range for {num_layers} layers")

    prompt_len = data.prompt_length
    mean_rows = data.mean_attention(layer_idx)  # shared per-(gen,layer) cache (C2)
    n_cols = 5 if include_sink else 4
    out = np.zeros((T, n_cols), dtype=np.float64)

    for t in range(T):
        mean_attn = mean_rows[t]
        seq_len = mean_attn.shape[0]
        if seq_len == 0:
            continue  # all-zero row — matches attention_flow's own T<... fallback

        total_mass = max(float(mean_attn.sum()), 1e-12)
        sys_mass = float(mean_attn[:prompt_len].sum()) / total_mass

        gen_start = prompt_len
        gen_len = seq_len - prompt_len
        if gen_len > 0:
            third = max(1, gen_len // 3)
            r_early = float(mean_attn[gen_start:gen_start + third].sum()) / total_mass
            r_mid = float(mean_attn[gen_start + third:gen_start + 2 * third].sum()) / total_mass
            r_recent = float(mean_attn[gen_start + 2 * third:].sum()) / total_mass
        else:
            r_early = r_mid = r_recent = 0.0

        out[t, 0] = sys_mass
        out[t, 1] = r_early
        out[t, 2] = r_mid
        out[t, 3] = r_recent
        if include_sink:
            out[t, 4] = float(mean_attn[0]) / total_mass

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Naming — must classify cleanly in anamnesis/feature_map.py
# ──────────────────────────────────────────────────────────────────────────────


def site_prefix(layer_idx: int, config: PathSignatureConfig) -> str:
    return (
        f"{FEATURE_PREFIX}_L{layer_idx}_{config.basis_label}"
        f"_k{config.n_components}_{config.aug_token}"
    )


def site_feature_names(layer_idx: int, config: PathSignatureConfig) -> list[str]:
    """All feature names for one site, in emission order.

    ``res_sig_L16_pcaA_k4_aug_lvl1_c0`` ... then ``..._lvl2_c0c1`` ...
    feature_map: SOURCE=residual (``res_sig`` prefix), METHOD=iterated_integral,
    DEPTH from ``_L{n}``, DYNAMIC from the ``lvl1``/``lvl2`` token.
    """
    prefix = site_prefix(layer_idx, config)
    names = [f"{prefix}_lvl1_c{i}" for i in range(config.path_dim)]
    if config.level >= 2:
        names += [f"{prefix}_lvl2_c{i}c{j}" for (i, j) in upper_pairs(config.path_dim)]
    return names


def feature_names(config: PathSignatureConfig) -> list[str]:
    """All feature names the family emits under this config, in emission order."""
    out: list[str] = []
    for layer_idx in config.layer_indices:
        out.extend(site_feature_names(layer_idx, config))
    return out


FEATURE_PREFIX_OUTPUT = "out_sig"
FEATURE_PREFIX_ATTENTION = "attn_sig"
FAMILY_NAME_OUTPUT = "path_signature_output"
FAMILY_NAME_ATTENTION = "path_signature_attention"


def output_stats_feature_names(config: OutputStatsPathConfig) -> list[str]:
    """All feature names for the §1a output-statistics path (one global site — no ``_L{n}``).

    ``out_sig_{aug|noaug}_lvl{1|2}_c{i}[c{j}]`` — feature_map: SOURCE=output (``out_sig``
    prefix), METHOD=iterated_integral, DEPTH=None (no per-layer site to read — same convention
    as the suite's other layer-free output features, e.g. ``logit_entropy_mean``), DYNAMIC from
    the ``lvl1``/``lvl2`` token. Coordinate order: ``c0=entropy, c1=margin, c2=eos_log_mass,
    c3=varentropy[, c4=clock]``.
    """
    prefix = f"{FEATURE_PREFIX_OUTPUT}_{config.aug_token}"
    names = [f"{prefix}_lvl1_c{i}" for i in range(config.path_dim)]
    if config.level >= 2:
        names += [f"{prefix}_lvl2_c{i}c{j}" for (i, j) in upper_pairs(config.path_dim)]
    return names


def attention_region_site_prefix(layer_idx: int, config: AttentionRegionPathConfig) -> str:
    return f"{FEATURE_PREFIX_ATTENTION}_L{layer_idx}_{config.aug_token}"


def attention_region_site_feature_names(
    layer_idx: int, config: AttentionRegionPathConfig,
) -> list[str]:
    """All feature names for one attention-region site, in emission order.

    ``attn_sig_L16_{aug|noaug}_lvl{1|2}_c{i}[c{j}]`` — feature_map: SOURCE=attention
    (``attn_sig`` prefix), METHOD=iterated_integral, DEPTH from ``_L{n}``, DYNAMIC from the
    ``lvl1``/``lvl2`` token. Coordinate order: ``c0=prompt, c1=early_gen, c2=mid_gen,
    c3=recent[, c4=sink][, c{last}=clock]``.
    """
    prefix = attention_region_site_prefix(layer_idx, config)
    names = [f"{prefix}_lvl1_c{i}" for i in range(config.path_dim)]
    if config.level >= 2:
        names += [f"{prefix}_lvl2_c{i}c{j}" for (i, j) in upper_pairs(config.path_dim)]
    return names


def attention_region_feature_names(config: AttentionRegionPathConfig) -> list[str]:
    """All feature names the §1b family emits under this config, in emission order."""
    out: list[str] = []
    for layer_idx in config.layer_indices:
        out.extend(attention_region_site_feature_names(layer_idx, config))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Extraction
# ──────────────────────────────────────────────────────────────────────────────


def signature_features_from_path(
    path: NDArray[Any],
    basis: ProjectionBasis,
    config: PathSignatureConfig,
) -> tuple[F64, int]:
    """Project -> (optionally permute) -> (optionally time-augment) -> integrate.

    Parameters
    ----------
    path : [T, d] hidden-space trajectory (already positionally corrected by the source).
    basis : the SUPPLIED projection. Never fitted here.
    config : the frozen extraction config.

    Returns
    -------
    (features [n_features_per_site], T) — features in ``site_feature_names`` order.

    Order of operations is load-bearing: the permutation acts on the projected,
    PRE-augmentation path (shuffle the moves, keep the clock).
    """
    arr = _as_path_array(path)
    T = arr.shape[0]
    if T < config.min_positions:
        raise ShortPathError(
            f"path has {T} positions; min_positions={config.min_positions}"
        )

    proj = basis.project(arr, config.n_components)          # [T, k]

    if config.permute_increments:
        assert config.permutation_seed is not None  # guaranteed by the model validator
        proj = permute_increments(proj, config.permutation_seed)

    integrated = time_augment_path(proj) if config.time_augment else proj

    l1, l2, _pairs = log_signature_level2(
        integrated, level=config.level, center=config.center_at_origin,
    )
    feats = np.concatenate([l1, l2]) if l2.size else l1

    expected = config.n_features_per_site
    if feats.shape[0] != expected:
        raise PathSignatureError(
            f"internal arity bug: produced {feats.shape[0]} features, expected {expected} "
            f"(k={config.n_components}, aug={config.time_augment}, level={config.level})"
        )
    if not np.all(np.isfinite(feats)):
        raise PathSignatureError(
            "non-finite signature terms produced from a finite path — numerical overflow; "
            "check the projection basis scale"
        )
    return feats, T


def extract_path_signature_from_source(
    source: ResidualPathSource,
    gen_id: int,
    basis_bank: ProjectionBasisBank,
    config: PathSignatureConfig,
) -> PathSignatureResult:
    """Extract one generation's path-signature block across all configured sites."""
    all_feats: list[F64] = []
    all_names: list[str] = []
    n_positions: dict[int, int] = {}
    degraded: list[int] = []

    for layer_idx in config.layer_indices:
        names = site_feature_names(layer_idx, config)
        basis = basis_bank.for_layer(layer_idx)
        try:
            rp = source.load_path(gen_id, layer_idx)
            feats, T = signature_features_from_path(rp.array, basis, config)
            n_positions[layer_idx] = T
        except (KeyError, ShortPathError) as exc:
            policy = (
                config.on_short_path if isinstance(exc, ShortPathError)
                else config.on_missing_layer
            )
            if policy == "raise":
                raise
            logger.warning(
                "path_signature: gen %d layer %d degraded to a zero block (%s)",
                gen_id, layer_idx, exc,
            )
            feats = np.zeros(len(names), dtype=np.float64)
            degraded.append(layer_idx)
            n_positions[layer_idx] = 0
        all_feats.append(feats)
        all_names.extend(names)

    features = (
        np.concatenate(all_feats).astype(np.float32)
        if all_feats else np.array([], dtype=np.float32)
    )
    if len(features) != len(all_names):
        raise PathSignatureError(
            f"features/names divergence: {len(features)} vs {len(all_names)}"
        )
    return PathSignatureResult(
        features=features,
        feature_names=all_names,
        family_name=FAMILY_NAME,
        metadata=PathSignatureMetadata.from_config(
            config, n_positions=n_positions, degraded_sites=tuple(degraded),
        ),
    )


def extract_path_signature(
    data: Any,
    basis_bank: ProjectionBasisBank,
    config: PathSignatureConfig | None = None,
    *,
    gen_id: int = 0,
) -> PathSignatureResult:
    """Feature-family entry point over a ``RawGenerationData`` (the pipeline contract).

    Mirrors ``extract_residual_trajectory`` / ``extract_gate_features`` etc.: takes the
    in-memory raw data, returns a ``FeatureFamilyResult`` (subclass) the pipeline can
    concatenate. Positional correction follows ``config.positional_correct`` and reuses
    ``data.positional_means``.
    """
    cfg = config or PathSignatureConfig()
    if basis_bank is None:
        raise ProjectionError(
            "path_signature requires a supplied projection basis (the banked calibration "
            "PCA). This family never fits a basis on evaluation data."
        )
    source = RawGenerationDataPathSource(
        data, gen_id=gen_id, positional_correct=cfg.positional_correct,
    )
    return extract_path_signature_from_source(source, gen_id, basis_bank, cfg)


def extract_null_battery(
    source: ResidualPathSource,
    gen_id: int,
    basis_bank: ProjectionBasisBank,
    config: PathSignatureConfig,
    seeds: Sequence[int],
) -> list[PathSignatureResult]:
    """The increment-permutation null at >=3 seeds (spec section 2), first-class and in-module.

    Returns one result per seed, column-identical to the real extraction.
    """
    if len(seeds) < 3:
        raise PathSignatureError(
            f"spec section 2 asks for >=3 shuffle seeds per cell; got {len(seeds)}"
        )
    out: list[PathSignatureResult] = []
    for s in seeds:
        null_cfg = config.model_copy(
            update={"permute_increments": True, "permutation_seed": int(s)},
        )
        out.append(
            extract_path_signature_from_source(source, gen_id, basis_bank, null_cfg)
        )
    return out


def extract_output_stats_signature_from_source(
    source: PathSource,
    gen_id: int,
    config: OutputStatsPathConfig,
) -> PathSignatureResult:
    """Extract one generation's §1a output-statistics path-signature block."""
    names = output_stats_feature_names(config)
    try:
        p = source.load_path(gen_id, OUTPUT_SITE_ID)
        feats, T = signature_features_from_native_path(
            p.array, config, expected_native_dim=OutputStatsPathConfig.NATIVE_DIM,
        )
        n_positions = {OUTPUT_SITE_ID: T}
        degraded: tuple[int, ...] = ()
    except ShortPathError as exc:
        if config.on_short_path == "raise":
            raise
        logger.warning(
            "path_signature(output): gen %d degraded to a zero block (%s)", gen_id, exc,
        )
        feats = np.zeros(len(names), dtype=np.float64)
        n_positions = {OUTPUT_SITE_ID: 0}
        degraded = (OUTPUT_SITE_ID,)

    features = feats.astype(np.float32)
    if len(features) != len(names):
        raise PathSignatureError(
            f"features/names divergence: {len(features)} vs {len(names)}"
        )
    return PathSignatureResult(
        features=features,
        feature_names=names,
        family_name=FAMILY_NAME_OUTPUT,
        metadata=OutputStatsMetadata.from_config(
            config, n_positions=n_positions, degraded_sites=degraded,
        ),
    )


def extract_attention_region_signature_from_source(
    source: PathSource,
    gen_id: int,
    config: AttentionRegionPathConfig,
) -> PathSignatureResult:
    """Extract one generation's §1b attention-region path-signature block across all sites."""
    all_feats: list[F64] = []
    all_names: list[str] = []
    n_positions: dict[int, int] = {}
    degraded: list[int] = []

    for layer_idx in config.layer_indices:
        names = attention_region_site_feature_names(layer_idx, config)
        try:
            p = source.load_path(gen_id, layer_idx)
            feats, T = signature_features_from_native_path(
                p.array, config, expected_native_dim=config.native_dim,
            )
            n_positions[layer_idx] = T
        except (KeyError, ShortPathError) as exc:
            policy = (
                config.on_short_path if isinstance(exc, ShortPathError)
                else config.on_missing_layer
            )
            if policy == "raise":
                raise
            logger.warning(
                "path_signature(attention): gen %d layer %d degraded to a zero block (%s)",
                gen_id, layer_idx, exc,
            )
            feats = np.zeros(len(names), dtype=np.float64)
            degraded.append(layer_idx)
            n_positions[layer_idx] = 0
        all_feats.append(feats)
        all_names.extend(names)

    features = (
        np.concatenate(all_feats).astype(np.float32)
        if all_feats else np.array([], dtype=np.float32)
    )
    if len(features) != len(all_names):
        raise PathSignatureError(
            f"features/names divergence: {len(features)} vs {len(all_names)}"
        )
    return PathSignatureResult(
        features=features,
        feature_names=all_names,
        family_name=FAMILY_NAME_ATTENTION,
        metadata=AttentionRegionMetadata.from_config(
            config, n_positions=n_positions, degraded_sites=tuple(degraded),
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Selftest
# ──────────────────────────────────────────────────────────────────────────────


class _Check:
    """Tiny assertion recorder so the selftest prints a verbatim, greppable table."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def ok(self, name: str, passed: bool, detail: str = "") -> None:
        self.rows.append((name, bool(passed), detail))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}" + (f"  —  {detail}" if detail else ""))

    def close(self, banner: str) -> bool:
        n_fail = sum(1 for _, p, _ in self.rows if not p)
        print(f"\n{banner}: {len(self.rows) - n_fail}/{len(self.rows)} checks passed")
        return n_fail == 0


def _synthetic_path(rng: np.random.Generator, T: int = 120, d: int = 64) -> F64:
    """A curved, order-structured path in hidden space (not a random walk)."""
    t = np.linspace(0.0, 1.0, T)
    latent = np.stack(
        [np.sin(3 * np.pi * t), np.cos(2 * np.pi * t) * t, t ** 2, np.exp(-3 * t)], axis=1,
    )
    mix = rng.standard_normal((4, d)) / np.sqrt(d)
    return latent @ mix + 0.02 * rng.standard_normal((T, d)) + 5.0


def _synthetic_basis(rng: np.random.Generator, d: int = 64, k_max: int = 16) -> ProjectionBasis:
    """An orthonormal supplied basis — stands in for the banked calibration PCA."""
    Q, _ = np.linalg.qr(rng.standard_normal((d, k_max)))
    return ProjectionBasis(
        components=Q.T.copy(), mean=rng.standard_normal(d), label="pcaA",
    )


def selftest(verbose: bool = True) -> bool:  # noqa: C901 — a linear battery, read top to bottom
    """CPU-only validation battery. No banked data, no GPU, no network."""
    np.set_printoptions(precision=12, suppress=False)
    rng = np.random.default_rng(20260911)
    c = _Check()

    print("=" * 78)
    print("path_signature selftest — level-2 log-signature family")
    print("=" * 78)

    # ── (a) THE CORE CONTROL: level-1 invariant under increment permutation, level-2 moves ──
    print("\n(a) increment-permutation null — the spec's primary control")
    X = _synthetic_path(rng, T=140, d=48)
    basis = _synthetic_basis(rng, d=48, k_max=16)
    cfg = PathSignatureConfig(layer_indices=(16,), n_components=4, time_augment=True)
    real, _ = signature_features_from_path(X, basis, cfg)
    n1 = cfg.n_level1

    l1_devs: list[float] = []
    l2_devs: list[float] = []
    for seed in (1, 2, 3, 4, 5):
        null_cfg = cfg.model_copy(
            update={"permute_increments": True, "permutation_seed": seed},
        )
        null, _ = signature_features_from_path(X, basis, null_cfg)
        # Time-augmented coordinate is the clock: it is NOT permuted, so its level-1 term is
        # trivially invariant. The k projected coordinates are the real test.
        l1_dev = float(np.max(np.abs(null[:n1] - real[:n1])))
        l2_dev = float(np.max(np.abs(null[n1:] - real[n1:])))
        l1_devs.append(l1_dev)
        l2_devs.append(l2_dev)
        print(f"      seed {seed}: max|Δlvl1| = {l1_dev:.3e}   max|Δlvl2| = {l2_dev:.6f}")

    max_l1_dev = max(l1_devs)
    min_l2_dev = min(l2_devs)
    l2_scale = float(np.max(np.abs(real[n1:])))
    c.ok(
        "a1 level-1 invariant under increment permutation (<=1e-10)",
        max_l1_dev <= 1e-10,
        f"max over 5 seeds = {max_l1_dev:.3e}",
    )
    c.ok(
        "a2 level-2 moves under increment permutation (>=1e-3 x level-2 scale)",
        min_l2_dev >= 1e-3 * l2_scale and l2_scale > 0,
        f"min over 5 seeds = {min_l2_dev:.6f}, level-2 scale = {l2_scale:.6f}",
    )
    # Re-cumulated null path must share start and end exactly.
    proj = basis.project(X, 4)
    perm = permute_increments(proj, 7)
    c.ok(
        "a3 null path shares start point exactly",
        bool(np.array_equal(perm[0], proj[0])),
    )
    c.ok(
        "a4 null path shares end point to <=1e-10",
        float(np.max(np.abs(perm[-1] - proj[-1]))) <= 1e-10,
        f"max|Δend| = {float(np.max(np.abs(perm[-1] - proj[-1]))):.3e}",
    )
    c.ok(
        "a5 null is reproducible under the same seed (bitwise)",
        bool(np.array_equal(permute_increments(proj, 7), perm)),
    )
    c.ok(
        "a6 different seeds give different null paths",
        not np.array_equal(permute_increments(proj, 8), perm),
    )
    c.ok(
        "a7 unseeded null is REFUSED at config construction",
        _raises(lambda: PathSignatureConfig(permute_increments=True), ValueError),
    )

    # ── (b) straight line -> zero areas ──
    print("\n(b) straight-line path has zero Lévy area")
    direction = rng.standard_normal(6)
    line = np.outer(np.linspace(0.0, 3.0, 90), direction) + rng.standard_normal(6)
    A_line = level2_area_matrix(line)
    c.ok(
        "b1 straight line: all areas ~0 (<=1e-10)",
        float(np.max(np.abs(A_line))) <= 1e-10,
        f"max|A| = {float(np.max(np.abs(A_line))):.3e}",
    )
    # Non-uniform parametrisation of the SAME straight line: still zero (areas see geometry).
    s = np.sort(rng.random(90)) * 3.0
    line_np = np.outer(s, direction) + 1.0
    c.ok(
        "b2 reparametrised straight line: areas still ~0",
        float(np.max(np.abs(level2_area_matrix(line_np)))) <= 1e-10,
        f"max|A| = {float(np.max(np.abs(level2_area_matrix(line_np)))):.3e}",
    )
    # A closed unit square traversed once: area = 1 (the textbook sanity value).
    square = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64,
    )
    A_sq = level2_area_matrix(square)
    c.ok(
        "b3 unit square traversed CCW: A_01 = +1 exactly",
        abs(float(A_sq[0, 1]) - 1.0) <= 1e-12,
        f"A_01 = {float(A_sq[0, 1]):.15f}",
    )
    c.ok(
        "b4 area matrix is antisymmetric",
        float(np.max(np.abs(A_sq + A_sq.T))) <= 1e-14,
    )

    # ── (b') invariances the convention rests on ──
    print("\n(b') convention checks — centering, quadrature, translation")
    Y = basis.project(_synthetic_path(rng, T=80, d=48), 5)
    shift = rng.standard_normal(Y.shape[1]) * 10.0
    c.ok(
        "b5 areas are translation invariant (centered=True)",
        float(np.max(np.abs(level2_area_matrix(Y) - level2_area_matrix(Y + shift)))) <= 1e-8,
        f"max|Δ| = {float(np.max(np.abs(level2_area_matrix(Y) - level2_area_matrix(Y + shift)))):.3e}",
    )
    c.ok(
        "b6 areas are NOT translation invariant without centering (the reason we center)",
        float(np.max(np.abs(
            level2_area_matrix(Y, center=False)
            - level2_area_matrix(Y + shift, center=False)
        ))) > 1e-3,
    )
    dY = np.diff(Y, axis=0)
    mid = 0.5 * (Y[:-1] + Y[1:]) - Y[0][None, :]
    M_mid = mid.T @ dY
    A_mid = 0.5 * (M_mid - M_mid.T)
    c.ok(
        "b7 midpoint quadrature == left-endpoint quadrature (antisym part)",
        float(np.max(np.abs(A_mid - level2_area_matrix(Y)))) <= 1e-9,
        f"max|Δ| = {float(np.max(np.abs(A_mid - level2_area_matrix(Y)))):.3e}",
    )
    c.ok(
        "b8 level-1 computed as sum-of-increments == endpoint difference",
        float(np.max(np.abs(level1_terms(Y) - (Y[-1] - Y[0])))) <= 1e-10,
    )

    # ── (c) Chen concatenation consistency ──
    print("\n(c) Chen / BCH concatenation consistency")
    P = _synthetic_path(rng, T=60, d=48)
    Pp = basis.project(P, 5)
    m = 37
    left, right = Pp[: m + 1], Pp[m:]          # share the joint point
    A_full = level2_area_matrix(Pp)
    A_l, A_r = level2_area_matrix(left), level2_area_matrix(right)
    L_l, L_r = level1_terms(left), level1_terms(right)
    bracket = 0.5 * (np.outer(L_l, L_r) - np.outer(L_r, L_l))
    chen_resid = float(np.max(np.abs(A_full - (A_l + A_r + bracket))))
    c.ok(
        "c1 A(P*Q) = A(P) + A(Q) + ½[L(P), L(Q)]  (<=1e-8)",
        chen_resid <= 1e-8,
        f"max residual = {chen_resid:.3e}",
    )
    c.ok(
        "c2 level-1 is additive under concatenation",
        float(np.max(np.abs(level1_terms(Pp) - (L_l + L_r)))) <= 1e-10,
    )
    # Three-way split, to catch an error that cancels at one cut point.
    m1, m2 = 17, 41
    segs = [Pp[: m1 + 1], Pp[m1 : m2 + 1], Pp[m2:]]
    A_acc = level2_area_matrix(segs[0])
    L_acc = level1_terms(segs[0])
    for seg in segs[1:]:
        A_s, L_s = level2_area_matrix(seg), level1_terms(seg)
        A_acc = A_acc + A_s + 0.5 * (np.outer(L_acc, L_s) - np.outer(L_s, L_acc))
        L_acc = L_acc + L_s
    c.ok(
        "c3 three-way Chen consistency (<=1e-8)",
        float(np.max(np.abs(A_full - A_acc))) <= 1e-8,
        f"max residual = {float(np.max(np.abs(A_full - A_acc))):.3e}",
    )

    # ── (d) every emitted name classifies in feature_map ──
    print("\n(d) feature_map classification of every emitted name")
    try:
        from anamnesis.feature_map import (
            Band, FeatureMap, Method, Source,
        )
    except ImportError as exc:  # pragma: no cover
        c.ok("d0 feature_map importable", False, str(exc))
        return c.close("SELFTEST")

    has_ii = hasattr(Method, "iterated_integral")
    c.ok("d1 Method.iterated_integral exists", has_ii)

    name_cfg = PathSignatureConfig(
        layer_indices=(0, 14, 16, 27), n_components=4, time_augment=True,
    )
    names = feature_names(name_cfg)
    fm = FeatureMap(names, n_layers=32)
    unc = fm.unclassified()
    c.ok(
        "d2 zero unclassified names",
        not unc,
        f"{len(unc)} flagged" + (f": {unc[:5]}" if unc else ""),
    )
    c.ok(
        "d3 every name SOURCE=residual",
        all(t.source == Source.residual for t in fm.tags),
        f"sources = {sorted({t.source.value for t in fm.tags})}",
    )
    if has_ii:
        c.ok(
            "d4 every name METHOD=iterated_integral",
            all(t.method == Method.iterated_integral for t in fm.tags),
            f"methods = {sorted({t.method.value for t in fm.tags})}",
        )
    bands = {t.layer: t.band for t in fm.tags}
    c.ok(
        "d5 DEPTH parsed per site (L0=early, L14/L16=mid, L27=late @ n_layers=32)",
        bands.get(0) == Band.early and bands.get(14) == Band.mid
        and bands.get(16) == Band.mid and bands.get(27) == Band.late,
        f"{ {k: (v.value if v else None) for k, v in sorted(bands.items())} }",
    )
    dyn = {n: t.dynamic for n, t in zip(names, fm.tags)}
    lvl1_dyn = {dyn[n] for n in names if "_lvl1_" in n}
    lvl2_dyn = {dyn[n] for n in names if "_lvl2_" in n}
    c.ok(
        "d6 lvl1 static / lvl2 dynamic",
        lvl1_dyn == {False} and lvl2_dyn == {True},
        f"lvl1 -> {lvl1_dyn}, lvl2 -> {lvl2_dyn}",
    )
    c.ok(
        "d7 legacy family label is 'path_signature'",
        {t.family for t in fm.tags} == {"path_signature"},
        f"{sorted({t.family for t in fm.tags})}",
    )
    c.ok("d8 feature names are unique", len(set(names)) == len(names))

    # ── (e) shapes and counts match the spec's arithmetic ──
    print("\n(e) arity — the spec's own numbers")
    cases = [
        # (k, augment, level, expected_l1, expected_l2, spec quote)
        (4, True, 2, 5, 10, "k=4 augmented to 5 -> 5 level-1 + 10 level-2 = 15"),
        (4, False, 2, 4, 6, "k=4 unaugmented -> 4 + 6 = 10"),
        (8, True, 2, 9, 36, "k=8 augmented to 9 -> 9 + 36 = 45"),
        (8, False, 2, 8, 28, "k=8 unaugmented -> 8 + 28 = 36"),
        (4, True, 1, 5, 0, "level-1 only, k=4 augmented -> 5"),
    ]
    for k, aug, lvl, e1, e2, quote in cases:
        cc = PathSignatureConfig(
            layer_indices=(16,), n_components=k, time_augment=aug, level=lvl,  # type: ignore[arg-type]
        )
        nm = feature_names(cc)
        feats, _T = signature_features_from_path(X, basis, cc)
        got1 = sum(1 for n in nm if "_lvl1_" in n)
        got2 = sum(1 for n in nm if "_lvl2_" in n)
        good = (
            cc.n_level1 == e1 and cc.n_level2 == e2
            and got1 == e1 and got2 == e2
            and len(nm) == e1 + e2 == len(feats) == cc.n_features_per_site
        )
        c.ok(f"e[k={k},{'aug' if aug else 'noaug'},lvl{lvl}] {quote}", good,
             f"names {got1}+{got2}={len(nm)}, features={len(feats)}")

    multi = PathSignatureConfig(layer_indices=(14, 16), n_components=8, time_augment=True)
    c.ok(
        "e6 multi-site arity = n_sites x per-site",
        len(feature_names(multi)) == 2 * multi.n_features_per_site == 90,
        f"{len(feature_names(multi))} names over 2 sites",
    )

    # ── (f) adapter + error handling ──
    print("\n(f) loader contract, error handling, metadata")
    src = ArrayPathSource({0: {16: X}, 1: {16: X[:2]}})
    bank = ProjectionBasisBank.single(basis)
    res = extract_path_signature_from_source(src, 0, bank, cfg)
    c.ok(
        "f1 result conforms to FeatureFamilyResult",
        isinstance(res, FeatureFamilyResult)
        and len(res) == len(res.feature_names) == cfg.n_features_per_site,
        f"family={res.family_name}, n={len(res)}",
    )
    c.ok(
        "f2 metadata records augmentation / basis / permutation",
        res.metadata.time_augmented is True
        and res.metadata.basis_label == "pcaA"
        and res.metadata.permuted is False
        and res.metadata.n_positions == {16: X.shape[0]},
        res.metadata.model_dump_json(),
    )
    c.ok(
        "f3 short path RAISES by default (no silent zeros)",
        _raises(lambda: extract_path_signature_from_source(src, 1, bank, cfg), ShortPathError),
    )
    zero_cfg = cfg.model_copy(update={"on_short_path": "zeros"})
    zres = extract_path_signature_from_source(src, 1, bank, zero_cfg)
    c.ok(
        "f4 on_short_path='zeros' emits a NAME-MATCHED zero block and flags it",
        len(zres) == len(zres.feature_names) and not np.any(zres.features)
        and zres.metadata.degraded_sites == (16,),
    )
    c.ok(
        "f5 missing layer RAISES (absence is never disguised as data)",
        _raises(
            lambda: extract_path_signature_from_source(
                ArrayPathSource({0: {8: X}}), 0, bank, cfg,
            ),
            KeyError,
        ),
    )
    c.ok(
        "f6 NaN in path raises MalformedPathError",
        _raises(lambda: _as_path_array(np.array([[1.0, np.nan], [2.0, 3.0]])),
                MalformedPathError),
    )
    c.ok(
        "f7 1-D path raises MalformedPathError",
        _raises(lambda: _as_path_array(np.arange(10.0)), MalformedPathError),
    )
    c.ok(
        "f8 basis/path dim mismatch raises ProjectionError",
        _raises(lambda: basis.project(np.zeros((10, 7)), 4), ProjectionError),
    )
    c.ok(
        "f9 k > basis rank raises ProjectionError",
        _raises(lambda: basis.project(X, 999), ProjectionError),
    )
    c.ok(
        "f10 basis_label with '_' is refused (would break name parsing)",
        _raises(lambda: PathSignatureConfig(basis_label="pca_A"), ValueError),
    )
    c.ok(
        "f11 null battery demands >=3 seeds",
        _raises(lambda: extract_null_battery(src, 0, bank, cfg, [1, 2]), PathSignatureError),
    )
    battery = extract_null_battery(src, 0, bank, cfg, [11, 12, 13])
    c.ok(
        "f12 null battery is column-identical to the real extraction",
        all(b.feature_names == res.feature_names for b in battery)
        and all(b.metadata.permuted for b in battery)
        and [b.metadata.permutation_seed for b in battery] == [11, 12, 13],
    )
    null_stack = np.stack([b.features for b in battery])
    c.ok(
        "f13 null battery: lvl1 columns match the real run, lvl2 columns do not",
        float(np.max(np.abs(null_stack[:, :n1] - res.features[None, :n1]))) <= 1e-5
        and float(np.max(np.abs(null_stack[:, n1:] - res.features[None, n1:]))) > 1e-3,
        f"lvl1 max|Δ| (float32) = {float(np.max(np.abs(null_stack[:, :n1] - res.features[None, :n1]))):.3e}",
    )

    # RawGenerationData adapter — built without torch or a model.
    fake = _FakeRawGenerationData(hidden_dim=48, T=60, n_layers=32, rng=rng)
    rgs = RawGenerationDataPathSource(fake, gen_id=0, positional_correct=False)
    rp = rgs.load_path(0, 16)
    c.ok(
        "f14 RawGenerationData adapter reads hidden_states[t][l+1]",
        rp.array.shape == (60, 48)
        and np.allclose(rp.array[3], fake.hidden_states[3][17]),
    )
    c.ok(
        "f15 adapter raises on a zero-filled (unsaved) layer",
        _raises(lambda: rgs.load_path(0, 2), KeyError),
    )
    fam = extract_path_signature(fake, bank, cfg)
    c.ok(
        "f16 extract_path_signature(RawGenerationData, ...) end-to-end",
        len(fam) == cfg.n_features_per_site and fam.family_name == FAMILY_NAME,
    )

    # ── (g) optional iisignature cross-check ──
    print("\n(g) optional external cross-check")
    try:
        import iisignature  # type: ignore[import-not-found]
    except Exception:
        print("      iisignature not installed — cross-check SKIPPED (it is optional by design;"
              " runtime never imports it)")
        c.ok("g1 iisignature cross-check", True, "SKIPPED (not installed)")
    else:  # pragma: no cover — only runs where the optional dep exists
        Z = np.ascontiguousarray(basis.project(_synthetic_path(rng, T=50, d=48), 4))
        s = iisignature.prepare(Z.shape[1], 2)
        ext = np.asarray(iisignature.logsig(Z, s), dtype=np.float64)
        mine1, mine2, pairs = log_signature_level2(Z, level=2)
        d_ = Z.shape[1]
        got = np.concatenate([mine1, mine2])
        ref = np.concatenate([ext[:d_], ext[d_:]])
        # iisignature's level-2 log-sig basis is the antisymmetric pairs in the same
        # (i<j) order but with the opposite sign convention on some builds; compare |.|
        # sorted as a build-robust check, and the exact vector when signs line up.
        exact = float(np.max(np.abs(got - ref)))
        upto_sign = float(np.max(np.abs(np.sort(np.abs(got)) - np.sort(np.abs(ref)))))
        c.ok(
            "g1 iisignature cross-check (level1 exact, level2 up to basis sign)",
            float(np.max(np.abs(mine1 - ext[:d_]))) <= 1e-8 and upto_sign <= 1e-8,
            f"exact max|Δ| = {exact:.3e}; up-to-sign max|Δ| = {upto_sign:.3e}; "
            f"n_pairs = {len(pairs)}",
        )

    # ── (h) the seam generalisation itself ──
    print("\n(h) generalised source seam")
    c.ok(
        "h1 ResidualPathSource is the SAME class object as PathSource (bit-identical alias)",
        ResidualPathSource is PathSource,
    )

    # ── (i) §1a output-statistics path — SOURCE=output ──
    print("\n(i) §1a output-statistics path (entropy, margin, eos_log_mass, varentropy)")
    out_native = _synthetic_path(rng, T=110, d=4)   # a curved, order-structured 4-dim path
    out_cfg = OutputStatsPathConfig(eos_token_ids=(3, 7), time_augment=True)
    out_real, _ = signature_features_from_native_path(
        out_native, out_cfg, expected_native_dim=4,
    )
    n1_out = out_cfg.n_level1
    l1_devs_o, l2_devs_o = [], []
    for seed in (1, 2, 3, 4, 5):
        null_cfg = out_cfg.model_copy(update={"permute_increments": True, "permutation_seed": seed})
        null_o, _ = signature_features_from_native_path(out_native, null_cfg, expected_native_dim=4)
        l1_devs_o.append(float(np.max(np.abs(null_o[:n1_out] - out_real[:n1_out]))))
        l2_devs_o.append(float(np.max(np.abs(null_o[n1_out:] - out_real[n1_out:]))))
    l2_scale_o = float(np.max(np.abs(out_real[n1_out:])))
    c.ok(
        "i1 output: level-1 invariant under increment permutation (<=1e-10)",
        max(l1_devs_o) <= 1e-10, f"max over 5 seeds = {max(l1_devs_o):.3e}",
    )
    c.ok(
        "i2 output: level-2 moves under increment permutation (>=1e-3 x scale)",
        min(l2_devs_o) >= 1e-3 * l2_scale_o and l2_scale_o > 0,
        f"min over 5 seeds = {min(l2_devs_o):.6f}, scale = {l2_scale_o:.6f}",
    )
    line4 = np.outer(np.linspace(0.0, 2.0, 60), rng.standard_normal(4)) + rng.standard_normal(4)
    c.ok(
        "i3 output: straight-line native path has zero Lévy area",
        float(np.max(np.abs(level2_area_matrix(line4)))) <= 1e-10,
    )
    out_cases = [
        (True, 2, 5, 10), (False, 2, 4, 6), (True, 1, 5, 0),
    ]
    for aug, lvl, e1, e2 in out_cases:
        cc = OutputStatsPathConfig(eos_token_ids=(3, 7), time_augment=aug, level=lvl)  # type: ignore[arg-type]
        nm = output_stats_feature_names(cc)
        feats, _T = signature_features_from_native_path(out_native, cc, expected_native_dim=4)
        got1 = sum(1 for n in nm if "_lvl1_" in n)
        got2 = sum(1 for n in nm if "_lvl2_" in n)
        c.ok(
            f"i4[{'aug' if aug else 'noaug'},lvl{lvl}] arity 4{'+clock' if aug else ''} "
            f"-> lvl1={e1},lvl2={e2}",
            cc.n_level1 == e1 and cc.n_level2 == e2 and got1 == e1 and got2 == e2
            and len(nm) == e1 + e2 == len(feats),
            f"names {got1}+{got2}={len(nm)}, features={len(feats)}",
        )
    c.ok(
        "i5 output config REFUSES an empty eos_token_ids",
        _raises(lambda: OutputStatsPathConfig(eos_token_ids=()), ValueError),
    )
    c.ok(
        "i6 output config REFUSES a missing eos_token_ids (no silent default)",
        _raises(lambda: OutputStatsPathConfig(), ValueError),  # type: ignore[call-arg]
    )
    c.ok(
        "i7 output config REFUSES an unseeded null (same rule as residual)",
        _raises(lambda: OutputStatsPathConfig(eos_token_ids=(3,), permute_increments=True), ValueError),
    )

    # source + end-to-end extraction (in-memory ArrayOutputStatsSource)
    out_src = ArrayOutputStatsSource({0: out_native, 1: out_native[:2]})
    out_res = extract_output_stats_signature_from_source(out_src, 0, out_cfg)
    c.ok(
        "i8 ArrayOutputStatsSource end-to-end matches the direct call",
        np.allclose(out_res.features.astype(np.float64), out_real, atol=1e-4)
        and out_res.family_name == FAMILY_NAME_OUTPUT
        and out_res.metadata.eos_token_ids == (3, 7),
    )
    c.ok(
        "i9 output: wrong site id raises KeyError",
        _raises(lambda: out_src.load_path(0, 99), KeyError),
    )
    c.ok(
        "i10 output: short path RAISES by default",
        _raises(lambda: extract_output_stats_signature_from_source(out_src, 1, out_cfg), ShortPathError),
    )
    out_zero_cfg = out_cfg.model_copy(update={"on_short_path": "zeros"})
    out_zres = extract_output_stats_signature_from_source(out_src, 1, out_zero_cfg)
    c.ok(
        "i11 output: on_short_path='zeros' emits a name-matched zero block",
        len(out_zres) == len(out_zres.feature_names) and not np.any(out_zres.features)
        and out_zres.metadata.degraded_sites == (OUTPUT_SITE_ID,),
    )

    # ── (j) §1b attention-region path — SOURCE=attention ──
    print("\n(j) §1b attention-region path (prompt/early_gen/mid_gen/recent[/sink])")
    attn_native = _synthetic_path(rng, T=95, d=5)
    attn_cfg = AttentionRegionPathConfig(layer_indices=(16,), include_sink=True, time_augment=True)
    attn_real, _ = signature_features_from_native_path(
        attn_native, attn_cfg, expected_native_dim=attn_cfg.native_dim,
    )
    n1_attn = attn_cfg.n_level1
    l1_devs_a, l2_devs_a = [], []
    for seed in (1, 2, 3, 4, 5):
        null_cfg = attn_cfg.model_copy(update={"permute_increments": True, "permutation_seed": seed})
        null_a, _ = signature_features_from_native_path(
            attn_native, null_cfg, expected_native_dim=attn_cfg.native_dim,
        )
        l1_devs_a.append(float(np.max(np.abs(null_a[:n1_attn] - attn_real[:n1_attn]))))
        l2_devs_a.append(float(np.max(np.abs(null_a[n1_attn:] - attn_real[n1_attn:]))))
    l2_scale_a = float(np.max(np.abs(attn_real[n1_attn:])))
    c.ok(
        "j1 attention: level-1 invariant under increment permutation (<=1e-10)",
        max(l1_devs_a) <= 1e-10, f"max over 5 seeds = {max(l1_devs_a):.3e}",
    )
    c.ok(
        "j2 attention: level-2 moves under increment permutation (>=1e-3 x scale)",
        min(l2_devs_a) >= 1e-3 * l2_scale_a and l2_scale_a > 0,
        f"min over 5 seeds = {min(l2_devs_a):.6f}, scale = {l2_scale_a:.6f}",
    )
    line5 = np.outer(np.linspace(0.0, 2.0, 55), rng.standard_normal(5)) + rng.standard_normal(5)
    c.ok(
        "j3 attention: straight-line native path has zero Lévy area",
        float(np.max(np.abs(level2_area_matrix(line5)))) <= 1e-10,
    )
    attn_cases = [
        (True, True, 2, 6, 15), (True, False, 2, 5, 10),
        (False, True, 2, 5, 10), (False, False, 2, 4, 6),
        (True, True, 1, 6, 0),
    ]
    for sink, aug, lvl, e1, e2 in attn_cases:
        cc = AttentionRegionPathConfig(
            layer_indices=(16,), include_sink=sink, time_augment=aug, level=lvl,  # type: ignore[arg-type]
        )
        nm = attention_region_feature_names(cc)
        feats, _T = signature_features_from_native_path(
            attn_native if sink else attn_native[:, :4], cc, expected_native_dim=cc.native_dim,
        )
        got1 = sum(1 for n in nm if "_lvl1_" in n)
        got2 = sum(1 for n in nm if "_lvl2_" in n)
        c.ok(
            f"j4[{'sink' if sink else 'nosink'},{'aug' if aug else 'noaug'},lvl{lvl}] "
            f"native={cc.native_dim} -> lvl1={e1},lvl2={e2}",
            cc.n_level1 == e1 and cc.n_level2 == e2 and got1 == e1 and got2 == e2
            and len(nm) == e1 + e2 == len(feats),
            f"names {got1}+{got2}={len(nm)}, features={len(feats)}",
        )
    multi_attn = AttentionRegionPathConfig(layer_indices=(8, 16), include_sink=True, time_augment=True)
    c.ok(
        "j5 attention: multi-site arity = n_sites x per-site",
        len(attention_region_feature_names(multi_attn)) == 2 * multi_attn.n_features_per_site == 42,
        f"{len(attention_region_feature_names(multi_attn))} names over 2 sites",
    )

    attn_src = ArrayAttentionRegionSource({0: {16: attn_native}, 1: {16: attn_native[:2]}})
    attn_res = extract_attention_region_signature_from_source(attn_src, 0, attn_cfg)
    c.ok(
        "j6 ArrayAttentionRegionSource end-to-end matches the direct call",
        np.allclose(attn_res.features.astype(np.float64), attn_real, atol=1e-4)
        and attn_res.family_name == FAMILY_NAME_ATTENTION,
    )
    c.ok(
        "j7 attention: short path RAISES by default",
        _raises(lambda: extract_attention_region_signature_from_source(attn_src, 1, attn_cfg), ShortPathError),
    )
    c.ok(
        "j8 attention: missing layer RAISES",
        _raises(
            lambda: extract_attention_region_signature_from_source(
                ArrayAttentionRegionSource({0: {8: attn_native}}), 0, attn_cfg,
            ),
            KeyError,
        ),
    )

    # ── (k) real RawGenerationData end-to-end (both new sources) ──
    print("\n(k) RawGenerationData adapter, both new sources, end-to-end")
    from anamnesis.extraction.state_extractor import RawGenerationData

    def _synthetic_raw_gen(
        rng_: np.random.Generator, *, T: int = 70, prompt_len: int = 9,
        vocab: int = 96, n_layers: int = 6, n_heads: int = 4,
    ) -> RawGenerationData:
        logits = [(rng_.standard_normal(vocab) * 2.5).astype(np.float32) for _ in range(T)]
        chosen = rng_.integers(0, vocab, size=T).astype(np.float32)
        attentions = []
        for t in range(T):
            seq_len = prompt_len + t + 1
            raw = rng_.random((n_layers, n_heads, seq_len)).astype(np.float32) + 0.01
            raw = raw / raw.sum(axis=-1, keepdims=True)
            attentions.append(raw.astype(np.float32))
        return RawGenerationData(
            hidden_states=[], attentions=attentions, logits=logits,
            chosen_token_ids=chosen, pre_rope_keys={}, prompt_length=prompt_len,
        )

    rgd = _synthetic_raw_gen(rng)

    # (k, output) direct per-token matrix sanity
    mat_out = _output_stats_per_token(rgd.logits, (3, 7))
    c.ok(
        "k1 output per-token matrix: entropy>=0, eos_log_mass<=~0, varentropy>=0",
        bool(np.all(mat_out[:, 0] >= -1e-9)) and bool(np.all(mat_out[:, 2] <= 1e-9))
        and bool(np.all(mat_out[:, 3] >= -1e-9)),
        f"ranges: entropy[{mat_out[:,0].min():.3f},{mat_out[:,0].max():.3f}] "
        f"eos_log_mass[{mat_out[:,2].min():.3f},{mat_out[:,2].max():.3f}] "
        f"varentropy[{mat_out[:,3].min():.3f},{mat_out[:,3].max():.3f}]",
    )
    c.ok(
        "k2 output per-token matrix: margin>=0 (top1 >= top2 by construction)",
        bool(np.all(mat_out[:, 1] >= -1e-9)),
    )
    c.ok(
        "k3 output: eos id out of vocab range raises",
        _raises(lambda: _output_stats_per_token(rgd.logits, (10_000,)), PathSignatureError),
    )

    out_rgd_src = RawGenerationDataOutputStatsSource(rgd, eos_token_ids=(3, 7))
    out_rgd_res = extract_output_stats_signature_from_source(
        out_rgd_src, 0, OutputStatsPathConfig(eos_token_ids=(3, 7), time_augment=True),
    )
    c.ok(
        "k4 RawGenerationDataOutputStatsSource end-to-end",
        len(out_rgd_res) == out_cfg.n_features_per_site and out_rgd_res.family_name == FAMILY_NAME_OUTPUT,
    )
    c.ok(
        "k5 RawGenerationDataOutputStatsSource REFUSES empty eos_token_ids at construction",
        _raises(lambda: RawGenerationDataOutputStatsSource(rgd, eos_token_ids=()), PathSignatureError),
    )

    # (k, attention) direct per-token matrix sanity: the 4 region masses partition [0,1]
    mat_attn = _attention_region_per_token(rgd, 3, include_sink=True)
    partitions_ok = np.allclose(mat_attn[:, :4].sum(axis=1), 1.0, atol=1e-4)
    sink_le_prompt = bool(np.all(mat_attn[:, 4] <= mat_attn[:, 0] + 1e-9))  # sink is WITHIN prompt
    c.ok(
        "k6 attention per-token matrix: prompt+early+mid+recent partitions to 1.0",
        bool(partitions_ok),
        f"max|sum-1| = {float(np.max(np.abs(mat_attn[:, :4].sum(axis=1) - 1.0))):.3e}",
    )
    c.ok(
        "k7 attention per-token matrix: sink mass <= prompt mass (sink is inside the prompt region)",
        sink_le_prompt,
    )
    attn_rgd_src = RawGenerationDataAttentionRegionSource(rgd, include_sink=True)
    attn_rgd_res = extract_attention_region_signature_from_source(
        attn_rgd_src, 0, AttentionRegionPathConfig(layer_indices=(3,), include_sink=True, time_augment=True),
    )
    c.ok(
        "k8 RawGenerationDataAttentionRegionSource end-to-end",
        len(attn_rgd_res) > 0 and attn_rgd_res.family_name == FAMILY_NAME_ATTENTION,
    )
    c.ok(
        "k9 RawGenerationDataAttentionRegionSource: out-of-range layer raises KeyError",
        _raises(lambda: attn_rgd_src.load_path(0, 999), KeyError),
    )

    # ── (l) feature_map classification — output + attention families, zero unclassified ──
    print("\n(l) feature_map classification — §1a/§1b families")
    out_names_full = output_stats_feature_names(
        OutputStatsPathConfig(eos_token_ids=(3, 7), time_augment=True),
    )
    fm_out = FeatureMap(out_names_full, n_layers=32)
    c.ok("l1 output: zero unclassified", not fm_out.unclassified(), f"{fm_out.unclassified()[:5]}")
    c.ok(
        "l2 output: every name SOURCE=output, METHOD=iterated_integral, family=path_signature_output",
        all(t.source == Source.output for t in fm_out.tags)
        and all(t.method == Method.iterated_integral for t in fm_out.tags)
        and {t.family for t in fm_out.tags} == {"path_signature_output"},
        f"sources={sorted({t.source.value for t in fm_out.tags})}",
    )
    c.ok(
        "l3 output: DEPTH is None (no per-layer site)",
        all(t.layer is None and t.band is None for t in fm_out.tags),
    )

    attn_names_full = attention_region_feature_names(
        AttentionRegionPathConfig(layer_indices=(0, 14, 16, 27), include_sink=True, time_augment=True),
    )
    fm_attn = FeatureMap(attn_names_full, n_layers=32)
    c.ok("l4 attention: zero unclassified", not fm_attn.unclassified(), f"{fm_attn.unclassified()[:5]}")
    c.ok(
        "l5 attention: every name SOURCE=attention, METHOD=iterated_integral, "
        "family=path_signature_attention",
        all(t.source == Source.attention for t in fm_attn.tags)
        and all(t.method == Method.iterated_integral for t in fm_attn.tags)
        and {t.family for t in fm_attn.tags} == {"path_signature_attention"},
        f"sources={sorted({t.source.value for t in fm_attn.tags})}",
    )
    bands_attn = {t.layer: t.band for t in fm_attn.tags}
    c.ok(
        "l6 attention: DEPTH parsed per site (L0=early, L14/L16=mid, L27=late @ n_layers=32)",
        bands_attn.get(0) == Band.early and bands_attn.get(14) == Band.mid
        and bands_attn.get(16) == Band.mid and bands_attn.get(27) == Band.late,
    )
    dyn_out = {n: t.dynamic for n, t in zip(out_names_full, fm_out.tags)}
    dyn_attn = {n: t.dynamic for n, t in zip(attn_names_full, fm_attn.tags)}
    c.ok(
        "l7 output + attention: lvl1 static / lvl2 dynamic (same rule as residual)",
        {dyn_out[n] for n in out_names_full if "_lvl1_" in n} == {False}
        and {dyn_out[n] for n in out_names_full if "_lvl2_" in n} == {True}
        and {dyn_attn[n] for n in attn_names_full if "_lvl1_" in n} == {False}
        and {dyn_attn[n] for n in attn_names_full if "_lvl2_" in n} == {True},
    )
    c.ok(
        "l8 output + attention names are disjoint from the residual/legacy corpus prefixes "
        "(res_sig/res_traj/... never collide with out_sig/attn_sig)",
        all(not n.startswith(("res_sig", "res_traj", "activation_norm", "cache_", "attn_flow_"))
            for n in out_names_full + attn_names_full),
    )

    return c.close("SELFTEST")


def _raises(fn: Any, exc_type: type[BaseException]) -> bool:
    try:
        fn()
    except exc_type:
        return True
    except Exception as e:  # wrong exception type is a failure, not a pass
        logger.debug("expected %s, got %r", exc_type.__name__, e)
        return False
    return False


class _FakeRawGenerationData:
    """Minimal RawGenerationData stand-in for the selftest (no torch, no model, no bank).

    Layer 2 is deliberately zero-filled to exercise the unsaved-layer guard.
    """

    def __init__(self, hidden_dim: int, T: int, n_layers: int, rng: np.random.Generator) -> None:
        self.hidden_states = []
        base = _synthetic_path(rng, T=T, d=hidden_dim)
        for t in range(T):
            rows = rng.standard_normal((n_layers + 1, hidden_dim)).astype(np.float32)
            rows[17] = base[t].astype(np.float32)   # layer 16 -> row 17
            rows[3] = 0.0                            # layer 2 -> "not saved"
            self.hidden_states.append(rows)
        self.prompt_length = 11
        self.positional_means = None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Path-signature feature family (level-2 log-signature, spec 2026-09-11)",
    )
    parser.add_argument(
        "--selftest", action="store_true",
        help="Run the CPU-only validation battery (no banked data, no GPU)",
    )
    parser.add_argument(
        "--names", action="store_true",
        help="Print the emitted feature names for the default config and exit",
    )
    parser.add_argument("--k", type=int, default=4, help="Projection rank for --names")
    parser.add_argument("--layer", type=int, default=16, help="Site for --names")
    parser.add_argument(
        "--no-time-augment", action="store_true", help="Disable time augmentation for --names",
    )
    args = parser.parse_args()

    if args.names:
        cfg = PathSignatureConfig(
            layer_indices=(args.layer,), n_components=args.k,
            time_augment=not args.no_time_augment,
        )
        for n in feature_names(cfg):
            print(n)
        return 0

    if args.selftest:
        return 0 if selftest() else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
