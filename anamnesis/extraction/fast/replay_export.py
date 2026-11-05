"""Replay exports: a signed engine record's raw surfaces, rebuilt on demand, as files any tool can read.

A signing server keeps a record's tokens and provenance, not its raw surfaces. Replaying the record through the
signing decoder rebuilds them bitwise, and the server's ``/replay`` endpoint writes the result as two files:

* ``<sign_id>.safetensors``: the surfaces, in their capture dtypes (bfloat16 kept as bfloat16);
* ``<sign_id>.json``: the header. It carries the schema (``anamnesis-replay-export-v1``), the token ids and span,
  the weights, engine-identity, pin-table and lever-plan digests, the lane id, the routing source counts, and
  every tensor's shape, dtype and axes. The safetensors metadata carries the same header under
  ``anamnesis_replay``.

The surfaces are the signed-row contract's (:mod:`anamnesis.extraction.fast.engine_reducer`), unchanged.
``committed`` is stored stacked, [blocks, steps, hidden], and loads back as the contract's list.

A server writes an export only when the replay reproduced the signed row (``replayed_bitwise``). This module
refuses an export that does not say so. :func:`load_export` returns the header and a row that
:class:`~anamnesis.extraction.fast.engine_reducer.EngineReducer` reduces like any signed row, so re-reducing
an export checks it against its signed features. :func:`save_export` writes the same format, for tests and
for tools that make exports themselves.

One caution for hidden-state tools: ``depth[:, l]`` is layer l's in-progress block sum after its MLP, unrouted.
It is not the residual stream that enters layer l + 1; that one is a routed mix over the committed blocks.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field

SCHEMA = "anamnesis-replay-export-v1"
"""The schema name an export's header carries."""

METADATA_KEY = "anamnesis_replay"
"""The safetensors metadata key the header is repeated under."""

SURFACES = ("depth", "committed", "route", "route_final", "q", "k", "v", "gate", "head_entropy",
            "mean_entropy", "head_scalars", "mean_rows", "logit_stats")
"""The signed surfaces an export of a head-agreement row holds."""

MIXTURE_SURFACES = ("mean_entropy", "head_collision")
"""The per-layer mixture surfaces: an export holds exactly one, the one its row was signed with
(``head_collision`` in place of ``mean_entropy`` for a lane whose ``head_disagreement`` is
``spread``)."""

SAMPLED_WITH = {"head_collision": ("head_spread",)}
"""The sampled-layer surfaces an export holds with its mixture surface: a ``spread`` row's
``head_spread [kept, sampled]``."""


def _surface_names(available) -> list[str]:
    """The surfaces an export holds, given the names a row or header carries.

    Raises
    ------
    ReplayExportError
        When the names carry neither mixture surface, or both, or a ``spread`` row's mixture
        without its sampled-layer surfaces."""
    mixture = [n for n in MIXTURE_SURFACES if n in available]
    if len(mixture) != 1:
        raise ReplayExportError(f"an export holds one of {list(MIXTURE_SURFACES)}, not {mixture}")
    extra = SAMPLED_WITH.get(mixture[0], ())
    missing = [n for n in extra if n not in available]
    if missing:
        raise ReplayExportError(f"an export with {mixture[0]} holds {missing} too")
    return [mixture[0] if n == "mean_entropy" else n for n in SURFACES] + list(extra)


class ReplayExportError(ValueError):
    """An export that is not one: a wrong schema, a missing surface, a shape its header does not name, or a
    replay that did not reproduce its signed row."""


class TensorSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    shape: tuple[int, ...]
    dtype: str
    axes: tuple[str, ...]


class ExportHeader(BaseModel):
    """An export's header (``<sign_id>.json``)."""

    model_config = ConfigDict(frozen=True, extra="allow")

    schema_: str = Field(alias="schema")
    sign_id: str = Field(min_length=1)
    request_id: str | None = None
    token_ids: tuple[int, ...]
    prompt_length: int = Field(ge=0)
    start: int = Field(ge=1)
    positions: int = Field(ge=1)
    weights_sha256: str = Field(min_length=1)
    engine_identity_sha256: str
    pins_sha256: str | None = None
    lever_plan_sha256: str | None = None
    lane_id: str
    replayed_bitwise: bool
    layers: dict[str, Any]
    route_sources: tuple[tuple[int, int], ...]
    final_sources: int = Field(ge=1)
    tensors: dict[str, TensorSpec]
    lever_slot: int = -1


@dataclass
class ExportedRow:
    """A signed row rebuilt from an export: the attributes EngineReducer reads."""

    positions: int
    kept: int
    start: int
    lever_slot: int
    steered: bool
    route_sources: tuple[tuple[int, int], ...]
    final_sources: int
    weights: str
    surfaces: dict[str, Any] = field(default_factory=dict)


def _paths(path: str | Path) -> tuple[Path, Path]:
    p = Path(path)
    stem = p.with_suffix("") if p.suffix in (".safetensors", ".json") else p
    return stem.with_suffix(".safetensors"), stem.with_suffix(".json")


def _dtype_name(t: torch.Tensor) -> str:
    return str(t.dtype).removeprefix("torch.")


def load_export(path: str | Path, device: str | torch.device = "cpu") -> tuple[ExportHeader, ExportedRow]:
    """An export's header and its row, with every surface on ``device``.

    Raises
    ------
    ReplayExportError
        When the schema is another one, the replay did not reproduce its signed row, a surface is missing, or a
        tensor's shape or dtype is not the one the header names.
    """
    from safetensors.torch import load_file

    tensors_path, header_path = _paths(path)
    try:
        header = ExportHeader.model_validate(json.loads(header_path.read_text()))
    except (OSError, json.JSONDecodeError, ValueError) as err:
        raise ReplayExportError(f"{header_path}: not a readable export header ({err})") from err
    if header.schema_ != SCHEMA:
        raise ReplayExportError(f"{header_path}: schema {header.schema_!r}, not {SCHEMA!r}")
    if not header.replayed_bitwise:
        raise ReplayExportError(
            f"{header.sign_id}: the replay did not reproduce its signed row; its surfaces describe some other pass")
    try:
        names = _surface_names(header.tensors)
    except ReplayExportError as err:
        raise ReplayExportError(f"{header.sign_id}: {err}") from err
    missing = [n for n in names if n not in header.tensors]
    if missing:
        raise ReplayExportError(f"{header.sign_id}: the header names no {missing}")
    loaded = load_file(str(tensors_path), device=str(device))
    surfaces: dict[str, Any] = {}
    for name in names:
        if name not in loaded:
            raise ReplayExportError(f"{header.sign_id}: {tensors_path.name} holds no {name!r}")
        t, spec = loaded[name], header.tensors[name]
        if tuple(t.shape) != spec.shape or _dtype_name(t) != spec.dtype:
            raise ReplayExportError(
                f"{header.sign_id}: {name} is {tuple(t.shape)} {_dtype_name(t)}, and the header names "
                f"{spec.shape} {spec.dtype}")
        surfaces[name] = list(t.unbind(0)) if name == "committed" else t
    row = ExportedRow(positions=header.positions, kept=header.positions, start=header.start,
                      lever_slot=header.lever_slot, steered=header.lever_plan_sha256 is not None,
                      route_sources=header.route_sources, final_sources=header.final_sources,
                      weights=header.weights_sha256, surfaces=surfaces)
    return header, row


def save_export(path: str | Path, row: Any, header: Mapping[str, Any], axes: Mapping[str, tuple[str, ...]]
                | None = None) -> tuple[Path, Path]:
    """Write ``row``'s surfaces and ``header`` in the export format; returns the two paths.

    ``header`` gives every field but ``schema`` and ``tensors``, which this fills in. ``axes`` names each
    tensor's axes (default: ``dim0``, ``dim1``, …).
    """
    from safetensors.torch import save_file

    tensors: dict[str, torch.Tensor] = {}
    for name in _surface_names(row.surfaces):
        x = row.surfaces[name]
        t = torch.stack(list(x)) if name == "committed" else x
        tensors[name] = t.detach().contiguous().cpu()
    full = dict(header, schema=SCHEMA, tensors={
        n: dict(shape=list(t.shape), dtype=_dtype_name(t),
                axes=list((axes or {}).get(n, tuple(f"dim{i}" for i in range(t.ndim)))))
        for n, t in tensors.items()})
    tensors_path, header_path = _paths(path)
    tensors_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(tensors_path), metadata={METADATA_KEY: json.dumps(full)})
    header_path.write_text(json.dumps(full, indent=1))
    return tensors_path, header_path


__all__ = ["METADATA_KEY", "MIXTURE_SURFACES", "SAMPLED_WITH", "SCHEMA", "SURFACES", "ExportHeader", "ExportedRow", "ReplayExportError",
           "TensorSpec", "load_export", "save_export"]
