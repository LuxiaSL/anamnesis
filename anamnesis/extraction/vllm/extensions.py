"""Extension lanes: a fine-tune of a shipped lane's model, declared by its owner.

:data:`LANES_ENV` names lane files, separated like ``PATH``, each
``{"lanes": {key: entry}}`` with entry fields as :class:`ExtensionLane` declares
them; relative paths resolve against the file's directory.
``anamnesis/scripts/transfer_vllm.py`` writes one on a passing transfer check
(:mod:`anamnesis.extraction.vllm.transfer`).

An entry adds a lane and never redefines one: a file is refused whole when an
entry reuses a shipped or declared key, declares one checkpoint on one base
twice, extends anything but a shipped lane, or sets the dtype, logprob handling or
an engine setting, all of which are inherited from the base. :func:`admit` is the
guard every use passes. An extension's calibration is verified against its
declared pins (:func:`verify_calibration`); shipped lanes keep
:mod:`anamnesis.extraction.vllm.hub`'s.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from anamnesis.extraction.vllm.conformance import FixtureSet, Tolerance
from anamnesis.extraction.vllm.envelope import (
    CONDITION_KEYS,
    LANE_MODELS,
    SETTINGS,
    extension_lane_id,
    lane_id,
)
from anamnesis.extraction.vllm.runtime import CALIBRATION_FILES, load_fixtures
from anamnesis.extraction.vllm.transfer import TransferReceipt
from anamnesis.provenance import digest_of_shas, file_sha

LANES_ENV = "ANAMNESIS_VLLM_LANES"
INHERITED = frozenset({"dtype", "logprob_wrapper", *SETTINGS, *CONDITION_KEYS})
STRUCTURAL_FIELDS = (
    "torch_dtype", "num_layers", "hidden_dim", "num_attention_heads", "num_kv_heads",
    "head_dim", "sampled_layers", "pca_layers", "pca_components_by_layer",
    "trajectory_layers", "contrastive_layers", "early_layer_cutoff", "late_layer_cutoff",
    "attention_layer_types",
)
"""Preset fields an extension's preset must share with its base's."""

HEX64 = "^[0-9a-f]{64}$"


class LaneFileError(ValueError):
    """A lane file is unreadable, malformed, or claims a name it may not."""


class CalibrationPin(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    size: int = Field(gt=0)
    sha256: str = Field(pattern=HEX64)


class ExtensionLane(BaseModel):
    """One declared extension lane."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1)
    extends: str = Field(min_length=1, description="The shipped lane it is a fine-tune of")
    preset: str = Field(min_length=1, description="Its registry preset")
    checkpoint_sha256: str = Field(pattern=HEX64)
    calibration_dir: Path
    calibration: dict[str, CalibrationPin]
    fixtures_dir: Path
    transfer_receipt: Path
    transfer_receipt_sha256: str = Field(pattern=HEX64)
    declared_in: Path

    @field_validator("calibration")
    @classmethod
    def _pins_every_file(cls, pins: dict[str, CalibrationPin]) -> dict[str, CalibrationPin]:
        if set(pins) != set(CALIBRATION_FILES):
            raise ValueError(f"calibration must pin exactly {list(CALIBRATION_FILES)}")
        return pins

    @property
    def lane_id(self) -> str:
        return extension_lane_id(self.extends, self.checkpoint_sha256)

    @property
    def calibration_sha256(self) -> str:
        return digest_of_shas({name: pin.sha256 for name, pin in self.calibration.items()})


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise LaneFileError(f"names {sorted({k for k in keys if keys.count(k) > 1})} more "
                            "than once")
    return dict(pairs)


def _read_file(path: Path) -> list[ExtensionLane]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"),
                             object_pairs_hook=_no_duplicate_keys)
    except (OSError, json.JSONDecodeError, LaneFileError) as exc:
        raise LaneFileError(f"{path}: unreadable or invalid JSON ({exc})") from exc
    if not isinstance(payload, dict) or set(payload) - {"description", "lanes"} \
            or not isinstance(payload.get("lanes"), dict):
        raise LaneFileError(f"{path}: a lane file is {{'lanes': {{key: entry}}}}")
    entries = []
    for key, row in payload["lanes"].items():
        overrides = sorted(INHERITED & set(row)) if isinstance(row, dict) else []
        if overrides:
            raise LaneFileError(f"{path}: lane {key!r} sets {overrides}; the dtype, logprob "
                                "handling and engine settings are inherited from its base")
        try:
            fields = {k: path.parent / Path(v).expanduser() if k in (
                "calibration_dir", "fixtures_dir", "transfer_receipt") else v
                for k, v in row.items()}
            entries.append(ExtensionLane.model_validate(dict(fields, key=key, declared_in=path)))
        except (AttributeError, TypeError, ValidationError) as exc:
            raise LaneFileError(f"{path}: lane {key!r} is not a valid entry ({exc})") from exc
    return entries


_CACHE: dict[tuple[tuple[str, str], ...], dict[str, ExtensionLane]] = {}


def declared_lanes() -> dict[str, ExtensionLane]:
    """Every extension lane the files in :data:`LANES_ENV` declare, keyed; cached by
    file content. Declaring is not admitting.

    Raises
    ------
    LaneFileError
        When a named file is missing or malformed, or claims a name it may not.
    """
    paths = [Path(p.strip()).expanduser()
             for p in os.environ.get(LANES_ENV, "").split(os.pathsep) if p.strip()]
    try:
        stamp = tuple((str(p), file_sha(p)) for p in paths)
    except OSError as exc:
        raise LaneFileError(f"lane file unreadable ({exc})") from exc
    if stamp in _CACHE:
        return _CACHE[stamp]
    lanes: dict[str, ExtensionLane] = {}
    by_lane_id: dict[str, str] = {}
    for entry in (e for p in paths for e in _read_file(p)):
        where = f"{entry.declared_in}: lane {entry.key!r}"
        if entry.key in LANE_MODELS or entry.key in lanes:
            raise LaneFileError(f"{where} is already a shipped or declared lane; a lane file "
                                "adds lanes and never redefines them")
        if entry.extends not in LANE_MODELS:
            raise LaneFileError(f"{where} extends {entry.extends!r}, which is not a shipped "
                                f"lane (shipped: {', '.join(sorted(LANE_MODELS))})")
        if entry.lane_id in by_lane_id:
            raise LaneFileError(f"{where} names {by_lane_id[entry.lane_id]!r}'s checkpoint "
                                "and base; one checkpoint on one base is one lane")
        lanes[entry.key], by_lane_id[entry.lane_id] = entry, entry.key
    _CACHE[stamp] = lanes
    return lanes


def declared_lane(key: str) -> ExtensionLane | None:
    """The extension lane declared as ``key``, or None."""
    return declared_lanes().get(key) if os.environ.get(LANES_ENV, "").strip() else None


def lane_keys() -> tuple[str, ...]:
    """Every lane key a command accepts: shipped, then declared."""
    declared = declared_lanes() if os.environ.get(LANES_ENV, "").strip() else {}
    return (*sorted(LANE_MODELS), *sorted(declared))


def check_preset(preset: str, extends: str) -> None:
    """Refuse a preset that does not descend from ``extends``'s through the registry's
    ``extends``, or that changes a :data:`STRUCTURAL_FIELDS` field.

    Raises
    ------
    ValueError
        Naming the preset, and its descent or the fields that differ.
    """
    from anamnesis.config.models import ModelRegistryError, UnknownPresetError, load_registry

    try:
        registry = load_registry()
        row = registry.resolve(preset)
    except (UnknownPresetError, ModelRegistryError) as exc:
        raise ValueError(f"preset {preset!r} does not resolve: {exc}") from exc
    base, key = registry.resolve(extends), row.name
    while key in registry.lineage and key != base.name:
        key = registry.lineage[key]
    if key != base.name or row.name == base.name:
        raise ValueError(f"preset {row.name!r} does not extend {base.name!r}")
    changed = [f for f in STRUCTURAL_FIELDS if getattr(row, f) != getattr(base, f)]
    if changed:
        raise ValueError(f"preset {row.name!r} changes {changed} from {base.name!r}; an "
                         "extension shares its base's architecture, dtype and layer plan")


@dataclass(frozen=True)
class AdmittedLane:
    entry: ExtensionLane
    fixtures: FixtureSet
    tolerance: Tolerance


def _refuse(key: str, reason: str) -> NoReturn:
    raise ValueError(f"extension lane {key!r} refused: {reason}")


def _require(key: str) -> ExtensionLane:
    entry = declared_lane(key)
    if entry is None:
        raise ValueError(f"{key!r} is not a declared extension lane (declared: "
                         f"{', '.join(lane_keys())})")
    return entry


def admit(key: str) -> AdmittedLane:
    """``key``'s entry, fixtures and tolerance, after the guard.

    Raises
    ------
    ValueError
        Naming ``key`` and the first failure among: an undeclared key; a transfer
        receipt missing, altered from its declared digest, refused, or naming another
        key, base, lane id, checkpoint or calibration; fixtures or a tolerance other
        than the receipt's or not carrying the extension's identity; a shipped base
        tolerance other than the one the receipt scored against; and a preset that is
        not a structural copy of the base's.
    """
    entry = _require(key)
    path = entry.transfer_receipt
    if not path.is_file():
        _refuse(key, f"its transfer receipt {path} does not exist")
    if file_sha(path) != entry.transfer_receipt_sha256:
        _refuse(key, f"its transfer receipt {path} differs from the declared digest")
    try:
        receipt = TransferReceipt.model_validate_json(path.read_text())
    except ValueError as exc:
        _refuse(key, f"its transfer receipt is not one ({exc})")
    if receipt.verdict != "pass":
        _refuse(key, f"its transfer receipt refused it: {'; '.join(receipt.reasons[:3])}")
    try:
        fixtures = FixtureSet.load(entry.fixtures_dir)
        tolerance = Tolerance.model_validate_json(
            (entry.fixtures_dir / "tolerance.json").read_text())
    except (OSError, ValueError) as exc:
        _refuse(key, f"its fixtures do not load ({exc})")
    expected = {
        "receipt key": (receipt.key, key), "receipt extends": (receipt.extends, entry.extends),
        "receipt base_lane_id": (receipt.base_lane_id, lane_id(entry.extends)),
        "receipt lane_id": (receipt.lane_id, entry.lane_id),
        "receipt checkpoint_sha256": (receipt.checkpoint_sha256, entry.checkpoint_sha256),
        "receipt calibration_sha256": (receipt.calibration_sha256, entry.calibration_sha256),
        "fixtures model": (fixtures.model, key), "tolerance model": (tolerance.model, key),
        "fixtures lane_id": (fixtures.lane_id, entry.lane_id),
        "fixtures checkpoint_sha256": (fixtures.checkpoint_sha256, entry.checkpoint_sha256),
        "fixtures calibration_sha256": (fixtures.calibration_sha256, entry.calibration_sha256),
        "fixtures digest": (fixtures.digest, receipt.fixture_digest),
        "tolerance digest": (tolerance.digest, receipt.tolerance_digest),
        "base tolerance digest": (load_fixtures(entry.extends)[1].digest,
                                  receipt.base_tolerance_digest),
    }
    for name, (found, wanted) in expected.items():
        if found != wanted:
            _refuse(key, f"{name} is {found!r}, expected {wanted!r}")
    try:
        check_preset(entry.preset, entry.extends)
    except ValueError as exc:
        _refuse(key, str(exc))
    return AdmittedLane(entry=entry, fixtures=fixtures, tolerance=tolerance)


def calibration_pins(directory: Path) -> dict[str, CalibrationPin]:
    """Each calibration file's size and sha256; a missing file raises ``OSError``."""
    return {name: CalibrationPin(size=(Path(directory) / name).stat().st_size,
                                 sha256=file_sha(Path(directory) / name))
            for name in CALIBRATION_FILES}


def verify_calibration(key: str, directory: Path | None = None) -> Path:
    """``key``'s calibration directory (the declared one by default), every file hashed
    against its declared pin before anything reads it.

    Raises
    ------
    ValueError
        Naming ``key`` and each file missing or differing from its pin.
    """
    entry = _require(key)
    directory = Path(directory) if directory is not None else entry.calibration_dir
    pins = {name: CalibrationPin(size=(directory / name).stat().st_size,
                                 sha256=file_sha(directory / name))
            if (directory / name).is_file() else None for name in CALIBRATION_FILES}
    wrong = [f"{n} {'missing' if pins[n] is None else 'differs from its declared pin'}"
             for n in CALIBRATION_FILES if pins[n] != entry.calibration[n]]
    if wrong:
        raise ValueError(f"{directory} is not extension lane {key!r}'s calibration: "
                         f"{', '.join(wrong)}")
    return directory
