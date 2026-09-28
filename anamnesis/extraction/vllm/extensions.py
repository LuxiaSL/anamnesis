"""Extension lanes: a fine-tune of a qualified model, declared by whoever owns it.

A shipped lane (:data:`anamnesis.extraction.vllm.envelope.LANE_MODELS`) was
qualified against the numeric anchor, and its fixtures and tolerance ship with
this package. A fine-tune of one of those models has the same architecture and
shapes, so everything that depends only on them carries over: the kernels, the
determinism of the dispatch, the feature schema and the capture routes. What the
weights can change is how far the engine drifts from the hook path on a row. The
transfer check (:mod:`anamnesis.extraction.vllm.transfer`) measures that drift on
a sample of the fine-tune's own rows and scores it against the base's recorded
tolerance; a pass produces the fine-tune's own fixtures and tolerance, and the
receipt that says so.

This module admits such a fine-tune as a lane of its own without editing the
package. :data:`LANES_ENV` names lane files, separated the way ``PATH`` is:

.. code-block:: json

    {"lanes": {"my-finetune": {
        "extends": "<a shipped lane key>",
        "preset": "my-finetune",
        "checkpoint_sha256": "<64 hex>",
        "calibration_dir": "calibration",
        "calibration": {"positional_means.npz": {"size": 123, "sha256": "<64 hex>"},
                        "pca_model.pkl": {"size": 456, "sha256": "<64 hex>"}},
        "fixtures_dir": "fixtures",
        "transfer_receipt": "<the receipt file>",
        "transfer_receipt_sha256": "<64 hex>"}}}

Relative paths resolve against the lane file's own directory.
``anamnesis/scripts/transfer_vllm.py`` writes an entry of this shape beside the
fixtures and receipt it produces.

**Declaring** an entry is refused, for the whole file, when its key is a shipped
lane or another entry's, when two entries name one checkpoint on one base, when
``extends`` is not a shipped lane, or when it sets the dtype, the logprob handling
or any engine setting: those are the base's, inherited unchanged, because the
base's qualification is what the transfer check leans on. An entry adds a lane
and never redefines one.

**Admitting** an entry (:func:`admit`) is the guard every use passes. It refuses
the entry by name unless its transfer receipt is present, has the declared
digest, says ``pass``, and names the same base lane id, checkpoint, fixture set
and tolerance as the entry and its fixtures; unless the fixtures carry the
extension's lane id, checkpoint and calibration; unless the base's shipped
tolerance is still the one the receipt was scored against; and unless the
registry preset the entry names descends from the base's preset through
``extends`` with the base's architecture and layer plan unchanged.

An extension's lane id is the digest of its base's identity, its base's key and
its checkpoint digest (:func:`anamnesis.extraction.vllm.envelope.extension_identity`):
distinct from the base's and from every other extension's, so the read-side
lane guard keeps their rows apart like any two lanes.

Its calibration is read from the declared directory and verified against the
declared sizes and digests by :func:`verify_calibration` before anything reads it.
The shipped lanes keep the pins in :mod:`anamnesis.extraction.vllm.hub`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from anamnesis.extraction.vllm.conformance import FixtureSet, Tolerance
from anamnesis.extraction.vllm.envelope import (
    CONDITION_KEYS,
    LANE_MODELS,
    SETTINGS,
    canonical_digest,
    extension_identity,
    lane_id,
)
from anamnesis.extraction.vllm.runtime import CALIBRATION_FILES, load_fixtures
from anamnesis.extraction.vllm.transfer import TransferReceipt
from anamnesis.provenance import digest_of_shas, file_sha

LANES_ENV = "ANAMNESIS_VLLM_LANES"
"""Environment variable naming extension-lane files, separated like ``PATH``."""

INHERITED = frozenset({"dtype", "logprob_wrapper", *SETTINGS, *CONDITION_KEYS})
"""Keys an entry may not set: its base's lane fixes every one of them."""

STRUCTURAL_FIELDS = (
    "torch_dtype", "num_layers", "hidden_dim", "num_attention_heads", "num_kv_heads",
    "head_dim", "sampled_layers", "pca_layers", "pca_components_by_layer",
    "trajectory_layers", "contrastive_layers", "early_layer_cutoff", "late_layer_cutoff",
    "attention_layer_types",
)
"""The preset fields an extension's preset must share with its base's: the
architecture, the dtype the reference loads at, and every layer the schema reads."""

HEX64 = "^[0-9a-f]{64}$"


class LaneFileError(ValueError):
    """An extension-lane file is unreadable, malformed, or claims a name it may not."""


class CalibrationPin(BaseModel):
    """One calibration file's size in bytes and sha256."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    size: int = Field(gt=0)
    sha256: str = Field(pattern=HEX64)


class ExtensionLane(BaseModel):
    """One declared extension lane: what it extends and where its evidence lives."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1, description="The lane key commands take")
    extends: str = Field(min_length=1, description="The shipped lane this fine-tune extends")
    preset: str = Field(min_length=1, description="The registry preset of the fine-tune")
    checkpoint_sha256: str = Field(
        pattern=HEX64, description="Digest of the checkpoint's config and safetensors shards, "
                                   "as the lane's install check computes it")
    calibration_dir: Path
    calibration: dict[str, CalibrationPin]
    fixtures_dir: Path
    transfer_receipt: Path
    transfer_receipt_sha256: str = Field(pattern=HEX64)
    declared_in: Path = Field(description="The lane file the entry was read from")

    @field_validator("calibration")
    @classmethod
    def _pins_every_calibration_file(cls, pins: dict[str, CalibrationPin]) -> dict[str, CalibrationPin]:
        if set(pins) != set(CALIBRATION_FILES):
            raise ValueError(f"calibration must pin exactly {list(CALIBRATION_FILES)}, "
                             f"got {sorted(pins)}")
        return pins

    @property
    def lane_id(self) -> str:
        """The extension's lane id."""
        return canonical_digest(extension_identity(self.extends, self.checkpoint_sha256))

    @property
    def calibration_sha256(self) -> str:
        """The calibration digest its fixtures record, from the declared file digests."""
        return digest_of_shas({name: pin.sha256 for name, pin in self.calibration.items()})


def lane_paths() -> tuple[Path, ...]:
    """Every file named in :data:`LANES_ENV`, read at call time."""
    return tuple(Path(entry.strip()).expanduser()
                 for entry in os.environ.get(LANES_ENV, "").split(os.pathsep) if entry.strip())


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [k for k, _ in pairs]
    repeated = sorted({k for k in keys if keys.count(k) > 1})
    if repeated:
        raise LaneFileError(f"names {repeated} more than once")
    return dict(pairs)


def _resolve(base: Path, value: Any) -> Any:
    if not isinstance(value, str):
        return value
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _read_file(path: Path) -> list[ExtensionLane]:
    """One lane file's entries, parsed; relative paths resolved against its directory.

    Raises
    ------
    LaneFileError
        When the file cannot be read or parsed, repeats a key, has another shape, or
        holds an entry that sets an inherited key or does not validate.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"),
                             object_pairs_hook=_no_duplicate_keys)
    except OSError as exc:
        raise LaneFileError(f"lane file unreadable: {path} ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise LaneFileError(f"{path}: invalid JSON at line {exc.lineno} ({exc.msg})") from exc
    except LaneFileError as exc:
        raise LaneFileError(f"{path}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) - {"description", "lanes"} \
            or not isinstance(payload.get("lanes"), dict):
        raise LaneFileError(f"{path}: a lane file is an object with a 'lanes' object and an "
                            "optional 'description'")
    entries = []
    for key, row in payload["lanes"].items():
        if not isinstance(row, dict):
            raise LaneFileError(f"{path}: lane {key!r} is not an object")
        overrides = sorted(INHERITED & set(row))
        if overrides:
            raise LaneFileError(
                f"{path}: lane {key!r} sets {overrides}; the dtype, the logprob handling and "
                f"every engine setting are inherited from the lane it extends and cannot be "
                "overridden")
        fields = {k: _resolve(path.parent, v) if k in ("calibration_dir", "fixtures_dir",
                                                       "transfer_receipt") else v
                  for k, v in row.items()}
        try:
            entries.append(ExtensionLane.model_validate(
                dict(fields, key=key, declared_in=path)))
        except ValidationError as exc:
            raise LaneFileError(f"{path}: lane {key!r} is not a valid entry ({exc})") from exc
    return entries


def _merge(files: list[tuple[Path, list[ExtensionLane]]]) -> dict[str, ExtensionLane]:
    """Every entry, keyed, after refusing each name it may not claim."""
    lanes: dict[str, ExtensionLane] = {}
    by_lane_id: dict[str, str] = {}
    for path, entries in files:
        for entry in entries:
            key = entry.key
            if key in LANE_MODELS:
                raise LaneFileError(
                    f"{path}: lane {key!r} is a shipped lane; an extension adds a lane under a "
                    "key of its own and never redefines a shipped one")
            if key in lanes:
                raise LaneFileError(
                    f"{path}: lane {key!r} is already declared in {lanes[key].declared_in}; a "
                    "lane file adds lanes and never redefines them")
            if entry.extends not in LANE_MODELS:
                raise LaneFileError(
                    f"{path}: lane {key!r} extends {entry.extends!r}, which is not a shipped "
                    f"lane (shipped: {', '.join(sorted(LANE_MODELS))}); an extension extends a "
                    "qualified base directly")
            held = by_lane_id.get(entry.lane_id)
            if held is not None:
                raise LaneFileError(
                    f"{path}: lane {key!r} names the checkpoint lane {held!r} already names on "
                    f"{entry.extends!r}; one checkpoint on one base is one lane")
            lanes[key] = entry
            by_lane_id[entry.lane_id] = key
    return lanes


_CACHE: dict[tuple[tuple[str, str], ...], dict[str, ExtensionLane]] = {}


def declared_lanes(paths: tuple[Path, ...] | None = None) -> dict[str, ExtensionLane]:
    """Every extension lane the files in :data:`LANES_ENV` declare, keyed.

    Cached against each file's path and content digest, so an edited file is
    read again however quickly it changed. Declaring is not admitting:
    :func:`admit` is the guard a use passes.

    Raises
    ------
    LaneFileError
        When a named file is missing or malformed, or claims a name it may not.
    """
    sources = lane_paths() if paths is None else tuple(paths)
    stamps = []
    for path in sources:
        try:
            stamps.append((str(path), file_sha(path)))
        except OSError as exc:
            raise LaneFileError(f"lane file unreadable: {path} ({exc})") from exc
    key = tuple(stamps)
    if key not in _CACHE:
        _CACHE[key] = _merge([(path, _read_file(path)) for path in sources])
    return dict(_CACHE[key])


def declared_lane(key: str) -> ExtensionLane | None:
    """The extension lane declared under ``key``, or None."""
    if not os.environ.get(LANES_ENV, "").strip():
        return None
    return declared_lanes().get(key)


def lane_keys() -> tuple[str, ...]:
    """Every lane key a command accepts: the shipped lanes, then the declared ones."""
    declared = declared_lanes() if os.environ.get(LANES_ENV, "").strip() else {}
    return (*sorted(LANE_MODELS), *sorted(declared))


def is_extension(key: str) -> bool:
    """Whether ``key`` names a declared extension lane rather than a shipped one."""
    return key not in LANE_MODELS and declared_lane(key) is not None


def _require(key: str) -> ExtensionLane:
    entry = declared_lane(key)
    if entry is None:
        raise ValueError(f"{key!r} is not a declared extension lane (declared: "
                         f"{', '.join(lane_keys())})")
    return entry


def check_preset(preset: str, extends: str) -> None:
    """Refuse a registry preset that is not a structural copy of ``extends``'s preset.

    The preset must descend from the base's preset through the registry's
    ``extends``, and keep every field in :data:`STRUCTURAL_FIELDS` equal to it.

    Raises
    ------
    ValueError
        Naming the preset, and either its descent or the fields that differ.
    """
    from anamnesis.config.models import ModelRegistryError, UnknownPresetError, load_registry

    try:
        registry = load_registry()
        row = registry.resolve(preset)
    except (UnknownPresetError, ModelRegistryError) as exc:
        raise ValueError(f"preset {preset!r} does not resolve: {exc}") from exc
    base = registry.resolve(extends)
    chain = registry.extends_chain(row.name)
    if base.name not in chain:
        raise ValueError(
            f"preset {row.name!r} does not extend {base.name!r} (its extends chain: "
            f"{list(chain) or 'none'}); an extension's preset is a variant row of its base's")
    changed = [f for f in STRUCTURAL_FIELDS if getattr(row, f) != getattr(base, f)]
    if changed:
        raise ValueError(
            f"preset {row.name!r} changes {changed} from {base.name!r}; an extension shares "
            "its base's architecture, dtype and layer plan")


@dataclass(frozen=True)
class AdmittedLane:
    """An extension lane that passed :func:`admit`, with what the guard read."""

    entry: ExtensionLane
    fixtures: FixtureSet
    tolerance: Tolerance
    receipt: TransferReceipt


def _refuse(key: str, reason: str) -> NoReturn:
    raise ValueError(f"extension lane {key!r} refused: {reason}")


def admit(key: str) -> AdmittedLane:
    """The guard: ``key``'s entry, its fixtures, tolerance and receipt, all verified.

    Raises
    ------
    ValueError
        Naming ``key`` and the first failed condition, among: an undeclared key, a
        missing or altered transfer receipt, a receipt that refused or names another
        base, checkpoint, lane, fixture set or tolerance, fixtures that do not carry
        the extension's identity, a base tolerance other than the one the receipt
        scored against, and a preset that is not a structural copy of the base's.
    """
    entry = _require(key)
    receipt_path = entry.transfer_receipt
    if not receipt_path.is_file():
        _refuse(key, f"its transfer receipt {receipt_path} does not exist")
    if file_sha(receipt_path) != entry.transfer_receipt_sha256:
        _refuse(key, f"its transfer receipt {receipt_path} differs from the declared digest")
    try:
        receipt = TransferReceipt.model_validate_json(receipt_path.read_text())
    except ValidationError as exc:
        _refuse(key, f"its transfer receipt is not a transfer receipt ({exc})")
    if receipt.verdict != "pass":
        _refuse(key, f"its transfer receipt refused it: {'; '.join(receipt.reasons[:3])}")
    expected = dict(key=key, extends=entry.extends, base_lane_id=lane_id(entry.extends),
                    lane_id=entry.lane_id, checkpoint_sha256=entry.checkpoint_sha256,
                    calibration_sha256=entry.calibration_sha256)
    for field, value in expected.items():
        if getattr(receipt, field) != value:
            _refuse(key, f"its transfer receipt's {field} is {getattr(receipt, field)!r}, "
                         f"the entry's is {value!r}")
    try:
        fixtures = FixtureSet.load(entry.fixtures_dir)
        tolerance = Tolerance.model_validate_json(
            (entry.fixtures_dir / "tolerance.json").read_text())
    except (OSError, ValueError) as exc:
        _refuse(key, f"its fixtures under {entry.fixtures_dir} do not load ({exc})")
    identity = dict(model=(fixtures.model, key), tolerance_model=(tolerance.model, key),
                    lane_id=(fixtures.lane_id, entry.lane_id),
                    checkpoint_sha256=(fixtures.checkpoint_sha256, entry.checkpoint_sha256),
                    calibration_sha256=(fixtures.calibration_sha256, entry.calibration_sha256),
                    fixture_digest=(fixtures.digest, receipt.fixture_digest),
                    tolerance_digest=(tolerance.digest, receipt.tolerance_digest))
    for field, (found, wanted) in identity.items():
        if found != wanted:
            _refuse(key, f"its fixtures' {field} is {found!r}, expected {wanted!r}")
    _, base_tolerance = load_fixtures(entry.extends)
    if base_tolerance.digest != receipt.base_tolerance_digest:
        _refuse(key, f"the {entry.extends} tolerance it was scored against is not the one "
                     "this package ships; run the transfer check again")
    try:
        check_preset(entry.preset, entry.extends)
    except ValueError as exc:
        _refuse(key, str(exc))
    return AdmittedLane(entry=entry, fixtures=fixtures, tolerance=tolerance, receipt=receipt)


def calibration_pins(directory: Path) -> dict[str, CalibrationPin]:
    """The size and sha256 of each file in :data:`CALIBRATION_FILES` under ``directory``.

    Raises
    ------
    ValueError
        When a file is missing.
    """
    pins = {}
    for name in CALIBRATION_FILES:
        path = Path(directory) / name
        if not path.is_file():
            raise ValueError(f"{directory} holds no {name}")
        pins[name] = CalibrationPin(size=path.stat().st_size, sha256=file_sha(path))
    return pins


def verify_calibration(key: str, directory: Path | None = None) -> Path:
    """``key``'s calibration directory, every file verified against its declared pin.

    ``directory`` defaults to the declared one. Files are hashed, never loaded,
    so nothing is unpickled before it is verified. Returns the directory.

    Raises
    ------
    ValueError
        Naming ``key`` and each missing file and each file whose size or sha256
        differs from the entry.
    """
    entry = _require(key)
    directory = Path(directory) if directory is not None else entry.calibration_dir
    wrong = []
    for name, pin in entry.calibration.items():
        path = directory / name
        if not path.is_file():
            wrong.append(f"{name} missing")
        elif path.stat().st_size != pin.size or file_sha(path) != pin.sha256:
            wrong.append(f"{name} differs from its declared pin")
    if wrong:
        raise ValueError(f"{directory} is not extension lane {key!r}'s calibration: "
                         f"{', '.join(wrong)}")
    return directory


def lane_entry(*, key: str, extends: str, preset: str, checkpoint_sha256: str,
               calibration_dir: Path, calibration: Mapping[str, CalibrationPin],
               fixtures_dir: str, transfer_receipt: str,
               transfer_receipt_sha256: str) -> dict[str, Any]:
    """A lane-file document declaring one entry, in the shape :func:`declared_lanes` reads.

    ``fixtures_dir`` and ``transfer_receipt`` are written as given, so a path
    relative to the file's directory stays relative; the entry is validated before
    it is returned.
    """
    row = dict(extends=extends, preset=preset, checkpoint_sha256=checkpoint_sha256,
               calibration_dir=str(calibration_dir),
               calibration={n: p.model_dump() for n, p in calibration.items()},
               fixtures_dir=fixtures_dir, transfer_receipt=transfer_receipt,
               transfer_receipt_sha256=transfer_receipt_sha256)
    ExtensionLane.model_validate(dict(row, key=key, declared_in=Path(".")))
    return dict(lanes={key: row})
