"""The registry of named extraction runs, and how a name becomes a directory.

Analysis is addressed by run name: a caller asks for ``8b_v2`` and gets the
signature directory it reads. The registry itself is data — :data:`RUNS_FILE`, a
JSON file beside this module — because adding a run is adding a row, and a row is
not a code change.

Resolution has two halves, kept apart on purpose. A :class:`RunSpec` is what the
file holds: a root token and a path relative to it. A :class:`ResolvedRun` is that
spec against this machine, with an absolute directory. So one registry serves a
tree read from the live outputs root and a tree read from the Phase-0 root
through ``ANAMNESIS_LEGACY_DATA``, and neither entry records anybody's absolute
layout.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from anamnesis.config.paths import DataRoot, resolve_data_path

RUNS_FILE: Path = Path(__file__).resolve().parent / "runs.json"
"""The registry file this module reads."""


class RunsRegistryError(RuntimeError):
    """The registry file is missing, unreadable, or not what this module expects."""


class UnknownRunError(KeyError):
    """A run name the registry does not hold."""


class RunSpec(BaseModel):
    """One registry row: where a run's signatures sit, relative to a root."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Registry key for this run")
    root: DataRoot = Field(description="Which data root the paths below hang from")
    signature_dir: str = Field(description="Signature directory, relative to the root")
    description: str = Field(default="", description="What this run is")

    def resolve(self) -> ResolvedRun:
        """This row as a directory on this machine."""
        return ResolvedRun(
            name=self.name,
            signature_dir=resolve_data_path(self.root, self.signature_dir),
            description=self.description,
        )


class ResolvedRun(BaseModel):
    """A registry row against this machine: an absolute directory, ready to read."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Registry key for this run")
    signature_dir: Path = Field(description="Directory holding one signature file per generation")
    description: str = Field(default="", description="What this run is")

    def missing_dirs(self) -> tuple[Path, ...]:
        """Every directory of this run that is absent.

        A caller reporting an unreadable run names directories rather than a
        boolean, so the message says which one it could not find.
        """
        return tuple(d for d in (self.signature_dir,) if not d.is_dir())


class _RunsFile(BaseModel):
    """The shape of the registry file."""

    model_config = ConfigDict(extra="forbid")

    runs: dict[str, RunSpec]


def load_runs(path: Path | None = None) -> dict[str, RunSpec]:
    """Every registry row, keyed by run name.

    Raises
    ------
    RunsRegistryError
        When the file is absent, is not readable as JSON, does not match the
        registry shape, or holds a row whose ``name`` disagrees with its key.
    """
    source = RUNS_FILE if path is None else path
    try:
        raw = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise RunsRegistryError(f"run registry unreadable: {source} ({exc})") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RunsRegistryError(f"{source}: invalid JSON at line {exc.lineno} ({exc.msg})") from exc
    try:
        parsed = _RunsFile.model_validate(payload)
    except ValidationError as exc:
        raise RunsRegistryError(f"{source}: not a run registry ({exc})") from exc
    for key, spec in parsed.runs.items():
        if spec.name != key:
            raise RunsRegistryError(
                f"{source}: run {key!r} carries name {spec.name!r}; the key and the name are one thing"
            )
    return dict(parsed.runs)


def run_names(path: Path | None = None) -> tuple[str, ...]:
    """Every run name the registry holds, in file order."""
    return tuple(load_runs(path))


def get_run(name: str, path: Path | None = None) -> RunSpec:
    """One registry row.

    Raises
    ------
    UnknownRunError
        Naming every run the registry holds.
    RunsRegistryError
        When the registry itself cannot be read.
    """
    runs = load_runs(path)
    try:
        return runs[name]
    except KeyError as exc:
        raise UnknownRunError(
            f"unknown run {name!r}; registry holds: {', '.join(sorted(runs))}"
        ) from exc


def resolve_run(name: str, path: Path | None = None) -> ResolvedRun:
    """One registry row as directories on this machine."""
    return get_run(name, path).resolve()
