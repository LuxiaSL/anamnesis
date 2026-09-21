"""The filesystem roots configuration resolves against.

Three roots locate everything the instrument reads and writes:

``package_root()``
    The installed package directory. Data shipped with the code — the prompt
    sets — lives under it.
``outputs_root()``
    Runs, calibration artifacts and analysis results. ``ANAMNESIS_OUTPUTS``
    names it; the default is ``outputs`` inside the package directory, which a
    deployment fills or points at its own data store.
``legacy_data_root()``
    The Phase-0 tree holding the 3B corpus and its calibration.
    ``ANAMNESIS_LEGACY_DATA`` names it; the default is ``phase_0`` beside the
    package directory.

Both environment variables are read at call time rather than at import, so a
process can point at another tree without reloading the package, and a test can
redirect either root with a temporary directory.

A stored path is a :data:`DataRoot` token plus a relative path, and
:func:`resolve_data_path` joins the two. Keeping the token beside the path is
what lets one registry entry address the legacy tree while the next addresses
the live one, without either entry hardcoding a machine's layout.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Literal, get_args

DataRoot = Literal["outputs", "legacy", "package"]
"""Which of the three roots a stored relative path hangs from."""

DATA_ROOTS: tuple[DataRoot, ...] = get_args(DataRoot)

OUTPUTS_ENV = "ANAMNESIS_OUTPUTS"
LEGACY_DATA_ENV = "ANAMNESIS_LEGACY_DATA"
RUN_NAME_ENV = "ANAMNESIS_RUN_NAME"

DEFAULT_RUN_NAME = "run_8b_baseline"
PROMPTS_DIRNAME = "prompts"
DEFAULT_PROMPT_SET = "prompt_sets.json"


class PathResolutionError(ValueError):
    """A stored path or root token that cannot be resolved to a location."""


def package_root() -> Path:
    """The directory of the installed ``anamnesis`` package."""
    return Path(__file__).resolve().parent.parent


def outputs_root() -> Path:
    """Where runs, calibration and analysis artifacts live."""
    override = os.environ.get(OUTPUTS_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return package_root() / "outputs"


def legacy_data_root() -> Path:
    """The Phase-0 tree the 3B corpus and its calibration are read from."""
    override = os.environ.get(LEGACY_DATA_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return package_root().parent / "phase_0"


def data_root(root: DataRoot) -> Path:
    """The directory a root token names.

    Raises
    ------
    PathResolutionError
        When the token is not one of :data:`DATA_ROOTS`.
    """
    if root == "outputs":
        return outputs_root()
    if root == "legacy":
        return legacy_data_root()
    if root == "package":
        return package_root()
    known = ", ".join(sorted(DATA_ROOTS))
    raise PathResolutionError(f"unknown data root {root!r}; known roots: {known}")


def resolve_data_path(root: DataRoot, relative: str) -> Path:
    """Join a relative POSIX path onto the directory a root token names.

    The relative path is a registry value, so it is checked rather than trusted:
    it must be non-empty, relative, and free of parent-directory steps, which
    keeps a registry entry from addressing anything outside its declared root.

    Raises
    ------
    PathResolutionError
        When the root token is unknown or the path is not a safe relative path.
    """
    text = relative.strip()
    if not text:
        raise PathResolutionError(f"empty relative path under root {root!r}")
    pure = PurePosixPath(text)
    if pure.is_absolute():
        raise PathResolutionError(
            f"path under root {root!r} must be relative, got {relative!r}"
        )
    if ".." in pure.parts:
        raise PathResolutionError(
            f"path under root {root!r} must not step outside it, got {relative!r}"
        )
    return data_root(root).joinpath(*pure.parts)


def run_name() -> str:
    """The run whose output directory is the default target of a generation."""
    override = os.environ.get(RUN_NAME_ENV, "").strip()
    return override or DEFAULT_RUN_NAME


def run_outputs_dir(name: str | None = None) -> Path:
    """The output directory of a named run under :func:`outputs_root`."""
    return outputs_root() / "runs" / (name or run_name())


def prompts_dir() -> Path:
    """The prompt-set directory shipped inside the package."""
    return package_root() / PROMPTS_DIRNAME


def prompts_path(name: str = DEFAULT_PROMPT_SET) -> Path:
    """A prompt-set file shipped inside the package."""
    return prompts_dir() / name


def legacy_prompts_path(name: str = DEFAULT_PROMPT_SET) -> Path:
    """The prompt-set file belonging to the Phase-0 tree.

    The 3B corpus was generated against its own copy of the prompt sets, so
    reproducing a legacy run reads the prompts from beside that data rather than
    from the package.
    """
    return legacy_data_root() / PROMPTS_DIRNAME / name


def resolve_prompts_path(name: str = DEFAULT_PROMPT_SET) -> Path:
    """The prompt-set file to read: the package copy, or the Phase-0 copy.

    The package ships the prompt sets, so the first branch is the ordinary
    answer. A checkout that reads a Phase-0 tree without the package data beside
    it falls through to that tree's own copy, which is where the 3B corpus keeps
    the prompts it was generated from. Neither path is required to exist; the
    caller that opens the file reports the miss.
    """
    shipped = prompts_path(name)
    if shipped.exists():
        return shipped
    legacy = legacy_prompts_path(name)
    return legacy if legacy.exists() else shipped
