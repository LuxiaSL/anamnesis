"""Check an installed ``anamnesis`` — the things only an installation can be wrong about.

The test suite runs against the source tree, where ``pythonpath = ["."]`` puts the
repository on the import path. That hides three kinds of packaging error, because in a
checkout every file is present whether or not the package declares it:

* **data that does not ship.** The run registry and the prompt sets are read at
  runtime. A wheel without them installs, imports, and then cannot name a run.
* **a default path inside the installation.** Artifacts written under the package
  directory land in ``site-packages``, where an upgrade removes them and every project
  on the machine shares them.
* **a module that ships by accident.** ``tools`` checks this repository; it is not part
  of the instrument, and a wheel that carries it puts a top-level ``tools`` on the
  import path of everything the user installs.

So this runs outside the checkout, against the installed package, and reports every
failure it finds rather than stopping at the first — a packaging fix is cheaper when the
whole list arrives at once.

Run it as ``python tools/smoke_installed.py`` with the interpreter that has the wheel
installed, from any directory that is not the repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path


def check_imports() -> list[str]:
    """The package and the submodules a caller reaches for first."""
    failures: list[str] = []
    modules = (
        "anamnesis",
        "anamnesis.config",
        "anamnesis.feature_map",
        "anamnesis.shortfall",
        "anamnesis.provenance",
        "anamnesis.modes",
        "anamnesis.extraction.state_extractor",
        "anamnesis.analysis.gauntlet",
    )
    for name in modules:
        try:
            __import__(name)
        except Exception as exc:  # noqa: BLE001 - the failure is the finding
            failures.append(f"{name} does not import from an installation: {exc!r}")
    return failures


def check_shipped_data() -> list[str]:
    """The data files the package reads at runtime, resolved the way it resolves them."""
    failures: list[str] = []
    try:
        from anamnesis.config import paths, run_names
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return [f"anamnesis.config does not import: {exc!r}"]

    for path in (
        paths.prompts_path(),
        paths.prompts_path("prompt_sets_narrative.json"),
    ):
        if not path.is_file():
            failures.append(f"shipped data missing from the installation: {path}")

    try:
        names = run_names()
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        failures.append(f"the run registry does not load from the installation: {exc!r}")
    else:
        if not names:
            failures.append("the run registry loaded but names no runs")

    return failures


def check_outputs_root_is_outside_the_installation() -> list[str]:
    """The default artifact location must not be the directory the code lives in."""
    try:
        from anamnesis.config import paths
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return [f"anamnesis.config does not import: {exc!r}"]

    package = paths.package_root().resolve()
    root = paths.outputs_root().resolve()
    if package == root or package in root.parents:
        return [f"the default outputs root is inside the installation: {root}"]
    return []


def check_gate_tooling_did_not_ship() -> list[str]:
    """``tools`` belongs to the repository, not to the wheel."""
    try:
        import tools
    except ImportError:
        return []
    location = getattr(tools, "__file__", "an unknown location")
    return [f"the gate tooling shipped in the wheel and is importable from {location}"]


def check_not_running_from_the_checkout() -> list[str]:
    """Guard the guard: inside the repository, every check above passes for free."""
    if Path("pyproject.toml").is_file() and Path("anamnesis").is_dir():
        return [
            "this ran from the repository root, where the source tree satisfies every "
            "check whether or not the wheel does; run it from another directory"
        ]
    return []


def main() -> int:
    failures: list[str] = []
    failures += check_not_running_from_the_checkout()
    failures += check_imports()
    failures += check_shipped_data()
    failures += check_outputs_root_is_outside_the_installation()
    failures += check_gate_tooling_did_not_ship()

    if failures:
        print(f"installed-package smoke test FAILED with {len(failures)} finding(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("installed-package smoke test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
