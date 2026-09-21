"""What importing the package does, and what it must not need to do it.

Two properties are load-bearing beyond tidiness:

* Importing :mod:`anamnesis` runs no work and pulls in no submodule, so a reader
  of one part of the instrument pays for that part alone.
* Configuration imports without torch. The numeric anchor, the analysis layer and
  every test describe runs on machines with no accelerator and no weights, which
  only holds while the description of a run is free of the framework that
  executes it. The development environment installs no torch, so the suite that
  proves this is the suite as run.
"""

from __future__ import annotations

import subprocess
import sys

import anamnesis
from anamnesis.config import ExperimentConfig, ModelConfig, ModelPreset, RunSpec


def run_probe(source: str) -> subprocess.CompletedProcess[str]:
    """Execute a probe in a fresh interpreter, where sys.modules starts clean."""
    return subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )


def test_the_package_root_exports_nothing_and_imports_nothing() -> None:
    assert anamnesis.__all__ == ()
    result = run_probe(
        "import anamnesis, sys; "
        "pulled = [m for m in sys.modules if m.startswith('anamnesis.')]; "
        "assert not pulled, pulled"
    )
    assert result.returncode == 0, result.stderr


def test_configuration_imports_without_torch() -> None:
    result = run_probe(
        "import anamnesis.config, sys; "
        "assert 'torch' not in sys.modules, 'torch reached the configuration import'"
    )
    assert result.returncode == 0, result.stderr


def test_torch_is_absent_from_the_environment_this_suite_runs_in() -> None:
    result = run_probe(
        "import importlib.util, sys; "
        "sys.exit(0 if importlib.util.find_spec('torch') is None else 3)"
    )
    assert result.returncode == 0, "torch is installed, so the import-free claim is untested here"


def test_configuration_does_not_import_the_mode_prompts() -> None:
    """The mode sets are a sibling of configuration, not a dependency of it.

    A configuration module that imported the modes package would execute that
    package's exports on every import of the instrument, which is how a module
    left behind in the frozen record becomes a hard dependency of everything.
    """
    result = run_probe(
        "import anamnesis.config, sys; "
        "pulled = [m for m in sys.modules if m.startswith('anamnesis.modes')]; "
        "assert not pulled, pulled"
    )
    assert result.returncode == 0, result.stderr


def test_the_public_surface_names_are_the_types_downstream_builds_on() -> None:
    for symbol in (ExperimentConfig, ModelConfig, ModelPreset, RunSpec):
        assert symbol.__module__.startswith("anamnesis.config.")
    assert hasattr(ModelConfig, "from_preset")
    assert hasattr(ExperimentConfig, "from_preset")
