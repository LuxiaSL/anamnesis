"""What importing the package does, and what it must not need to do it.

Two properties are load-bearing beyond tidiness:

* Importing :mod:`anamnesis` runs no work and pulls in no submodule, so a reader
  of one part of the instrument pays for that part alone.
* Configuration imports without torch. The numeric anchor, the analysis layer and
  every test describe runs on machines with no accelerator and no weights, which
  only holds while the description of a run is free of the framework that
  executes it.

Torch is a hard dependency of the instrument, because the capture layer is a model
runtime and guarding its imports would be a redesign rather than a metadata
choice. That makes the import-graph claims here and in
`tests/test_extraction_purity.py` load-bearing rather than incidental: torch is
installed and importable in this environment, so a probe finding it absent from
`sys.modules` found a real property of the graph, not an empty shelf. The check
that torch is installed is therefore part of the pair — without it, every claim
below about what an import does not pull in could pass for the wrong reason.
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


def test_torch_is_installed_so_the_import_free_claims_are_not_vacuous() -> None:
    """The counterpart to the probe above: torch is here, and still not reached.

    A probe that looks for torch in `sys.modules` passes trivially when torch
    cannot be imported at all. The capture layer needs it, so it is installed,
    and that is what makes "configuration imports without torch" a statement
    about the import graph.
    """
    result = run_probe(
        "import importlib.util, sys; "
        "sys.exit(0 if importlib.util.find_spec('torch') is not None else 3)"
    )
    assert result.returncode == 0, (
        "torch is not installed, so every claim here about what an import does not "
        "pull in would pass for the wrong reason"
    )


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


def test_naming_the_judging_package_pulls_no_vendor_client() -> None:
    """Judging is the one layer that talks to a provider, and it says so by extra.

    The two client libraries ship in `[judge]` and are imported at first use, so
    reading the judging package's docstring, or the prompt table, costs nothing
    and needs no account.
    """
    result = run_probe(
        "import anamnesis.judging, anamnesis.judging.prompts, sys; "
        "pulled = [m for m in sys.modules if m.split('.')[0] in ('anthropic', 'requests')]; "
        "assert not pulled, pulled"
    )
    assert result.returncode == 0, result.stderr


def test_the_harness_imports_without_the_provider_libraries_installed() -> None:
    """The import-time claim is not vacuous only because the harness is importable
    in an environment that has neither library, which is this one."""
    result = run_probe(
        "import importlib.util as u, anamnesis.judging.harness as h, sys; "
        "missing = [m for m in ('anthropic', 'requests') if u.find_spec(m) is None]; "
        "assert missing, 'both client libraries are installed, so this proves nothing'; "
        "pulled = [m for m in sys.modules if m.split('.')[0] in ('anthropic', 'requests')]; "
        "assert not pulled, pulled; "
        "assert h.AnthropicBackend().api_key_present() in (True, False)"
    )
    assert result.returncode == 0, result.stderr


def test_the_public_surface_names_are_the_types_downstream_builds_on() -> None:
    for symbol in (ExperimentConfig, ModelConfig, ModelPreset, RunSpec):
        assert symbol.__module__.startswith("anamnesis.config.")
    assert hasattr(ModelConfig, "from_preset")
    assert hasattr(ExperimentConfig, "from_preset")
