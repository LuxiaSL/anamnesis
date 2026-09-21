"""The numeric anchor is pure numpy, enforced here rather than asserted in a comment.

`state_extractor` and its reference are the definition of what a signature's numbers
are. That definition has to be reproducible on a machine with no GPU, no model
weights and no deep-learning stack — otherwise the fast paths have nothing to be
checked against, and a contributor cannot run the anchor at all. Under a
GPU-primary extraction lane the constraint carries more weight, not less: the lane
is only trustworthy because something independent of it says what the answer is.

Two readings, because either one alone can be fooled:

* **At runtime**, in a subprocess: import the modules and look at what actually
  landed in `sys.modules`. This catches a real import no matter how it is spelled,
  but says nothing about an import guarded behind a function that the import itself
  never calls.
* **Statically**, over the AST: follow every import inside the package, including
  the ones written inside functions, and collect what they reach. This catches a
  deferred import, and it keeps working in an environment where torch is not
  installed and so could not have leaked into `sys.modules` anyway.

The families are held to the same rule. They run over banked tensors, which is what
lets a feature set be revised without re-running a model.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "anamnesis"
REPO_ROOT = PACKAGE_ROOT.parent

FORBIDDEN_TOPS = frozenset({"torch", "transformers", "accelerate", "sklearn"})
"""Tops the anchor and the families must not reach: a model runtime, or a fitter.

`sklearn` is here with the rest because a fitted model belongs to calibration, which
hands the extractor arrays. An extractor that could fit would be able to fit on the
data it is measuring.
"""

ANCHOR_MODULES = (
    "anamnesis.extraction.state_extractor",
    "anamnesis.extraction.state_extractor_reference",
)

FAMILY_ENTRY_POINTS = (
    "anamnesis.extraction.feature_pipeline",
    "anamnesis.extraction.raw_saver",
    "anamnesis.feature_map",
)


def _module_path(name: str) -> Path | None:
    """The file a dotted `anamnesis.*` name lives in, or None when it is not ours."""
    if name != "anamnesis" and not name.startswith("anamnesis."):
        return None
    parts = name.split(".")[1:]
    direct = REPO_ROOT / "anamnesis" / Path(*parts).with_suffix(".py") if parts else None
    if direct is not None and direct.is_file():
        return direct
    package = REPO_ROOT / "anamnesis" / Path(*parts) / "__init__.py"
    return package if package.is_file() else None


def _resolve_relative(source: str, level: int, module: str | None) -> str:
    """The absolute name a `from .. import x` in `source` refers to."""
    anchor = source.split(".")
    if _module_path(source) is not None and not _module_path(source).name == "__init__.py":
        anchor = anchor[:-1]
    base = anchor[: len(anchor) - (level - 1)] if level > 1 else anchor
    return ".".join([*base, module]) if module else ".".join(base)


def _imports_of(name: str) -> set[str]:
    """Every module name imported anywhere in one of our modules, functions included."""
    path = _module_path(name)
    if path is None:
        return set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                found.add(_resolve_relative(name, node.level, node.module))
            elif node.module:
                found.add(node.module)
    return found


def static_closure(roots: tuple[str, ...]) -> tuple[set[str], dict[str, set[str]]]:
    """Walk our own modules from `roots`; return (internal modules, external top → importers)."""
    seen: set[str] = set()
    external: dict[str, set[str]] = {}
    queue = list(roots)
    while queue:
        current = queue.pop()
        if current in seen or _module_path(current) is None:
            continue
        seen.add(current)
        for target in _imports_of(current):
            if _module_path(target) is not None:
                queue.append(target)
                continue
            if target.startswith("anamnesis"):
                pytest.fail(f"{current} imports {target}, which is not a module in this package")
            external.setdefault(target.split(".")[0], set()).add(current)
    return seen, external


def _runtime_leaks(modules: tuple[str, ...]) -> list[str]:
    """Forbidden tops that end up in `sys.modules` when `modules` are imported."""
    program = (
        "import sys\n"
        f"for name in {list(modules)!r}:\n"
        "    __import__(name)\n"
        f"forbidden = {sorted(FORBIDDEN_TOPS)!r}\n"
        "print(' '.join(sorted({m.split('.')[0] for m in sys.modules "
        "if m.split('.')[0] in forbidden})))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300, check=False,
    )
    assert proc.returncode == 0, f"import of {modules} failed:\n{proc.stderr}"
    return proc.stdout.split()


# ── the anchor ────────────────────────────────────────────────────────────────


def test_anchor_imports_no_model_runtime_at_runtime() -> None:
    leaked = _runtime_leaks(ANCHOR_MODULES)
    assert leaked == [], f"importing the numeric anchor pulled in {leaked}"


def test_anchor_imports_no_model_runtime_statically() -> None:
    _, external = static_closure(ANCHOR_MODULES)
    hits = {top: sorted(where) for top, where in external.items() if top in FORBIDDEN_TOPS}
    assert hits == {}, f"the numeric anchor's import closure reaches {hits}"


def test_anchor_depends_only_on_configuration() -> None:
    """Inside the package the anchor reads configuration and nothing else.

    The tighter statement is what keeps the constraint true as the package grows:
    a later module in `anamnesis.extraction` may import a model runtime, and this
    holds the anchor apart from it by name rather than by hoping nobody links them.
    """
    internal, _ = static_closure(ANCHOR_MODULES)
    allowed = set(ANCHOR_MODULES)
    unexpected = sorted(
        name for name in internal
        if name not in allowed and not name.startswith("anamnesis.config")
    )
    assert unexpected == [], f"the anchor reaches {unexpected}; it may only reach configuration"


def test_anchor_third_party_surface_is_three_packages() -> None:
    """Outside the standard library the anchor needs numpy, scipy and pydantic.

    Stating the whole list, rather than only the forbidden part, is what makes a
    new dependency on the anchor a decision somebody takes on purpose: an install
    that can run this can reproduce the numbers.
    """
    _, external = static_closure(ANCHOR_MODULES)
    third_party = {top for top in external if top not in sys.stdlib_module_names}
    assert third_party == {"numpy", "scipy", "pydantic"}, (
        f"the anchor's third-party imports are {sorted(third_party)}"
    )


# ── the families ──────────────────────────────────────────────────────────────


def _family_modules() -> list[str]:
    directory = PACKAGE_ROOT / "extraction" / "feature_families"
    return [
        f"anamnesis.extraction.feature_families.{path.stem}"
        for path in sorted(directory.glob("*.py"))
    ]


def test_every_family_is_discovered() -> None:
    """The check below is only worth anything if it sees the whole package."""
    families = _family_modules()
    assert "anamnesis.extraction.feature_families.attn_res" in families
    assert "anamnesis.extraction.feature_families.path_signature" in families
    assert len(families) >= 12


@pytest.mark.parametrize("module", _family_modules())
def test_family_imports_no_model_runtime_statically(module: str) -> None:
    _, external = static_closure((module,))
    hits = {top: sorted(where) for top, where in external.items() if top in FORBIDDEN_TOPS}
    assert hits == {}, f"{module}'s import closure reaches {hits}"


def test_recompute_path_imports_no_model_runtime_at_runtime() -> None:
    """The recompute-from-raw path is the CPU lane; it loads no model runtime."""
    leaked = _runtime_leaks(FAMILY_ENTRY_POINTS + tuple(_family_modules()))
    assert leaked == [], f"importing the recompute path pulled in {leaked}"


def test_recompute_path_imports_no_model_runtime_statically() -> None:
    _, external = static_closure(FAMILY_ENTRY_POINTS + tuple(_family_modules()))
    hits = {top: sorted(where) for top, where in external.items() if top in FORBIDDEN_TOPS}
    assert hits == {}, f"the recompute path's import closure reaches {hits}"
