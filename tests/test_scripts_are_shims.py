"""The rule that keeps `anamnesis/scripts/` from growing back into a pile.

The extraction repository's script directory reached two hundred and forty-four
files, most of them near-copies of each other, because a script was the easiest
place to put a function and the next script's author copied it. The structural fix
is not discipline, it is a boundary: **capability lives in the package, and a
script may not define a function another script imports.**

This test is that boundary, made mechanical, in four assertions over the parsed
sources:

1. No module under `anamnesis/scripts/` imports another one. This is the rule as
   written, and it is the one that matters: the moment one script imports a
   sibling, the sibling is a library filed in the wrong place.
2. No module outside `anamnesis/scripts/` imports a script. Capability never
   flows *upward* out of an entry point, which is the same rule seen from the
   package's side.
3. A function a script defines is used by that script and by its own test, and
   nowhere else. This is the thinness proxy, and it is a judgement: a diagnostic
   command's own logic is legitimately its own — `qualify_box.py`'s verdict
   rendering is not capability anybody else wants — while a function a second
   reader needs is capability, and belongs in the package where a stranger can
   find it. Reading "used by its own test" as allowed is deliberate: pinning a
   command's refusals by test is how the qualification boundaries stay honest,
   and forbidding it would push those tests into subprocess calls that assert
   less.
4. Two scripts do not define the same name, apart from the command-line roles
   every script has. A shared name across two entry points is the copy-paste
   signature itself — it is how the calibration reader came to exist twice — so
   it fails here rather than being found later by eye.

What this test does not check is duplication under two different names. Nothing
mechanical catches that; the port map is where it is recorded when it is found.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO / "anamnesis" / "scripts"
PACKAGE_DIR = REPO / "anamnesis"
TESTS_DIR = REPO / "tests"

SCRIPTS_PREFIX = "anamnesis.scripts"

CLI_ROLES = frozenset({"main", "parser", "build_parser"})
"""Names every entry point is allowed to share: the two argument-parsing and
dispatch roles a command has by being a command."""


def _script_paths() -> list[Path]:
    return sorted(p for p in SCRIPTS_DIR.glob("*.py") if p.name != "__init__.py")


def _module_name(path: Path) -> str:
    return ".".join(path.relative_to(REPO).with_suffix("").parts)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _imported_modules(tree: ast.Module) -> set[str]:
    """Every dotted module name the source imports, at any indentation.

    Deferred imports inside a function count: an import that happens at call time
    is still an edge, and the entry points here defer their heavy imports by
    habit.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def _imported_names_from(tree: ast.Module, module: str) -> set[str]:
    """The names a source imports out of one specific module."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module and not node.level:
            found.update(alias.name for alias in node.names)
    return found


def _public_functions(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }


def test_there_are_scripts_to_check() -> None:
    """A rule over an empty directory proves nothing, so say how many it covers."""
    assert len(_script_paths()) >= 2


@pytest.mark.parametrize("path", _script_paths(), ids=lambda p: p.name)
def test_no_script_imports_a_sibling_script(path: Path) -> None:
    """Assertion 1 — the rule as written."""
    imports = _imported_modules(_parse(path))
    own = _module_name(path)
    siblings = {
        name for name in imports
        if name.startswith(f"{SCRIPTS_PREFIX}.") and not name.startswith(own)
    }
    assert not siblings, (
        f"{path.name} imports {sorted(siblings)}. A script that another script needs is a "
        f"library filed in the wrong place: move the capability into the package and have "
        f"both entry points call it there."
    )


def test_no_package_module_imports_a_script() -> None:
    """Assertion 2 — capability does not flow upward out of an entry point."""
    offenders: dict[str, list[str]] = {}
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        if SCRIPTS_DIR in path.parents or path.parent == SCRIPTS_DIR:
            continue
        reached = sorted(
            name for name in _imported_modules(_parse(path))
            if name == SCRIPTS_PREFIX or name.startswith(f"{SCRIPTS_PREFIX}.")
        )
        if reached:
            offenders[_module_name(path)] = reached
    assert not offenders, (
        f"package modules import scripts: {offenders}. A package module that needs a "
        f"script's logic needs that logic in the package."
    )


def test_a_scripts_functions_are_used_by_it_and_its_own_test() -> None:
    """Assertion 3 — the thinness proxy, as stated in this module's header."""
    scripts = {_module_name(p): p for p in _script_paths()}
    offenders: dict[str, dict[str, list[str]]] = {}
    for reader in sorted(TESTS_DIR.glob("*.py")) + [
        p for p in PACKAGE_DIR.rglob("*.py") if p.parent != SCRIPTS_DIR
    ]:
        tree = _parse(reader)
        for module, script in scripts.items():
            names = _imported_names_from(tree, module)
            if not names:
                continue
            expected_test = TESTS_DIR / f"test_{script.stem}.py"
            if reader == expected_test:
                continue
            offenders.setdefault(module, {})[str(reader.relative_to(REPO))] = sorted(names)
    assert not offenders, (
        f"a script's functions are read from outside the script and its own test: "
        f"{offenders}. That makes them capability two readers depend on, so they belong "
        f"in the package."
    )


def test_no_two_scripts_define_the_same_name() -> None:
    """Assertion 4 — the copy-paste signature, caught by collision."""
    seen: dict[str, list[str]] = {}
    for path in _script_paths():
        tree = _parse(path)
        defined = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        for name in defined - CLI_ROLES:
            seen.setdefault(name, []).append(path.name)
    shared = {name: files for name, files in seen.items() if len(files) > 1}
    assert not shared, (
        f"the same name is defined in more than one script: {shared}. Two entry points "
        f"holding one name is how a reader came to exist twice; give the package the one "
        f"definition and import it."
    )


@pytest.mark.parametrize("path", _script_paths(), ids=lambda p: p.name)
def test_every_script_has_a_main(path: Path) -> None:
    """A script is a command. One without an entry point is a library in disguise."""
    assert "main" in _public_functions(_parse(path)), (
        f"{path.name} defines no main(); a module under scripts/ is a command"
    )
