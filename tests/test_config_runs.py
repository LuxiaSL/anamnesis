"""The run registry: what the file holds, and what a name resolves to."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from anamnesis.config import paths
from anamnesis.config.runs import (
    RUNS_FILE,
    ResolvedRun,
    RunSpec,
    RunsRegistryError,
    UnknownRunError,
    get_run,
    load_runs,
    resolve_run,
    run_names,
)

EXPECTED_RUNS: dict[str, tuple[str, str, int]] = {
    "8b_baseline": ("outputs", "runs/run_8b_baseline/signatures", 0),
    "3b_run4": ("legacy", "outputs/runs/run4_format_controlled/signatures", 0),
    "8b_v2": ("outputs", "runs/8b_fat_01/signatures_v2", 2),
    "3b_v2": ("outputs", "runs/3b_fat_01/signatures_v2", 1),
    "synthetic_demo": ("outputs", "runs/synthetic_demo/signatures", 0),
}


@pytest.fixture
def data_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv(paths.OUTPUTS_ENV, str(tmp_path / "outputs"))
    monkeypatch.setenv(paths.LEGACY_DATA_ENV, str(tmp_path / "phase_0"))
    return tmp_path


def test_registry_ships_with_the_package() -> None:
    assert RUNS_FILE.is_file()
    assert RUNS_FILE.parent == paths.package_root() / "config"


def test_registry_holds_the_named_runs() -> None:
    runs = load_runs()
    assert set(runs) == set(EXPECTED_RUNS)
    assert run_names() == tuple(runs)
    for name, (root, signature_dir, addons) in EXPECTED_RUNS.items():
        spec = runs[name]
        assert spec.root == root
        assert spec.signature_dir == signature_dir
        assert len(spec.addon_dirs) == addons
        assert spec.description


def test_a_run_resolves_under_the_outputs_root(data_roots: Path) -> None:
    resolved = resolve_run("8b_v2")
    outputs = data_roots / "outputs"
    assert resolved.signature_dir == outputs / "runs" / "8b_fat_01" / "signatures_v2"
    assert resolved.addon_dirs == (
        outputs / "runs" / "8b_fat_01" / "signatures_v2_addon",
        outputs / "runs" / "8b_fat_01" / "signatures_v2_contrastive",
    )


def test_the_legacy_run_resolves_through_the_hatch(data_roots: Path) -> None:
    resolved = resolve_run("3b_run4")
    assert resolved.signature_dir == (
        data_roots / "phase_0" / "outputs" / "runs" / "run4_format_controlled" / "signatures"
    )


def test_resolution_follows_a_moved_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(paths.OUTPUTS_ENV, str(tmp_path / "first"))
    first = resolve_run("3b_v2").signature_dir
    monkeypatch.setenv(paths.OUTPUTS_ENV, str(tmp_path / "second"))
    second = resolve_run("3b_v2").signature_dir
    assert first != second
    assert first.relative_to(tmp_path / "first") == second.relative_to(tmp_path / "second")


def test_missing_dirs_reports_signatures_first(data_roots: Path) -> None:
    resolved = resolve_run("8b_v2")
    assert resolved.missing_dirs() == (resolved.signature_dir, *resolved.addon_dirs)
    for directory in resolved.missing_dirs():
        directory.mkdir(parents=True)
    assert resolve_run("8b_v2").missing_dirs() == ()


def test_present_run_reports_nothing_missing(data_roots: Path) -> None:
    resolved = resolve_run("3b_run4")
    resolved.signature_dir.mkdir(parents=True)
    assert resolve_run("3b_run4").missing_dirs() == ()


def test_unknown_run_names_the_known_ones() -> None:
    with pytest.raises(UnknownRunError) as caught:
        get_run("8b_v9")
    message = str(caught.value)
    assert "8b_v9" in message
    for name in EXPECTED_RUNS:
        assert name in message


def test_an_absent_registry_names_the_path(tmp_path: Path) -> None:
    missing = tmp_path / "runs.json"
    with pytest.raises(RunsRegistryError) as caught:
        load_runs(missing)
    assert str(missing) in str(caught.value)


def test_invalid_json_names_the_line(tmp_path: Path) -> None:
    broken = tmp_path / "runs.json"
    broken.write_text('{"runs": {,}}', encoding="utf-8")
    with pytest.raises(RunsRegistryError) as caught:
        load_runs(broken)
    assert "invalid JSON" in str(caught.value)


@pytest.mark.parametrize(
    "payload",
    [
        {"entries": {}},
        {"runs": {"a": {"name": "a", "root": "outputs"}}},
        {"runs": {"a": {"name": "a", "root": "elsewhere", "signature_dir": "x"}}},
        {"runs": {"a": {"name": "a", "root": "outputs", "signature_dir": "x", "extra": 1}}},
    ],
)
def test_a_registry_of_the_wrong_shape_is_refused(tmp_path: Path, payload: dict[str, Any]) -> None:
    path = tmp_path / "runs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RunsRegistryError) as caught:
        load_runs(path)
    assert "not a run registry" in str(caught.value)


def test_a_row_whose_name_disagrees_with_its_key_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "runs.json"
    path.write_text(
        json.dumps({"runs": {"a": {"name": "b", "root": "outputs", "signature_dir": "x"}}}),
        encoding="utf-8",
    )
    with pytest.raises(RunsRegistryError) as caught:
        load_runs(path)
    assert "'b'" in str(caught.value)


def test_a_custom_registry_loads_from_an_explicit_path(
    data_roots: Path, tmp_path: Path
) -> None:
    path = tmp_path / "runs.json"
    path.write_text(
        json.dumps(
            {
                "runs": {
                    "probe": {
                        "name": "probe",
                        "root": "outputs",
                        "signature_dir": "runs/probe/signatures",
                        "addon_dirs": ["runs/probe/signatures_addon"],
                        "description": "A run declared outside the shipped registry.",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    assert run_names(path) == ("probe",)
    resolved = get_run("probe", path).resolve()
    assert resolved.signature_dir == data_roots / "outputs" / "runs" / "probe" / "signatures"
    assert len(resolved.addon_dirs) == 1


def test_a_row_cannot_address_outside_its_root() -> None:
    spec = RunSpec(name="escape", root="outputs", signature_dir="../elsewhere")
    with pytest.raises(paths.PathResolutionError):
        spec.resolve()


def test_registry_rows_and_resolved_runs_are_immutable(data_roots: Path) -> None:
    spec = get_run("8b_v2")
    resolved = spec.resolve()
    assert isinstance(resolved, ResolvedRun)
    with pytest.raises(Exception):
        spec.signature_dir = "elsewhere"
    with pytest.raises(Exception):
        resolved.signature_dir = Path("elsewhere")
