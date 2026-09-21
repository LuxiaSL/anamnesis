"""The three data roots, their environment hatches, and path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from anamnesis.config import paths


def test_package_root_is_the_installed_package() -> None:
    root = paths.package_root()
    assert root.name == "anamnesis"
    assert (root / "config" / "paths.py").is_file()


def test_outputs_root_defaults_beside_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(paths.OUTPUTS_ENV, raising=False)
    assert paths.outputs_root() == paths.package_root() / "outputs"


def test_outputs_root_follows_its_environment_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(paths.OUTPUTS_ENV, str(tmp_path / "store"))
    assert paths.outputs_root() == tmp_path / "store"


def test_blank_environment_variable_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(paths.OUTPUTS_ENV, "   ")
    assert paths.outputs_root() == paths.package_root() / "outputs"


def test_legacy_data_root_defaults_beside_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(paths.LEGACY_DATA_ENV, raising=False)
    assert paths.legacy_data_root() == paths.package_root().parent / "phase_0"


def test_legacy_data_root_is_the_hatch_onto_phase_zero_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(paths.LEGACY_DATA_ENV, str(tmp_path / "phase_0"))
    assert paths.legacy_data_root() == tmp_path / "phase_0"


def test_roots_are_read_at_call_time(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(paths.OUTPUTS_ENV, str(tmp_path / "one"))
    first = paths.outputs_root()
    monkeypatch.setenv(paths.OUTPUTS_ENV, str(tmp_path / "two"))
    assert paths.outputs_root() != first


def test_data_root_covers_every_declared_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(paths.OUTPUTS_ENV, str(tmp_path / "out"))
    monkeypatch.setenv(paths.LEGACY_DATA_ENV, str(tmp_path / "legacy"))
    resolved = {token: paths.data_root(token) for token in paths.DATA_ROOTS}
    assert resolved["outputs"] == tmp_path / "out"
    assert resolved["legacy"] == tmp_path / "legacy"
    assert resolved["package"] == paths.package_root()


def test_unknown_root_names_the_known_ones() -> None:
    with pytest.raises(paths.PathResolutionError) as caught:
        paths.data_root("elsewhere")  # type: ignore[arg-type]
    message = str(caught.value)
    assert "elsewhere" in message
    for token in paths.DATA_ROOTS:
        assert token in message


def test_resolve_data_path_joins_posix_segments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(paths.OUTPUTS_ENV, str(tmp_path))
    assert paths.resolve_data_path("outputs", "runs/8b_fat_01/signatures_v2") == (
        tmp_path / "runs" / "8b_fat_01" / "signatures_v2"
    )


@pytest.mark.parametrize("relative", ["", "   ", "/absolute/path", "runs/../../escape"])
def test_resolve_data_path_rejects_paths_that_leave_their_root(relative: str) -> None:
    with pytest.raises(paths.PathResolutionError):
        paths.resolve_data_path("outputs", relative)


def test_run_name_and_run_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(paths.OUTPUTS_ENV, str(tmp_path))
    monkeypatch.delenv(paths.RUN_NAME_ENV, raising=False)
    assert paths.run_name() == paths.DEFAULT_RUN_NAME
    assert paths.run_outputs_dir() == tmp_path / "runs" / paths.DEFAULT_RUN_NAME
    monkeypatch.setenv(paths.RUN_NAME_ENV, "8b_fat_01")
    assert paths.run_name() == "8b_fat_01"
    assert paths.run_outputs_dir() == tmp_path / "runs" / "8b_fat_01"
    assert paths.run_outputs_dir("other") == tmp_path / "runs" / "other"


def test_prompt_sets_ship_with_the_package() -> None:
    assert paths.prompts_dir() == paths.package_root() / "prompts"
    assert paths.prompts_path().is_file()
    assert paths.prompts_path().name == paths.DEFAULT_PROMPT_SET


def test_prompt_resolution_prefers_the_package_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    legacy = tmp_path / "phase_0" / "prompts"
    legacy.mkdir(parents=True)
    (legacy / paths.DEFAULT_PROMPT_SET).write_text("{}", encoding="utf-8")
    monkeypatch.setenv(paths.LEGACY_DATA_ENV, str(tmp_path / "phase_0"))
    assert paths.resolve_prompts_path() == paths.prompts_path()


def test_prompt_resolution_falls_through_to_the_legacy_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    legacy = tmp_path / "phase_0" / "prompts"
    legacy.mkdir(parents=True)
    (legacy / "prompt_sets_run3.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv(paths.LEGACY_DATA_ENV, str(tmp_path / "phase_0"))
    assert paths.resolve_prompts_path("prompt_sets_run3.json") == legacy / "prompt_sets_run3.json"


def test_prompt_resolution_reports_the_package_path_when_neither_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(paths.LEGACY_DATA_ENV, str(tmp_path / "absent"))
    resolved = paths.resolve_prompts_path("prompt_sets_absent.json")
    assert resolved == paths.prompts_path("prompt_sets_absent.json")
    assert not resolved.exists()
