"""The vLLM lane's calibration pins, their verification and the fetch that honours them.

Every model whose fixtures ship must pin calibration files whose digests combine to
the calibration digest its fixtures record: the install check compares a host with
vectors reduced under exactly that calibration. The fetch is exercised against a
stand-in download, so nothing here touches the network: a file matching its pin is
kept, a missing one is fetched and verified, and a download that does not match is
deleted before the refusal.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from anamnesis.extraction.vllm import hub, runtime
from anamnesis.provenance import digest_of_shas


def test_every_shipped_fixture_set_records_the_digest_its_pins_combine_to():
    shipped = sorted(p.name for p in runtime.FIXTURES_ROOT.iterdir()
                     if (p / "fixtures.json").is_file())
    assert shipped, "no fixture set ships"
    for model in shipped:
        fixtures, _ = runtime.load_fixtures(model)
        pins = {name: digest for name, (_, digest) in hub.CALIBRATIONS[model].items()}
        assert digest_of_shas(pins) == fixtures.calibration_sha256, model
        assert set(pins) == set(runtime.CALIBRATION_FILES)


def _pinned(monkeypatch, files: dict[str, bytes]) -> None:
    monkeypatch.setitem(hub.CALIBRATIONS, "toy", {
        name: (len(data), hashlib.sha256(data).hexdigest()) for name, data in files.items()})


FILES = {"positional_means.npz": b"means", "pca_model.pkl": b"basis"}


def test_a_matching_directory_verifies(tmp_path, monkeypatch):
    _pinned(monkeypatch, FILES)
    for name, data in FILES.items():
        (tmp_path / name).write_bytes(data)
    hub.verify_calibration("toy", tmp_path)


def test_a_missing_or_altered_file_is_refused_by_name(tmp_path, monkeypatch):
    _pinned(monkeypatch, FILES)
    (tmp_path / "positional_means.npz").write_bytes(b"other")
    with pytest.raises(ValueError, match="positional_means.npz differs from its pin, "
                                         "pca_model.pkl missing"):
        hub.verify_calibration("toy", tmp_path)


def test_a_model_without_pins_is_refused():
    with pytest.raises(ValueError, match="no calibration is pinned"):
        hub.verify_calibration("absent", Path("."))
    with pytest.raises(ValueError, match="no calibration is pinned"):
        hub.fetch_calibration("absent", Path("."))


def _stand_in_download(monkeypatch, tmp_path, served: dict[str, bytes]) -> list[str]:
    import huggingface_hub

    requested = []

    def download(repo_id, filename, *, repo_type):
        assert (repo_id, repo_type) == (hub.CALIBRATION_REPO, "dataset")
        requested.append(filename)
        path = tmp_path / "hub-cache" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(served[Path(filename).name])
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return requested


def test_the_fetch_keeps_matching_files_and_downloads_the_rest(tmp_path, monkeypatch):
    _pinned(monkeypatch, FILES)
    target = tmp_path / "calib"
    target.mkdir()
    (target / "pca_model.pkl").write_bytes(FILES["pca_model.pkl"])
    requested = _stand_in_download(monkeypatch, tmp_path, FILES)
    assert hub.fetch_calibration("toy", target) == target
    assert requested == ["calibration/toy/positional_means.npz"]
    hub.verify_calibration("toy", target)


def test_a_download_that_misses_its_pin_is_deleted_and_refused(tmp_path, monkeypatch):
    _pinned(monkeypatch, FILES)
    target = tmp_path / "calib"
    _stand_in_download(monkeypatch, tmp_path, dict(FILES, **{"positional_means.npz": b"x"}))
    with pytest.raises(ValueError, match="does not match its pin"):
        hub.fetch_calibration("toy", target)
    assert not (target / "positional_means.npz").exists()


def test_the_default_directory_is_under_the_output_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ANAMNESIS_OUTPUTS", str(tmp_path))
    assert hub.default_calibration_dir("8b") == tmp_path / "vllm_calibration" / "8b"
