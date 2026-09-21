"""Unit tests for the G1 array-content manifest and its diff."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.g1_hash_manifest import (
    FORMAT_LINE,
    ManifestError,
    build_manifest,
    compare_manifests,
    main,
    parse_manifest,
    render_manifest,
)


def make_signatures(
    directory: Path,
    features: np.ndarray | None = None,
    sidecar: dict[str, object] | None = None,
) -> Path:
    """A two-generation signatures directory with npz arrays and JSON sidecars."""
    directory.mkdir(parents=True, exist_ok=True)
    first = np.arange(4, dtype=np.float32) if features is None else features
    np.savez(
        directory / "gen_000.npz",
        features=first,
        feature_names=np.array(["a", "b", "c", "d"], dtype="<U4"),
    )
    np.savez(directory / "gen_001.npz", features=np.ones(3, dtype=np.float64))
    payload = {"gen_id": "gen_000", "mode": "socratic"} if sidecar is None else sidecar
    (directory / "gen_000.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


def test_manifest_covers_every_array_and_sidecar(tmp_path: Path) -> None:
    signatures = make_signatures(tmp_path / "sig")
    entries = build_manifest(signatures)
    identities = [entry.identity for entry in entries]
    assert identities == [
        ("gen_000.npz", "feature_names"),
        ("gen_000.npz", "features"),
        ("gen_001.npz", "features"),
        ("gen_000.json", "json:content"),
    ]
    features = next(e for e in entries if e.identity == ("gen_000.npz", "features"))
    assert features.dtype == "float32"
    assert features.shape == "(4,)"
    assert len(features.digest) == 64


def test_manifest_text_is_deterministic(tmp_path: Path) -> None:
    signatures = make_signatures(tmp_path / "sig")
    first = render_manifest(build_manifest(signatures))
    second = render_manifest(build_manifest(signatures))
    assert first == second
    assert first.splitlines()[0] == FORMAT_LINE


def test_identical_directories_compare_equal(tmp_path: Path) -> None:
    left = build_manifest(make_signatures(tmp_path / "a"))
    right = build_manifest(make_signatures(tmp_path / "b"))
    differences, only_left, only_right = compare_manifests(left, right)
    assert (differences, only_left, only_right) == ([], [], [])


def test_a_changed_value_is_caught(tmp_path: Path) -> None:
    left = build_manifest(make_signatures(tmp_path / "a"))
    changed = np.arange(4, dtype=np.float32)
    changed[2] = np.nextafter(np.float32(2.0), np.float32(3.0))
    right = build_manifest(make_signatures(tmp_path / "b", features=changed))
    differences, _, _ = compare_manifests(left, right)
    assert [(d.key, d.field) for d in differences] == [("features", "digest")]


def test_a_changed_dtype_is_caught(tmp_path: Path) -> None:
    left = build_manifest(make_signatures(tmp_path / "a"))
    right = build_manifest(make_signatures(tmp_path / "b", features=np.arange(4, dtype=np.float64)))
    differences, _, _ = compare_manifests(left, right)
    fields = {d.field for d in differences}
    assert "dtype" in fields


def test_a_changed_shape_is_caught(tmp_path: Path) -> None:
    left = build_manifest(make_signatures(tmp_path / "a"))
    right = build_manifest(make_signatures(tmp_path / "b", features=np.arange(5, dtype=np.float32)))
    differences, _, _ = compare_manifests(left, right)
    fields = {d.field for d in differences}
    assert "shape" in fields


def test_a_missing_array_is_reported_on_the_side_that_has_it(tmp_path: Path) -> None:
    left = build_manifest(make_signatures(tmp_path / "a"))
    right_dir = make_signatures(tmp_path / "b")
    np.savez(right_dir / "gen_001.npz", features=np.ones(3, dtype=np.float64), extra=np.zeros(2))
    right = build_manifest(right_dir)
    differences, only_left, only_right = compare_manifests(left, right)
    assert differences == []
    assert only_left == []
    assert only_right == [("gen_001.npz", "extra")]


def test_sidecar_key_order_does_not_register_as_drift(tmp_path: Path) -> None:
    left_dir = make_signatures(tmp_path / "a")
    right_dir = make_signatures(tmp_path / "b")
    (right_dir / "gen_000.json").write_text(
        json.dumps({"mode": "socratic", "gen_id": "gen_000"}), encoding="utf-8"
    )
    differences, _, _ = compare_manifests(build_manifest(left_dir), build_manifest(right_dir))
    assert differences == []


def test_sidecar_value_change_is_drift(tmp_path: Path) -> None:
    left_dir = make_signatures(tmp_path / "a")
    right_dir = make_signatures(tmp_path / "b", sidecar={"gen_id": "gen_000", "mode": "linear"})
    differences, _, _ = compare_manifests(build_manifest(left_dir), build_manifest(right_dir))
    assert [(d.name, d.key) for d in differences] == [("gen_000.json", "json:content")]


def test_skip_json_omits_sidecars(tmp_path: Path) -> None:
    signatures = make_signatures(tmp_path / "sig")
    entries = build_manifest(signatures, include_json=False)
    assert all(entry.key != "json:content" for entry in entries)


def test_empty_directory_fails_loudly(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ManifestError):
        build_manifest(empty)


def test_missing_directory_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(ManifestError):
        build_manifest(tmp_path / "absent")


def test_corrupt_npz_names_the_file(tmp_path: Path) -> None:
    signatures = make_signatures(tmp_path / "sig")
    (signatures / "gen_002.npz").write_bytes(b"not an npz")
    with pytest.raises(ManifestError) as caught:
        build_manifest(signatures)
    assert "gen_002.npz" in str(caught.value)


def test_invalid_sidecar_json_names_the_file(tmp_path: Path) -> None:
    signatures = make_signatures(tmp_path / "sig")
    (signatures / "gen_001.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ManifestError) as caught:
        build_manifest(signatures)
    assert "gen_001.json" in str(caught.value)


def test_manifest_round_trips(tmp_path: Path) -> None:
    signatures = make_signatures(tmp_path / "sig")
    entries = build_manifest(signatures)
    path = tmp_path / "manifest.txt"
    path.write_text(render_manifest(entries), encoding="utf-8")
    assert parse_manifest(path) == entries


def test_manifest_without_the_format_line_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "manifest.txt"
    path.write_text("gen_000.npz\tfeatures\tfloat32\t(4,)\tdeadbeef\n", encoding="utf-8")
    with pytest.raises(ManifestError):
        parse_manifest(path)


def test_manifest_with_a_short_row_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "manifest.txt"
    path.write_text(f"{FORMAT_LINE}\ngen_000.npz\tfeatures\n", encoding="utf-8")
    with pytest.raises(ManifestError) as caught:
        parse_manifest(path)
    assert "5 tab-separated fields" in str(caught.value)


def test_cli_build_then_compare(tmp_path: Path) -> None:
    left_dir = make_signatures(tmp_path / "a")
    right_dir = make_signatures(tmp_path / "b")
    left = tmp_path / "left.txt"
    right = tmp_path / "right.txt"
    assert main(["--signatures", str(left_dir), "--out", str(left)]) == 0
    assert main(["--signatures", str(right_dir), "--out", str(right)]) == 0
    assert left.read_bytes() == right.read_bytes()
    assert main(["--compare", str(left), str(right)]) == 0

    np.savez(right_dir / "gen_001.npz", features=np.zeros(3, dtype=np.float64))
    assert main(["--signatures", str(right_dir), "--out", str(right)]) == 0
    assert main(["--compare", str(left), str(right)]) == 1


def test_cli_rejects_both_modes_at_once(tmp_path: Path) -> None:
    signatures = make_signatures(tmp_path / "sig")
    with pytest.raises(SystemExit):
        main(["--signatures", str(signatures), "--compare", "a.txt", "b.txt"])


def test_cli_requires_a_mode() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_cli_missing_directory_returns_two(tmp_path: Path) -> None:
    assert main(["--signatures", str(tmp_path / "absent"), "--out", str(tmp_path / "m.txt")]) == 2
