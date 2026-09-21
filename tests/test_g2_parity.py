"""The G2 parity table: ported files measured beside their donors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.g2_loc_report import BaselineError, ParityRow, build_parity, load_parity


def write(path: Path, lines: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x = 1\n" * lines, encoding="utf-8")


def test_parity_measures_both_sides(tmp_path: Path) -> None:
    repo = tmp_path / "new"
    old = tmp_path / "old"
    write(repo / "anamnesis" / "mod.py", 10)
    write(old / "anamnesis" / "mod.py", 14)
    rows = build_parity(repo, old, {"anamnesis/mod.py": "anamnesis/mod.py"})
    assert rows == [
        ParityRow(new_path="anamnesis/mod.py", old_path="anamnesis/mod.py", new_loc=10, old_loc=14)
    ]
    assert rows[0].delta == -4


def test_parity_growth_is_visible(tmp_path: Path) -> None:
    repo = tmp_path / "new"
    old = tmp_path / "old"
    write(repo / "grew.py", 20)
    write(old / "donor.py", 5)
    rows = build_parity(repo, old, {"grew.py": "donor.py"})
    assert rows[0].delta == 15


def test_parity_missing_files_read_as_none(tmp_path: Path) -> None:
    repo = tmp_path / "new"
    old = tmp_path / "old"
    old.mkdir()
    repo.mkdir()
    write(old / "donor.py", 5)
    rows = build_parity(repo, old, {"pending.py": "donor.py"})
    assert rows[0].new_loc is None
    assert rows[0].old_loc == 5
    assert rows[0].delta is None


def test_load_parity_validates_shape(tmp_path: Path) -> None:
    old = tmp_path / "old"
    old.mkdir()
    good = tmp_path / "parity.json"
    good.write_text(
        json.dumps({"old_root": str(old), "pairs": {"a.py": "b.py"}}), encoding="utf-8"
    )
    root, pairs = load_parity(good)
    assert root == old
    assert pairs == {"a.py": "b.py"}

    for payload in (
        {"pairs": {}},
        {"old_root": str(old)},
        {"old_root": str(old / "missing"), "pairs": {}},
        {"old_root": str(old), "pairs": {"a.py": 3}},
    ):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(BaselineError):
            load_parity(bad)
