"""The surface report: code tokens and doc words, split correctly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.surface_report import (
    FileMeasure,
    SurfaceError,
    main,
    measure_markdown,
    measure_python,
    run,
)


def test_code_tokens_exclude_docs() -> None:
    source = '"""Module doc, four words here."""\n\n# a comment of five words\nvalue = 1 + 2\n'
    m = measure_python(source)
    assert m.doc_words == 5 + 5
    # value = 1 + 2 -> five code tokens; the docstring contributes none.
    assert m.code_tokens == 5
    assert m.loc == 4


def test_data_strings_are_code_not_docs() -> None:
    source = 'message = "previously seen 2026-01-01 words"\n'
    m = measure_python(source)
    assert m.doc_words == 0
    assert m.code_tokens == 3  # message, =, the string literal


def test_function_and_class_docstrings_count_as_docs() -> None:
    source = (
        "class A:\n"
        '    """Two words."""\n'
        "    def f(self):\n"
        '        """Three more words."""\n'
        "        return 1\n"
    )
    m = measure_python(source)
    assert m.doc_words == 2 + 3


def test_markdown_counts_words() -> None:
    m = measure_markdown("# Title\n\nSome prose here.\n")
    assert m.doc_words == 5
    assert m.code_tokens == 0
    assert m.loc == 3


def test_run_groups_by_top_level_dir(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("hello world\n", encoding="utf-8")
    report = run([tmp_path])
    names = set(report.groups)
    assert any(name.endswith("/pkg/") for name in names)
    total = report.total()
    assert total.files == 2
    assert total.code_tokens == 3
    assert total.doc_words == 2


def test_skip_dirs_excluded(tmp_path: Path) -> None:
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "big.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "keep.py").write_text("y = 2\n", encoding="utf-8")
    report = run([tmp_path])
    assert report.total().files == 1


def test_missing_root_raises() -> None:
    with pytest.raises(SurfaceError):
        run([Path("/nonexistent/surface/root")])


def test_main_writes_json_and_compares_baseline(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "m.py").write_text("x = 1\n", encoding="utf-8")
    out = tmp_path / "surface.json"
    assert main(["--root", str(root), "--json", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["total"]["code_tokens"] == 3

    (root / "n.py").write_text("y = 2 + 3\n", encoding="utf-8")
    assert main(["--root", str(root), "--baseline", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "Against baseline:" in printed
    assert "code_tokens: 3 -> 8 (+5)" in printed


def test_main_rejects_unreadable_baseline(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "m.py").write_text("x = 1\n", encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert main(["--root", str(root), "--baseline", str(bad)]) == 2


def test_unreadable_python_is_reported_not_fatal(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "broken.py").write_text("def f(:\n    (\n", encoding="utf-8")
    (root / "fine.py").write_text("x = 1\n", encoding="utf-8")
    report = run([root])
    assert report.total().files == 1
    assert len(report.unreadable) == 1


def test_filemeasure_defaults() -> None:
    m = FileMeasure()
    assert (m.loc, m.code_tokens, m.doc_words) == (0, 0, 0)
