"""Unit tests for the G3 timelessness checker.

The fixtures spell the flagged phrases out in full, which is the point of the
test; the checker's own patterns are written so they do not match themselves.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from tools.check_timelessness import (
    DEFAULT_ALLOWLIST,
    TimelessnessError,
    build_report,
    load_allowlist,
    main,
    used_to_violation,
)


def write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return path


def rules_hit(report: object) -> set[str]:
    return {violation.rule for violation in report.violations}  # type: ignore[attr-defined]


def test_shipped_allowlist_compiles() -> None:
    patterns = load_allowlist(DEFAULT_ALLOWLIST)
    assert len(patterns) > 10


def test_marker_comments_are_flagged(tmp_path: Path) -> None:
    write(
        tmp_path / "pkg" / "mod.py",
        """
        # TODO: tighten this
        VALUE = 1  # FIXME
        OTHER = 2  # HACK around the loader
        """,
    )
    report = build_report([tmp_path / "pkg"])
    assert len(report.violations) == 3
    assert rules_hit(report) == {"marker-comment"}
    assert report.passed is False


def test_changelog_phrases_in_comments_and_docstrings(tmp_path: Path) -> None:
    write(
        tmp_path / "pkg" / "mod.py",
        '''
        """This loop was previously two passes over the same file."""

        # we now cache the matrix once per generation
        VALUE = 1

        def f() -> str:
            """The split is no longer identical to the banked one."""
            return "ok"
        ''',
    )
    report = build_report([tmp_path / "pkg"])
    assert rules_hit(report) == {
        "changelog-prior-state",
        "changelog-we-now",
        "changelog-no-longer",
    }


def test_changelog_phrase_inside_an_f_string_is_flagged(tmp_path: Path) -> None:
    write(
        tmp_path / "pkg" / "mod.py",
        """
        def report(n: int) -> str:
            return f"{n} paths dropped - split no longer identical to the bank"
        """,
    )
    report = build_report([tmp_path / "pkg"])
    assert rules_hit(report) == {"changelog-no-longer"}


def test_changed_in_is_flagged(tmp_path: Path) -> None:
    write(tmp_path / "pkg" / "mod.py", "# the layer list changed in the second run\nVALUE = 1\n")
    report = build_report([tmp_path / "pkg"])
    assert rules_hit(report) == {"changelog-changed-in"}


def test_code_identifier_is_not_mistaken_for_prose(tmp_path: Path) -> None:
    write(tmp_path / "pkg" / "mod.py", "previously_seen = set()\nno_longer_valid = False\n")
    report = build_report([tmp_path / "pkg"])
    assert report.violations == []
    assert report.passed is True


def test_used_to_narrating_an_edit_is_flagged(tmp_path: Path) -> None:
    write(
        tmp_path / "pkg" / "mod.py",
        '''
        """Summaries the temporal_dynamics family used to host."""
        VALUE = 1
        ''',
    )
    report = build_report([tmp_path / "pkg"])
    assert rules_hit(report) == {"changelog-used-to"}


def test_used_to_as_a_present_tense_constraint_is_not_flagged(tmp_path: Path) -> None:
    write(
        tmp_path / "pkg" / "mod.py",
        '''
        """A lower bound is never reported as a floor or used to quote effect sizes."""
        VALUE = 1
        ''',
    )
    report = build_report([tmp_path / "pkg"])
    assert report.violations == []


def test_used_to_guard_directly() -> None:
    assert used_to_violation("the family used to host these summaries") == "used to host"
    assert used_to_violation("these states are used to compute features") is None
    assert used_to_violation("nothing followed by a verb here: used to") is None


def test_dated_comment_without_a_citation_marker_is_flagged(tmp_path: Path) -> None:
    write(tmp_path / "pkg" / "mod.py", "# rewrote the loader 2026-07-18\nVALUE = 1\n")
    report = build_report([tmp_path / "pkg"])
    assert rules_hit(report) == {"dated-comment"}


def test_dated_comment_with_a_citation_marker_is_exempt(tmp_path: Path) -> None:
    write(
        tmp_path / "pkg" / "mod.py",
        """
        # vmb matrix completion pass 1 (prereg Stage A(ii), census 2026-07-12)
        VALUE = 1
        """,
    )
    report = build_report([tmp_path / "pkg"])
    assert report.violations == []
    assert len(report.exempted) == 1
    assert report.passed is True


def test_dated_docstring_without_a_comment_marker_is_not_a_dated_comment(tmp_path: Path) -> None:
    write(tmp_path / "pkg" / "mod.py", '"""Banked 2026-07-12."""\nVALUE = 1\n')
    report = build_report([tmp_path / "pkg"])
    assert report.violations == []


def test_custom_allowlist_is_honoured(tmp_path: Path) -> None:
    write(tmp_path / "pkg" / "mod.py", "# rebuilt for the widget 2026-07-18\nVALUE = 1\n")
    allowlist = tmp_path / "allow.txt"
    allowlist.write_text("# only widgets are citable here\n\\bwidget\\b\n", encoding="utf-8")
    report = build_report([tmp_path / "pkg"], allowlist_path=allowlist)
    assert report.violations == []
    assert len(report.exempted) == 1


def test_invalid_allowlist_regex_raises(tmp_path: Path) -> None:
    allowlist = tmp_path / "allow.txt"
    allowlist.write_text("[unclosed\n", encoding="utf-8")
    write(tmp_path / "pkg" / "mod.py", "VALUE = 1\n")
    with pytest.raises(TimelessnessError):
        build_report([tmp_path / "pkg"], allowlist_path=allowlist)


def test_missing_allowlist_raises(tmp_path: Path) -> None:
    write(tmp_path / "pkg" / "mod.py", "VALUE = 1\n")
    with pytest.raises(TimelessnessError):
        build_report([tmp_path / "pkg"], allowlist_path=tmp_path / "absent.txt")


def test_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(TimelessnessError):
        build_report([tmp_path / "absent"])


def test_unparseable_file_is_reported_and_fails(tmp_path: Path) -> None:
    write(tmp_path / "pkg" / "mod.py", "def broken(:\n")
    report = build_report([tmp_path / "pkg"])
    assert len(report.read_errors) == 1
    assert "mod.py" in report.read_errors[0]
    assert report.passed is False


def test_a_single_file_root_is_accepted(tmp_path: Path) -> None:
    target = write(tmp_path / "pkg" / "mod.py", "# TODO: later\nVALUE = 1\n")
    report = build_report([target])
    assert len(report.violations) == 1


def test_cli_exit_codes_and_json_receipt(tmp_path: Path) -> None:
    root = tmp_path / "pkg"
    write(root / "clean.py", "VALUE = 1\n")
    receipt = tmp_path / "out" / "g3.json"
    argv = ["--root", str(root), "--json", str(receipt)]
    assert main(argv) == 0
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["gate"] == "G3"
    assert payload["passed"] is True

    write(root / "dirty.py", "# TODO: fix\nVALUE = 1\n")
    assert main(argv) == 1
    assert json.loads(receipt.read_text(encoding="utf-8"))["counts"]["violations"] == 1


def test_cli_reports_a_bad_allowlist_with_status_two(tmp_path: Path) -> None:
    root = tmp_path / "pkg"
    write(root / "mod.py", "VALUE = 1\n")
    assert main(["--root", str(root), "--allowlist", str(tmp_path / "absent.txt")]) == 2
