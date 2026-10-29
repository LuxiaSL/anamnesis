"""Unit tests for G3's public-voice check.

The parser is not needed here: a stand-in measurer counts marker words, so the
ratchet, the new-page limits, the flourish fences and page discovery are pinned on
a machine without spaCy. The measurement itself runs in the gate's own CI job.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import pytest

from tools import check_legibility as gate
from tools.check_legibility import (
    DEFAULT_BASELINE,
    DEFAULT_REPO,
    GATED,
    INSTANCE_ALLOWANCE,
    NEW_PAGE_MIN_WORDS,
    RELATIVE_TOLERANCE,
    build_report,
    discover_pages,
    load_baseline,
    metrics_entry,
    page_prose,
    regressions,
    strip_flourishes,
    write_baseline,
)
from tools.legibility_measure import (
    COUNTED,
    REFERENCED,
    LegibilityError,
    PageMetrics,
    load_reference,
)

MARKERS = {"REDUCED": "orc_reduced", "OVERT": "orc", "STACK": "stacked", "MINDS": "inanimate"}


def fake_measure(text: str) -> PageMetrics:
    """Words are words; each marker word counts once toward its feature."""
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", text)
    counts = {feature: 0 for feature in COUNTED}
    for word in words:
        if word in MARKERS:
            counts[MARKERS[word]] += 1
    counts["orc_all"] = counts["orc"] + counts["orc_reduced"]
    return PageMetrics(words=len(words), sentences=max(1, text.count(".")), dep_len=2.3, counts=counts)


def filler(n: int) -> str:
    return " ".join(["plain"] * n)


def write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return path


@pytest.fixture
def ref():
    return load_reference()


# ---------------------------------------------------------------------------- fences
def test_a_wellformed_fence_removes_its_passage_and_keeps_the_reason() -> None:
    text = "Intro words here.\n\n<!-- legibility: flourish — the thesis line -->\nVoice REDUCED REDUCED.\n<!-- /legibility -->\n\nOutro."
    fenced = strip_flourishes(text, "page.md")
    assert fenced.problems == []
    assert "REDUCED" not in fenced.text
    assert "Intro" in fenced.text and "Outro" in fenced.text
    [flourish] = fenced.flourishes
    assert flourish.reason == "the thesis line"
    assert flourish.line == 3
    assert flourish.words == 3


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("<!-- legibility: flourish -->\nx\n<!-- /legibility -->", "states no reason"),
        ("<!-- legibility: flourish — why -->\nnever closed", "never closed"),
        ("text\n<!-- /legibility -->", "closes without opening"),
        ("<!-- legibility: flourish — a -->\n<!-- legibility: flourish — b -->\n<!-- /legibility -->", "opens inside"),
        ("<!-- legibility: skip this -->", "unreadable legibility marker"),
    ],
)
def test_a_malformed_fence_is_a_problem(text: str, problem: str) -> None:
    fenced = strip_flourishes(text, "page.md")
    assert any(problem in p for p in fenced.problems), fenced.problems


def test_a_fence_shown_as_code_is_an_example_not_a_fence() -> None:
    text = textwrap.dedent(
        """
        How to fence:

        ```markdown
        <!-- legibility: flourish — example -->
        A quoted line.
        <!-- /legibility -->
        ```

        Inline, `<!-- /legibility -->` closes one.
        """
    )
    fenced = strip_flourishes(text, "page.md")
    assert fenced.problems == []
    assert fenced.flourishes == []
    assert fenced.text == text


# ---------------------------------------------------------------------------- pages
def test_discovery_finds_markdown_and_the_docstrings_it_links(tmp_path: Path) -> None:
    write(tmp_path / "README.md", "See [the package](pkg/__init__.py) and [gone](pkg/missing.py).\n")
    write(tmp_path / "docs" / "GUIDE.md", "A [sibling](../pkg/mod.py#L3) link.\n")
    write(tmp_path / "pkg" / "__init__.py", '"""Package prose."""\n')
    write(tmp_path / "pkg" / "mod.py", '"""Module prose."""\n')
    write(tmp_path / ".venv" / "lib" / "NOTES.md", "not a page\n")
    write(tmp_path / "outputs" / "run" / "REPORT.md", "not a page\n")
    assert discover_pages(tmp_path) == ["README.md", "docs/GUIDE.md", "pkg/__init__.py", "pkg/mod.py"]


def test_a_docstring_page_is_its_module_docstring_with_roles_as_code(tmp_path: Path) -> None:
    write(
        tmp_path / "pkg" / "__init__.py",
        '''
        """The package. :mod:`pkg.inner` holds ``THING``.

        Second paragraph.
        """

        # a comment is not part of the page
        VALUE = "a string literal is not part of the page"
        ''',
    )
    prose = page_prose(tmp_path, "pkg/__init__.py").text
    assert "CODE" in prose and "pkg.inner" not in prose and "THING" not in prose
    assert "Second paragraph." in prose
    assert "comment" not in prose and "literal" not in prose


# ---------------------------------------------------------------------------- the ratchet
def entry_for(text: str) -> dict[str, object]:
    return metrics_entry(fake_measure(text))


def test_one_more_occurrence_on_a_page_is_within_tolerance() -> None:
    base = entry_for(filler(200) + " REDUCED")
    now = fake_measure(filler(200) + " REDUCED REDUCED")
    assert regressions("p.md", now, base) == []


def test_a_rise_past_the_tolerance_fails_and_names_the_feature() -> None:
    base = entry_for(filler(500) + " STACK")
    now = fake_measure(filler(500) + " STACK STACK STACK STACK")
    excess = regressions("p.md", now, base)
    assert [e.feature for e in excess] == ["stacked"]
    [e] = excess
    assert e.allowed == pytest.approx(
        base["rates"]["stacked"] * (1 + RELATIVE_TOLERANCE) + INSTANCE_ALLOWANCE * 1000 / now.words  # type: ignore[index]
    )


def test_giving_a_relative_its_that_never_fails() -> None:
    """Reduced to overt is the rewrite the guide asks for; the overt count is not held."""
    assert "orc" not in GATED
    base = entry_for(filler(300) + " REDUCED REDUCED REDUCED REDUCED")
    now = fake_measure(filler(300) + " OVERT OVERT OVERT OVERT")
    assert regressions("p.md", now, base) == []


def test_arc_length_has_its_own_relative_tolerance() -> None:
    base = entry_for(filler(300))
    now = fake_measure(filler(300))
    now.dep_len = 2.3 * (1 + gate.DEP_LEN_TOLERANCE) + 0.01
    assert [e.feature for e in regressions("p.md", now, base)] == ["dep_len"]


def test_an_improvement_is_reported_as_tightenable() -> None:
    base = entry_for(filler(300) + " " + " ".join(["MINDS"] * 6))
    now = fake_measure(filler(300))
    assert "inanimate" in gate.tightenable(now, base)


# ---------------------------------------------------------------------------- reports
def baseline_file(path: Path, pages: dict[str, str]) -> Path:
    path.write_text(json.dumps({"pages": {page: entry_for(text) for page, text in pages.items()}}), encoding="utf-8")
    return path


def test_a_new_page_is_held_to_the_limits(tmp_path: Path, ref) -> None:
    write(tmp_path / "GOOD.md", filler(400) + "\n")
    write(tmp_path / "BAD.md", filler(400) + " " + " ".join(["MINDS"] * 20) + "\n")
    write(tmp_path / "SHORT.md", filler(NEW_PAGE_MIN_WORDS - 50) + " " + " ".join(["MINDS"] * 20) + "\n")
    report = build_report(tmp_path, tmp_path / "baseline.json", ref, fake_measure)
    status = {r.page: (r.status, [e.feature for e in r.excess]) for r in report.pages}
    assert status == {"BAD.md": ("new", ["inanimate"]), "GOOD.md": ("new", []), "SHORT.md": ("short", [])}
    assert not report.passed


def test_a_baseline_entry_without_a_page_is_stale(tmp_path: Path, ref) -> None:
    write(tmp_path / "README.md", filler(200) + "\n")
    baseline = baseline_file(tmp_path / "baseline.json", {"README.md": filler(200), "GONE.md": filler(200)})
    report = build_report(tmp_path, baseline, ref, fake_measure)
    assert report.stale == ["GONE.md"]
    assert not report.passed


def test_a_fence_problem_fails_the_report(tmp_path: Path, ref) -> None:
    write(tmp_path / "README.md", filler(200) + "\n<!-- legibility: flourish — why -->\nopen\n")
    report = build_report(tmp_path, tmp_path / "baseline.json", ref, fake_measure)
    assert report.fence_problems and not report.passed


def test_baseline_writes_bootstrap_then_refuse_then_record_a_reason(tmp_path: Path, ref) -> None:
    write(tmp_path / "README.md", filler(400) + " " + " ".join(["MINDS"] * 20) + "\n")
    baseline = tmp_path / "baseline.json"

    report = build_report(tmp_path, baseline, ref, fake_measure)
    assert report.failing
    assert write_baseline(baseline, report, reason=None) == []
    assert load_baseline(baseline)["README.md"]["rates"]["inanimate"] == pytest.approx(20 * 1000 / 420, abs=1e-3)

    write(tmp_path / "README.md", filler(400) + " " + " ".join(["MINDS"] * 40) + "\n")
    report = build_report(tmp_path, baseline, ref, fake_measure)
    assert [r.page for r in report.failing] == ["README.md"]
    with pytest.raises(LegibilityError, match="--reason"):
        write_baseline(baseline, report, reason=None)
    assert write_baseline(baseline, report, reason="the new section quotes the protocol") == ["README.md"]
    assert load_baseline(baseline)["README.md"]["accepted_regression"] == "the new section quotes the protocol"


def test_cli_exit_codes_and_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "load_parser", lambda: object())
    monkeypatch.setattr(gate, "measure", lambda text, nlp, ref: fake_measure(text))
    write(tmp_path / "README.md", filler(300) + "\n")
    baseline = baseline_file(tmp_path / "baseline.json", {"README.md": filler(300)})
    receipt = tmp_path / "receipt.json"
    assert gate.main(["--repo", str(tmp_path), "--baseline", str(baseline), "--json", str(receipt)]) == 0
    assert json.loads(receipt.read_text())["passed"] is True

    write(tmp_path / "README.md", filler(300) + " " + " ".join(["STACK"] * 10) + "\n")
    assert gate.main(["--repo", str(tmp_path), "--baseline", str(baseline)]) == 1
    assert gate.main(["--repo", str(tmp_path), "--baseline", str(baseline), "--page", "NOPE.md"]) == 2


def test_a_missing_parser_is_unrunnable_not_a_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_parser() -> object:
        raise LegibilityError("spaCy is not installed")

    monkeypatch.setattr(gate, "load_parser", no_parser)
    assert gate.main(["--repo", str(tmp_path)]) == 2


# ---------------------------------------------------------------------------- shipped data
def test_the_shipped_reference_defines_every_ratio() -> None:
    ref = load_reference()
    for feature in REFERENCED:
        assert ref.rates[feature] > 0
    assert ref.rates["orc_all"] == pytest.approx(ref.rates["orc"] + ref.rates["orc_reduced"])


def test_the_shipped_baseline_names_only_pages_that_exist() -> None:
    baseline = load_baseline(DEFAULT_BASELINE)
    assert baseline, "the shipped baseline is empty"
    assert sorted(set(baseline) - set(discover_pages(DEFAULT_REPO))) == []
    for entry in baseline.values():
        assert set(GATED) <= set(entry["rates"])  # type: ignore[arg-type]


def test_the_shipped_pages_fences_are_wellformed() -> None:
    problems = [p for page in discover_pages(DEFAULT_REPO) for p in page_prose(DEFAULT_REPO, page).problems]
    assert problems == []
