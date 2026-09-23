"""The sub-perceptual census: the bars, the asymmetry, and the gate that enforces it.

The census is arithmetic over banked arm records, and every one of these tests is
about a way the arithmetic could quietly say the wrong thing:

  * the bars are declared, so they are checked at their own boundaries rather than
    approximately;
  * the gap is ``internals - max(content, likelihood)``, which means a strong
    *second* rung defeats membership even when the first is weak — the monotone-
    conservative reading, and the one an added detector can only tighten;
  * the content rung of a mode row is the maximum over the declared detector set
    *including the judge*, so a judge that reads a mode well raises the bar;
  * a judge failure is not symmetric with a judge success: where a forced-choice
    table shows the judge discriminating, the blind-judge gap is annotated as a
    contrast artifact and is not quotable, and where no such table exists the row
    says the hardening is pending rather than passed;
  * the class object is the census's subject, and an empty entry for a model is a
    reading rather than missing data;
  * the document cannot be produced without the judge-defense gate running.

The command is covered too: it refuses a root with no records rather than writing an
empty census.

CPU only; no model, no network, no banked data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anamnesis.analysis.battery.census import (
    BORDERLINE,
    BORDERLINE_BAR,
    EXCLUDED,
    HARDENED_2AFC_BAR,
    MEMBER,
    MEMBER_BAR,
    MODES,
    a1_rows,
    a3_rows,
    census_document,
    census_markdown,
    class_object,
    classify,
    load_2afc_rates,
    run_census,
)
from anamnesis.scripts.census import main


def write_a1(root: Path, *, model: str = "3b", internals: float = 0.95,
             content: float = 0.60, likelihood: float = 0.70, record: str = "A1") -> None:
    directory = root / record
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "a1_results.json").write_text(
        json.dumps(
            {
                "models": {
                    model: {
                        "dissociation": {
                            "tfidf_groupkfold_auc_t03_vs_t09": content,
                            "likelihood_mean_surprise_auc": likelihood,
                            "signature_output_groupkfold_auc": internals,
                            "n": 120,
                        }
                    }
                }
            }
        )
    )


def write_a3(root: Path, *, model: str = "3b", judge: dict[str, float] | None = None,
             internals: float = 0.90, record: str = "A3") -> None:
    directory = root / record
    directory.mkdir(parents=True, exist_ok=True)
    per_mode = {mode: internals for mode in MODES}
    (directory / "a3_results.json").write_text(
        json.dumps(
            {
                "models": {
                    model: {
                        "hierarchy": {
                            "content_tfidf": {"per_mode_recall": {m: 0.40 for m in MODES}},
                            "likelihood_surprise": {"per_mode_recall": {m: 0.30 for m in MODES}},
                            "internals_rf": {"per_mode_recall": per_mode},
                        },
                        "judge": (
                            {"per_mode": {m: {"judge_recall": v} for m, v in judge.items()}}
                            if judge
                            else {}
                        ),
                    }
                }
            }
        )
    )


def write_2afc(root: Path, *, model: str = "3b", accuracy: float = 0.80) -> None:
    directory = root / "A3" / "judge"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "socratic_2afc_fable.json").write_text(
        json.dumps({"models": {model: {"acc_2afc": accuracy}}})
    )


def test_the_bars_are_checked_at_their_own_boundaries() -> None:
    assert classify(MEMBER_BAR) == MEMBER
    assert classify(MEMBER_BAR - 1e-9) == BORDERLINE
    assert classify(BORDERLINE_BAR) == BORDERLINE
    assert classify(BORDERLINE_BAR - 1e-9) == EXCLUDED
    assert classify(-1.0) == EXCLUDED


def test_the_higher_of_the_two_other_rungs_is_what_the_gap_is_against(tmp_path: Path) -> None:
    write_a1(tmp_path, internals=0.95, content=0.60, likelihood=0.90)
    row = a1_rows(tmp_path)[0]
    assert row.gap == pytest.approx(0.05)
    assert row.status == BORDERLINE
    assert row.binding_rung == "likelihood"
    assert "exempt" in row.hardening, "no judge is involved in a likelihood-bound row"


def test_a_content_bound_row_names_content_as_its_binding_rung(tmp_path: Path) -> None:
    write_a1(tmp_path, internals=0.95, content=0.80, likelihood=0.40)
    row = a1_rows(tmp_path)[0]
    assert row.binding_rung == "content"
    assert row.gap == pytest.approx(0.15)
    assert row.status == MEMBER


def test_every_a1_record_directory_present_contributes_its_models(tmp_path: Path) -> None:
    write_a1(tmp_path, model="3b", record="A1")
    write_a1(tmp_path, model="8b", record="A1_m3m4")
    rows = a1_rows(tmp_path)
    assert {row.model for row in rows} == {"3b", "8b"}
    assert {row.source_record for row in rows} == {
        "arms/A1/a1_results.json",
        "arms/A1_m3m4/a1_results.json",
    }


def test_a_judge_that_reads_a_mode_well_raises_the_bar_it_has_to_clear(tmp_path: Path) -> None:
    write_a3(tmp_path, internals=0.90, judge={"linear": 0.85})
    rows = {row.row: row for row in a3_rows(tmp_path)}
    linear = rows["A3:mode:linear"]
    socratic = rows["A3:mode:socratic"]
    assert linear.content == pytest.approx(0.85), "the content rung is the max over the set"
    assert linear.gap == pytest.approx(0.05)
    assert linear.status == BORDERLINE
    assert socratic.content == pytest.approx(0.40), "a mode with no judge keeps the trained rung"
    assert socratic.status == MEMBER


def test_a_forced_choice_table_voids_the_blind_judge_gap(tmp_path: Path) -> None:
    write_a3(tmp_path, internals=0.95, judge={"socratic": 0.20})
    write_2afc(tmp_path, accuracy=HARDENED_2AFC_BAR + 0.1)
    rows = {row.row: row for row in a3_rows(tmp_path)}
    socratic = rows["A3:mode:socratic"]
    assert socratic.judge_gap == pytest.approx(0.75)
    assert "FAILED hardening" in socratic.hardening
    assert "NOT" in socratic.hardening, "a voided judge-gap says so in the row"
    assert socratic.status == MEMBER, "membership still binds on the trained detector"


def test_a_judge_below_the_forced_choice_bar_leaves_the_row_surviving(tmp_path: Path) -> None:
    write_a3(tmp_path, internals=0.95, judge={"socratic": 0.20})
    write_2afc(tmp_path, accuracy=HARDENED_2AFC_BAR - 0.1)
    rows = {row.row: row for row in a3_rows(tmp_path)}
    assert "SURVIVES" in rows["A3:mode:socratic"].hardening


def test_without_a_forced_choice_table_the_hardening_is_pending(tmp_path: Path) -> None:
    write_a3(tmp_path, internals=0.95, judge={"linear": 0.20})
    assert load_2afc_rates(tmp_path) == {}
    rows = {row.row: row for row in a3_rows(tmp_path)}
    assert rows["A3:mode:linear"].hardening.startswith("PENDING")


def test_the_class_object_is_per_model_and_may_be_empty(tmp_path: Path) -> None:
    write_a1(tmp_path, model="3b", internals=0.95, content=0.60, likelihood=0.60)
    write_a1(tmp_path, model="8b", internals=0.62, content=0.60, likelihood=0.60,
             record="A1_m5")
    rows = a1_rows(tmp_path)
    classes = class_object(rows)
    assert classes["3b"]["members"] == ["A1:temperature(t03|t09)"]
    assert classes["8b"]["members"] == []
    assert classes["8b"]["borderline"] == []


def test_the_document_states_its_bars_and_runs_the_gate(tmp_path: Path) -> None:
    write_a1(tmp_path)
    write_a3(tmp_path)
    rows = a1_rows(tmp_path) + a3_rows(tmp_path)
    document = census_document(rows)
    assert str(MEMBER_BAR) in document["bars"] and str(BORDERLINE_BAR) in document["bars"]
    assert "internals - max(content, likelihood)" in document["definition"]
    assert len(document["rows"]) == len(rows)
    assert set(document) == {"bars", "definition", "rows", "class_object"}, (
        "the census carries only what it computes from records; nothing hand-written rides along"
    )
    table = census_markdown(rows)
    data_rows = [line for line in table.splitlines() if line.startswith("| A")]
    assert any("\\|" in line for line in data_rows), (
        "a row label spelling a contrast with a pipe escapes it, or it splits its own cell"
    )
    cells = [
        [c.strip() for c in line.replace("\\|", "/").split("|")] for line in data_rows
    ]
    assert all(len(row) == 10 for row in cells), "every row has the header's columns"
    gaps = [float(row[7]) for row in cells]
    assert gaps == sorted(gaps, reverse=True), "the widest gaps are read first"


def test_the_command_refuses_a_root_with_no_records(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no record directory from"):
        run_census(tmp_path / "empty", tmp_path / "out")
    with pytest.raises(FileNotFoundError):
        main(["--arms-root", str(tmp_path / "empty"), "--out-dir", str(tmp_path / "out")])


def test_the_command_writes_both_artifacts(tmp_path: Path) -> None:
    write_a1(tmp_path / "arms")
    write_a3(tmp_path / "arms")
    assert main(["--arms-root", str(tmp_path / "arms"), "--out-dir", str(tmp_path / "out")]) == 0
    document = json.loads((tmp_path / "out" / "subperceptual_census.json").read_text())
    assert document["rows"] and document["class_object"]
    assert (tmp_path / "out" / "subperceptual_census.md").read_text().startswith(
        "# Sub-perceptual census"
    )
