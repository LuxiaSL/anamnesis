"""Unit tests for the G2 donor-LOC report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.g2_loc_report import (
    DEFAULT_BASELINE,
    BaselineError,
    build_report,
    count_lines,
    load_baseline,
    main,
)


def write_lines(path: Path, count: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"line {n}\n" for n in range(count)), encoding="utf-8")
    return path


def baseline_payload(module: str = "pkg/consolidated.py", consumers: object = 2) -> dict[str, object]:
    return {
        "metric": "physical lines",
        "total_python_loc_before": 100,
        "consolidations": {
            "K1": {
                "capability": "demo capability",
                "module": module,
                "justification": "consolidation",
                "ported_consumers": consumers,
                "donor_total": 30,
                "donors": {"pkg/donor_a.py": 10, "pkg/donor_b.py": 20},
            }
        },
        "tests": {"paths": ["tests"], "floor_loc": 12},
    }


def write_baseline(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_count_lines_matches_newline_count(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "f.py", 7)
    assert count_lines(path) == 7
    unterminated = tmp_path / "g.py"
    unterminated.write_text("one\ntwo", encoding="utf-8")
    assert count_lines(unterminated) == 1


def test_shipped_baseline_is_internally_consistent() -> None:
    payload = load_baseline(DEFAULT_BASELINE)
    consolidations = payload["consolidations"]
    assert isinstance(consolidations, dict)
    assert set(consolidations) == {"C1", "C2", "C3", "C4", "C5"}
    totals = {key: entry["donor_total"] for key, entry in consolidations.items()}
    assert totals == {"C1": 932, "C2": 1700, "C3": 1387, "C4": 826, "C5": 648}
    assert consolidations["C4"]["ported_consumers"] == 1
    assert payload["tests"]["floor_loc"] == 4559


def test_baseline_with_a_wrong_stated_total_is_refused(tmp_path: Path) -> None:
    payload = baseline_payload()
    payload["consolidations"]["K1"]["donor_total"] = 31  # type: ignore[index]
    path = write_baseline(tmp_path / "b.json", payload)
    with pytest.raises(BaselineError) as caught:
        load_baseline(path)
    assert "rows sum to 30" in str(caught.value)


def test_baseline_missing_a_key_is_refused(tmp_path: Path) -> None:
    payload = baseline_payload()
    del payload["tests"]
    path = write_baseline(tmp_path / "b.json", payload)
    with pytest.raises(BaselineError):
        load_baseline(path)


def test_baseline_that_is_not_json_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "b.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(BaselineError):
        load_baseline(path)


def test_absent_module_reports_pending_and_does_not_fail(tmp_path: Path) -> None:
    baseline = write_baseline(tmp_path / "b.json", baseline_payload())
    write_lines(tmp_path / "tests" / "test_a.py", 12)
    report = build_report(tmp_path, baseline)
    assert [(item.key, item.status) for item in report.consolidations] == [("K1", "pending")]
    assert report.consolidations[0].actual_loc is None
    assert report.passed is True


def test_smaller_module_passes_and_reports_the_reduction(tmp_path: Path) -> None:
    baseline = write_baseline(tmp_path / "b.json", baseline_payload())
    write_lines(tmp_path / "pkg" / "consolidated.py", 25)
    write_lines(tmp_path / "tests" / "test_a.py", 12)
    report = build_report(tmp_path, baseline)
    item = report.consolidations[0]
    assert (item.status, item.actual_loc, item.reduction) == ("pass", 25, 5)
    assert report.passed is True


def test_equal_module_fails_because_the_rule_is_strictly_less(tmp_path: Path) -> None:
    baseline = write_baseline(tmp_path / "b.json", baseline_payload())
    write_lines(tmp_path / "pkg" / "consolidated.py", 30)
    write_lines(tmp_path / "tests" / "test_a.py", 12)
    report = build_report(tmp_path, baseline)
    assert report.consolidations[0].status == "fail"
    assert report.passed is False


def test_test_loc_below_the_floor_fails(tmp_path: Path) -> None:
    baseline = write_baseline(tmp_path / "b.json", baseline_payload())
    write_lines(tmp_path / "pkg" / "consolidated.py", 20)
    write_lines(tmp_path / "tests" / "test_a.py", 11)
    report = build_report(tmp_path, baseline)
    assert report.test_loc == 11
    assert report.tests_pass is False
    assert report.passed is False


def test_test_loc_sums_only_python_files(tmp_path: Path) -> None:
    baseline = write_baseline(tmp_path / "b.json", baseline_payload())
    write_lines(tmp_path / "tests" / "test_a.py", 8)
    write_lines(tmp_path / "tests" / "nested" / "test_b.py", 4)
    (tmp_path / "tests" / "fixture.txt").write_text("a\nb\nc\n", encoding="utf-8")
    report = build_report(tmp_path, baseline)
    assert report.test_loc == 12


def test_cache_directories_are_excluded_from_test_loc(tmp_path: Path) -> None:
    baseline = write_baseline(tmp_path / "b.json", baseline_payload())
    write_lines(tmp_path / "tests" / "test_a.py", 12)
    write_lines(tmp_path / "tests" / "__pycache__" / "stale.py", 500)
    report = build_report(tmp_path, baseline)
    assert report.test_loc == 12


def test_unrecorded_consumer_count_stays_unrecorded(tmp_path: Path) -> None:
    baseline = write_baseline(tmp_path / "b.json", baseline_payload(consumers=None))
    write_lines(tmp_path / "tests" / "test_a.py", 12)
    report = build_report(tmp_path, baseline)
    assert report.consolidations[0].ported_consumers is None


def test_consumer_count_is_carried_through_to_the_receipt(tmp_path: Path) -> None:
    baseline = write_baseline(tmp_path / "b.json", baseline_payload(consumers=3))
    write_lines(tmp_path / "tests" / "test_a.py", 12)
    report = build_report(tmp_path, baseline)
    assert report.to_json()["consolidations"][0]["ported_consumers"] == 3  # type: ignore[index]


def test_missing_repo_raises(tmp_path: Path) -> None:
    baseline = write_baseline(tmp_path / "b.json", baseline_payload())
    with pytest.raises(BaselineError):
        build_report(tmp_path / "absent", baseline)


def test_cli_exit_codes_and_json_receipt(tmp_path: Path) -> None:
    baseline = write_baseline(tmp_path / "b.json", baseline_payload())
    write_lines(tmp_path / "tests" / "test_a.py", 12)
    receipt = tmp_path / "out" / "g2.json"
    argv = ["--repo", str(tmp_path), "--baseline", str(baseline), "--json", str(receipt)]
    assert main(argv) == 0
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["gate"] == "G2"
    assert payload["tests"]["loc"] == 12

    write_lines(tmp_path / "pkg" / "consolidated.py", 40)
    assert main(argv) == 1


def test_cli_bad_baseline_returns_two(tmp_path: Path) -> None:
    assert main(["--repo", str(tmp_path), "--baseline", str(tmp_path / "absent.json")]) == 2
