#!/usr/bin/env python3
"""G2 — net line reduction where a consolidation was promised.

The port is an allow-list, so "the new repo is smaller" is true by construction
and carries no information. G2 is a claim about the five consolidations only:
each extracted module must be strictly smaller than the summed donors it
replaces, and the test suite must not shrink to make a number move.

The donor table is data, not code: `g2_baseline.json` holds each consolidation's
capability, its target module path, its donors with per-file LOC, the donor sum,
and the test-LOC floor. The tool re-adds the donors and refuses a baseline whose
stated sum disagrees with its own rows, so the table cannot drift silently.

A consolidation whose module does not exist yet reports `pending` and does not
fail the gate; that is the normal state until its PR lands. `ported_consumers`
is printed beside the arithmetic because a consolidation that passes on LOC while
serving nobody has not delivered the capability it was extracted for; where the
count is not yet recorded the field says so rather than implying zero.

LOC is physical lines, as `wc -l` counts them. Blank and comment lines count,
which makes the metric gameable by reformatting and trivially reproducible by
anyone; the arithmetic is reported per donor so a reader can check it.

Usage
-----
    python tools/g2_loc_report.py --repo . --json g2.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, TextIO

DEFAULT_BASELINE = Path(__file__).resolve().parent / "g2_baseline.json"

SKIP_DIRS = frozenset({"__pycache__", ".git", ".venv", "venv", "node_modules", ".mypy_cache"})

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_PENDING = "pending"


class BaselineError(RuntimeError):
    """A malformed baseline, or a repository the report cannot be built over."""


@dataclass(frozen=True)
class ConsolidationResult:
    """One consolidation's arithmetic and verdict."""

    key: str
    capability: str
    module: str
    justification: str
    ported_consumers: int | None
    donor_total: int
    donor_loc: dict[str, int]
    actual_loc: int | None
    status: str

    @property
    def reduction(self) -> int | None:
        return None if self.actual_loc is None else self.donor_total - self.actual_loc


@dataclass(frozen=True)
class ParityRow:
    """One ported file's LOC beside its donor's — the drift detector for
    PORT-as-is items, which G2's consolidation arithmetic does not cover."""

    new_path: str
    old_path: str
    new_loc: int | None
    old_loc: int | None

    @property
    def delta(self) -> int | None:
        if self.new_loc is None or self.old_loc is None:
            return None
        return self.new_loc - self.old_loc


@dataclass
class LocReport:
    """The result of a G2 pass, serialisable as the receipt."""

    repo: str
    baseline: str
    metric: str
    consolidations: list[ConsolidationResult] = field(default_factory=list)
    test_loc: int = 0
    test_floor: int = 0
    test_paths: list[str] = field(default_factory=list)
    total_python_loc: int = 0
    total_python_loc_before: int | None = None
    total_loc_method: str = ""
    parity: list[ParityRow] = field(default_factory=list)

    @property
    def tests_pass(self) -> bool:
        return self.test_loc >= self.test_floor

    @property
    def passed(self) -> bool:
        failing = any(item.status == STATUS_FAIL for item in self.consolidations)
        return not failing and self.tests_pass

    def to_json(self) -> dict[str, object]:
        return {
            "gate": "G2",
            "repo": self.repo,
            "baseline": self.baseline,
            "metric": self.metric,
            "consolidations": [
                {
                    "key": item.key,
                    "capability": item.capability,
                    "module": item.module,
                    "justification": item.justification,
                    "ported_consumers": item.ported_consumers,
                    "donor_total": item.donor_total,
                    "donor_loc": item.donor_loc,
                    "actual_loc": item.actual_loc,
                    "reduction": item.reduction,
                    "status": item.status,
                }
                for item in self.consolidations
            ],
            "tests": {
                "paths": self.test_paths,
                "loc": self.test_loc,
                "floor": self.test_floor,
                "passed": self.tests_pass,
            },
            "total_python_loc": self.total_python_loc,
            "total_python_loc_before": self.total_python_loc_before,
            "total_loc_method": self.total_loc_method,
            "parity": [
                {
                    "new_path": row.new_path,
                    "old_path": row.old_path,
                    "new_loc": row.new_loc,
                    "old_loc": row.old_loc,
                    "delta": row.delta,
                }
                for row in self.parity
            ],
            "passed": self.passed,
        }


def count_lines(path: Path) -> int:
    """Physical line count, matching `wc -l`: one per newline byte."""
    try:
        return path.read_bytes().count(b"\n")
    except OSError as exc:
        raise BaselineError(f"{path}: unreadable ({exc})") from exc


def python_files_under(root: Path) -> list[Path]:
    """Every `.py` file under a directory, excluding caches and environments."""
    if root.is_file():
        return [root] if root.suffix == ".py" else []
    out: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        out.append(path)
    return out


def tracked_python_files(repo: Path) -> tuple[list[Path], str]:
    """Python files git tracks, falling back to a filesystem walk."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "*.py"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return python_files_under(repo), "filesystem walk (git ls-files unavailable)"
    paths = [repo / line for line in completed.stdout.splitlines() if line.strip()]
    existing = [path for path in paths if path.is_file()]
    return existing, "git ls-files"


def load_baseline(path: Path) -> dict[str, object]:
    """Read and validate the donor table."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BaselineError(f"baseline unreadable: {path} ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise BaselineError(f"{path}: invalid JSON at line {exc.lineno} ({exc.msg})") from exc
    if not isinstance(payload, dict):
        raise BaselineError(f"{path}: top level must be an object")
    for required in ("consolidations", "tests", "metric"):
        if required not in payload:
            raise BaselineError(f"{path}: missing required key {required!r}")
    consolidations = payload["consolidations"]
    if not isinstance(consolidations, dict) or not consolidations:
        raise BaselineError(f"{path}: 'consolidations' must be a non-empty object")
    for key, entry in consolidations.items():
        if not isinstance(entry, dict):
            raise BaselineError(f"{path}: consolidation {key} must be an object")
        for required in ("capability", "module", "donors", "donor_total"):
            if required not in entry:
                raise BaselineError(f"{path}: consolidation {key} missing {required!r}")
        donors = entry["donors"]
        if not isinstance(donors, dict) or not donors:
            raise BaselineError(f"{path}: consolidation {key} has no donors")
        summed = 0
        for donor, loc in donors.items():
            if not isinstance(loc, int) or loc < 0:
                raise BaselineError(f"{path}: consolidation {key} donor {donor} has a non-integer LOC")
            summed += loc
        if summed != entry["donor_total"]:
            raise BaselineError(
                f"{path}: consolidation {key} states donor_total {entry['donor_total']} "
                f"but its rows sum to {summed}"
            )
    tests = payload["tests"]
    if not isinstance(tests, dict) or "floor_loc" not in tests or "paths" not in tests:
        raise BaselineError(f"{path}: 'tests' needs 'paths' and 'floor_loc'")
    if not isinstance(tests["floor_loc"], int) or tests["floor_loc"] < 0:
        raise BaselineError(f"{path}: tests.floor_loc must be a non-negative integer")
    return payload


def load_parity(path: Path) -> tuple[Path, dict[str, str]]:
    """Read a parity map: `{"old_root": <path>, "pairs": {new: old}}`.

    A pair is a ported file in this tree beside its donor's path relative to
    `old_root` (a checkout of the frozen record, linked from `README.md`)."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BaselineError(f"parity map unreadable: {path} ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise BaselineError(f"{path}: invalid JSON at line {exc.lineno} ({exc.msg})") from exc
    if not isinstance(payload, dict) or "old_root" not in payload or "pairs" not in payload:
        raise BaselineError(f"{path}: parity map needs 'old_root' and 'pairs'")
    pairs = payload["pairs"]
    if not isinstance(pairs, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in pairs.items()
    ):
        raise BaselineError(f"{path}: 'pairs' must map new path -> old path (strings)")
    old_root = Path(str(payload["old_root"]))
    if not old_root.is_dir():
        raise BaselineError(f"{path}: old_root is not a directory: {old_root}")
    return old_root, {str(k): str(v) for k, v in pairs.items()}


def build_parity(repo: Path, old_root: Path, pairs: dict[str, str]) -> list[ParityRow]:
    """LOC on both sides of every parity pair; a missing file reads as None
    rather than failing the receipt, so pending ports stay visible."""
    rows: list[ParityRow] = []
    for new_rel in sorted(pairs):
        old_rel = pairs[new_rel]
        new_path = repo / new_rel
        old_path = old_root / old_rel
        rows.append(
            ParityRow(
                new_path=new_rel,
                old_path=old_rel,
                new_loc=count_lines(new_path) if new_path.is_file() else None,
                old_loc=count_lines(old_path) if old_path.is_file() else None,
            )
        )
    return rows


def build_report(repo: Path, baseline_path: Path = DEFAULT_BASELINE) -> LocReport:
    """Measure the repository against the donor table."""
    repo = repo.resolve()
    if not repo.is_dir():
        raise BaselineError(f"--repo is not a directory: {repo}")
    payload = load_baseline(baseline_path)
    consolidations = payload["consolidations"]
    tests = payload["tests"]
    assert isinstance(consolidations, dict) and isinstance(tests, dict)

    report = LocReport(
        repo=str(repo),
        baseline=str(baseline_path.resolve()),
        metric=str(payload["metric"]),
        test_floor=int(tests["floor_loc"]),
        test_paths=[str(p) for p in tests["paths"]],
        total_python_loc_before=payload.get("total_python_loc_before"),
    )

    for key in sorted(consolidations):
        entry = consolidations[key]
        module_path = repo / str(entry["module"])
        exists = module_path.is_file()
        actual = count_lines(module_path) if exists else None
        if actual is None:
            status = STATUS_PENDING
        else:
            status = STATUS_PASS if actual < int(entry["donor_total"]) else STATUS_FAIL
        consumers = entry.get("ported_consumers")
        report.consolidations.append(
            ConsolidationResult(
                key=key,
                capability=str(entry["capability"]),
                module=str(entry["module"]),
                justification=str(entry.get("justification", "consolidation")),
                ported_consumers=consumers if isinstance(consumers, int) else None,
                donor_total=int(entry["donor_total"]),
                donor_loc={str(k): int(v) for k, v in entry["donors"].items()},
                actual_loc=actual,
                status=status,
            )
        )

    test_files: list[Path] = []
    for relative in report.test_paths:
        candidate = repo / relative
        if candidate.exists():
            test_files.extend(python_files_under(candidate))
    report.test_loc = sum(count_lines(path) for path in sorted(set(test_files)))

    tracked, method = tracked_python_files(repo)
    report.total_python_loc = sum(count_lines(path) for path in tracked)
    report.total_loc_method = method
    return report


def print_report(report: LocReport, stream: TextIO | None = None) -> None:
    """Human-readable receipt, matching the JSON."""
    out = sys.stdout if stream is None else stream

    def line(text: str = "") -> None:
        out.write(text + "\n")

    line("G2 net line reduction per consolidation")
    line(f"  repo: {report.repo}")
    line(f"  baseline: {report.baseline}")
    line(f"  metric: {report.metric}")
    for item in report.consolidations:
        consumers = "unrecorded" if item.ported_consumers is None else str(item.ported_consumers)
        line(f"  {item.key} — {item.capability} -> {item.module}")
        line(f"    donors: {len(item.donor_loc)} files, {item.donor_total} LOC")
        for donor in sorted(item.donor_loc):
            line(f"      {item.donor_loc[donor]:>5}  {donor}")
        if item.actual_loc is None:
            line(f"    module LOC: pending (module not present)   status: {item.status}")
        else:
            line(
                f"    module LOC: {item.actual_loc}  (donor sum {item.donor_total}, "
                f"reduction {item.reduction})   status: {item.status}"
            )
        line(f"    ported consumers: {consumers}   justification: {item.justification}")
    line(f"  tests: {report.test_loc} LOC over {', '.join(report.test_paths)} (floor {report.test_floor})")
    line(f"    status: {'pass' if report.tests_pass else 'fail'}")
    before = "unrecorded" if report.total_python_loc_before is None else str(report.total_python_loc_before)
    line(f"  total Python LOC: {report.total_python_loc} via {report.total_loc_method} (before: {before})")
    if report.parity:
        grew = sum(1 for row in report.parity if row.delta is not None and row.delta > 0)
        line(f"  port parity (informational): {len(report.parity)} pairs, {grew} grew")
        for row in report.parity:
            new_loc = "absent" if row.new_loc is None else str(row.new_loc)
            old_loc = "absent" if row.old_loc is None else str(row.old_loc)
            delta = "" if row.delta is None else f"  ({row.delta:+d}{'  GREW' if row.delta > 0 else ''})"
            line(f"    {row.new_path}: {new_loc} vs {row.old_path}: {old_loc}{delta}")
    line(f"  verdict: {'PASS' if report.passed else 'FAIL'}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="g2_loc_report.py",
        description="G2: each consolidated module strictly smaller than its donors; tests not shrinking.",
    )
    parser.add_argument("--repo", type=Path, default=Path("."), help="repository root to measure")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE, help="donor table (JSON)")
    parser.add_argument("--json", type=Path, default=None, help="write the receipt here")
    parser.add_argument(
        "--parity",
        type=Path,
        default=None,
        help="parity map JSON ({old_root, pairs}); adds the informational ported-vs-donor LOC table",
    )
    args = parser.parse_args(argv)

    try:
        report = build_report(repo=args.repo, baseline_path=args.baseline)
        if args.parity is not None:
            old_root, pairs = load_parity(args.parity)
            report.parity = build_parity(args.repo.resolve(), old_root, pairs)
    except BaselineError as exc:
        print(f"g2_loc_report: {exc}", file=sys.stderr)
        return 2

    print_report(report)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"  receipt: {args.json}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
