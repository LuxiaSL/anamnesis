#!/usr/bin/env python3
"""G3 — documentation timelessness: code says what is true now, not what changed.

The gate targets documentation, which here means comments and docstrings. Every
other string literal is data — a fixture, a log line, an error message, a JSON
payload — and data is free to carry a date or a phrase that prose may not; a
checker that policed it would cry wolf on ported code. `documentation_lines`
draws that boundary, and all three rules apply only inside it:

  1. **Marker comments.** A comment opening with one of the three deferral
     markers is a note to a future reader that the code does not keep.
  2. **Changelog phrasing.** Prose that narrates an edit rather than the state.
     The phrase set lives in `CHANGELOG_RULES`.
  3. **Dated comments.** A date in a comment, unless the line also carries a
     citation marker from `timelessness_allowlist.txt`. A date on evidence
     stays; a date on an edit goes. The allowlist is versioned in the repo
     beside this checker, one regex per line, because which idioms count as
     citation is a judgment that accumulates rather than a closed set.

The patterns are written with the last character of each phrase bracketed
(`previousl[y]`), so this file does not match itself and can be scanned by the
same gate it implements.

The past-habitual rule carries a guard: the phrase is flagged only when a word
follows it and no passive auxiliary precedes it on the same line. In the passive
voice the same two words mean "for the purpose of" and state a present-tense
constraint, which passes; said of a subject they narrate a state the code has
left behind, which does not.

Usage
-----
    python tools/check_timelessness.py --root anamnesis tools --json g3.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence, TextIO

DEFAULT_ALLOWLIST = Path(__file__).resolve().parent / "timelessness_allowlist.txt"

SKIP_DIRS = frozenset({"__pycache__", ".git", ".venv", "venv", "node_modules", ".mypy_cache"})

MARKER_RULE = "marker-comment"
DATED_RULE = "dated-comment"

MARKER_RE = re.compile(r"#\s*(TOD[O]|FIXM[E]|HAC[K])\b")
DATE_COMMENT_RE = re.compile(r"#.*(?:19|20)[0-9]{2}-[0-9]{2}-[0-9]{2}")
USED_TO_RE = re.compile(r"\bused t[o]\s+([A-Za-z]+)", re.IGNORECASE)
PASSIVE_AUXILIARY_RE = re.compile(
    r"\b(is|are|was|were|be|been|being|get|gets|got|become|becomes)\b", re.IGNORECASE
)

CHANGELOG_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("changelog-prior-state", re.compile(r"\bpreviousl[y]\b", re.IGNORECASE)),
    ("changelog-changed-in", re.compile(r"\bchanged i[n]\b", re.IGNORECASE)),
    ("changelog-we-now", re.compile(r"\bwe no[w]\b", re.IGNORECASE)),
    ("changelog-no-longer", re.compile(r"\bno longe[r]\b", re.IGNORECASE)),
)

USED_TO_RULE = "changelog-used-to"


class TimelessnessError(RuntimeError):
    """A condition that makes the check unrunnable rather than failing."""


@dataclass(frozen=True)
class Violation:
    """One flagged line."""

    rule: str
    file: str
    line: int
    match: str
    text: str


@dataclass
class TimelessnessReport:
    """The result of a timelessness pass, serialisable as the G3 receipt."""

    roots: list[str]
    allowlist: str
    allowlist_size: int
    files_scanned: int = 0
    violations: list[Violation] = field(default_factory=list)
    read_errors: list[str] = field(default_factory=list)
    exempted: list[Violation] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations and not self.read_errors

    def counts_by_rule(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for violation in self.violations:
            counts[violation.rule] = counts.get(violation.rule, 0) + 1
        return dict(sorted(counts.items()))

    def to_json(self) -> dict[str, object]:
        return {
            "gate": "G3",
            "roots": self.roots,
            "allowlist": self.allowlist,
            "allowlist_size": self.allowlist_size,
            "counts": {
                "files_scanned": self.files_scanned,
                "violations": len(self.violations),
                "by_rule": self.counts_by_rule(),
                "exempted_dated_comments": len(self.exempted),
                "read_errors": len(self.read_errors),
            },
            "violations": [vars(v) for v in self.violations],
            "exempted_dated_comments": [vars(v) for v in self.exempted],
            "read_errors": self.read_errors,
            "passed": self.passed,
        }


def load_allowlist(path: Path) -> list[re.Pattern[str]]:
    """Compile the citation-marker allowlist: one regex per line, `#` comments."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TimelessnessError(f"allowlist unreadable: {path} ({exc})") from exc
    patterns: list[re.Pattern[str]] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            patterns.append(re.compile(stripped, re.IGNORECASE))
        except re.error as exc:
            raise TimelessnessError(f"{path}:{number}: invalid regex {stripped!r} ({exc})") from exc
    return patterns


def iter_python_files(root: Path) -> list[Path]:
    """Every `.py` file under `root`, excluding caches and virtual environments."""
    if root.is_file():
        return [root]
    out: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        out.append(path)
    return out


DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def documentation_lines(path: Path, source: str) -> set[int]:
    """Line numbers that hold documentation: a comment, or a docstring.

    Comments come from the tokenizer, docstrings from the parse tree — the
    first-statement string of a module, class or function. Every other string
    literal is data: a fixture, a log message, a JSON payload, an error string.
    Data may legitimately contain a date or a phrase this checker forbids in
    prose, so the rules stop at the documentation boundary the tokenizer and the
    parse tree draw between them.
    """
    lines: set[int] = set()
    try:
        tokens = tokenize.generate_tokens(iter(source.splitlines(keepends=True)).__next__)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                lines.add(token.start[0])
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        raise TimelessnessError(f"{path}: tokenize failed ({exc})") from exc
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise TimelessnessError(f"{path}: syntax error at line {exc.lineno} ({exc.msg})") from exc
    for node in ast.walk(tree):
        if not isinstance(node, DOCSTRING_OWNERS):
            continue
        body = node.body
        if not body or not isinstance(body[0], ast.Expr):
            continue
        value = body[0].value
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            continue
        end = value.end_lineno or value.lineno
        lines.update(range(value.lineno, end + 1))
    return lines


def used_to_violation(text: str) -> str | None:
    """The matched phrase when `used to` narrates an edit, else None."""
    match = USED_TO_RE.search(text)
    if match is None:
        return None
    prefix = text[: match.start()]
    if PASSIVE_AUXILIARY_RE.search(prefix):
        return None
    return match.group(0)


def scan_source(path: Path, source: str, allowlist: Sequence[re.Pattern[str]]) -> tuple[list[Violation], list[Violation]]:
    """Flag one file; returns (violations, exempted dated comments)."""
    doc_lines = documentation_lines(path, source)
    violations: list[Violation] = []
    exempted: list[Violation] = []
    file_str = str(path)
    for number, raw in enumerate(source.splitlines(), start=1):
        if number not in doc_lines:
            continue
        text = raw.rstrip("\n")
        marker = MARKER_RE.search(text)
        if marker is not None:
            violations.append(
                Violation(rule=MARKER_RULE, file=file_str, line=number, match=marker.group(0), text=text.strip())
            )
        for rule, pattern in CHANGELOG_RULES:
            found = pattern.search(text)
            if found is not None:
                violations.append(
                    Violation(rule=rule, file=file_str, line=number, match=found.group(0), text=text.strip())
                )
        phrase = used_to_violation(text)
        if phrase is not None:
            violations.append(
                Violation(rule=USED_TO_RULE, file=file_str, line=number, match=phrase, text=text.strip())
            )
        dated = DATE_COMMENT_RE.search(text)
        if dated is not None:
            record = Violation(
                rule=DATED_RULE, file=file_str, line=number, match=dated.group(0), text=text.strip()
            )
            if any(pattern.search(text) for pattern in allowlist):
                exempted.append(record)
            else:
                violations.append(record)
    return violations, exempted


def build_report(
    roots: Sequence[Path],
    allowlist_path: Path = DEFAULT_ALLOWLIST,
) -> TimelessnessReport:
    """Run the check over every `.py` file under the given roots."""
    resolved: list[Path] = []
    for root in roots:
        candidate = root.resolve()
        if not candidate.exists():
            raise TimelessnessError(f"--root does not exist: {candidate}")
        resolved.append(candidate)
    if not resolved:
        raise TimelessnessError("at least one --root is required")

    allowlist = load_allowlist(allowlist_path)
    report = TimelessnessReport(
        roots=[str(r) for r in resolved],
        allowlist=str(allowlist_path.resolve()),
        allowlist_size=len(allowlist),
    )

    seen: set[Path] = set()
    for root in resolved:
        for path in iter_python_files(root):
            if path in seen:
                continue
            seen.add(path)
            report.files_scanned += 1
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                report.read_errors.append(f"{path}: unreadable ({exc})")
                continue
            try:
                violations, exempted = scan_source(path, source, allowlist)
            except TimelessnessError as exc:
                report.read_errors.append(str(exc))
                continue
            report.violations.extend(violations)
            report.exempted.extend(exempted)
    report.violations.sort(key=lambda v: (v.file, v.line, v.rule))
    report.exempted.sort(key=lambda v: (v.file, v.line))
    return report


def print_report(report: TimelessnessReport, stream: TextIO | None = None) -> None:
    """Human-readable receipt, matching the JSON."""
    out = sys.stdout if stream is None else stream

    def line(text: str = "") -> None:
        out.write(text + "\n")

    line("G3 documentation timelessness")
    line(f"  roots: {', '.join(report.roots)}")
    line(f"  allowlist: {report.allowlist} ({report.allowlist_size} patterns)")
    line(f"  files scanned: {report.files_scanned}")
    line(f"  dated comments exempted as citations: {len(report.exempted)}")
    line(f"  violations: {len(report.violations)}")
    for rule, count in report.counts_by_rule().items():
        line(f"    {rule}: {count}")
    for violation in report.violations:
        line(f"    {violation.file}:{violation.line} [{violation.rule}] {violation.text}")
    if report.read_errors:
        line(f"  READ ERRORS: {len(report.read_errors)}")
        for error in report.read_errors:
            line(f"    {error}")
    line(f"  verdict: {'PASS' if report.passed else 'FAIL'}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_timelessness.py",
        description="G3: no marker comments, no changelog phrasing, no uncited dated comments.",
    )
    parser.add_argument("--root", required=True, nargs="+", type=Path, help="directories or files to scan")
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=DEFAULT_ALLOWLIST,
        help="citation-marker allowlist (one regex per line)",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the receipt here")
    args = parser.parse_args(argv)

    try:
        report = build_report(roots=args.root, allowlist_path=args.allowlist)
    except TimelessnessError as exc:
        print(f"check_timelessness: {exc}", file=sys.stderr)
        return 2

    print_report(report)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"  receipt: {args.json}")
    return 0 if report.passed else 1


def rule_names() -> Iterable[str]:
    """Every rule identifier this checker can emit."""
    return (MARKER_RULE, USED_TO_RULE, DATED_RULE, *(rule for rule, _ in CHANGELOG_RULES))


if __name__ == "__main__":
    raise SystemExit(main())
