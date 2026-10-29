#!/usr/bin/env python3
"""G3 — public voice: the pages a stranger reads stay at least as legible as they are.

The two G3 rules above this one police what documentation says. This check reads
how it says it. It measures the processing-difficulty features in
`tools/legibility_measure.py` — arc length, object relatives with and without
"that", stacked nouns, things as the subjects of mind verbs, verbs made of nouns,
sentences over 30 words — on every public page, and holds each page to what it
measured when it was last accepted.

What a public page is
---------------------
Every Markdown file in the repository, and the module docstring of every Python
file one of those pages links to. A docstring the README sends a reader to is read
as a page, so it is measured as one. Code blocks, inline code, tables and link
targets are not prose and are dropped before anything is counted.

The ratchet
-----------
`tools/legibility_baseline.json` holds each page's measurement. A page in the
baseline fails when a feature's rate rises past

    baseline rate x (1 + RELATIVE_TOLERANCE)  +  INSTANCE_ALLOWANCE per page

where the per-page allowance is that many occurrences at the page's current length,
or when its arc length rises by more than DEP_LEN_TOLERANCE of the baseline. The
ten percent absorbs a parse that differs in its last digits on another machine; the
one occurrence lets a sentence be added to a short page without the gate reading a
single clause as a trend. A page that gets better passes, and the check names the
features that could be tightened; `--update-baseline` writes the new numbers in.

A page not yet in the baseline is held to fixed limits instead (`NEW_PAGE_LIMITS`):
no feature worse than the median page the baseline was first written from. A page
under NEW_PAGE_MIN_WORDS is reported and not held to them, because one clause moves
a short page's rate further than any limit is wide. A baseline entry whose page has
gone is stale and fails, so the file stays a list of pages that exist.

The first `--update-baseline`, with no baseline file present, records every page
as it stands. After that, a regression is accepted only in writing: `--update-baseline --reason "..."` writes
the numbers and records the reason on each page that would have failed. Without a
reason the update refuses.

Flourishes
----------
A passage written as voice rather than exposition — a definition meant to be
quoted, a line that carries the project's stance — is fenced, and the fence states
why:

    <!-- legibility: flourish — the one-line definition every page quotes -->
    ...
    <!-- /legibility -->

The fenced text is skipped. A fence without a reason, an unclosed fence, a close
without an open, or a nested open fails the check, and every fence and its reason
is listed in the receipt, so the exemptions stay few and visible.

Usage
-----
    python -m tools.check_legibility --json g3-legibility.json
    python -m tools.check_legibility --page README.md          # one page, with its flags
    python -m tools.check_legibility --update-baseline [--reason "..."]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence, TextIO

from tools.check_timelessness import SKIP_DIRS
from tools.legibility_measure import (
    COUNTED,
    DEFAULT_REFERENCE,
    PARSER_MODEL,
    REFERENCED,
    LegibilityError,
    PageMetrics,
    Reference,
    docstring_markdown,
    load_parser,
    load_reference,
    md_text,
    measure,
)

DEFAULT_REPO = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = Path(__file__).resolve().parent / "legibility_baseline.json"

#: Directories that hold no public page: caches, environments, run outputs, build products.
PAGE_SKIP_DIRS = SKIP_DIRS | frozenset({"outputs", "receipts", "dist", "build", ".pytest_cache"})

#: The counted features the gate holds. An overt object relative ("the gates that a
#: change passes") is not among them on its own: giving a reduced relative its "that"
#: is the rewrite the contributing guide asks for, and it turns one `orc_reduced` into
#: one `orc`. So the gate holds the reduced ones, and all object relatives together,
#: and reports the overt count without holding it.
GATED = tuple(feature for feature in COUNTED if feature != "orc")
GATED_REFERENCED = tuple(feature for feature in REFERENCED if feature != "orc")
#: The flagged sentences that explain a derived feature.
FLAG_SOURCES: dict[str, tuple[str, ...]] = {"orc_all": ("orc", "orc_reduced")}

RELATIVE_TOLERANCE = 0.10
INSTANCE_ALLOWANCE = 1
DEP_LEN_TOLERANCE = 0.03

#: A new page below this many words is reported, not held to `NEW_PAGE_LIMITS`.
NEW_PAGE_MIN_WORDS = 150

#: Ceilings for a page the baseline does not hold yet. The arc length and the four
#: referenced features are multiples of the reference; `denominal` and `long` have no
#: reference rate and are per 1,000 words. Each is the median of the ten public pages
#: the baseline was first written from, rounded up, so a new page is held to the
#: middle of what is already here rather than to an ideal nothing here meets. The
#: counted features also get the ratchet's `INSTANCE_ALLOWANCE` on top, so a single
#: clause on a short page is not read as a rate.
NEW_PAGE_LIMITS: dict[str, float] = {
    "dep_len": 1.05,
    "orc_reduced": 8.0,
    "orc_all": 8.0,
    "stacked": 2.0,
    "inanimate": 3.5,
    "denominal": 1.0,
    "long": 11.0,
}

FENCE_OPEN_RE = re.compile(r"<!--\s*legibility:\s*flourish\s*(?:[—–-]\s*(?P<reason>.*?))?\s*-->", re.S)
FENCE_CLOSE_RE = re.compile(r"<!--\s*/legibility\s*-->")
#: Anything that looks like a fence marker, so a malformed one is caught rather than
#: silently read as an ordinary comment.
FENCE_ANY_RE = re.compile(r"<!--\s*/?\s*legibility\b.*?-->", re.S)
MD_LINK_RE = re.compile(r"\]\(([^)\s#]+\.py)(?:#[^)]*)?\)")


@dataclass(frozen=True)
class Flourish:
    """One fenced passage: where it is, why, and how many words it takes out of the count."""

    page: str
    line: int
    reason: str
    words: int


@dataclass
class Fenced:
    """A page's text with its flourishes removed, and what the fences said."""

    text: str
    flourishes: list[Flourish] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def code_ranges(text: str) -> list[tuple[int, int]]:
    """The character ranges `md_text` drops as code: fenced blocks, then inline spans.

    A fence marker inside one of them is an example being shown, not a fence.
    """
    blocks = [(m.start(), m.end()) for m in re.finditer(r"```.*?```", text, flags=re.S)]
    masked = list(text)
    for start, end in blocks:
        masked[start:end] = "x" * (end - start)
    inline = [(m.start(), m.end()) for m in re.finditer(r"`[^`]*`", "".join(masked))]
    return blocks + inline


def strip_flourishes(text: str, page: str) -> Fenced:
    """Remove every fenced passage, and report each fence that is not well formed."""
    out: list[str] = []
    flourishes: list[Flourish] = []
    problems: list[str] = []
    cursor = 0
    open_at: tuple[int, int, str] | None = None
    code = code_ranges(text)
    for marker in FENCE_ANY_RE.finditer(text):
        if any(start <= marker.start() < end for start, end in code):
            continue
        line = text.count("\n", 0, marker.start()) + 1
        opened = FENCE_OPEN_RE.fullmatch(marker.group(0))
        closed = FENCE_CLOSE_RE.fullmatch(marker.group(0))
        if opened is not None:
            reason = (opened.group("reason") or "").strip()
            if open_at is not None:
                problems.append(f"{page}:{line}: a flourish fence opens inside the one opened at line {open_at[1]}")
                continue
            if not reason:
                problems.append(f"{page}:{line}: a flourish fence states no reason")
            out.append(text[cursor:marker.start()])
            open_at = (marker.end(), line, reason)
        elif closed is not None:
            if open_at is None:
                problems.append(f"{page}:{line}: a flourish fence closes without opening")
                continue
            body = text[open_at[0]:marker.start()]
            words = len(re.findall(r"[A-Za-z][A-Za-z'\-]*", md_text(body)))
            flourishes.append(Flourish(page=page, line=open_at[1], reason=open_at[2], words=words))
            cursor = marker.end()
            open_at = None
        else:
            problems.append(
                f"{page}:{line}: unreadable legibility marker {marker.group(0)!r}; "
                "a fence is `<!-- legibility: flourish — <reason> -->` … `<!-- /legibility -->`"
            )
    if open_at is not None:
        problems.append(f"{page}:{open_at[1]}: a flourish fence is never closed")
        out.append(text[open_at[0]:])
    else:
        out.append(text[cursor:])
    return Fenced(text="".join(out), flourishes=flourishes, problems=problems)


def discover_pages(repo: Path) -> list[str]:
    """Every public page, as a repository-relative POSIX path, sorted.

    The Markdown files, and each Python file a Markdown file links to.
    """
    markdown: list[Path] = []
    for path in sorted(repo.rglob("*.md")):
        relative = path.relative_to(repo)
        if any(part in PAGE_SKIP_DIRS or part.startswith(".") for part in relative.parts[:-1]):
            continue
        markdown.append(path)
    pages = {path.relative_to(repo).as_posix() for path in markdown}
    root = repo.resolve()
    for path in markdown:
        for link in MD_LINK_RE.findall(path.read_text(encoding="utf-8")):
            target = (path.parent / link).resolve()
            if target.is_file() and target.is_relative_to(root):
                pages.add(target.relative_to(root).as_posix())
    return sorted(pages)


def page_prose(repo: Path, page: str) -> Fenced:
    """A page's prose, flourishes removed, ready to measure."""
    path = repo / page
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise LegibilityError(f"{page}: unreadable ({exc})") from exc
    if page.endswith(".py"):
        source = docstring_markdown(source, filename=page)
    fenced = strip_flourishes(source, page)
    fenced.text = md_text(fenced.text)
    return fenced


# ---------------------------------------------------------------------------- the ratchet
def metrics_entry(metrics: PageMetrics) -> dict[str, object]:
    """A page's measurement as it is written into the baseline."""
    return {
        "words": metrics.words,
        "sentences": metrics.sentences,
        "dep_len": round(metrics.dep_len, 4),
        "rates": {feature: round(metrics.rate(feature), 4) for feature in COUNTED},
    }


@dataclass(frozen=True)
class Excess:
    """One feature over its allowance."""

    page: str
    feature: str
    baseline: float | None
    now: float
    allowed: float


def regressions(page: str, metrics: PageMetrics, entry: dict[str, object]) -> list[Excess]:
    """Each feature of a baselined page that has risen past its tolerance."""
    out: list[Excess] = []
    base_dep = float(entry["dep_len"])  # type: ignore[arg-type]
    allowed_dep = base_dep * (1 + DEP_LEN_TOLERANCE)
    if metrics.dep_len > allowed_dep + 1e-9:
        out.append(Excess(page, "dep_len", base_dep, metrics.dep_len, allowed_dep))
    rates = entry["rates"]
    assert isinstance(rates, dict)
    per_instance = 1000.0 / metrics.words if metrics.words else 0.0
    for feature in GATED:
        base = float(rates.get(feature, 0.0))
        allowed = base * (1 + RELATIVE_TOLERANCE) + INSTANCE_ALLOWANCE * per_instance
        now = metrics.rate(feature)
        if now > allowed + 1e-9:
            out.append(Excess(page, feature, base, now, allowed))
    return out


def tightenable(metrics: PageMetrics, entry: dict[str, object]) -> list[str]:
    """Features that have improved by more than the tolerance: the baseline can follow them down."""
    out: list[str] = []
    if metrics.dep_len < float(entry["dep_len"]) * (1 - DEP_LEN_TOLERANCE):  # type: ignore[arg-type]
        out.append("dep_len")
    rates = entry["rates"]
    assert isinstance(rates, dict)
    per_instance = 1000.0 / metrics.words if metrics.words else 0.0
    for feature in GATED:
        base = float(rates.get(feature, 0.0))
        if metrics.rate(feature) < base * (1 - RELATIVE_TOLERANCE) - INSTANCE_ALLOWANCE * per_instance:
            out.append(feature)
    return out


def over_new_page_limits(page: str, metrics: PageMetrics, ref: Reference) -> list[Excess]:
    """Each feature of a page outside the baseline that is past `NEW_PAGE_LIMITS`."""
    ratios = metrics.ratios(ref)
    per_instance = INSTANCE_ALLOWANCE * 1000.0 / metrics.words if metrics.words else 0.0
    out: list[Excess] = []
    if ratios["dep_len"] > NEW_PAGE_LIMITS["dep_len"] + 1e-9:
        out.append(Excess(page, "dep_len", None, ratios["dep_len"], NEW_PAGE_LIMITS["dep_len"]))
    for feature in GATED_REFERENCED:
        allowed = NEW_PAGE_LIMITS[feature] + per_instance / ref.rates[feature]
        if ratios[feature] > allowed + 1e-9:
            out.append(Excess(page, feature, None, ratios[feature], allowed))
    for feature in ("denominal", "long"):
        allowed = NEW_PAGE_LIMITS[feature] + per_instance
        if metrics.rate(feature) > allowed + 1e-9:
            out.append(Excess(page, feature, None, metrics.rate(feature), allowed))
    return out


# ---------------------------------------------------------------------------- the report
@dataclass
class PageResult:
    page: str
    metrics: PageMetrics
    status: str  # "baselined", "new", "short"
    excess: list[Excess] = field(default_factory=list)
    tighten: list[str] = field(default_factory=list)


@dataclass
class LegibilityReport:
    """The result of a pass, serialisable as the receipt."""

    repo: str
    baseline: str
    reference: str
    pages: list[PageResult] = field(default_factory=list)
    flourishes: list[Flourish] = field(default_factory=list)
    fence_problems: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)

    @property
    def failing(self) -> list[PageResult]:
        return [result for result in self.pages if result.excess]

    @property
    def passed(self) -> bool:
        return not self.failing and not self.fence_problems and not self.stale

    def to_json(self, ref: Reference) -> dict[str, object]:
        return {
            "gate": "G3-legibility",
            "repo": self.repo,
            "baseline": self.baseline,
            "reference": self.reference,
            "tolerance": {
                "relative": RELATIVE_TOLERANCE,
                "instances": INSTANCE_ALLOWANCE,
                "dep_len_relative": DEP_LEN_TOLERANCE,
            },
            "new_page_limits": NEW_PAGE_LIMITS,
            "pages": {
                result.page: {
                    "status": result.status,
                    **metrics_entry(result.metrics),
                    "ratio_to_reference": {
                        key: round(value, 3) for key, value in result.metrics.ratios(ref).items()
                    },
                    "excess": [vars(e) for e in result.excess],
                    "tightenable": result.tighten,
                }
                for result in self.pages
            },
            "flourishes": [vars(f) for f in self.flourishes],
            "fence_problems": self.fence_problems,
            "stale_baseline_entries": self.stale,
            "passed": self.passed,
        }


def load_baseline(path: Path) -> dict[str, dict[str, object]]:
    """The baselined pages, or an empty mapping when no baseline file exists yet."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        pages = raw["pages"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise LegibilityError(f"baseline unreadable: {path} ({exc})") from exc
    if not isinstance(pages, dict):
        raise LegibilityError(f"baseline malformed: {path} (`pages` is not a mapping)")
    for page, entry in pages.items():
        if not isinstance(entry, dict) or "dep_len" not in entry or not isinstance(entry.get("rates"), dict):
            raise LegibilityError(f"baseline malformed: {path} (entry {page!r})")
    return pages


Measurer = Callable[[str], PageMetrics]


def build_report(
    repo: Path,
    baseline_path: Path,
    ref: Reference,
    measurer: Measurer,
    only: Sequence[str] | None = None,
) -> LegibilityReport:
    """Measure every public page (or only the named ones) and hold each to its rule."""
    repo = repo.resolve()
    baseline = load_baseline(baseline_path)
    discovered = discover_pages(repo)
    if only:
        unknown = [page for page in only if page not in discovered]
        if unknown:
            raise LegibilityError(f"not a public page: {unknown}; public pages are {discovered}")
        targets = list(only)
    else:
        targets = discovered
    report = LegibilityReport(repo=str(repo), baseline=str(baseline_path), reference=ref.source)
    for page in targets:
        fenced = page_prose(repo, page)
        report.flourishes.extend(fenced.flourishes)
        report.fence_problems.extend(fenced.problems)
        metrics = measurer(fenced.text)
        if page in baseline:
            result = PageResult(page, metrics, "baselined", regressions(page, metrics, baseline[page]),
                                tightenable(metrics, baseline[page]))
        elif metrics.words < NEW_PAGE_MIN_WORDS:
            result = PageResult(page, metrics, "short")
        else:
            result = PageResult(page, metrics, "new", over_new_page_limits(page, metrics, ref))
        report.pages.append(result)
    if not only:
        report.stale = sorted(set(baseline) - set(discovered))
    return report


def write_baseline(path: Path, report: LegibilityReport, reason: str | None) -> list[str]:
    """Write every measured page into the baseline; returns the pages the reason covers.

    A page that fails, or a fence problem, refuses the update unless a reason is
    given, and the reason is recorded on each failing page's entry. When no baseline
    file exists, the first one records every page as it stands: there is nothing yet
    for a page to have regressed from.
    """
    failing = [] if not path.exists() else [result.page for result in report.failing]
    if report.fence_problems:
        raise LegibilityError("fix the flourish fences before writing a baseline: " + "; ".join(report.fence_problems))
    if failing and not reason:
        raise LegibilityError(
            f"{failing} would fail; a regression is accepted only with --reason, which is recorded beside it"
        )
    pages: dict[str, object] = {}
    for result in sorted(report.pages, key=lambda r: r.page):
        entry = metrics_entry(result.metrics)
        if result.page in failing:
            entry["accepted_regression"] = reason
        pages[result.page] = entry
    payload = {
        "reference": report.reference,
        "parser": PARSER_MODEL,
        "pages": pages,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return failing


def print_report(report: LegibilityReport, ref: Reference, show_flags: bool, stream: TextIO | None = None) -> None:
    """Human-readable receipt, matching the JSON."""
    out = sys.stdout if stream is None else stream

    def line(text: str = "") -> None:
        out.write(text + "\n")

    line("G3 public voice (legibility)")
    line(f"  reference: {report.reference}")
    line(f"  baseline: {report.baseline}")
    line(f"  pages: {len(report.pages)}  flourishes: {len(report.flourishes)}")
    for result in report.pages:
        m = result.metrics
        ratios = m.ratios(ref)
        verdict = "FAIL" if result.excess else "ok"
        line(f"  [{verdict}] {result.page} ({result.status}; {m.words} words, {m.sentences} sentences)")
        line(
            f"      dep_len ×{ratios['dep_len']:.2f}  "
            + "  ".join(f"{feature} ×{ratios[feature]:.2f}" for feature in REFERENCED)
            + f"  denominal {m.rate('denominal'):.2f}/1k  long {m.rate('long'):.2f}/1k"
        )
        for excess in result.excess:
            was = "limit" if excess.baseline is None else f"baseline {excess.baseline:.3f}"
            line(f"      {excess.feature}: {excess.now:.3f} > allowed {excess.allowed:.3f} ({was})")
        if result.tighten:
            line(f"      improved past tolerance, baseline can tighten: {', '.join(result.tighten)}")
        if show_flags or result.excess:
            features = [e.feature for e in result.excess] if not show_flags else list(m.flags)
            sources = [source for feature in features for source in FLAG_SOURCES.get(feature, (feature,))]
            for feature in dict.fromkeys(sources):
                for flag in m.flags.get(feature, [])[:12]:
                    line(f"        {feature} [{flag['match']}] {flag['sentence'][:160]}")
    for flourish in report.flourishes:
        line(f"  flourish {flourish.page}:{flourish.line} ({flourish.words} words): {flourish.reason}")
    for problem in report.fence_problems:
        line(f"  FENCE {problem}")
    for page in report.stale:
        line(f"  STALE baseline entry with no page: {page}")
    line(f"  verdict: {'PASS' if report.passed else 'FAIL'}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_legibility.py",
        description="G3 public voice: no public page gets harder to read than its baseline.",
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO, help="repository root")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE, help="per-page baseline (JSON)")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="reference statistics (JSON)")
    parser.add_argument("--page", action="append", default=None, help="measure only this page (repeatable)")
    parser.add_argument("--flags", action="store_true", help="print every flagged sentence, not only the failing ones")
    parser.add_argument("--update-baseline", action="store_true", help="write the current measurements as the baseline")
    parser.add_argument("--reason", default=None, help="why a regression is accepted (recorded in the baseline)")
    parser.add_argument("--json", type=Path, default=None, help="write the receipt here")
    args = parser.parse_args(argv)

    if args.update_baseline and args.page:
        print("check_legibility: --update-baseline measures every page; drop --page", file=sys.stderr)
        return 2
    try:
        ref = load_reference(args.reference)
        nlp = load_parser()
        report = build_report(
            repo=args.repo,
            baseline_path=args.baseline,
            ref=ref,
            measurer=lambda text: measure(text, nlp, ref),
            only=args.page,
        )
    except LegibilityError as exc:
        print(f"check_legibility: {exc}", file=sys.stderr)
        return 2

    print_report(report, ref, show_flags=args.flags)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_json(ref), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                             encoding="utf-8")
        print(f"  receipt: {args.json}")
    if args.update_baseline:
        try:
            covered = write_baseline(args.baseline, report, args.reason)
        except LegibilityError as exc:
            print(f"check_legibility: {exc}", file=sys.stderr)
            return 1
        print(f"  baseline written: {args.baseline}" + (f" (reason recorded on {covered})" if covered else ""))
        return 0
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
