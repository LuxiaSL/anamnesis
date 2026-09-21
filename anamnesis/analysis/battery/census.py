"""The sub-perceptual census: which rows the internals see and the cheap readers miss.

A row of the battery is a **class member** when its internals rung materially
exceeds both of the rungs that do not read internals. The three rungs are the
tested detectors of record, and the order matters more than any one of them:

* **content** — a trained text detector (TF-IDF under GroupKFold-by-topic), and
  where a zero-shot judge was run, whichever of the two is *higher*.
* **likelihood** — the banked surprisal probe, which reads the model's own
  distribution rather than the text.
* **internals** — the signature classifier.

``gap = internals - max(content, likelihood)``, and the bars are declared rather
than fitted: ``>= 0.10`` is a member, ``>= 0.03`` is borderline, below that is
excluded. Units are whatever the row's rungs are measured in — AUC for a binary
row, per-mode recall for a k-way one — which is why a row carries its unit and
rows in different units are never averaged.

Two asymmetries are load-bearing and are the reason this is code rather than a
spreadsheet. First, the content rung is a **maximum over a declared detector
set**: adding a detector can only raise it, so membership is monotone-conservative
— "not yet defeated by any detector we ran". Second, a **judge failure** is not
symmetric with a judge success. A judge that fails to tell two texts apart may
simply have been asked badly, so its failure defends membership only at the
hardened reading (a forced-choice contrast, plus a second judge family), while a
judge *success* defeats membership directly, because a blind reader succeeding is
a lower bound on what a hardened reader would do. That rule is enforced by
:func:`~anamnesis.analysis.battery.gates.reject_blind_judge_defense`, which this
module calls before it will produce a census at all.

The census re-runs at every scale point. Its object is not a row but the *set*:
how many members there are and how large their gaps are across models. A single
row licenses no claim about the class.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from anamnesis.analysis.battery.gates import reject_blind_judge_defense

logger = logging.getLogger(__name__)

MEMBER_BAR = 0.10
"""Gap at or above which a row is a class member (declared, not fitted)."""

BORDERLINE_BAR = 0.03
"""Gap at or above which a row is borderline rather than excluded."""

HARDENED_2AFC_BAR = 0.65
"""Forced-choice rate at which a judge counts as having discriminated. Above it,
that judge's *failure* elsewhere cannot defend a row."""

MODES: tuple[str, ...] = ("linear", "analogical", "socratic", "contrastive", "dialectical")
"""The five format-controlled modes a k-way row is read per-mode over."""

A1_RECORD_DIRS: tuple[str, ...] = ("A1", "A1_m3m4", "A1_m5", "A1_dsv2")
"""Arm directories the temperature row is read from, in reading order."""

A3_RECORD_DIRS: tuple[str, ...] = ("A3", "A3_m3", "A3_m5", "A3_dsv2")
"""Arm directories the mode rows are read from, in reading order."""

A3_ROW_N = 160
"""Generations behind one mode row's per-mode recall, by the arm's own design."""

MEMBER = "MEMBER"
BORDERLINE = "BORDERLINE"
EXCLUDED = "EXCLUDED"

PENDING_AND_APPENDIX: tuple[dict[str, str], ...] = (
    {
        "row": "A2:cell-ii:unexecuted-instruction-carriage",
        "status": "PENDING",
        "note": "embargoed behind the length-matched prefix control (Wave-2); "
                "enters the census when the control lands",
    },
    {
        "row": "A4/exp11:P3 eviction-kind vs token-KL",
        "status": "APPENDIX(pre-battery)",
        "note": "banked at n=12, kv-rotation exp11 (prereg p=0.0029); re-enters as a "
                "battery row when A4 runs; likelihood-rung analog = token-KL (exempt "
                "from judge hardening)",
    },
    {
        "row": "pre-battery:Run-1 uncertain/confident",
        "status": "APPENDIX(pre-battery)",
        "note": "phase-0 era; pointer only — no battery-grade rungs",
    },
    {
        "row": "pre-battery:wolf (subliminal)",
        "status": "APPENDIX(pre-battery)",
        "note": "subliminal_anamnesis repo; behavioral metric was the false negative — "
                "the class's founding exemplar; pointer only",
    },
)
"""Rows that are not census rows yet, carried so their absence is a statement
rather than a silence: one waiting on a control, three predating the battery."""


def classify(gap: float) -> str:
    """Member, borderline or excluded, by the declared bars."""
    if gap >= MEMBER_BAR:
        return MEMBER
    if gap >= BORDERLINE_BAR:
        return BORDERLINE
    return EXCLUDED


class CensusRow(BaseModel):
    """One row: three rungs, the gap between them, and what that makes it.

    ``judge`` is its own column rather than folded into ``content`` silently: the
    content rung is the maximum over the declared set including the judge, and the
    judge-gap is a separate quotable whose hardening status travels with it.
    """

    model_config = ConfigDict(extra="allow")

    row: str = Field(description="What was discriminated, in the arm's own vocabulary")
    model: str
    units: str = Field(description="AUC for a binary row, per-mode recall for a k-way one")
    content: float
    likelihood: float
    internals: float
    judge: float | None = None
    gap: float
    status: str
    n: Any = Field(default=None, description="What the row's rungs rest on")
    source_record: str = Field(description="The banked arm record this row was read from")


def _row_dicts(rows: Sequence[CensusRow]) -> list[dict[str, Any]]:
    return [row.model_dump() for row in rows]


def a1_rows(arms_root: Path, *, record_dirs: Iterable[str] = A1_RECORD_DIRS) -> list[CensusRow]:
    """The temperature row, per model, from every A1 record present.

    The binding rung is named: where the likelihood probe is the higher of the two
    non-internals rungs, the row binds on likelihood and no judge was involved, so
    judge hardening does not apply to it. Saying that in the row is what keeps a
    reader from asking for a hardening pass that would answer nothing.
    """
    rows: list[CensusRow] = []
    for record_dir in record_dirs:
        path = Path(arms_root) / record_dir / "a1_results.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for model, entry in payload["models"].items():
            dissociation = entry["dissociation"]
            content = float(dissociation["tfidf_groupkfold_auc_t03_vs_t09"])
            likelihood = float(dissociation["likelihood_mean_surprise_auc"])
            internals = float(dissociation["signature_output_groupkfold_auc"])
            gap = internals - max(content, likelihood)
            binds_on_likelihood = likelihood >= content
            rows.append(
                CensusRow(
                    row="A1:temperature(t03|t09)",
                    model=model,
                    units="AUC",
                    content=content,
                    likelihood=likelihood,
                    internals=internals,
                    judge=None,
                    gap=round(gap, 4),
                    status=classify(gap),
                    binding_rung="likelihood" if binds_on_likelihood else "content",
                    hardening=(
                        "exempt (likelihood-rung binding; no judge involved)"
                        if binds_on_likelihood
                        else "n/a"
                    ),
                    n=dissociation["n"],
                    source_record=f"arms/{record_dir}/a1_results.json",
                )
            )
    return rows


def load_2afc_rates(arms_root: Path) -> dict[str, float]:
    """Forced-choice rates per model, from whichever hardening tables are banked.

    An empty result means the hardening pass has not run, which is a different
    state from a judge having failed it — every row reads the absence as *pending*
    rather than as a pass.
    """
    rates: dict[str, float] = {}
    judge_dir = Path(arms_root) / "A3" / "judge"
    for path in sorted(judge_dir.glob("socratic_2afc_fable*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rates.update({model: float(v["acc_2afc"]) for model, v in payload["models"].items()})
    return rates


def _hardening_note(
    *,
    mode: str,
    model: str,
    judge_gap: float | None,
    rates: Mapping[str, float],
) -> str:
    """What a quotable judge-gap is allowed to be read as, in this row's state."""
    if judge_gap is None or judge_gap < MEMBER_BAR:
        return "n/a (judge-gap below member bar or no judge)"
    if mode == "socratic" and model in rates:
        if rates[model] >= HARDENED_2AFC_BAR:
            return (
                f"FAILED hardening (a): 2AFC {rates[model]:.3f} >> chance — the "
                "blind-k-way judge-gap is a contrast artifact; judge-gap NOT "
                "quotable as class evidence (membership rests on the "
                "trained-detector rung only)"
            )
        return "SURVIVES 2AFC (a); second-family (b) pending"
    return (
        "PENDING 2AFC + second judge family — required before any JUDGE-GAP quote "
        "as class evidence"
    )


def a3_rows(
    arms_root: Path,
    *,
    record_dirs: Iterable[str] = A3_RECORD_DIRS,
    modes: Sequence[str] = MODES,
) -> list[CensusRow]:
    """One row per mode per model, from every A3 record present.

    The content rung here is ``max(trained TF-IDF, zero-shot judge)``: extending
    the detector set can only raise the bar a row has to clear, which is what
    makes membership a conservative reading rather than an optimistic one.
    """
    rates = load_2afc_rates(arms_root)
    rows: list[CensusRow] = []
    for record_dir in record_dirs:
        path = Path(arms_root) / record_dir / "a3_results.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for model, entry in payload["models"].items():
            hierarchy = entry["hierarchy"]
            judged = entry.get("judge", {}).get("per_mode", {})
            for mode in modes:
                tfidf = float(hierarchy["content_tfidf"]["per_mode_recall"][mode])
                likelihood = float(hierarchy["likelihood_surprise"]["per_mode_recall"][mode])
                internals = float(hierarchy["internals_rf"]["per_mode_recall"][mode])
                judge = judged.get(mode, {}).get("judge_recall")
                content = max(tfidf, judge) if judge is not None else tfidf
                gap = internals - max(content, likelihood)
                judge_gap = (internals - judge) if judge is not None else None
                rows.append(
                    CensusRow(
                        row=f"A3:mode:{mode}",
                        model=model,
                        units="per-mode recall (5-way)",
                        content=content,
                        content_tfidf=tfidf,
                        likelihood=likelihood,
                        internals=internals,
                        judge=judge,
                        gap=round(gap, 4),
                        status=classify(gap),
                        judge_gap=round(judge_gap, 4) if judge_gap is not None else None,
                        hardening=_hardening_note(
                            mode=mode, model=model, judge_gap=judge_gap, rates=rates
                        ),
                        n=A3_ROW_N,
                        source_record=f"arms/{record_dir}/a3_results.json",
                    )
                )
    return rows


def class_object(rows: Sequence[CensusRow]) -> dict[str, dict[str, list[str]]]:
    """The set, per model: its members and its borderline rows.

    This is the census's actual object. A model's entry can be empty, and an empty
    entry is a reading — the class is nearly empty at that scale — not a gap in
    the data.
    """
    return {
        model: {
            "members": [r.row for r in rows if r.model == model and r.status == MEMBER],
            "borderline": [r.row for r in rows if r.model == model and r.status == BORDERLINE],
        }
        for model in sorted({r.model for r in rows})
    }


def census_document(rows: Sequence[CensusRow]) -> dict[str, Any]:
    """The banked census, with the bars and the definition beside the rows.

    The hardening gate runs here rather than at the call site: a census that has
    not been checked against the judge-defense rule must not be writable.
    """
    payload = _row_dicts(rows)
    reject_blind_judge_defense(payload)
    return {
        "bars": (
            f"MEMBER >= {MEMBER_BAR}, BORDERLINE >= {BORDERLINE_BAR} "
            "(declared implementation ruling; changeable by addendum only)"
        ),
        "definition": "gap = internals - max(content, likelihood); the three rungs of record",
        "rows": payload,
        "pending_and_appendix": [dict(entry) for entry in PENDING_AND_APPENDIX],
        "class_object": class_object(rows),
    }


def _escape_cell(text: str) -> str:
    """A row label with the table's own delimiter escaped.

    Row names spell a contrast the way the arm did — ``A1:temperature(t03|t09)`` — and
    an unescaped pipe splits that cell into two, which shifts every column after it.
    """
    return text.replace("|", "\\|")


def census_markdown(rows: Sequence[CensusRow]) -> str:
    """The census as a table, sorted by gap: the widest gaps read first."""
    lines = [
        "# Sub-perceptual census",
        "",
        f"gap = internals - max(content, likelihood); MEMBER >= {MEMBER_BAR}, "
        f"BORDERLINE >= {BORDERLINE_BAR}.",
        "content rung = MAX over the declared content-class detectors (trained TF-IDF,",
        "zero-shot judge); the judge-GAP (internals - judge) is a separate quotable whose",
        "hardening status travels with it.",
        "",
        "| row | model | content | likelihood | internals | judge | gap | status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: -r.gap):
        judge = f"{row.judge:.3f}" if row.judge is not None else "—"
        lines.append(
            f"| {_escape_cell(row.row)} | {row.model} | {row.content:.3f} | {row.likelihood:.3f} | "
            f"{row.internals:.3f} | {judge} | {row.gap:+.3f} | {row.status} |"
        )
    lines += ["", "## Pending / appendix"]
    for entry in PENDING_AND_APPENDIX:
        lines.append(f"- **{_escape_cell(entry['row'])}** [{entry['status']}] — {entry['note']}")
    return "\n".join(lines)


def run_census(arms_root: Path, out_dir: Path) -> list[CensusRow]:
    """Read every banked arm record, bank the census beside it, return the rows."""
    rows = a1_rows(Path(arms_root)) + a3_rows(Path(arms_root))
    if not rows:
        raise FileNotFoundError(
            f"no A1 or A3 records under {arms_root} — a census over no rows is not a census"
        )
    document = census_document(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "subperceptual_census.json").write_text(
        json.dumps(document, indent=1), encoding="utf-8"
    )
    (out_dir / "subperceptual_census.md").write_text(census_markdown(rows), encoding="utf-8")
    logger.info(f"census -> {out_dir} ({len(rows)} rows)")
    return rows


__all__ = [
    "A1_RECORD_DIRS",
    "A3_RECORD_DIRS",
    "BORDERLINE",
    "BORDERLINE_BAR",
    "CensusRow",
    "EXCLUDED",
    "HARDENED_2AFC_BAR",
    "MEMBER",
    "MEMBER_BAR",
    "MODES",
    "PENDING_AND_APPENDIX",
    "a1_rows",
    "a3_rows",
    "census_document",
    "census_markdown",
    "class_object",
    "classify",
    "load_2afc_rates",
    "run_census",
]
