"""Code gates: emission discipline an analyzer cannot talk its way past.

A rule held only in prose gets broken even with a vigilant reviewer, so the rules
below are assertions instead. An analyzer CANNOT emit what they forbid, so the
discipline survives a model swap.

Every gate raises GateError (never warns) — a blocked emission is a bug in the
caller, to be fixed at authoring time, not silenced.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


PENDING_HARDENING = "PENDING"
"""The hardening spelling that means the work has not been done.

A census row carries it while its hardening is outstanding, so a judge gap large
enough to quote may not rest on it.
"""


class GateError(AssertionError):
    """An emission-discipline gate refused a row."""


REQUIRED_STAMP_KEYS = ("n", "M", "law", "floor_type")


def require_stamp(row: Mapping[str, Any], context: str = "") -> None:
    """Every emitted number carries (n, M, law, floor_type).

    Call before writing any per-row result to a record file.
    """
    stamp = row.get("stamp")
    if not isinstance(stamp, Mapping):
        raise GateError(f"unstamped row{' in ' + context if context else ''}: "
                        f"{dict(row).get('cell', dict(row).get('row', row))!r}")
    missing = [k for k in REQUIRED_STAMP_KEYS if k not in stamp]
    if missing:
        raise GateError(f"stamp missing {missing}{' in ' + context if context else ''}")


def require_gated_outcome(row: Mapping[str, Any], outcome_key: str,
                          gate_keys: Sequence[str], context: str = "") -> None:
    """A categorical verdict exists ONLY alongside its own-tail BH gate fields.
    Point direction never ships as a verdict.
    """
    if outcome_key in row:
        missing = [k for k in gate_keys if k not in row]
        if missing:
            raise GateError(
                f"outcome {row[outcome_key]!r} emitted without gate fields "
                f"{missing}{' in ' + context if context else ''}")


def reject_blind_judge_defense(rows: Sequence[Mapping[str, Any]]) -> None:
    """A blind-k-way judge FAILURE may never be the evidence that makes a row a
    class member. Judge failures defend membership only at the hardened (2AFC)
    reading; judge successes may defeat (raise the rung).

    Census rows must satisfy: any MEMBER/BORDERLINE row whose judge value is
    LOW (below internals by the member bar) either (a) binds its membership on
    a non-judge detector (content >= tfidf, i.e. the max was not lowered by the
    judge — structurally guaranteed by max()), AND (b) carries a hardening
    annotation not spelled ``PENDING`` if its judge_gap is quoted-eligible.
    """
    for r in rows:
        if r.get("status") not in ("MEMBER", "BORDERLINE"):
            continue
        jg = r.get("judge_gap")
        if jg is None or jg < 0.10:
            continue
        hardening = str(r.get("hardening", ""))
        if hardening.startswith(PENDING_HARDENING):
            raise GateError(
                f"row {r.get('row')}/{r.get('model')}: judge gap {jg} is quotable, "
                f"but its hardening is {hardening!r} — a gap this size may not rest on "
                "hardening that has not been done")
        if "blind" in hardening.lower() and "artifact" not in hardening.lower():
            raise GateError(
                f"row {r.get('row')}/{r.get('model')}: a blind k-way judge reading "
                "is being used as the class defense; a judge failure may not be the "
                "evidence that makes a row a member")


__all__ = ["GateError", "require_stamp", "require_gated_outcome",
           "reject_blind_judge_defense"]
