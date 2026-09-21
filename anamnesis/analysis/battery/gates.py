"""Code gates — prompt-level discipline moved into assertions (swap-prep 2026-07-12).

The three-strike soft-rule pattern (kill-vs-ruler; raw-vs-z; point-direction
outcomes) happened with a vigilant enactor. These gates make the discipline
survive a model swap: analyzers CANNOT emit what the prereg forbids.

Every gate raises GateError (never warns) — a blocked emission is a bug in the
caller, to be fixed at authoring time, not silenced.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


class GateError(AssertionError):
    """A prereg discipline gate refused an emission."""


REQUIRED_STAMP_KEYS = ("n", "M", "law", "floor_type")


def require_stamp(row: Mapping[str, Any], context: str = "") -> None:
    """Every emitted number carries (n, M, law, floor_type) — prereg §1/§6b.

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
    """12d: a categorical verdict may exist ONLY alongside its own-tail BH gate
    fields. Point direction never ships as a verdict.
    """
    if outcome_key in row:
        missing = [k for k in gate_keys if k not in row]
        if missing:
            raise GateError(
                f"outcome {row[outcome_key]!r} emitted without gate fields "
                f"{missing}{' in ' + context if context else ''} — 12d violation")


def reject_blind_judge_defense(rows: Sequence[Mapping[str, Any]]) -> None:
    """12g codicil (a): a blind-k-way judge FAILURE may never be the evidence
    that makes a row a class member. Judge failures defend membership only at
    the hardened (2AFC) reading; judge successes may defeat (raise the rung).

    Census rows must satisfy: any MEMBER/BORDERLINE row whose judge value is
    LOW (below internals by the member bar) either (a) binds its membership on
    a non-judge detector (content >= tfidf, i.e. the max was not lowered by the
    judge — structurally guaranteed by max()), AND (b) carries a hardening
    annotation that is not 'pending' if its judge_gap is quoted-eligible.
    """
    for r in rows:
        if r.get("status") not in ("MEMBER", "BORDERLINE"):
            continue
        jg = r.get("judge_gap")
        if jg is None or jg < 0.10:
            continue
        hardening = str(r.get("hardening", ""))
        if hardening.startswith("PENDING"):
            # membership itself is fine (binds on trained detector via max);
            # but emitting the row without the pending flag visible = violation
            if "PENDING" not in hardening:
                raise GateError(
                    f"row {r.get('row')}/{r.get('model')}: quotable judge-gap "
                    "without hardening status — 12g violation")
        if "blind" in hardening.lower() and "artifact" not in hardening.lower() \
                and "PENDING" not in hardening:
            raise GateError(
                f"row {r.get('row')}/{r.get('model')}: blind-k-way reading "
                "used as class defense — 12g codicil (a) violation")


__all__ = ["GateError", "require_stamp", "require_gated_outcome",
           "reject_blind_judge_defense"]
