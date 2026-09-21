"""Dissociation column (§0 item 4) — Wave-1 implementation.

For each arm: what token-space sees (token-KL, TF-IDF, judge hooks) vs what the
signature sees. exp11's P3 (signature separates matched-token conditions that
token-KL is structurally blind to) is the template. Detector-class blindness
(cheap per-turn text detectors) feeds THIS column and must never be cited as an
instrument null (§3 contrast-arm note).
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from anamnesis.analysis.battery.stats import StampedValue


class DissociationRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    cell_id: str
    token_kl: StampedValue
    signature_effect: StampedValue
    tfidf_auc: StampedValue | None = None
    judge_auc: StampedValue | None = None
    direction: str                 # "visible-to-both" | "signature-only" | "token-only" | "neither"


def dissociation_row(cell_id: str, token_outputs: object, signature_deltas: object) -> DissociationRow:
    raise NotImplementedError("Wave-1: dissociation column (prereg §6b dissoc.py)")
