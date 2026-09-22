"""The dissociation column: what token-space sees against what the signature sees.

Per arm, the token-space readers (token-KL, TF-IDF, judge hooks) beside the
signature effect. The case it exists to name: a signature separating two
matched-token conditions that token-KL is blind to by construction, the tokens
being identical. A detector failing feeds THIS column and is never an instrument
null — it says the detector saw nothing, not that there was nothing to see.
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
    raise NotImplementedError(
        "the dissociation column is not implemented; it would name, per cell, "
        "whether a change is visible to the tokens, to the signature, to both or to neither"
    )
