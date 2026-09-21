"""Channel decomposition (§1, fourth readout) — implemented 2026-07-13 (addendum 13d).

Every arm: compare the matched-token delta (replay channel; deformation at fixed
tokens = DIRECT component) against the free-generation delta; the remainder is
the TOKEN-MEDIATED component. Structural predictions live in §2c (A1 = 100%
token-mediated — matched-token delta AT faithfulness floor, parameter-free;
A4/A5 have direct components by construction). Output: the map's channel column.

The typed `ChannelSplit` (below) is the report-layer container (referenced by
`report.py`); the working readout is `decompose_channel`, which returns a plain
dict of per-cell magnitudes for the analyzer JSON. The two are kept distinct so a
per-family dict can be emitted cheaply while `ChannelSplit` carries stamps at the
report layer.

DIRECT = mean matched-token delta (steered-replay − unsteered-replay of the SAME
tokens); TOKEN-MEDIATED = free-gen centroid shift − direct. CAVEATS the caller must
carry: the two halves use different references (matched-token vs stage0; free-gen vs
riders) and different continuations, so `direct` is an ESTIMATE of the fixed-token
component, not an exact subtraction. Magnitudes are per-feature RMS in floor-z units.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from anamnesis.analysis.battery.stats import StampedValue


class ChannelSplit(BaseModel):
    model_config = ConfigDict(frozen=True)

    cell_id: str
    direct: StampedValue           # matched-token delta vs faithfulness floor
    token_mediated: StampedValue   # free-gen delta minus the direct component
    direct_at_floor: bool          # True = direct component indistinguishable from floor


def _rms(v: NDArray) -> float:
    return float(np.linalg.norm(v) / np.sqrt(max(len(v), 1)))


def decompose_channel(
    direct_vec: NDArray,
    free_gen_vec: NDArray,
    mask: NDArray,
    faithfulness_floor: float = 0.0,
) -> dict:
    """Split a free-gen deformation into DIRECT (fixed-token) + TOKEN-MEDIATED remainder.

    Args:
      direct_vec:   signed mean matched-token delta (steered-replay − unsteered), z-space.
      free_gen_vec: signed free-gen centroid shift (steered − rider), z-space.
      mask:         boolean feature mask for the family cell.
      faithfulness_floor: replay floor for this cell (bitwise-zero on the anchors per 12b,
        so any nonzero direct clears it — direct_at_floor is then True only at exactly 0).

    Returns a dict: direct/token-mediated/free-gen RMS (floor-z), a rough magnitude split
    `fraction_direct`, the alignment `cos_direct_freegen` (is the fixed-token deformation
    pointed where the free-gen state ends up?), and `direct_at_floor`.
    """
    d = np.asarray(direct_vec, dtype=np.float64)[mask]
    f = np.asarray(free_gen_vec, dtype=np.float64)[mask]
    tm = f - d
    direct_rms, tm_rms, free_rms = _rms(d), _rms(tm), _rms(f)
    denom = float(np.linalg.norm(d) * np.linalg.norm(f))
    cos_df = float(d @ f / denom) if denom > 1e-12 else 0.0
    return {
        "direct_rms_z": direct_rms,
        "token_mediated_rms_z": tm_rms,
        "free_gen_rms_z": free_rms,
        "fraction_direct": float(direct_rms / max(direct_rms + tm_rms, 1e-12)),
        "cos_direct_freegen": cos_df,
        "direct_at_floor": bool(direct_rms <= faithfulness_floor),
    }
