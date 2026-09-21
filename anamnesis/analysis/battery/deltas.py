"""Paired-delta construction (§6b) — Wave-1 implementation (first used by arm A1).

The unit of analysis is the PAIRED DELTA, never raw signature position (§1).
Two pairing rules implemented here:

  - WITHIN-condition (matched history): same prompt class + same condition,
    different seed → the floor analog. floors.pair_deltas_by_class does this for
    the Stage-0 corpus; arm runs reuse it for the addendum-12a item-2 variance check.
  - CROSS-condition (the arm effect): same prompt class, one gen from each of two
    conditions (e.g. T=0.3 vs T=0.9) → the effect distribution compared against the
    Stage-0 floor per cell.

Standardization: ALWAYS the Stage-0 stochastic-floor scale (median/MAD-floored) of
the same model, so arm deltas, arm-conditional floors, and Stage-0 floors share one
z space. Deltas are mean |Δz| over a cell's features (the exp11 aggregation).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from anamnesis.analysis.battery.floors import (
    build_cells,
    load_class_labels,
    load_signature_matrix,
    robust_scale,
)

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]


class ConditionCorpus:
    """One condition's standardized signatures + class labels, on a shared z scale."""

    def __init__(self, sig_dir: Path, metadata_path: Path,
                 med: F32, scale: F32, name: str):
        X, names, gen_ids = load_signature_matrix(sig_dir)
        if X.shape[1] != len(med):
            raise ValueError(
                f"{name}: feature dim {X.shape[1]} != floor scale dim {len(med)} — "
                "conditions must share the extraction feature set"
            )
        self.name = name
        self.feature_names = names
        self.Z = (X - med) / scale
        self.gen_ids = gen_ids          # row-aligned with Z (modal-vector gens only)
        labels = load_class_labels(metadata_path)
        self.rows_by_class: dict[tuple[int, str], list[int]] = {}
        for row, gid in enumerate(gen_ids):
            if gid not in labels:
                logger.warning(f"{name}: gen {gid} missing from metadata — excluded")
                continue
            self.rows_by_class.setdefault(labels[gid], []).append(row)


def load_floor_scale(floor_sig_dir: Path) -> tuple[F32, F32]:
    """The model's frozen z space: robust scale of its Stage-0 stochastic floor corpus."""
    X, _names, _ids = load_signature_matrix(floor_sig_dir)
    return robust_scale(X)


def cross_condition_deltas(
    a: ConditionCorpus,
    b: ConditionCorpus,
    cells: dict[str, NDArray],
    max_pairs_per_class: int | None = None,
    seed: int = 0,
) -> dict[str, F32]:
    """Same-class cross-condition pair deltas per cell: mean |Δz| over cell features.

    All (row_a, row_b) same-class combinations, optionally subsampled per class
    (deterministic RNG) — n is stamped by the caller from the returned lengths.
    """
    rng = np.random.RandomState(seed)
    out: dict[str, list[float]] = {c: [] for c in cells}
    shared = sorted(set(a.rows_by_class) & set(b.rows_by_class))
    if not shared:
        raise ValueError(f"no shared prompt classes between {a.name} and {b.name}")
    for cls in shared:
        pairs = [(ra, rb) for ra in a.rows_by_class[cls] for rb in b.rows_by_class[cls]]
        if max_pairs_per_class is not None and len(pairs) > max_pairs_per_class:
            idx = rng.choice(len(pairs), size=max_pairs_per_class, replace=False)
            pairs = [pairs[i] for i in idx]
        for ra, rb in pairs:
            dz = np.abs(a.Z[ra] - b.Z[rb])
            for cname, mask in cells.items():
                out[cname].append(float(dz[mask].mean()))
    return {c: np.asarray(v, dtype=np.float32) for c, v in out.items()}


def within_condition_deltas(
    a: ConditionCorpus,
    cells: dict[str, NDArray],
) -> dict[str, F32]:
    """Same-class same-condition pair deltas (the arm-conditional floor)."""
    out: dict[str, list[float]] = {c: [] for c in cells}
    for _cls, rows in sorted(a.rows_by_class.items()):
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                dz = np.abs(a.Z[rows[i]] - a.Z[rows[j]])
                for cname, mask in cells.items():
                    out[cname].append(float(dz[mask].mean()))
    return {c: np.asarray(v, dtype=np.float32) for c, v in out.items()}


def location_dispersion(
    a: ConditionCorpus,
    b: ConditionCorpus,
    cells: dict[str, NDArray],
) -> dict[str, dict[str, float]]:
    """First-class effect decomposition (addendum 2026-07-12c item 2): per cell,
    - centroid_shift: mean |Δ| of per-feature centroid difference between conditions
      (a location statistic, in floor-z units — same scale as the pair deltas)
    - dispersion_ratio: median within-b pair delta / median within-a pair delta
      (>1 = b's state cloud is WIDER than a's; the mover-vs-spreader axis)
    Convention: call with a = reference/native, b = dose/perturbed.
    """
    wa = within_condition_deltas(a, cells)
    wb = within_condition_deltas(b, cells)
    mu_a = a.Z.mean(axis=0)
    mu_b = b.Z.mean(axis=0)
    dmu = np.abs(mu_b - mu_a)
    out: dict[str, dict[str, float]] = {}
    for cname, mask in cells.items():
        med_a = float(np.median(wa[cname])) if len(wa[cname]) else 0.0
        med_b = float(np.median(wb[cname])) if len(wb[cname]) else 0.0
        out[cname] = {
            "centroid_shift": float(dmu[mask].mean()),
            "dispersion_ratio": (med_b / med_a) if med_a > 1e-12 else float("inf"),
            "n_a": int(a.Z.shape[0]), "n_b": int(b.Z.shape[0]),
        }
    return out


__all__ = [
    "ConditionCorpus",
    "build_cells",
    "cross_condition_deltas",
    "load_floor_scale",
    "location_dispersion",
    "within_condition_deltas",
]
