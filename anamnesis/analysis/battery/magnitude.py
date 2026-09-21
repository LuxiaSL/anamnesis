"""Decomposed magnitude ruler for FREE-GEN cells — addendum 2026-07-12e item 2.

Twice-observed geometry (A1 kill contrast; A3 all-60 rows): the median
cross-condition pairwise delta for two clouds separated by δ with internal
spread σ scales like sqrt(δ² + 2σ²) against a floor of sqrt(2)·σ — location
shifts are square-root-compressed, so the k× pairwise bar implicitly demands
an enormous δ. From 12e, magnitude verdicts for free-gen cells are stated on
the 12c decomposition instead:

  - centroid_shift  — mean |Δμ| over the cell's features, floor-z units
  - dispersion_ratio — median within-b pair delta / median within-a pair delta

each against a PERMUTATION NULL (condition labels shuffled within prompt
class, group sizes preserved). The raw pairwise ratio stays in emitted JSON
as a conservative lower bound, never as the verdict column. Replay /
matched-token cells are unaffected (12b seed-floor ruler: per-gen deltas are
direct displacements — no pairwise compression).

Permutation efficiency: a pair's |Δz| does not change under relabeling — only
its subset membership (within-a / within-b / cross) does. All pooled
within-class pair deltas are computed ONCE per cell; each permutation just
re-partitions the precomputed values. Centroid shifts are recomputed from
per-class group sums (vectorized).
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from anamnesis.analysis.battery.deltas import ConditionCorpus

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]


def decomposed_magnitude(
    a: ConditionCorpus,
    b: ConditionCorpus,
    cells: dict[str, NDArray],
    n_perm: int = 1000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """12e magnitude readout per cell. Convention: a = reference, b = condition.

    Returns per cell:
      centroid_shift, p_shift (perm, one-sided greater),
      dispersion_ratio, p_disp_wider, p_disp_narrower (perm, each one-sided),
      n_a, n_b, n_perm.
    """
    rng = np.random.RandomState(seed)
    shared = sorted(set(a.rows_by_class) & set(b.rows_by_class))
    if not shared:
        raise ValueError(f"no shared prompt classes between {a.name} and {b.name}")

    cell_names = list(cells)
    masks = [cells[c] for c in cell_names]

    # ── per-class pooled structures ──
    # pooled rows per class; label vector marks a (0) / b (1)
    pooled_Z: list[F32] = []
    class_slices: list[tuple[int, int, int]] = []   # (start, n_a_cls, n_b_cls)
    for cls in shared:
        ra, rb = a.rows_by_class[cls], b.rows_by_class[cls]
        start = sum(s[1] + s[2] for s in class_slices)
        pooled_Z.append(np.vstack([a.Z[ra], b.Z[rb]]))
        class_slices.append((start, len(ra), len(rb)))
    P = np.vstack(pooled_Z)                              # [N, D]

    # all within-class pair deltas per cell, precomputed once
    pair_cell: dict[str, list[float]] = {c: [] for c in cell_names}
    pair_idx: list[tuple[int, int]] = []                 # pooled-row indices
    for (start, na, nb) in class_slices:
        n = na + nb
        for i in range(n):
            for j in range(i + 1, n):
                dz = np.abs(P[start + i] - P[start + j])
                pair_idx.append((start + i, start + j))
                for cname, m in zip(cell_names, masks):
                    pair_cell[cname].append(float(dz[m].mean()))
    pair_arr = {c: np.asarray(v, dtype=np.float32) for c, v in pair_cell.items()}
    pi = np.asarray(pair_idx, dtype=np.int64)            # [n_pairs, 2]

    def stats_for(labels: NDArray) -> tuple[F32, dict[str, tuple[float, float]]]:
        """labels: 0/1 per pooled row → (per-feature |Δμ|, per-cell floor medians)."""
        mu_a = P[labels == 0].mean(axis=0)
        mu_b = P[labels == 1].mean(axis=0)
        dmu = np.abs(mu_b - mu_a)
        la, lb = labels[pi[:, 0]], labels[pi[:, 1]]
        within_a = (la == 0) & (lb == 0)
        within_b = (la == 1) & (lb == 1)
        med = {}
        for cname in cell_names:
            pa = pair_arr[cname][within_a]
            pb = pair_arr[cname][within_b]
            med[cname] = (float(np.median(pa)) if len(pa) else np.nan,
                          float(np.median(pb)) if len(pb) else np.nan)
        return dmu, med

    # observed
    labels_obs = np.zeros(P.shape[0], dtype=np.int8)
    for (start, na, nb) in class_slices:
        labels_obs[start + na: start + na + nb] = 1
    dmu_obs, med_obs = stats_for(labels_obs)
    obs_shift = {c: float(dmu_obs[m].mean()) for c, m in zip(cell_names, masks)}
    obs_ratio = {}
    for cname in cell_names:
        ma, mb = med_obs[cname]
        obs_ratio[cname] = (mb / ma) if (ma and ma > 1e-12) else float("inf")

    # permutation null: shuffle labels WITHIN class, preserving group sizes
    ge_shift = {c: 1 for c in cell_names}          # +1 correction included via init
    ge_wider = {c: 1 for c in cell_names}
    ge_narrower = {c: 1 for c in cell_names}
    for _ in range(n_perm):
        lab = labels_obs.copy()
        for (start, na, nb) in class_slices:
            seg = lab[start:start + na + nb]
            rng.shuffle(seg)
            lab[start:start + na + nb] = seg
        dmu_p, med_p = stats_for(lab)
        for cname, m in zip(cell_names, masks):
            if float(dmu_p[m].mean()) >= obs_shift[cname]:
                ge_shift[cname] += 1
            ma, mb = med_p[cname]
            r = (mb / ma) if (ma and ma > 1e-12) else np.nan
            if np.isfinite(r):
                if r >= obs_ratio[cname]:
                    ge_wider[cname] += 1
                if r <= obs_ratio[cname]:
                    ge_narrower[cname] += 1

    denom = n_perm + 1
    out = {}
    for cname in cell_names:
        out[cname] = {
            "centroid_shift": obs_shift[cname],
            "p_shift": ge_shift[cname] / denom,
            "dispersion_ratio": obs_ratio[cname],
            "p_disp_wider": ge_wider[cname] / denom,
            "p_disp_narrower": ge_narrower[cname] / denom,
            "n_a": int(sum(s[1] for s in class_slices)),
            "n_b": int(sum(s[2] for s in class_slices)),
            "n_perm": n_perm,
        }
    return out


__all__ = ["decomposed_magnitude"]
