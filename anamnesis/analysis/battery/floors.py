"""Stage-0 floor estimation and the noise law.

Two floor types:
  - STOCHASTIC: matched-history, different-seed pair deltas within a prompt class
    (topic × task stratum) — every within-class pair the floor corpus affords.
  - FAITHFULNESS: replay-vs-replay deltas of an identical continuation, STRATIFIED
    into within-device (pinned repeats — pure replay determinism) and cross-device
    (operational jitter) components.

Delta metric, over feature_map cells: per-feature robust z (median/MAD over the
floor corpus) → per-cell delta of a pair = mean |z_i − z_j| over the cell's
features. Whole-vector = the all-feature cell. Cells at four granularities:
whole_vector, legacy family, source, source × band.

The law: for each (cell × model), n_min = smallest n at which a two-sample
comparison of delta distributions detects a k=2× floor-median shift with power 0.9
at the effective per-test α. Published as a TABLE over α_test ∈ {0.05, 0.01, 1e-3,
1e-4}; a planner reads the row α = 0.05 / m, where m is the confirmatory-cell count
its comparison declares — ``BatteryManifest.confirmatory_m`` totals it, and reading
it as a Bonferroni bound is conservative for BH. Shift definition (the conservative
reading): the arm delta's location exceeds the floor
median by (k−1)×median — i.e. "the arm sits at k× floor", the floor-ruler
criterion — so effect size d = (k−1)·median/σ_floor. A rank-test variant inflates
n by 1/0.955 (Mann-Whitney ARE vs t).

Every emitted number is stamped (n, M, law, floor_type), and localization is a
feature_map cell rather than a coarse feature block.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import norm

from anamnesis.feature_map import FeatureMap
from anamnesis.analysis.battery.manifest import FloorType
from anamnesis.analysis.lane_guard import gate_banked_signatures

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]

#: α grid the law table is published over.
ALPHA_GRID: tuple[float, ...] = (0.05, 0.01, 1e-3, 1e-4)

#: Mann-Whitney asymptotic relative efficiency vs the t-test (normal shift).
RANK_ARE: float = 0.955


class LawParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    k: float = Field(default=2.0, description="Floor-ruler multiple: arm sits at k× floor median.")
    power: float = 0.9
    alpha_grid: tuple[float, ...] = ALPHA_GRID
    shift_reading: str = Field(
        default="(k-1)*floor_median location shift (conservative; floor-ruler criterion)",
        description="Documented interpretation of 'detects a k× floor-median shift'.",
    )


class FloorCell(BaseModel):
    """One cell's floor distribution + its n_min row of the law table."""

    model_config = ConfigDict(frozen=True)

    cell: str                      # e.g. "whole_vector", "family:attention_flow", "source_band:attention|mid"
    model: str                     # M: which model this floor characterizes
    floor_type: FloorType
    n_pairs: int                   # n: paired deltas in the distribution
    n_features: int                # features aggregated by this cell
    median: float
    mad: float
    std: float
    q10: float
    q90: float
    n_min_by_alpha: dict[str, int]         # α_test (str) → n_min per group (t-approx, σ-based)
    n_min_by_alpha_rank: dict[str, int]    # rank-test variant (ARE-inflated)
    n_min_by_alpha_robust: dict[str, int]  # σ_robust = MAD(deltas)·1.4826 (heavy-tail resistant)
    effect_d: float                # (k−1)·median / σ_floor
    effect_d_robust: float         # (k−1)·median / σ_robust — planning uses max(n_min, n_min_robust)
    exact_zero: bool = False       # floor distribution is EXACTLY zero, which replay-
                                   # deterministic hardware gives: any nonzero delta exceeds
                                   # floor, so n is set by effect-side estimation alone and the
                                   # k×-floor ruler degenerates to "nonzero at all".
    law: LawParams


class FloorReport(BaseModel):
    """The Stage-0 deliverable for one (model × floor_type): floors + law table."""

    model: str
    floor_type: FloorType
    n_gens: int
    n_pairs_total: int
    corpus: str                    # provenance path
    law: LawParams
    cells: list[FloorCell]
    notes: list[str] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        logger.info(f"floor report → {path} ({len(self.cells)} cells)")


# ---------------------------------------------------------------------------- loading

def load_signature_matrix(
    sig_dir: Path, *, allow_mixed_lanes: bool = False
) -> tuple[F32, list[str], list[int]]:
    """Load gen_*.npz signatures → (X [n, d], feature_names, gen_ids). Skips empty/failed.

    The rows that reach the matrix must all come from one arithmetic lane, so
    :func:`anamnesis.analysis.lane_guard.gate_banked_signatures` runs over exactly
    those rows before they are stacked: the gate covers the join, and the join is
    the matrix. ``allow_mixed_lanes`` names the bypass and logs the conflict.
    """
    paths = sorted(sig_dir.glob("gen_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    if not paths:
        raise FileNotFoundError(f"no gen_*.npz under {sig_dir}")
    loaded: list[tuple[int, F32, tuple[str, ...], Path]] = []
    for p in paths:
        z = np.load(p, allow_pickle=True)
        if "features" not in z:
            logger.warning(f"{p.name}: no features key — skipped")
            continue
        f = np.asarray(z["features"], dtype=np.float32)
        loaded.append((int(p.stem.split("_")[1]),
                       f, tuple(str(x) for x in z["feature_names"]), p))
    if not loaded:
        raise ValueError(f"no usable signatures under {sig_dir}")
    # The model's STANDARD vector = the modal feature-name set. Generations too
    # short to emit it — a 2-token instant-EOS gen drops the gate-drift, STFT and
    # CKA families, which need a window — are EXCLUDED, loudly: a truncated vector
    # is not the same measurement.
    name_counts: dict[tuple[str, ...], int] = {}
    for _, _, nm, _ in loaded:
        name_counts[nm] = name_counts.get(nm, 0) + 1
    modal = max(name_counts, key=lambda k: name_counts[k])
    modal_len = len(modal)
    # A signature is standard iff its names ARE the modal set AND its feature VECTOR
    # length matches. The second guard catches a malformed signature whose family
    # emitted features and names of divergent length while the name TUPLE still
    # matched the modal set: grouping by names alone lets it reach np.stack, which
    # then raises. Drop it loudly, same as a short gen.
    def _standard(f: F32, nm: tuple[str, ...]) -> bool:
        return nm == modal and int(f.shape[0]) == modal_len
    dropped = [gid for gid, f, nm, _ in loaded if not _standard(f, nm)]
    if dropped:
        logger.warning(
            f"{sig_dir}: {len(dropped)}/{len(loaded)} signatures excluded — "
            f"non-standard feature vector (short gens or features/names length mismatch); "
            f"gen_ids={dropped}"
        )
    kept = [(gid, f, p) for gid, f, nm, p in loaded if _standard(f, nm)]
    gate_banked_signatures([p for _, _, p in kept], allow_mixed_lanes=allow_mixed_lanes)
    return (
        np.stack([f for _, f, _ in kept]),
        list(modal),
        [gid for gid, _, _ in kept],
    )


def load_class_labels(metadata_path: Path) -> dict[int, tuple[int, str]]:
    """gen_id → (topic_idx, stratum) from a Stage-0 run's metadata.json."""
    md = json.loads(metadata_path.read_text())
    gens = md["generations"] if isinstance(md, dict) and "generations" in md else md
    return {int(g["generation_id"]): (int(g["topic_idx"]), str(g["mode"])) for g in gens}


# ---------------------------------------------------------------------------- deltas

def robust_scale(X: F32) -> tuple[F32, F32]:
    """Per-feature (median, scale) over the floor corpus.

    Scale = max(MAD·1.4826, 0.05·std): MAD-primary for outlier resistance, but
    floored at a fraction of std so NEAR-CONSTANT features — a saturated
    spectral-entropy feature has a MAD orders of magnitude below its std — cannot
    inflate |z| astronomically on a rare deviation. One such feature is enough to
    poison every cell aggregate that contains it. For healthy features
    MAD·1.4826 ≈ std, so the floor never binds. Degenerate both → 1.
    """
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) * 1.4826
    std = X.std(axis=0)
    scale = np.maximum(mad, 0.05 * std)
    scale = np.where(scale > 1e-12, scale, np.where(std > 1e-12, std, 1.0))
    return med.astype(np.float32), scale.astype(np.float32)


def build_cells(names: list[str], n_layers: int) -> dict[str, NDArray]:
    """Cell name → boolean feature mask, at four floor granularities:
    whole_vector, legacy family, source (which is the only granularity covering a
    source that carries no band, such as output), source × band."""
    fm = FeatureMap(names, n_layers)
    cells: dict[str, NDArray] = {"whole_vector": np.ones(len(names), dtype=bool)}
    for i, t in enumerate(fm.tags):
        fam_key = f"family:{t.family}"
        cells.setdefault(fam_key, np.zeros(len(names), dtype=bool))[i] = True
        src_key = f"source:{t.source.value}"
        cells.setdefault(src_key, np.zeros(len(names), dtype=bool))[i] = True
        if t.band is not None:
            sb_key = f"source_band:{t.source.value}|{t.band.value}"
            cells.setdefault(sb_key, np.zeros(len(names), dtype=bool))[i] = True
    return cells


def pair_deltas_by_class(
    Z: F32,
    gen_ids: list[int],
    labels: dict[int, tuple[int, str]],
    cells: dict[str, NDArray],
) -> dict[str, F32]:
    """All same-class pair deltas per cell: mean |Δz| over the cell's features."""
    by_class: dict[tuple[int, str], list[int]] = {}
    for row, gid in enumerate(gen_ids):
        if gid not in labels:
            logger.warning(f"gen {gid} missing from metadata — excluded from pairing")
            continue
        by_class.setdefault(labels[gid], []).append(row)

    out: dict[str, list[float]] = {c: [] for c in cells}
    for _cls, rows in sorted(by_class.items()):
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                dz = np.abs(Z[rows[i]] - Z[rows[j]])
                for cname, mask in cells.items():
                    out[cname].append(float(dz[mask].mean()))
    return {c: np.asarray(v, dtype=np.float32) for c, v in out.items()}


# ---------------------------------------------------------------------------- the law

def min_n_permutation(alpha: float) -> int:
    """Smallest per-group n at which a two-sample permutation test can even REACH α:
    C(2n, n) must exceed ~5/α so the attainable p-value floor sits comfortably below α.
    (Power math alone returns n=2 for huge effects, but no test resolves α=0.05 with
    C(4,2)=6 relabelings.)"""
    need = 5.0 / alpha
    n = 2
    while math.comb(2 * n, n) < need:
        n += 1
    return n


def n_min_for(effect_d: float, alpha: float, power: float) -> int:
    """Two-sample n per group: normal-approximation power bound, floored at the
    permutation-resolution minimum for this α."""
    if effect_d <= 0:
        return 10**9  # degenerate: no detectable shift definition → effectively unpowerable
    za = norm.ppf(1.0 - alpha / 2.0)
    zb = norm.ppf(power)
    n_power = math.ceil(2.0 * ((za + zb) / effect_d) ** 2)
    return max(min_n_permutation(alpha), n_power)


def floor_cell_from_deltas(
    cell: str,
    deltas: F32,
    n_features: int,
    model: str,
    floor_type: FloorType,
    law: LawParams,
) -> FloorCell:
    med = float(np.median(deltas))
    mad = float(np.median(np.abs(deltas - med)) * 1.4826)
    std = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
    exact_zero = bool(np.all(deltas == 0.0))
    effect = ((law.k - 1.0) * med / std) if std > 1e-12 else 0.0
    effect_robust = ((law.k - 1.0) * med / mad) if mad > 1e-12 else 0.0
    n_by_a = {str(a): n_min_for(effect, a, law.power) for a in law.alpha_grid}
    n_by_a_rank = {a: min(10**9, math.ceil(n / RANK_ARE)) for a, n in n_by_a.items()}
    n_by_a_robust = {str(a): n_min_for(effect_robust, a, law.power) for a in law.alpha_grid}
    return FloorCell(
        cell=cell, model=model, floor_type=floor_type,
        n_pairs=len(deltas), n_features=n_features,
        median=med, mad=mad, std=std,
        q10=float(np.quantile(deltas, 0.10)), q90=float(np.quantile(deltas, 0.90)),
        n_min_by_alpha=n_by_a, n_min_by_alpha_rank=n_by_a_rank,
        n_min_by_alpha_robust=n_by_a_robust,
        effect_d=float(effect), effect_d_robust=float(effect_robust),
        exact_zero=exact_zero, law=law,
    )


# ---------------------------------------------------------------------------- entry points

def compute_stochastic_floors(
    sig_dir: Path,
    metadata_path: Path,
    model: str,
    n_layers: int,
    law: LawParams | None = None,
) -> FloorReport:
    """The Stage-0 stochastic-floor pipeline for one model."""
    law = law or LawParams()
    X, names, gen_ids = load_signature_matrix(sig_dir)
    labels = load_class_labels(metadata_path)
    med, scale = robust_scale(X)
    Z = (X - med) / scale
    cells = build_cells(names, n_layers)
    deltas = pair_deltas_by_class(Z, gen_ids, labels, cells)
    fcells = [
        floor_cell_from_deltas(c, d, int(cells[c].sum()), model, FloorType.stochastic, law)
        for c, d in deltas.items() if len(d) > 0
    ]
    return FloorReport(
        model=model, floor_type=FloorType.stochastic,
        n_gens=len(gen_ids), n_pairs_total=len(deltas["whole_vector"]),
        corpus=str(sig_dir), law=law, cells=fcells,
        notes=[
            "bare floors (no system prompt) — planning estimates; arm-time within-condition "
            "variance >~1.5x these floors triggers a flag + n top-up",
        ],
    )


def compute_faithfulness_floors(
    sig_dir: Path,
    replay_index_path: Path,
    model: str,
    n_layers: int,
    law: LawParams | None = None,
    scale_from: tuple[F32, F32] | None = None,
) -> list[FloorReport]:
    """Faithfulness floors, stratified within-device and cross-device.

    replay_index_path: JSON list of {"sig": "gen_XXX", "continuation_id": int,
    "replay_idx": int, "device": str} — written by the Stage-0 replay driver.
    Standardization: pass scale_from=(median, scale) FROM THE STOCHASTIC FLOOR
    CORPUS so faithfulness deltas live on the same z scale as everything else.
    """
    law = law or LawParams()
    X, names, gen_ids = load_signature_matrix(sig_dir)
    index = {e["sig"]: e for e in json.loads(replay_index_path.read_text())}
    if scale_from is None:
        med, scale = robust_scale(X)
        logger.warning("faithfulness floors standardized on their OWN corpus — pass the "
                       "stochastic-floor scale for cross-floor comparability")
    else:
        med, scale = scale_from
    Z = (X - med) / scale
    cells = build_cells(names, n_layers)

    by_cont: dict[int, list[tuple[int, str]]] = {}   # continuation → [(row, device)]
    for row, gid in enumerate(gen_ids):
        e = index.get(f"gen_{gid:03d}")
        if e is None:
            logger.warning(f"replay sig gen_{gid:03d} missing from index — skipped")
            continue
        by_cont.setdefault(int(e["continuation_id"]), []).append((row, str(e["device"])))

    within: dict[str, list[float]] = {c: [] for c in cells}
    cross: dict[str, list[float]] = {c: [] for c in cells}
    for _cont, rows in sorted(by_cont.items()):
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                (ri, di), (rj, dj) = rows[i], rows[j]
                dz = np.abs(Z[ri] - Z[rj])
                bucket = within if di == dj else cross
                for cname, mask in cells.items():
                    bucket[cname].append(float(dz[mask].mean()))

    reports: list[FloorReport] = []
    for ftype, bucket in (
        (FloorType.faithfulness_within_device, within),
        (FloorType.faithfulness_cross_device, cross),
    ):
        fcells = [
            floor_cell_from_deltas(c, np.asarray(v, dtype=np.float32), int(cells[c].sum()),
                                   model, ftype, law)
            for c, v in bucket.items() if len(v) > 0
        ]
        n_pairs = len(bucket["whole_vector"]) if bucket["whole_vector"] else 0
        reports.append(FloorReport(
            model=model, floor_type=ftype,
            n_gens=len(gen_ids), n_pairs_total=n_pairs,
            corpus=str(sig_dir), law=law, cells=fcells,
            notes=["arm-time floor-ruling uses the component matching how that cell's "
                   "pairs were actually scheduled"],
        ))
    return reports
