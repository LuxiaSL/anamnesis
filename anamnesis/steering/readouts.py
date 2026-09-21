"""What an intervention did — read in signature space, against matched nulls.

A steered bank is a set of signatures. Turning it into a claim means asking two
questions that a single distance cannot answer together: did the state move *the
way the vector was supposed to move it*, and how much else moved.

**The lever.** Build the axis from two pure corpora in the model's own
floor-normalized signature space, then split each cell's displacement from an
unsteered baseline into its component along that axis (``target_shift``) and the
norm of what is left (``off_target``). The lever ratio is the target shift of the
real vector over the mean target shift of the random controls at the same dose: a
steering vector that moves the state no further along the axis than a random
direction of the same magnitude has not steered anything.

**Matched-support efficiency.** The same decomposition, with the discipline that
makes it citable: nulls are never pooled across supports. A tail-confined vector
is mechanically high-deformation and a band-confined one is not, so a baseline
that mixes top, tail, band and full-support randoms corrupts both the numerator
and the denominator. Support is inferred from the vector's name, nulls are
aggregated only within a support at a matched dose, and the coherence statistics
of the generated text ride alongside, because a cell whose text collapsed has not
shown what its signature delta looks like it has shown.

**Checkpoint series.** The install analogue: per-checkpoint displacement from a
base bank, projected onto the install axis, with sign-flip permutation for
significance and the seed floor as the visibility bar. The contrast frame is
stamped into the artifact — against a matched control the generic install drift
cancels and what is left is trait-specific, against base it is total displacement
— because the two answer different questions and only the stamp keeps them apart.

**The identity sidecar.** A mixture-of-experts model's expert-usage histogram is
a preference fingerprint: *which* experts it likes. That is content identity,
hereditary under fine-tuning, and it is explicitly not a how-processing signal.
It is banked separately so the channel is available for provenance work without
ever entering a signature.
"""

from __future__ import annotations

import itertools
import json
import logging
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from anamnesis.analysis.battery.deltas import load_floor_scale
from anamnesis.analysis.battery.floors import load_signature_matrix
from anamnesis.analysis.text_stats import text_stats
from anamnesis.steering.vectors import F64, unit

logger = logging.getLogger(__name__)

#: Visibility bar in seed-floor units: below this, a displacement is inside the
#: model's own stochastic floor and the axis it defines is noise.
VISIBILITY_BAR = 0.1

#: The standing promotion rule: an effect clears when it is this many times its
#: matched null.
PROMOTION_RATIO = 1.5

#: Cell-directory names, longest alternative first so a longer vector name binds
#: before a prefix of it. The dose suffix is optional and signed, in either of
#: the two banked spellings: ``_a0.1`` / ``_an0.1``, and ``_p03`` / ``_m01``
#: where the digits are the fraction times ten.
CELL_RE = re.compile(
    r"^(?P<vec>V3sel_bare|Rband\d|V3|V4|V5|V7|RA|R1|R2|R3|V1|rider|baseline)"
    r"(?:_L(?P<site>\d+))?"
    r"(?:_a(?P<neg>n?)(?P<a>[0-9.]+)|_(?P<pm>[pm])(?P<pma>\d+))?$"
)

#: The other banked cell-directory spelling, where the vector name is open and
#: the dose is unsigned: ``V1_L14_a0.3``, the doubled-tag rider ``V1_L14_L14_a0.0``,
#: and the cross-site form ``V4_L14_at7_a0.3``.
CELL_ALT_RE = re.compile(
    r"^(?P<vec>[A-Za-z][A-Za-z0-9]*?)(?:_L(?P<vl>\d+))?(?:_L\d+)?"
    r"(?:_at(?P<at>\d+))?_a(?P<a>[\d.]+)$"
)


def parse_cell_name(name: str, default_site: int) -> dict[str, Any] | None:
    """A steered cell directory's name into vector, site and signed dose.

    Both banked spellings are accepted, the signed-dose one first. A name that
    matches neither returns ``None``: a directory that is not a cell is skipped
    rather than parsed into a cell with default values, which would quietly enter
    the wrong dose into an aggregation.
    """
    match = CELL_RE.match(name)
    if match is not None:
        if match.group("pma") is not None:
            alpha = (-1.0 if match.group("pm") == "m" else 1.0) * int(match.group("pma")) / 10.0
        elif match.group("a") is not None:
            alpha = (-1.0 if match.group("neg") else 1.0) * float(match.group("a"))
        else:
            alpha = 0.0
        return {
            "vector": match.group("vec"),
            "site": int(match.group("site") or default_site),
            "alpha_frac": alpha,
        }
    alt = CELL_ALT_RE.match(name)
    if alt is None:
        return None
    return {
        "vector": alt.group("vec"),
        "site": int(alt.group("at") or alt.group("vl") or default_site),
        "alpha_frac": float(alt.group("a")),
    }


def floor_z(sig_dir: Path | str, median: NDArray[Any], scale: NDArray[Any]) -> F64:
    """A bank's signatures in the model's frozen floor-normalized z space."""
    X, _names, _ids = load_signature_matrix(Path(sig_dir))
    return ((np.asarray(X, dtype=np.float64) - median) / scale)


# ── The lever ─────────────────────────────────────────────────────────────────
def pole_axis(pole_a: NDArray[Any], pole_b: NDArray[Any]) -> tuple[F64, float, float, float]:
    """The unit axis between two pure corpora, with their projections and midpoint.

    The midpoint is the behavioural threshold: a steered generation counts as
    having moved toward pole A when its projection passes the point halfway
    between the two pure corpora's means, which is a decision rule stated in the
    corpora rather than in the steered data.
    """
    A = np.asarray(pole_a, dtype=np.float64)
    B = np.asarray(pole_b, dtype=np.float64)
    axis = unit(A.mean(axis=0) - B.mean(axis=0))
    projection_a, projection_b = float((A @ axis).mean()), float((B @ axis).mean())
    return axis, projection_a, projection_b, 0.5 * (projection_a + projection_b)


def decompose_shift(shift: NDArray[Any], axis: NDArray[Any]) -> dict[str, float]:
    """A displacement split into movement along an axis and movement off it.

    ``effect_per_offtarget`` is the ratio of the two, and it is the metric that
    survives dose: both parts scale with the injection, so their ratio says
    whether a direction is *selective* rather than merely large.
    """
    shift64 = np.asarray(shift, dtype=np.float64)
    axis64 = np.asarray(axis, dtype=np.float64)
    target = float(shift64 @ axis64)
    off_target = float(np.linalg.norm(shift64 - target * axis64))
    return {
        "target_shift": target,
        "off_target": off_target,
        "effect_per_offtarget": float(target / max(off_target, 1e-9)),
    }


def lever_readout(
    run_dir: Path | str,
    pole_a_dir: Path | str,
    pole_b_dir: Path | str,
    floor_dir: Path | str,
    *,
    map_site: int,
    baseline_cell: str | None = None,
    lever_vector: str = "V3",
    sig_subdir: str = "signatures_v3",
    pole_a_name: str = "pole_a",
) -> dict[str, Any]:
    """Target shift, off-target movement and lever ratio for every steered cell.

    ``lever_ratio`` is the named vector's target shift over the mean target shift
    of the random controls at the same dose and site. The baseline cell is the
    unsteered rider the displacements are measured from; its absence is an error
    rather than a fallback to the pooled mean, because a mis-chosen origin moves
    every number in the table.
    """
    median, scale = load_floor_scale(Path(floor_dir))
    axis, projection_a, projection_b, threshold = pole_axis(
        floor_z(pole_a_dir, median, scale), floor_z(pole_b_dir, median, scale)
    )
    baseline_name = baseline_cell or f"{lever_vector}_L{map_site}_a0.0"

    cells: dict[str, dict[str, Any]] = {}
    for directory in sorted(Path(run_dir).iterdir()):
        parsed = parse_cell_name(directory.name, map_site)
        sig = directory / sig_subdir
        if parsed is None or not sig.exists():
            continue
        Z = floor_z(sig, median, scale)
        projection = Z @ axis
        cells[directory.name] = dict(
            parsed,
            z_mean=Z.mean(axis=0),
            proj_mean=float(projection.mean()),
            frac_pole_a=float((projection > threshold).mean()),
            n=int(len(Z)),
        )
    if baseline_name not in cells:
        raise ValueError(f"baseline cell {baseline_name} not found among {sorted(cells)}")
    baseline = cells[baseline_name]

    rows: list[dict[str, Any]] = []
    for name, cell in cells.items():
        decomposed = decompose_shift(cell["z_mean"] - baseline["z_mean"], axis)
        rows.append({
            "cell": name, "vector": cell["vector"], "site": cell["site"],
            "alpha_frac": cell["alpha_frac"], "n": cell["n"],
            "target_shift": round(decomposed["target_shift"], 4),
            "off_target": round(decomposed["off_target"], 4),
            "effect_per_offtarget": round(decomposed["effect_per_offtarget"], 4),
            "frac_pole_a": round(cell["frac_pole_a"], 4),
            "frac_pole_a_vs_baseline": round(cell["frac_pole_a"] - baseline["frac_pole_a"], 4),
        })

    lever: dict[str, dict[str, float]] = {}
    for site in sorted({row["site"] for row in rows}):
        doses = sorted({r["alpha_frac"] for r in rows if r["site"] == site and r["alpha_frac"] > 0})
        for dose in doses:
            target = next(
                (r["target_shift"] for r in rows
                 if r["vector"] == lever_vector and r["site"] == site and r["alpha_frac"] == dose),
                None,
            )
            controls = [
                r["target_shift"] for r in rows
                if r["vector"].startswith("R") and r["alpha_frac"] == dose
            ]
            if target is not None and controls:
                mean_control = float(np.mean(controls))
                lever[f"L{site}_a{dose}"] = {
                    "vector_target": target,
                    "mean_control_target": round(mean_control, 4),
                    "lever_ratio": round(target / max(mean_control, 1e-9), 3),
                }
    return {
        "readout": "steering lever (model-agnostic)",
        "pole_a": pole_a_name, "map_site": int(map_site), "baseline_cell": baseline_name,
        "law": (
            "axis = unit(mean(pole A) - mean(pole B)) in floor-z signature space; target_shift is "
            "the displacement from the unsteered baseline along it; lever_ratio is the vector's "
            "target shift over the random controls' mean at the same dose; frac_pole_a counts "
            "generations past the pure corpora's midpoint"
        ),
        "pole_projections": {
            "pole_a": round(projection_a, 4), "pole_b": round(projection_b, 4),
            "threshold": round(threshold, 4),
        },
        "lever_ratio_by_dose": lever,
        "per_cell": sorted(rows, key=lambda r: (r["vector"], r["site"], r["alpha_frac"])),
    }


# ── Matched-support efficiency ────────────────────────────────────────────────
def support_of(vector: str) -> str:
    """Which spectral support a vector's name declares it belongs to.

    Support is read from the name because it is a fact about the construction,
    and the construction is what a name records. Mixing supports in one null pool
    is the failure this exists to prevent.
    """
    lowered = vector.lower()
    if "top" in lowered:
        return "top"
    if "tail" in lowered:
        return "tail"
    if "band" in lowered or vector == "V7":
        return "band"
    return "full"


def is_null(vector: str) -> bool:
    """Whether a vector name denotes a random control rather than a target."""
    return vector.upper().startswith("R")


def matched_support_efficiency(
    cells: Mapping[str, Mapping[str, Any]],
    reference_centroid: NDArray[Any],
    axis: NDArray[Any],
    *,
    promotion_ratio: float = PROMOTION_RATIO,
    parity_vector: str = "V3",
) -> list[dict[str, Any]]:
    """Targeting over deformation per cell, each read against its own matched null.

    ``cells`` maps a cell name to ``{"z": [n, d], "vector": str, "site": int,
    "alpha_frac": float, "metadata": Path | None}``. ``efficiency`` is targeting
    over total deformation and ``effect_per_offtarget`` is targeting over what
    moved off-axis; both are reported, because the first compares across
    subspaces and the second is the selectivity metric within one.

    Null aggregation is grouped by ``(support, dose)`` and by nothing else. A
    target with no matched null at its dose reports ``None`` rather than
    borrowing one from a neighbouring support.
    """
    centroid = np.asarray(reference_centroid, dtype=np.float64)
    axis64 = np.asarray(axis, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for name, cell in sorted(cells.items()):
        Z = np.asarray(cell["z"], dtype=np.float64)
        shift = Z.mean(axis=0) - centroid
        deformation = float(np.linalg.norm(shift))
        targeting = float(abs(shift @ axis64))
        off_target = float(np.sqrt(max(deformation**2 - targeting**2, 0.0)))
        vector = str(cell["vector"])
        row: dict[str, Any] = {
            "cell": name, "vector": vector, "support": support_of(vector),
            "is_null": is_null(vector), "site": int(cell["site"]),
            "alpha_frac": float(cell["alpha_frac"]), "n": int(len(Z)),
            "targeting": targeting, "deformation": deformation, "off_target": off_target,
            "effect_per_offtarget": float(targeting / max(off_target, 1e-9)),
            "efficiency": float(targeting / max(deformation, 1e-9)),
        }
        metadata = cell.get("metadata")
        if metadata is not None:
            stats = text_stats(metadata)
            row["coherence"] = {k: stats[k] for k in ("mean_len", "mean_ttr", "mean_trigram_rep")}
        rows.append(row)

    def null_mean(metric: str, support: str, dose: float) -> float | None:
        values = [
            r[metric] for r in rows
            if r["is_null"] and r["support"] == support and r["alpha_frac"] == dose
        ]
        return float(np.mean(values)) if values else None

    for row in rows:
        if row["is_null"]:
            continue
        for metric in ("effect_per_offtarget", "efficiency", "targeting"):
            base = null_mean(metric, row["support"], row["alpha_frac"])
            row[f"{metric}_over_matched_null"] = (
                None if base is None else float(row[metric] / max(base, 1e-9))
            )
        ratio = row.get("efficiency_over_matched_null")
        row[f"clears_{promotion_ratio}x_matched_null"] = bool(
            ratio is not None and ratio >= promotion_ratio
        )
        parity = next(
            (r for r in rows if r["vector"] == parity_vector and r["alpha_frac"] == row["alpha_frac"]),
            None,
        )
        if parity is not None and row["vector"] != parity_vector:
            row["effect_vs_parity"] = float(
                row["effect_per_offtarget"] / max(parity["effect_per_offtarget"], 1e-9)
            )
            row["efficiency_vs_parity"] = float(row["efficiency"] / max(parity["efficiency"], 1e-9))
    return rows


def construction_mahalanobis(v: NDArray[Any], evals: NDArray[Any], evecs: NDArray[Any]) -> float:
    """``vᵀΣ⁻¹v`` from a banked eigendecomposition — a check, never a denominator.

    It confirms the expected ordering, that a top-confined direction is cheap and
    a tail-confined one expensive. Dividing an effect by it would be dividing by a
    property of the construction rather than by a measured null.
    """
    coeff = np.asarray(evecs, dtype=np.float64).T @ np.asarray(v, dtype=np.float64)
    clipped = np.clip(np.asarray(evals, dtype=np.float64), 1e-12, None)
    return float(np.sum(coeff * coeff / clipped))


# ── Checkpoint series ─────────────────────────────────────────────────────────
def seed_floor(base_z: NDArray[Any], gen_ids: Sequence[int], group_size: int = 10,
               per_group: int = 4) -> float:
    """The model's own stochastic floor: median mean absolute z distance within a class.

    Generations sharing a prompt differ only by sampling, so the typical distance
    between two of them is the floor a real displacement has to clear. Any
    checkpoint whose displacement sits under it has moved within the model's own
    noise.
    """
    Z = np.asarray(base_z, dtype=np.float64)
    index = {int(g): i for i, g in enumerate(gen_ids)}
    distances: list[float] = []
    for start in range(0, max(index) + 1 if index else 0, group_size):
        rows = [index[g] for g in range(start, start + group_size) if g in index][:per_group]
        distances.extend(
            float(np.abs(Z[i] - Z[j]).mean()) for i, j in itertools.combinations(rows, 2)
        )
    if not distances:
        raise ValueError("no within-class pairs available to estimate a seed floor")
    return float(np.median(distances))


def sign_flip_p(projections: NDArray[Any], n_perm: int = 5000, seed: int = 20260713) -> float:
    """One-sided p for a mean projection under sign-flip permutation.

    The null is that the sign of each generation's projection is arbitrary, which
    is the right null for a directional claim: it tests whether the displacements
    agree on a direction, not merely whether they are large.
    """
    values = np.asarray(projections, dtype=np.float64)
    rng = np.random.default_rng(seed)
    observed = float(values.mean())
    null = (rng.choice([-1.0, 1.0], size=(int(n_perm), len(values))) * values).mean(axis=1)
    return float((np.sum(null >= observed) + 1) / (int(n_perm) + 1))


def directional_series(
    fields_by_step: Mapping[str, NDArray[Any]],
    steps: Sequence[str],
    floor: float,
    *,
    axis: NDArray[Any] | None = None,
    n_perm: int = 5000,
    seed: int = 20260713,
    visibility_bar: float = VISIBILITY_BAR,
) -> dict[str, Any]:
    """Per-checkpoint alignment, projection and significance along an install axis.

    The axis defaults to the final checkpoint's mean displacement — the direction
    the series ended up going — so earlier checkpoints are asked whether they
    were already going there. ``directional_onset`` is the first step whose
    sign-flip p falls below 0.05, which is the quantity the readout exists to
    produce; it leads the magnitude ratio, because agreeing on a direction
    happens before the displacement is large.
    """
    if not steps:
        raise ValueError("a directional series needs at least one step")
    u = np.asarray(axis, dtype=np.float64) if axis is not None else np.asarray(
        fields_by_step[steps[-1]], dtype=np.float64
    ).mean(axis=0)
    norm = float(np.linalg.norm(u))
    u = u / norm if norm > 0 else u
    rows: list[dict[str, Any]] = []
    onset: str | None = None
    for step in steps:
        field = np.asarray(fields_by_step[step], dtype=np.float64)
        mean_field = field.mean(axis=0)
        projections = field @ u
        p = sign_flip_p(projections, n_perm=n_perm, seed=seed)
        ratio = float(np.median(np.abs(field).mean(axis=1)) / floor)
        rows.append({
            "step": step,
            "field_mag": round(float(np.linalg.norm(mean_field)), 4),
            "ratio_seed_floor": round(ratio, 4),
            "above_visibility_bar": bool(ratio >= visibility_bar),
            "align_cos": (
                None if norm <= 0
                else round(float(mean_field @ u / max(np.linalg.norm(mean_field), 1e-12)), 4)
            ),
            "proj_mean": round(float(projections.mean()), 4),
            "proj_p_signflip": round(p, 5),
            "n": int(len(projections)),
        })
        if onset is None and p < 0.05:
            onset = step
    return {
        "install_axis": "external" if axis is not None else f"step-{steps[-1]} mean field",
        "seed_floor_median": round(floor, 5),
        "visibility_bar": visibility_bar,
        "final_ratio_seed_floor": rows[-1]["ratio_seed_floor"],
        "directional_onset_step": onset,
        "law": "install-axis projection with sign-flip permutation; floor = within-class seed distance",
        "per_checkpoint": rows,
    }


def contrast_fields(
    arm_by_step: Mapping[str, tuple[NDArray[Any], Sequence[int]]],
    control_by_step: Mapping[str, tuple[NDArray[Any], Sequence[int]]] | None,
    base: tuple[NDArray[Any], Sequence[int]],
    steps: Sequence[str],
) -> tuple[dict[str, F64], str]:
    """Per-step displacement fields, and the contrast frame they were built under.

    With a matched control, the field is the arm minus the control on each shared
    probe: the base cancels exactly and the generic drift of any install of the
    same shape cancels too, leaving what is specific to the arm. Without one, the
    field is the arm minus the base — total displacement. The frame is returned
    so it can be stamped, because the two are not comparable and a table without
    the stamp does not say which it holds.
    """
    base_z, base_ids = np.asarray(base[0], dtype=np.float64), base[1]
    base_index = {int(g): i for i, g in enumerate(base_ids)}
    frame = "vs-matched-control" if control_by_step is not None else "vs-base"
    fields: dict[str, F64] = {}
    for step in steps:
        arm_z, arm_ids = arm_by_step[step]
        arm_z = np.asarray(arm_z, dtype=np.float64)
        if control_by_step is not None:
            control_z, control_ids = control_by_step[step]
            control_z = np.asarray(control_z, dtype=np.float64)
            control_index = {int(g): i for i, g in enumerate(control_ids)}
            arm_index = {int(g): i for i, g in enumerate(arm_ids)}
            shared = [int(g) for g in arm_ids if int(g) in control_index]
            if not shared:
                raise ValueError(f"step {step}: no probes shared between the arm and its control")
            fields[step] = np.stack([arm_z[arm_index[g]] - control_z[control_index[g]] for g in shared])
        else:
            fields[step] = np.stack([
                arm_z[i] - base_z[base_index[int(g)]]
                for i, g in enumerate(arm_ids) if int(g) in base_index
            ])
    return fields, frame


# ── The identity sidecar ──────────────────────────────────────────────────────
def _entropy(p: NDArray[Any]) -> float:
    positive = np.asarray(p, dtype=np.float64)
    positive = positive[positive > 0]
    return float(-(positive * np.log(positive)).sum())


def _gini(x: NDArray[Any]) -> float:
    values = np.sort(np.asarray(x, dtype=np.float64))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(values) / (n * values.sum()))


def expert_usage_histogram(
    raw_npz_paths: Sequence[Path | str], top_k: int = 6, router_key: str = "router_dist"
) -> dict[str, Any]:
    """A mixture-of-experts model's expert-preference fingerprint, two ways.

    The soft view is mean routing mass per expert, the hard view is top-k
    selection frequency. Both are identity, not processing: *which* experts a
    checkpoint prefers is content-side and hereditary under fine-tuning, so this
    is a sidecar and never a signature feature. The warning travels with the
    artifact for that reason.
    """
    soft_sum: NDArray[Any] | None = None
    hard_counts: NDArray[Any] | None = None
    layer_indices: list[int] | None = None
    n_tokens, n_gens = 0, 0
    for path in raw_npz_paths:
        data = np.load(Path(path), allow_pickle=True)
        if router_key not in data.files:
            continue
        router = np.asarray(data[router_key], dtype=np.float64)
        if router.ndim != 3 or router.size == 0:
            continue
        n_positions, n_layers, n_experts = router.shape
        if soft_sum is None:
            soft_sum = np.zeros((n_layers, n_experts))
            hard_counts = np.zeros((n_layers, n_experts))
            layer_indices = (
                [int(i) for i in data["router_layer_indices"]]
                if "router_layer_indices" in data.files
                else list(range(n_layers))
            )
        assert hard_counts is not None
        soft_sum += router.sum(0)
        k = min(int(top_k), n_experts)
        selected = np.argpartition(-router, k - 1, axis=-1)[..., :k]
        for layer in range(n_layers):
            index, counts = np.unique(selected[:, layer, :], return_counts=True)
            hard_counts[layer, index] += counts
        n_tokens += n_positions
        n_gens += 1
    if soft_sum is None or hard_counts is None:
        raise ValueError(f"no {router_key!r} array found in any of the given files")

    n_layers, n_experts = soft_sum.shape
    soft = soft_sum / max(n_tokens, 1)
    soft_pooled = soft.mean(0)
    hard = hard_counts / max(n_tokens, 1)
    hard_pooled = hard_counts.sum(0)
    hard_pooled = hard_pooled / max(hard_pooled.sum(), 1)
    order = np.argsort(-hard_pooled)
    return {
        "sidecar": "mixture-of-experts expert-usage identity histogram",
        "WARNING": (
            "IDENTITY channel: expert preference is content-side and hereditary. It must never "
            "enter a signature — this is not a how-processing feature."
        ),
        "lineage": {
            "n_gens": n_gens, "n_tokens": n_tokens, "n_router_layers": int(n_layers),
            "n_experts": int(n_experts), "top_k": int(top_k), "router_layer_indices": layer_indices,
        },
        "hard_usage_pooled": {
            "freq": [round(float(x), 5) for x in hard_pooled],
            "entropy_nats": round(_entropy(hard_pooled), 4),
            "entropy_frac_of_uniform": round(_entropy(hard_pooled) / np.log(n_experts), 4),
            "gini": round(_gini(hard_pooled), 4),
            "top5_experts": [(int(i), round(float(hard_pooled[i]), 5)) for i in order[:5]],
            "bottom5_experts": [(int(i), round(float(hard_pooled[i]), 5)) for i in order[-5:]],
            "n_experts_unused": int((hard_pooled == 0).sum()),
        },
        "soft_usage_pooled": {
            "mass": [round(float(x), 5) for x in soft_pooled],
            "entropy_frac_of_uniform": round(
                _entropy(soft_pooled / soft_pooled.sum()) / np.log(n_experts), 4
            ),
        },
        "per_layer_hard_entropy_frac": [
            round(_entropy(hard[layer] / max(hard[layer].sum(), 1e-12)) / np.log(n_experts), 4)
            for layer in range(n_layers)
        ],
    }


def write_json(path: Path | str, payload: Mapping[str, Any]) -> Path:
    """Write a readout artifact, creating its directory."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload), indent=1))
    logger.info(f"wrote {out}")
    return out


# ── The qualitative readout ───────────────────────────────────────────────────
DOSE_LADDER: tuple[float, ...] = (0.0, 0.03, 0.1, 0.3)
"""The doses a free-generation ladder banks. It stops at 0.3, and a mode-induction
peak sits higher (analogy markers peaked at 0.45 in the wave-1 adjudication), so the
strongest collected cell is pre-peak: the ladder shows a trend, not a ceiling."""

QUALITATIVE_CHARS = 600
"""Characters shown per generation. Enough to read the register and the structure,
short enough that a block of six cells fits on a screen."""


def cell_ladder(
    site: int,
    *,
    vector: str = "V3",
    controls: Sequence[tuple[str, str]] = (("V1", "formality control"), ("R1", "random-vector control")),
    doses: Sequence[float] = DOSE_LADDER,
    control_dose: float = 0.3,
) -> list[tuple[str, str]]:
    """The cells a qualitative read lays out, in reading order: ``(label, directory)``.

    The directory spelling is the banked one — ``<vector>_L<site>_L<site>_a<dose>``
    with the dose written as a float, and a random control carrying one site rather
    than two — because these names address directories that already exist on disk.

    A dose ladder of one vector plus one control per axis at a matched dose. The
    controls are what make the ladder readable: a formality vector says what a
    *different* real direction does at the same dose, and a random one says what mere
    magnitude does — so an effect that appears in all three is dose, not steering.
    """
    ladder = [
        (
            f"{vector} alpha={dose}" + (" (baseline)" if dose == 0.0 else ""),
            f"{vector}_L{site}_L{site}_a{float(dose)}",
        )
        for dose in doses
    ]
    return ladder + [
        (
            f"{name} alpha={control_dose} ({description})",
            f"{name}_L{site}_L{site}_a{float(control_dose)}" if name != "R1"
            else f"{name}_L{site}_a{float(control_dose)}",
        )
        for name, description in controls
    ]


def matched_generations(cell_dir: Path | str) -> dict[int, dict[str, Any]]:
    """One cell's banked generations, keyed by generation id.

    Returns an empty mapping where the cell has no metadata, because a ladder read
    over the cells that exist is the normal case — a dose that was not collected is a
    gap in the ladder, and the readout says so by leaving it out.
    """
    path = Path(cell_dir) / "metadata.json"
    if not path.exists():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    generations = (
        document["generations"]
        if isinstance(document, dict) and "generations" in document
        else document
    )
    return {int(g["generation_id"]): g for g in generations}


def qualitative_markdown(
    run_dir: Path | str,
    *,
    model: str,
    site: int,
    gen_ids: Sequence[int],
    ladder: Sequence[tuple[str, str]] | None = None,
    chars: int = QUALITATIVE_CHARS,
) -> str:
    """Matched prompts across a dose ladder and its controls, as a readable document.

    This is the eyeball channel beside the numbers, and what it is for is one
    question: does the vector induce the *mode* while the random control merely
    degrades? A lever ratio cannot answer that and a reader can.

    Matching is by generation id, which is the prompt. Seeds differ per cell
    namespace, so two cells' texts for one id are the same prompt written twice rather
    than a token-level control — a style comparison, and the document says so where it
    is read rather than in a note somewhere else.
    """
    cells = list(ladder) if ladder is not None else cell_ladder(site)
    loaded = {directory: matched_generations(Path(run_dir) / directory) for _, directory in cells}
    lines = [
        f"# Qualitative steering readout — {model} (site L{site})",
        "",
        "**Dose caveat:** the ladder tops out at the strongest dose that was collected, "
        "and the mode-induction peak sits above it. Read the trend, not the ceiling.",
        "",
        "Matched by prompt (generation id). Seeds differ per cell, so this is a "
        "style-and-mode comparison rather than a token-level control. **For each block: "
        "does the vector shift the mode with dose, while the random control only drifts?**",
        "",
    ]
    for gen_id in gen_ids:
        reference = next(
            (loaded[d][gen_id] for _, d in cells if gen_id in loaded[d]), None
        )
        if reference is None:
            continue
        lines += [
            "---",
            f"## generation {gen_id} — topic: *{reference.get('topic', '?')}* | "
            f"stratum: {reference.get('mode', '?')}",
            "",
        ]
        for label, directory in cells:
            generation = loaded[directory].get(gen_id)
            if not generation:
                continue
            text = str(generation.get("generated_text", "")).strip().replace("\n", " ")
            lines += [
                f"**{label}**  ({generation.get('num_generated_tokens', '?')} tokens)",
                "",
                f"> {text[:chars]}{'…' if len(text) > chars else ''}",
                "",
            ]
    return "\n".join(lines)
