"""Four gates a steering result passes before it is a result.

Each of these exists because the corresponding prose reminder kept failing. A
reflex that became a function fires whether or not anyone remembers it; a reflex
that stayed a sentence in a document is remembered everywhere except at the
moment of temptation. That asymmetry is the whole argument for this module.

**On-policy agreement.** A matched-token comparison forces the steered model
through a continuation it did not produce. That is only meaningful while those
tokens are still plausible *for the steered model*: past some dose they are not,
and the deltas measure an incoherent counterfactual rather than an intervention.
The gate is top-1 agreement with the banked continuation, and the bar is 0.85.
The unsteered baseline is reported beside it, because a low-dose pass against a
model whose own agreement is already near the bar is baseline-dominated.

**Upstream zero.** Under bitwise-deterministic replay at matched tokens, every
feature reading strictly upstream of the injection site must differ by exactly
zero from the unsteered signature of the same generation — identical tokens mean
the layers below the site ran identically. One check certifies pairing, replay
determinism, absence of leakage and the delta algebra at once, and it has no free
parameter. Any nonzero value quarantines the bank.

**A direction's own matched null.** Every construction carries a spectral bias,
and the bias points *differently* for different constructions: a whitened
direction is pushed toward the tail by ``Σ⁻¹``, a mean difference toward the top
by Σ. So a spectral position is unreadable without the null for the construction
that produced it, and the construction is a required argument with no default —
asking where a vector sits without saying how it was built is the failure this
prevents. Where no matched null exists, the gate raises rather than returning a
number: a value that renders as a verdict is worse than a crash, because it looks
exactly like an answer.

**Shape.** Location and scale audits of a score against its covariates are
necessary and not sufficient. An axis can hold most of a corpus's variance, pass
every nuisance-covariate check, and be a handful of degenerate rows — kurtosis in
the thousands, a minority cluster of one row, and the middle 98% of the data
spanning a fraction of a standard deviation. No covariate audit sees that; only
the shape of the score distribution does.

The two array-level checks take arrays and return verdicts. Loading a corpus, or
fitting the axes a shape audit runs over, is the caller's business — which is
what makes these gates usable on any corpus rather than on the one they were
written beside.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from anamnesis.steering.vectors import Spectrum, dose_alpha, site_of_key, unit

logger = logging.getLogger(__name__)

F64 = NDArray[np.float64]

#: The matched-token validity bar: below this top-1 agreement, forced tokens are
#: off-policy for the steered model and the cell's deltas are not interpretable.
GATE_BAR = 0.85

#: Dose fractions the pilot gate is run at.
PILOT_ALPHA_FRACTIONS: tuple[float, ...] = (0.03, 0.1)

#: A vector's construction. Required, with no default, because the entire failure
#: mode is asking where a direction sits without declaring how it was built.
Construction = Literal["diff_of_means", "lda_whitened", "gradient", "isotropic_random"]

NULL_AVAILABILITY: dict[str, str] = {
    "diff_of_means": (
        "ANALYTIC: under a random split dmu ~ N(0, cΣ), so E[top-k energy] is the top-k share "
        "of the trace. Free, and needs no data."
    ),
    "isotropic_random": "ANALYTIC: E[top-k energy] = k/d.",
    "lda_whitened": (
        "REQUIRES a shuffled-label null: Σ⁻¹ biases toward the tail by construction. Not "
        "derivable from Σ alone — the construction has to be refit on shuffled labels."
    ),
    "gradient": (
        "NO KNOWN NULL. The spectral position of a gauge gradient depends on the gauge, and "
        "measured gradients sit at both ends of the spectrum, so the class has no "
        "characteristic position. Do not state one."
    ),
}

#: Shape tripwires. They flag; a reader looks. A threshold is not a proof.
KURTOSIS_FLAG = 10.0
MINORITY_FLAG = 0.02
SPAN_FLAG = 0.5

_LAYER_RE = re.compile(r"_L(\d+)")


# ── On-policy agreement ───────────────────────────────────────────────────────
def teacher_forced_agreement(model: Any, input_ids: Sequence[int] | Any, prompt_length: int) -> float:
    """Top-1 agreement between the model's argmax and a forced continuation.

    One full-sequence forward, so any registered write hooks fire. The token at
    position ``t`` is predicted from the logits at ``t-1``, which puts the first
    generated token's prediction at the last prompt position — before injection,
    matching generation-side semantics where steering starts at the first
    generated position. Returns a fraction in ``[0, 1]``.
    """
    import torch

    device = next(model.parameters()).device
    ids = torch.as_tensor(input_ids, dtype=torch.long, device=device)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    prompt = int(prompt_length)
    length = int(ids.shape[1])
    if not 0 < prompt < length:
        raise ValueError(f"prompt_length {prompt} out of range for sequence length {length}")
    with torch.no_grad():
        out = model(ids, use_cache=False, return_dict=True)
    predicted = out.logits[0, prompt - 1:length - 1].argmax(dim=-1)
    return float((predicted == ids[0, prompt:length]).float().mean().item())


def pilot_generations(
    entries: Mapping[str, Mapping[str, Any]], count: int = 20, stride: int = 40, min_generated: int = 32
) -> list[Mapping[str, Any]]:
    """A strided pilot subset of a bank: one generation per ``stride`` ids.

    Striding rather than taking the first ``count`` spreads the pilot across the
    protocol's classes, so an agreement number is not a property of whichever
    class happens to be banked first. Generations too short to carry the
    measurement are dropped.
    """
    chosen: list[Mapping[str, Any]] = []
    for k in range(count):
        entry = entries.get(str(k * stride))
        if entry and (len(entry["input_ids"]) - entry["prompt_length"]) >= min_generated:
            chosen.append(entry)
    return chosen


def on_policy_gate(
    model: Any,
    bank: Mapping[str, NDArray[Any]],
    median_residual_norms: Mapping[str, float],
    pilots: Sequence[Mapping[str, Any]],
    *,
    map_site: int,
    vector_keys: Sequence[str] | None = None,
    alpha_fractions: Sequence[float] = PILOT_ALPHA_FRACTIONS,
    gate_bar: float = GATE_BAR,
) -> dict[str, Any]:
    """Run the matched-token pilot at each (vector, dose) and report pass or fail.

    One residual-write handle per site is registered once and its alpha mutated
    per cell, so the site being tested carries the dose and every other site
    carries zero. The ``alpha = 0`` baseline runs first, through the same hooks:
    it is nearly free and it is what says whether a low-dose pass means the
    injection was tolerable or merely small.
    """
    import torch

    from anamnesis.extraction.model_loader import ResidualWriteSpec, attach_residual_write

    keys = sorted(vector_keys) if vector_keys is not None else sorted(bank)
    if not pilots:
        raise ValueError("the on-policy gate needs at least one pilot continuation")
    hidden_dim = int(np.asarray(bank[keys[0]]).shape[0])
    handles = {
        site: attach_residual_write(
            model,
            ResidualWriteSpec(
                layer_idx=site, vector=torch.zeros(hidden_dim), alpha=0.0, normalize=True
            ),
        )
        for site in sorted({site_of_key(key, map_site) for key in keys})
    }
    report: dict[str, Any] = {
        "gate_bar": float(gate_bar), "n_pilot": len(pilots), "site": int(map_site), "cells": {}
    }
    try:
        for handle in handles.values():
            handle.spec.alpha = 0.0
        baseline = []
        for entry in pilots:
            for handle in handles.values():
                handle.spec.start_pos = int(entry["prompt_length"])
            baseline.append(
                teacher_forced_agreement(model, entry["input_ids"], int(entry["prompt_length"]))
            )
        report["alpha0_baseline_mean"] = float(np.mean(baseline))
        report["alpha0_baseline_min"] = float(np.min(baseline))
        logger.info(
            f"alpha=0 baseline agreement {np.mean(baseline):.4f} (min {np.min(baseline):.4f})"
        )

        for key in keys:
            site = site_of_key(key, map_site)
            vector = torch.from_numpy(np.asarray(bank[key], dtype=np.float32))
            for fraction in alpha_fractions:
                alpha = dose_alpha(fraction, float(median_residual_norms[f"L{site}"]))
                for handle_site, handle in handles.items():
                    handle.spec.alpha = alpha if handle_site == site else 0.0
                    if handle_site == site:
                        handle.spec.vector = vector
                agreements = []
                for entry in pilots:
                    handles[site].spec.start_pos = int(entry["prompt_length"])
                    agreements.append(
                        teacher_forced_agreement(
                            model, entry["input_ids"], int(entry["prompt_length"])
                        )
                    )
                mean_agreement = float(np.mean(agreements))
                report["cells"][f"{key}_a{fraction}"] = {
                    "alpha_frac": float(fraction), "alpha_abs": alpha, "site": site,
                    "agreement_mean": mean_agreement,
                    "agreement_min": float(np.min(agreements)),
                    "PASS": bool(mean_agreement >= gate_bar),
                }
                logger.info(
                    f"{key}_a{fraction}: agreement {mean_agreement:.4f} "
                    f"{'PASS' if mean_agreement >= gate_bar else 'FAIL'}"
                )
    finally:
        for handle in handles.values():
            handle.remove()
    return report


# ── Upstream zero ─────────────────────────────────────────────────────────────
def deepest_layer_read(name: str) -> int | None:
    """The deepest layer a feature name reads, or ``None`` when it names none.

    A cross-layer feature names more than one layer and reads the deeper one too,
    so a feature is upstream-safe only when *all* its layers are upstream, which
    makes the maximum the key. A name with no layer token is not
    layer-attributable and is excluded from the check rather than guessed at.
    """
    layers = [int(match.group(1)) for match in _LAYER_RE.finditer(name)]
    return max(layers) if layers else None


def is_upstream(name: str, layer: int, site: int) -> bool:
    """Whether a feature at ``layer`` reads strictly below an injection at ``site``.

    A plain feature ``X_L{n}`` reads the output of model layer ``n``. A delta
    feature reads two adjacent blocks, so it is upstream one block shallower.
    """
    if name.startswith("delta") or "_delta" in name:
        return layer <= site - 2
    return layer < site


def upstream_zero_check(
    steered_signatures: Mapping[int, tuple[Sequence[str], NDArray[Any]]],
    unsteered_signatures: Mapping[int, tuple[Sequence[str], NDArray[Any]]],
    site: int,
) -> dict[str, Any]:
    """Compare matched generations' upstream features; anything nonzero is a failure.

    Both arguments map generation id to ``(feature_names, features)``. The bar is
    exact equality, not a tolerance: identical tokens under deterministic replay
    make the layers below the injection byte-identical, so the expected
    difference is the float ``0.0`` and anything else is a defect in the bank
    rather than numerical noise to be absorbed.
    """
    shared = sorted(set(steered_signatures) & set(unsteered_signatures))
    if not shared:
        return {"error": "no generation ids shared between the steered and unsteered banks"}
    first_names = list(steered_signatures[shared[0]][0])
    mask = np.array([
        (deepest_layer_read(name) is not None)
        and is_upstream(name, int(deepest_layer_read(name) or 0), int(site))
        for name in first_names
    ])
    upstream_names = np.asarray(first_names)[mask]
    worst, worst_feature, n_nonzero = 0.0, None, 0
    offenders: list[str] = []
    for gen_id in shared:
        names, steered = steered_signatures[gen_id]
        base_names, unsteered = unsteered_signatures[gen_id]
        if list(names) != list(base_names):
            return {"error": f"feature-name mismatch for generation {gen_id}"}
        diff = np.abs(np.asarray(steered, dtype=np.float64) - np.asarray(unsteered, dtype=np.float64))[mask]
        nonzero = np.nonzero(diff)[0]
        if nonzero.size:
            n_nonzero += int(nonzero.size)
            offenders.extend(f"{upstream_names[i]}={diff[i]:.3e}(gen{gen_id})" for i in nonzero[:5])
            j = int(np.argmax(diff))
            if diff[j] > worst:
                worst, worst_feature = float(diff[j]), str(upstream_names[j])
    return {
        "site": int(site), "n_upstream_features": int(mask.sum()), "n_gens": len(shared),
        "n_nonzero_upstream": n_nonzero, "max_abs_upstream_delta": worst,
        "worst_feature": worst_feature, "offenders_sample": offenders[:10],
        "PASS": n_nonzero == 0,
    }


def load_signature_pair(path: Path | str) -> tuple[list[str], F64]:
    """``(feature_names, features)`` from one banked ``gen_*.npz`` signature."""
    data = np.load(Path(path), allow_pickle=True)
    return (
        [str(name) for name in data["feature_names"]],
        np.asarray(data["features"], dtype=np.float64),
    )


def load_signature_directory(
    sig_dir: Path | str, sample: int | None = None
) -> dict[int, tuple[list[str], F64]]:
    """Banked signatures under a directory, keyed by generation id."""
    paths = sorted(Path(sig_dir).glob("gen_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    if sample is not None:
        paths = paths[:sample]
    return {int(p.stem.split("_")[1]): load_signature_pair(p) for p in paths}


# ── A direction's own matched null ────────────────────────────────────────────
@dataclass(frozen=True)
class NullVerdict:
    """A spectral position, reported only beside the null that licenses it."""

    construction: str
    k: int
    observed_topk: float
    null_mean: float
    null_p05: float
    null_p95: float
    percentile: float
    mahalanobis: float
    verdict: str

    def __str__(self) -> str:
        return (
            f"{self.construction:16s} top{self.k}={self.observed_topk:.4f}  "
            f"null={self.null_mean:.4f} [{self.null_p05:.4f},{self.null_p95:.4f}]  "
            f"pct={self.percentile:5.1f}%  mahal={self.mahalanobis:8.1f}  → {self.verdict}"
        )


def analytic_null_topk(spectrum: Spectrum, construction: Construction, k: int) -> float:
    """Expected top-k energy under a construction's null, where one exists in closed form.

    For a mean difference the derivation makes the null free: under a random
    split of one distribution, ``dmu ~ N(0, cΣ)``, so a label-free mean
    difference *is* a Σ-weighted random vector and its expected top-k energy is
    the top-k share of the trace. No data, no labels, no shuffling.
    """
    if construction == "diff_of_means":
        return float(spectrum.evals[:k].sum() / spectrum.evals.sum())
    if construction == "isotropic_random":
        return float(k / spectrum.d)
    raise NotImplementedError(
        f"no analytic null for construction={construction!r}. {NULL_AVAILABILITY[construction]}"
    )


def empirical_null_topk(
    spectrum: Spectrum, construction: Construction, k: int, n_draws: int = 200, seed: int = 0
) -> F64:
    """Top-k energies under a construction's null across ``n_draws`` samples.

    Sampled rather than analytic because a percentile says more than a mean: the
    question is whether an observed position sits inside the null's spread, and a
    point estimate cannot answer it.
    """
    rng = np.random.default_rng(seed)
    if construction == "diff_of_means":
        sd = np.sqrt(np.maximum(spectrum.evals, 0.0))
        draw = lambda: sd * rng.normal(size=spectrum.d)
    elif construction == "isotropic_random":
        draw = lambda: rng.normal(size=spectrum.d)
    else:
        raise NotImplementedError(
            f"no samplable null for construction={construction!r}. {NULL_AVAILABILITY[construction]}"
        )
    out = np.empty(int(n_draws), dtype=np.float64)
    for i in range(int(n_draws)):
        out[i] = float((unit(draw())[:k] ** 2).sum())
    return out


def assert_against_own_null(
    v: NDArray[Any],
    construction: Construction,
    spectrum: Spectrum,
    *,
    k: int = 256,
    n_draws: int = 200,
    seed: int = 0,
) -> NullVerdict:
    """A direction's spectral position, reported only against its own matched null.

    Raises rather than returning a number when the construction has no matched
    null. A verdict inside the null band is the important case: it means the
    position *is* the construction, and there is nothing about the vector to
    report.
    """
    if construction not in NULL_AVAILABILITY:
        raise ValueError(
            f"unknown construction {construction!r}; declare one of {sorted(NULL_AVAILABILITY)}"
        )
    observed = float(spectrum.energy_profile(v)[:k].sum())
    null = empirical_null_topk(spectrum, construction, k, n_draws=n_draws, seed=seed)
    analytic = analytic_null_topk(spectrum, construction, k)
    percentile = 100.0 * float((null < observed).mean())
    if percentile < 5:
        verdict = f"BELOW its own null (p<.05) — {construction} does not explain this position"
    elif percentile > 95:
        verdict = f"ABOVE its own null (p<.05) — {construction} does not explain this position"
    else:
        verdict = "INSIDE its own null — unreadable; the position IS the construction"
    return NullVerdict(
        construction=construction, k=int(k), observed_topk=observed, null_mean=float(analytic),
        null_p05=float(np.percentile(null, 5)), null_p95=float(np.percentile(null, 95)),
        percentile=percentile, mahalanobis=spectrum.mahalanobis(v), verdict=verdict,
    )


# ── Shape ─────────────────────────────────────────────────────────────────────
def minority_cluster(z: NDArray[Any]) -> int:
    """Size of the smaller cluster under a one-dimensional two-means split.

    A real axis splits a corpus roughly in half; an artifact peels off a shard.
    The size of the shard is the readout, and a cluster of a handful of rows out
    of tens of thousands is the shape of a degenerate-row artifact.
    """
    scores = np.asarray(z, dtype=np.float64)
    centres = np.array([scores.min() / 2, scores.max() / 2], dtype=np.float64)
    labels = np.zeros(len(scores), dtype=np.int64)
    for _ in range(50):
        labels = np.abs(scores[:, None] - centres[None, :]).argmin(1)
        for k in (0, 1):
            if (labels == k).any():
                centres[k] = scores[labels == k].mean()
    return int(min((labels == 0).sum(), (labels == 1).sum()))


def audit_axis(score: NDArray[Any], covariates: Mapping[str, NDArray[Any]]) -> dict[str, Any]:
    """Location, scale and shape of one axis's scores, with tripwires.

    The covariate correlations are the location and scale legs: a score
    correlated with a nuisance variable, or whose *magnitude* is, is reading that
    variable. The kurtosis, minority cluster and percentile span are the shape
    leg, and they are what catch an axis that passes both covariate checks and is
    still a handful of rows. A covariate with no variance is reported as
    controlled by construction rather than as a correlation of nothing.
    """
    values = np.asarray(score, dtype=np.float64)
    z = (values - values.mean()) / max(float(values.std()), 1e-12)
    kurtosis = float((z**4).mean())
    minority = minority_cluster(z)
    p1, p25, p50, p75, p99 = (float(x) for x in np.percentile(z, [1, 25, 50, 75, 99]))
    span = p99 - p1
    out: dict[str, Any] = {
        "kurtosis": round(kurtosis, 2),
        "minority_cluster": minority,
        "minority_frac": round(minority / len(z), 5),
        "percentiles": {"p1": round(p1, 3), "p25": round(p25, 3), "p50": round(p50, 3),
                        "p75": round(p75, 3), "p99": round(p99, 3)},
        "p1_p99_span_sd": round(span, 3),
        "n_abs_z_gt5": int((np.abs(z) > 5).sum()),
        "n_abs_z_gt10": int((np.abs(z) > 10).sum()),
        "covariates": {},
    }
    for name, raw in covariates.items():
        covariate = np.asarray(raw, dtype=np.float64)
        if covariate.std() < 1e-12:
            out["covariates"][name] = {
                "location": None, "scale": None, "note": "no variance (controlled by construction)"
            }
            continue
        out["covariates"][name] = {
            "location": round(float(np.corrcoef(z, covariate)[0, 1]), 4),
            "scale": round(float(np.corrcoef(np.abs(z), covariate)[0, 1]), 4),
        }
    flags: list[str] = []
    if kurtosis > KURTOSIS_FLAG:
        flags.append(f"KURTOSIS {kurtosis:.1f} > {KURTOSIS_FLAG}")
    if minority / len(z) < MINORITY_FLAG:
        flags.append(f"MINORITY {minority}/{len(z)} < {MINORITY_FLAG:.0%}")
    if span < SPAN_FLAG:
        flags.append(f"p1..p99 SPAN {span:.3f} sd < {SPAN_FLAG}")
    for name, entry in out["covariates"].items():
        if entry.get("scale") is not None and abs(entry["scale"]) > 0.3:
            flags.append(f"SCALE vs {name} = {entry['scale']:+.3f}")
    out["flags"] = flags
    out["verdict"] = "ARTIFACT-SHAPED — do not name this axis" if flags else "shape OK"
    return out


def audit_axes(
    scores: Mapping[str, NDArray[Any]],
    covariates: Mapping[str, NDArray[Any]],
    variance_ratios: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Audit several named axes at once, keeping each one's variance share beside it.

    ``scores`` maps an axis name to its per-row projection. Whatever produced
    those projections — a principal component, a discriminant, a steering axis —
    is the caller's, which is what lets this run on any corpus: the audit is a
    property of a score distribution, not of the pipeline that made it.
    """
    audited: dict[str, Any] = {}
    for name, values in scores.items():
        row = audit_axis(values, covariates)
        if variance_ratios is not None and name in variance_ratios:
            row["var_ratio"] = round(float(variance_ratios[name]), 4)
        audited[name] = row
    return {
        "thresholds": {
            "kurtosis": KURTOSIS_FLAG, "minority_frac": MINORITY_FLAG, "p1_p99_span_sd": SPAN_FLAG,
            "note": "tripwires, not proofs — they flag, a reader looks",
        },
        "law": (
            "location + scale against every covariate, plus the shape of the score distribution "
            "itself: an axis can pass both covariate legs and still be a handful of degenerate rows"
        ),
        "axes": audited,
    }
