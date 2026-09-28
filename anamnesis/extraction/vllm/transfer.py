"""The transfer check: does a fine-tune stay inside its base lane's measured regime?

A shipped lane was qualified by comparing the vLLM lane with the numeric anchor on
the base model, and its effects were retained at the deviations that comparison
measured. The comparison's recorded tolerance is that regime: per row, a
component's distance over the row's path floor, and per feature family, the
largest standardized coordinate deviation. A fine-tune of the base has the same
architecture and shapes, so the kernels, the determinism of the dispatch and the
schema carry over; what its weights can change is how far the engine drifts from
the hook path on a row, through threshold proximity, large activations and gate
sparsity, all statistics of the activations. A fine-tune whose engine-versus-hook
deviation lies inside its base's measured regime therefore inherits effect
preservation for the same reason a conformant host does: preservation was measured
as a function of the size of the perturbation, and the fine-tune's perturbation is
inside the tested size.

This module is the decision, and it never touches an engine or a model. It takes,
on a sample of the fine-tune's own rows:

* the fast lane's vectors (the reference) and the vLLM lane's vectors (the candidate);
* the fine-tune's own ruler: a per-coordinate scale σ_cal and each row's path floor,
  how far the numeric anchor's full replay and its incremental path disagree
  (:func:`path_ruler` computes both from the two paths' vectors);
* each row's stratum (:func:`select_strata`): 16 ordinary rows fixed from the
  generation ids alone, before anything is measured, and the rest labelled for
  the fixture set by the shipped selection rules;
* captures of the rows repeated one at a time and batched in eights.

:func:`check_transfer` scores the deviation with the **base's** tolerance, reusing
the install check's scoring (:func:`anamnesis.extraction.vllm.conformance.score_rows`,
:func:`~anamnesis.extraction.vllm.conformance.tolerance_reasons`): every row's
component ratios under the base's ceilings and every family under the base's
maxima, and the ordinary stratum's median under the base's p90. Rows whose path
floor exceeds the base's :data:`BASE_MAX_FLOOR` are named and left out of that
scoring, because on a fragile reference the deviation measures the reference
rather than the lane. The lane must also reproduce itself byte for byte across
the repeat and the batch.

**The ordinary stratum is chosen from ids, never from deviations.** The median
and tail gates compare a sample statistic with a population quantile of the
base, and that comparison only means something over rows drawn without regard to
how far they deviate. Rows ranked by deviation are the sample's largest movers,
so the rows left after them would be its calmest, and a median over those would
pass a fine-tune that drifts everywhere.

**The tail rule is a count, not a bound on every row.** The install check bounds
every ordinary row by the base's p99 because a host reproducing the qualified
lane deviates from its fixtures by far less than the qualification's own
deviations. Here the deviation *is* a full engine-versus-hook deviation on new
weights, expected to be distributed like the base's, so each ordinary row
exceeds the base's p99 with probability about 0.01 even when the fine-tune is
exactly in regime, and requiring all 16 under it would refuse such a fine-tune
about 15% of the time. At most :data:`TAIL_ALLOWANCE` ordinary rows may exceed
it; an in-regime fine-tune has two or more above it with probability about 1%.

A pass returns a :class:`TransferReceipt` together with the fine-tune's own
fixture set and tolerance, built from the sample by :func:`build_extension_fixtures`
under the same contracts as the shipped ones, so a host's install check works for
the fine-tune afterwards. A refusal returns the receipt alone, with its reasons.
:mod:`anamnesis.extraction.vllm.extensions` admits a lane only on a passing receipt
that matches its entry.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from anamnesis.extraction.vllm.conformance import (
    CapturedFixture,
    FixtureRow,
    FixtureSet,
    RowResult,
    Tolerance,
    check_coordinates,
    component_from_ratios,
    score_rows,
    tolerance_reasons,
)
from anamnesis.extraction.vllm.envelope import (
    LANE_MODELS,
    canonical_digest,
    extension_identity,
    lane_id,
)

TRANSFER_CONTRACT = "vllm-transfer-receipt/1"

SAMPLE_ROWS = (44, 60)
"""The fewest and most rows a transfer sample holds: enough for every selection
rule's count and a 16-row ordinary stratum, few enough that a host's install check
over the resulting fixtures stays cheap."""

DETERMINISM_ROWS = 16
"""The fewest rows whose repeated and batched captures must match byte for byte."""

BASE_MAX_FLOOR: dict[str, float | None] = {"3b": None, "8b": None, "70b": 2.39}
"""Per shipped lane, the largest path floor a row may have and still be scored.

A row whose anchor paths disagree by more than this has a reference too fragile
to divide by. At 70B the limit is 2.39, the largest path floor measured on the
8B model's qualification rows, and the shipped 70B tolerance names exactly its
fixture rows above it in ``excluded_rows``. The 3B and 8B tolerances exclude no
row, so their lanes have no limit and every row of a fine-tune of them is scored."""

ATTENTION_COMPONENT = "attention"
SPECTRAL_FAMILY = "attn-spectral"
COVERAGE_COORDINATE = re.compile(r"^cache_cache_coverage_")
GATE_SPARSITY_COORDINATE = re.compile(r"^gate_L\d+_sparsity_")
ORDINARY_RULE = "evenly-spaced"
ORDINARY_ROWS = 16
"""How many rows the ordinary stratum holds."""

UNRANKED = "unranked"
"""The label of a row neither ordinary nor taken by a ranked rule: it is a fixture,
gated by the ceilings and family maxima, and read by no stratum gate."""

TAIL_ALLOWANCE = 1
"""How many ordinary rows may exceed the base's p99 ratio, per component."""

RULES: tuple[tuple[str, int], ...] = (
    ("largest-attention-shift", 8),
    ("largest-coverage-shift", 8),
    ("largest-gate-sparsity-displacement", 8),
    ("largest-spectral-deviation", 4),
)
"""The ranked selection rules, in order, with how many rows each takes from the
rows outside the ordinary stratum. Their labels shape the fixture set and never
feed a gate."""

SUBSTRATE_LABEL = "transfer check: fast lane vs vLLM lane distance / path floor"

HEX64 = "^[0-9a-f]{64}$"
F32 = NDArray[np.float32]
F64 = NDArray[np.float64]


class SampleRow(BaseModel):
    """One sampled row: its tokens, where its prompt ends, and its path floor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    generation_id: int = Field(ge=0)
    input_ids: tuple[int, ...] = Field(min_length=2)
    prompt_length: int = Field(gt=0)
    end: int
    floor_b: float = Field(gt=0, description="The row's path floor, in σ_cal units")

    @model_validator(mode="after")
    def _span_partitions_ids(self) -> Self:
        if self.end != len(self.input_ids) or not 0 < self.prompt_length < self.end - 1:
            raise ValueError(f"row {self.generation_id}: span does not partition its ids")
        return self


class TransferSample(BaseModel):
    """A fine-tune's sample: its ruler, its rows, and both lanes' vectors for them."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    feature_names: tuple[str, ...] = Field(min_length=1)
    sigma_cal: F64
    weights: F64
    rows: tuple[SampleRow, ...] = Field(min_length=1)
    reference: dict[int, F32] = Field(description="The fast lane's vector per row")
    candidate: dict[int, F32] = Field(description="The vLLM lane's vector per row")

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        n = len(self.feature_names)
        if len(set(self.feature_names)) != n:
            raise ValueError("feature names repeat")
        if self.sigma_cal.shape != (n,) or self.weights.shape != (n,):
            raise ValueError("the ruler must cover every feature name")
        if not (np.isfinite(self.sigma_cal).all() and (self.sigma_cal > 0).all()):
            raise ValueError("σ_cal must be finite and positive everywhere")
        if not (np.isfinite(self.weights).all() and (self.weights > 0).all()):
            raise ValueError("weights must be finite and positive everywhere")
        ids = [r.generation_id for r in self.rows]
        if len(set(ids)) != len(ids) or set(ids) != set(self.reference) \
                or set(ids) != set(self.candidate):
            raise ValueError("every sampled row needs exactly one vector from each lane")
        for side in (self.reference, self.candidate):
            for gid, vector in side.items():
                if vector.shape != (n,) or vector.dtype != np.float32 \
                        or not np.isfinite(vector).all():
                    raise ValueError(f"vector {gid} is not a finite float32 [features]")
        return self

    @property
    def ids(self) -> tuple[int, ...]:
        """The sampled generation ids, ascending."""
        return tuple(sorted(r.generation_id for r in self.rows))

    @property
    def floors(self) -> dict[int, float]:
        """Each row's path floor."""
        return {r.generation_id: r.floor_b for r in self.rows}

    @property
    def digest(self) -> str:
        """Content digest: the rows, the ruler and both lanes' vector bytes."""
        h = hashlib.sha256()
        h.update(json.dumps(dict(feature_names=list(self.feature_names),
                                 rows=[r.model_dump(mode="json") for r in self.rows]),
                            sort_keys=True).encode())
        h.update(self.sigma_cal.tobytes())
        h.update(self.weights.tobytes())
        for side in (self.reference, self.candidate):
            for gid in sorted(side):
                h.update(gid.to_bytes(8, "little"))
                h.update(side[gid].tobytes())
        return h.hexdigest()

    def standardized(self, gid: int) -> F64:
        """Row ``gid``'s deviation of the candidate from the reference, over σ_cal."""
        return (self.candidate[gid].astype(np.float64)
                - self.reference[gid].astype(np.float64)) / self.sigma_cal


class TransferReceipt(BaseModel):
    """The transfer check's verdict for one fine-tune of one shipped lane."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract: Literal["vllm-transfer-receipt/1"] = TRANSFER_CONTRACT
    verdict: Literal["pass", "refuse"]
    reasons: tuple[str, ...]
    key: str = Field(min_length=1, description="The extension lane's key")
    extends: str = Field(min_length=1)
    base_lane_id: str = Field(min_length=1)
    base_tolerance_digest: str = Field(pattern=HEX64,
                                       description="The base tolerance the sample was scored "
                                                   "against")
    lane_id: str = Field(min_length=1, description="The extension lane's id")
    checkpoint_sha256: str = Field(pattern=HEX64)
    calibration_sha256: str = Field(pattern=HEX64)
    sample_sha256: str = Field(pattern=HEX64)
    max_floor: float | None = Field(description="The base's path-floor limit; None scores "
                                                "every row")
    fragile_rows: tuple[int, ...] = Field(description="Rows over the limit, left out of the "
                                                      "scoring")
    determinism_rows: int = Field(ge=0)
    ordinary_rows: tuple[int, ...] = Field(
        default=(), description="The id-chosen rows the median and tail gates read")
    ordinary_over_p90: dict[str, int] = Field(
        default_factory=dict, description="Per component, scored ordinary rows over the "
                                          "base's p90 ratio")
    ordinary_over_p99: dict[str, int] = Field(
        default_factory=dict, description="Per component, scored ordinary rows over the "
                                          "base's p99 ratio")
    rows: tuple[RowResult, ...]
    family_report: dict[str, float] = Field(
        description="Per family, the worst |δ|/σ over the scored rows, over the base's maximum")
    fixture_digest: str | None = Field(default=None, pattern=HEX64)
    tolerance_digest: str | None = Field(default=None, pattern=HEX64)

    @model_validator(mode="after")
    def _verdict_matches_contents(self) -> Self:
        if self.verdict == "pass":
            if self.reasons or self.fixture_digest is None or self.tolerance_digest is None:
                raise ValueError("a pass carries no reasons and names the fixtures and "
                                 "tolerance it produced")
        elif not self.reasons:
            raise ValueError("a refusal carries its reasons")
        return self

    @property
    def digest(self) -> str:
        return canonical_digest(self.model_dump(mode="json"))


@dataclass(frozen=True)
class TransferResult:
    """A receipt, and on a pass the fine-tune's fixture set and tolerance."""

    receipt: TransferReceipt
    fixtures: FixtureSet | None
    tolerance: Tolerance | None


def extension_lane_id(extends: str, checkpoint_sha256: str) -> str:
    """The lane id an extension of ``extends`` at this checkpoint carries."""
    return canonical_digest(extension_identity(extends, checkpoint_sha256))


def path_ruler(replay: F32, incremental: F32) -> tuple[F64, F64]:
    """σ_cal and each row's path floor from the numeric anchor's two paths.

    ``replay`` and ``incremental`` are ``[rows, features]``: each row's anchor
    vector from one teacher-forced forward and from prefill plus one-token steps.
    σ_cal is the per-coordinate standard deviation of the replay vectors over the
    rows, floored at the 5th percentile of its positive values so that a
    coordinate constant over the sample is not divided by zero. A row's floor is
    the L2 norm of its two paths' difference over σ_cal.

    Raises
    ------
    ValueError
        On mismatched or nonfinite inputs, fewer than two rows, a sample with no
        varying coordinate, or a row whose two paths agree exactly, which leaves
        nothing to divide by.
    """
    first = np.asarray(replay, dtype=np.float64)
    second = np.asarray(incremental, dtype=np.float64)
    if first.ndim != 2 or first.shape != second.shape or first.shape[0] < 2:
        raise ValueError("two [rows, features] arrays of one shape and at least two rows "
                         "are required")
    if not (np.isfinite(first).all() and np.isfinite(second).all()):
        raise ValueError("the anchor paths produced a nonfinite feature")
    sigma = first.std(axis=0)
    positive = sigma[sigma > 0]
    if not positive.size:
        raise ValueError("no coordinate varies over the sample")
    sigma = np.maximum(sigma, float(np.percentile(positive, 5)))
    floors = np.linalg.norm((second - first) / sigma, axis=1)
    flat = [i for i, f in enumerate(floors) if not f > 0]
    if flat:
        raise ValueError(f"rows at positions {flat}: the two anchor paths agree exactly, so "
                         "no ratio can be read against their floor")
    return sigma, floors


def _rank(scores: Mapping[int, float], rule: str) -> list[int]:
    if len(set(scores.values())) < 2:
        raise ValueError(f"{rule}: every row scores the same, so it would rank by id alone")
    return sorted(scores, key=lambda g: (-scores[g], g))


def evenly_spaced(ids: Sequence[int], k: int) -> list[int]:
    """``k`` of ``ids`` at even strides through them, in order: position
    ``round((i + 0.5) · n / k − 0.5)`` for i < k; all of them when there are at most k."""
    n = len(ids)
    if n <= k:
        return list(ids)
    return [ids[round((i + 0.5) * n / k - 0.5)] for i in range(k)]


def ordinary_rows(ids: Collection[int]) -> tuple[int, ...]:
    """The ordinary stratum of a sample: :data:`ORDINARY_ROWS` rows evenly spaced
    over its sorted generation ids, fixed before anything is measured."""
    return tuple(evenly_spaced(sorted(ids), ORDINARY_ROWS))


def select_strata(sample: TransferSample, base_tolerance: Tolerance) -> dict[int, str]:
    """Each sampled row's stratum: ordinary rows from ids, the rest ranked for the fixtures.

    The ordinary stratum is :func:`ordinary_rows`, a function of the sampled ids
    alone. From the other rows, the rules in :data:`RULES` take, in order, the rows
    with the largest attention distance, the largest coverage shift, the largest
    gate-sparsity displacement (summed |δ|/σ_cal over the discrete gate-sparsity
    coordinates) and the largest spectral-family deviation, all read from the
    candidate's deviation from the reference. A row already taken is skipped and
    the next rank taken; ties break by generation id. A row left over is
    :data:`UNRANKED`.

    Raises
    ------
    ValueError
        When the sample is outside :data:`SAMPLE_ROWS`, the base tolerance lacks a
        coordinate a rule reads, or every candidate row scores the same under a rule.
    """
    ids = sample.ids
    low, high = SAMPLE_ROWS
    if not low <= len(ids) <= high:
        raise ValueError(f"a transfer sample holds {low} to {high} rows, not {len(ids)}")
    index = check_coordinates(sample.feature_names, base_tolerance)
    components = {c.name: c for c in base_tolerance.components}
    if ATTENTION_COMPONENT not in components:
        raise ValueError(f"the base tolerance has no {ATTENTION_COMPONENT!r} component")
    attention = np.asarray([index[n] for n in components[ATTENTION_COMPONENT].feature_names])
    coverage = np.asarray([i for n, i in index.items() if COVERAGE_COORDINATE.match(n)],
                          dtype=np.int64)
    gate = np.asarray([index[n] for n in base_tolerance.discrete
                       if GATE_SPARSITY_COORDINATE.match(n)], dtype=np.int64)
    spectral = np.asarray([index[n] for n, f in base_tolerance.families.items()
                           if f == SPECTRAL_FAMILY], dtype=np.int64)
    for name, ix in (("coverage", coverage), ("gate-sparsity", gate), ("spectral", spectral)):
        if not ix.size:
            raise ValueError(f"the schema has no {name} coordinate to rank by")
    taken: dict[int, str] = {gid: ORDINARY_RULE for gid in ordinary_rows(ids)}
    pool = [gid for gid in ids if gid not in taken]
    scores: dict[str, dict[int, float]] = {rule: {} for rule, _ in RULES}
    shift, cov, gates, spec = (rule for rule, _ in RULES)
    for gid in pool:
        z = sample.standardized(gid)
        scores[shift][gid] = float(np.sqrt(np.sum(sample.weights[attention] * z[attention] ** 2)))
        scores[cov][gid] = float(np.abs(z[coverage]).max())
        scores[gates][gid] = float(np.abs(z[gate]).sum())
        scores[spec][gid] = float(np.abs(z[spectral]).max())
    for rule, count in RULES:
        added = 0
        for gid in _rank(scores[rule], rule):
            if added == count:
                break
            if gid not in taken:
                taken[gid] = rule
                added += 1
    return {gid: taken.get(gid, UNRANKED) for gid in ids}


def _determinism_reasons(sample: TransferSample,
                         determinism: Sequence[CapturedFixture]) -> list[str]:
    reasons = []
    by_gid = {c.generation_id: c for c in determinism}
    if len(by_gid) != len(determinism):
        reasons.append("the determinism check names a row twice")
    if len(by_gid) < DETERMINISM_ROWS:
        reasons.append(f"the determinism check covers {len(by_gid)} rows; at least "
                       f"{DETERMINISM_ROWS} are required")
    unknown = sorted(set(by_gid) - set(sample.ids))
    if unknown:
        reasons.append(f"the determinism check names rows outside the sample: {unknown[:5]}")
    for gid in sorted(set(by_gid) & set(sample.ids)):
        c = by_gid[gid]
        if c.first.tobytes() != sample.candidate[gid].tobytes():
            reasons.append(f"row {gid}: the scored vLLM vector is not the determinism "
                           "check's first capture")
        elif c.repeat.tobytes() != c.first.tobytes() or c.batched.tobytes() != c.first.tobytes():
            reasons.append(f"row {gid}: lane disagrees with itself (repeat or batched "
                           "capture differs)")
    return reasons


def build_extension_fixtures(
    *,
    key: str,
    extends: str,
    checkpoint_sha256: str,
    calibration_sha256: str,
    sample: TransferSample,
    strata: Mapping[int, str],
    rows: Sequence[RowResult],
    base_tolerance: Tolerance,
    fragile: Collection[int],
) -> tuple[FixtureSet, Tolerance]:
    """The fine-tune's fixture set and tolerance, from its scored sample.

    Every sampled row is a fixture, carrying its stratum and its path floor, and
    its vector is the vLLM lane's. The tolerance keeps the base's components,
    families, discrete coordinates and median-gate stratum, so a host's install
    check gates its median and tail on the same id-chosen ordinary rows; its ceilings are the
    maximum, 90th and 99th percentile of the sample's own ratios, and its family
    maxima the sample's own largest |δ|/σ_cal, each over the rows not in
    ``fragile``, which it names as excluded.
    """
    skip = set(fragile)
    ratios = {r.generation_id: r.component_ratios for r in rows}
    components = tuple(
        component_from_ratios(c.name, c.feature_names,
                              {g: ratios[g][c.name] for g in ratios if g not in skip},
                              SUBSTRATE_LABEL)
        for c in base_tolerance.components)
    index = check_coordinates(sample.feature_names, base_tolerance)
    members: dict[str, list[int]] = {}
    for name, family in base_tolerance.families.items():
        members.setdefault(family, []).append(index[name])
    kept = [g for g in sample.ids if g not in skip]
    peaks = np.max(np.abs(np.stack([sample.standardized(g) for g in kept])), axis=0)
    family_max = {family: float(peaks[np.asarray(ix)].max()) for family, ix in members.items()}
    tolerance = Tolerance(model=key, components=components, families=dict(base_tolerance.families),
                          family_max_abs_sigma=family_max, discrete=base_tolerance.discrete,
                          median_gate_stratum=base_tolerance.median_gate_stratum,
                          excluded_rows=tuple(sorted(skip)),
                          sources={"vllm-transfer/sample": sample.digest,
                                   "vllm-transfer/base-tolerance": base_tolerance.digest})
    ruler = hashlib.sha256()
    ruler.update(sample.sigma_cal.tobytes())
    ruler.update(sample.weights.tobytes())
    ruler.update(json.dumps(sorted(sample.floors.items())).encode())
    fixtures = FixtureSet(
        model=key, lane_id=extension_lane_id(extends, checkpoint_sha256),
        checkpoint_sha256=checkpoint_sha256, calibration_sha256=calibration_sha256,
        ruler_sha256=ruler.hexdigest(),
        authority=dict(transfer_sample_sha256=sample.digest, base_lane_id=lane_id(extends),
                       base_tolerance_sha256=base_tolerance.digest),
        feature_names=sample.feature_names, sigma_cal=sample.sigma_cal, weights=sample.weights,
        rows=tuple(FixtureRow(generation_id=r.generation_id, population="native",
                              input_ids=r.input_ids, prompt_length=r.prompt_length, end=r.end,
                              floor_b=r.floor_b, selected_by=strata[r.generation_id])
                   for r in sorted(sample.rows, key=lambda r: r.generation_id)),
        vectors={g: sample.candidate[g] for g in sample.ids})
    return fixtures, tolerance


def check_transfer(
    *,
    key: str,
    extends: str,
    base_tolerance: Tolerance,
    base_feature_names: Sequence[str],
    checkpoint_sha256: str,
    calibration_sha256: str,
    sample: TransferSample,
    strata: Mapping[int, str],
    determinism: Sequence[CapturedFixture],
    max_floor: float | None,
) -> TransferResult:
    """Score a fine-tune's sample against its base's tolerance.

    ``key`` is the extension lane's key and ``extends`` the shipped lane whose
    ``base_tolerance`` and ``base_feature_names`` are given. ``max_floor`` is the
    base's path-floor limit (:data:`BASE_MAX_FLOOR`). ``strata`` assigns every
    sampled row its stratum (:func:`select_strata`), and ``determinism`` holds the
    vLLM lane's captures of at least :data:`DETERMINISM_ROWS` sampled rows, once
    alone, repeated, and batched; each first capture must be the vector scored.

    Refuses, with every reason found, a schema other than the base's, a sample
    outside :data:`SAMPLE_ROWS`, strata that do not cover the sample, a lane that
    disagrees with itself, and any deviation outside the base's tolerance on the
    rows at or under ``max_floor``. A pass also returns the fine-tune's fixtures and
    tolerance (:func:`build_extension_fixtures`).

    Raises
    ------
    ValueError
        When ``extends`` is not a shipped lane, ``key`` is one, or the base tolerance
        belongs to another model: those are a caller's error, not a verdict.
    """
    if extends not in LANE_MODELS:
        raise ValueError(f"{extends!r} is not a shipped vLLM lane")
    if key in LANE_MODELS:
        raise ValueError(f"{key!r} is a shipped lane; an extension takes a key of its own")
    if base_tolerance.model != extends:
        raise ValueError(f"the tolerance given is {base_tolerance.model!r}'s, not {extends!r}'s")
    if base_tolerance.median_gate_stratum != ORDINARY_RULE:
        raise ValueError(f"the {extends} tolerance gates stratum "
                         f"{base_tolerance.median_gate_stratum!r}, not {ORDINARY_RULE!r}")
    reasons: list[str] = []
    if tuple(sample.feature_names) != tuple(base_feature_names):
        reasons.append(f"the sample's feature schema is not the {extends} lane's")
    low, high = SAMPLE_ROWS
    if not low <= len(sample.rows) <= high:
        reasons.append(f"the sample holds {len(sample.rows)} rows; a transfer sample holds "
                       f"{low} to {high}")
    ordinary = ordinary_rows(sample.ids)
    if set(strata) != set(sample.ids):
        reasons.append("the strata must name every sampled row and no other")
    elif {g for g, s in strata.items() if s == ORDINARY_RULE} != set(ordinary):
        reasons.append("the ordinary stratum must be the rows evenly spaced over the sampled "
                       "ids, chosen before any deviation is read")
    reasons += _determinism_reasons(sample, determinism)
    fragile = tuple(sorted(g for g, f in sample.floors.items()
                           if max_floor is not None and f > max_floor))
    rows: list[RowResult] = []
    family_report: dict[str, float] = {}
    over_p90: dict[str, int] = {}
    over_p99: dict[str, int] = {}
    if not reasons:
        if len(fragile) == len(sample.rows):
            reasons.append(f"every sampled row's path floor exceeds {max_floor}")
        else:
            rows, family_report = score_rows(
                sample.feature_names, sample.sigma_cal, sample.weights, sample.floors,
                sample.reference, sample.candidate, base_tolerance,
                exclude_from_report=fragile)
            no_row_tail = base_tolerance.model_copy(update=dict(components=tuple(
                c.model_copy(update=dict(p99_ratio=None)) for c in base_tolerance.components)))
            reasons += tolerance_reasons(rows, no_row_tail, strata, exclude=fragile)
            scored = [r for r in rows
                      if r.generation_id in set(ordinary) and r.generation_id not in fragile]
            for component in base_tolerance.components:
                ratios = [r.component_ratios[component.name] for r in scored]
                over_p90[component.name] = sum(x > component.p90_ratio for x in ratios)
                if component.p99_ratio is None:
                    continue
                over_p99[component.name] = sum(x > component.p99_ratio for x in ratios)
                if over_p99[component.name] > TAIL_ALLOWANCE:
                    reasons.append(
                        f"{component.name}: {over_p99[component.name]} of {len(scored)} "
                        f"ordinary rows exceed the recorded p99 {component.p99_ratio:.3g}; "
                        f"at most {TAIL_ALLOWANCE} may")
    fixtures = tolerance = None
    if not reasons:
        fixtures, tolerance = build_extension_fixtures(
            key=key, extends=extends, checkpoint_sha256=checkpoint_sha256,
            calibration_sha256=calibration_sha256, sample=sample, strata=strata, rows=rows,
            base_tolerance=base_tolerance, fragile=fragile)
    receipt = TransferReceipt(
        verdict="refuse" if reasons else "pass", reasons=tuple(reasons), key=key,
        extends=extends, base_lane_id=lane_id(extends),
        base_tolerance_digest=base_tolerance.digest,
        lane_id=extension_lane_id(extends, checkpoint_sha256),
        checkpoint_sha256=checkpoint_sha256, calibration_sha256=calibration_sha256,
        sample_sha256=sample.digest, max_floor=max_floor, fragile_rows=fragile,
        determinism_rows=len({c.generation_id for c in determinism}),
        ordinary_rows=ordinary, ordinary_over_p90=over_p90, ordinary_over_p99=over_p99,
        rows=tuple(rows),
        family_report=family_report,
        fixture_digest=None if fixtures is None else fixtures.digest,
        tolerance_digest=None if tolerance is None else tolerance.digest)
    return TransferResult(receipt=receipt, fixtures=fixtures, tolerance=tolerance)
