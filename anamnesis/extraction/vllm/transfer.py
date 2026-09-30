"""The lane-agreement audit: how far do two lanes sit apart on matched tokens?

No lane is ground truth: a lane is self-consistency plus provenance, and the hook
path is the reference lane by convention. Whether two lanes agree is measured
when a claim needs it: before claiming a finding generalizes beyond one lane,
when comparing deployments or checking that a serving change left the computation
alone, and when joining conclusions from two lanes at the effect level. It is
never a condition of using a lane.

:func:`check_transfer` scores any sample of matched vectors from two lanes
(reference and candidate, standardized by the sample's own σ_cal and path floors)
against the regime a shipped lane's qualification measured between the vLLM lane
and the hook path while that lane's effects were retained: the **base's**
tolerance, through the install check's scoring. ``anamnesis/scripts/transfer_vllm.py``
runs it for a fine-tune of a shipped lane's model, whose architecture, kernels,
determinism and schema are its base's, pairing the fine-tune's vLLM lane with its
fast lane. Its verdict is information: ``pass`` reads the sample inside the
base's measured regime. Every scored sample, whatever the verdict, yields the
fine-tune's fixtures and tolerance (:func:`build_extension_fixtures`): they are the
fine-tune's own lane output, the reference its hosts' install checks read, not a
certificate of agreement. A sample that cannot be scored (a lane that disagrees with
itself, a schema or sample outside the check's scope) yields none. An extension lane
records the receipt when it names one (:mod:`anamnesis.extraction.vllm.extensions`).

Its readings differ from the install check's because the deviation here is a full
engine-versus-hook deviation on new weights, distributed like the base's own:

* the median and tail gates read 16 **ordinary rows chosen from the sampled ids
  alone** (:func:`ordinary_rows`); rows ranked by deviation would leave the
  calmest rows ordinary and make those gates lenient;
* its component gates are counted: an in-regime sample of 44 rows puts some row over
  a 190-row maximum about one time in five.

**Each base carries its own rule set** (:class:`TransferRules`, shipped beside its
tolerance), measured on that base's own records: simulated in-regime samples drawn
from the base's rows, scored against ceilings read from the rows left out, give each
rule set's false-refusal rate. An audit against a base without one is refused by
name. A rule set holds:

* per component, at most ``component_max_allowance`` rows over the base's max ratio,
  the ordinary median at or below the p90, and at most ``p99_allowance`` ordinary rows
  over the p99;
* per family, a **drift gate**: the ordinary rows' median of each row's largest
  family |δ|/σ_cal at or below the base's family p90 (``family_p90_abs_sigma``).

A single row's extremity is reported, never gated, whatever its size: every row over a
component's or a family's base maximum is listed in the receipt with its multiple, beside
the bifurcation diagnostic (:func:`bifurcation_report`). At 70B the largest single-row
deviations are single tokens where one engine forms a massive activation and the other
does not, which did not affect retained effects, and no per-row bound read from about
190 rows separated them from a departure without refusing most in-regime samples; the
gates that remain read distributions.

The families where the drift gate catches a 1.5× drift less than 80% of the time are
named in the rule set and in every receipt, with the scale at which it does.

Rows whose path floor exceeds :data:`BASE_MAX_FLOOR` are named and left out of the
scoring: on a fragile reference the deviation measures the reference.
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
    RowComponent,
    RowResult,
    Tolerance,
    check_coordinates,
    score_rows,
    tolerance_reasons,
)
from anamnesis.extraction.vllm.envelope import (
    LANE_MODELS,
    extension_lane_id,
    lane_id,
)

SAMPLE_ROWS = (44, 60)
"""The fewest and most rows a transfer sample holds."""

DETERMINISM_ROWS = 16
"""The fewest rows whose repeated and batched captures must match byte for byte."""

BASE_MAX_FLOOR: dict[str, float | None] = {"3b": None, "8b": None, "70b": 2.489361708872322}
"""Per shipped lane, the largest path floor a scored row may have. At 70B it is the
largest path floor of the 8B qualification rows, measured under the lane's
arithmetic like every floor here, and the shipped 70B tolerance excludes exactly
its fixture rows above it; the 3B and 8B tolerances exclude none."""

ORDINARY_RULE = "evenly-spaced"
ORDINARY_ROWS = 16
UNRANKED = "unranked"
RULES_FILE = "transfer_rules.json"
RULES: tuple[tuple[str, int], ...] = (
    ("largest-attention-shift", 8),
    ("largest-coverage-shift", 8),
    ("largest-gate-sparsity-displacement", 8),
    ("largest-spectral-deviation", 4),
)
"""The shipped selection rules, applied to the non-ordinary rows to label the
fixture set; they feed no gate."""

COVERAGE_COORDINATE = re.compile(r"^cache_cache_coverage_")
GATE_SPARSITY_COORDINATE = re.compile(r"^gate_L\d+_sparsity_")
SOURCE_LABEL = "transfer check: fast lane vs vLLM lane distance / path floor"
HEX64 = "^[0-9a-f]{64}$"
F32 = NDArray[np.float32]
F64 = NDArray[np.float64]


class TransferRules(BaseModel):
    """A base's transfer rule set, qualified on that base's own records."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract: Literal["vllm-transfer-rules/1"] = "vllm-transfer-rules/1"
    model: str = Field(min_length=1)
    tolerance_digest: str = Field(pattern=HEX64, description="The base tolerance the rule set "
                                                             "was qualified with")
    component_max_allowance: int = Field(ge=0)
    p99_allowance: int = Field(ge=0)
    diagnostic_from_depth: float = Field(gt=0, lt=1, description="The bifurcation diagnostic "
                                                                 "reads blocks from this "
                                                                 "fraction of depth")
    family_p90_abs_sigma: dict[str, float] = Field(min_length=1)
    weakly_guarded: dict[str, float] = Field(
        default_factory=dict, description="Family → the drift scale the gate catches 80% of "
                                          "the time, where 1.5× is caught less often")
    qualification: dict[str, float | int | str | list[int]] = Field(
        description="The false-refusal probe the rule set passed on the base's records")

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"), sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()


def load_transfer_rules(base: str, tolerance: Tolerance) -> TransferRules:
    """``base``'s qualified rule set, checked against the tolerance it was qualified with.

    Raises
    ------
    ValueError
        When the base ships no rule set, or one for another model or tolerance.
    """
    from anamnesis.extraction.vllm.runtime import fixtures_dir

    path = fixtures_dir(base) / RULES_FILE
    if not path.is_file():
        raise ValueError(f"no transfer rules are qualified for {base!r}; its extensions are "
                         "refused until a rule set passes a probe on its own records")
    rules = TransferRules.model_validate_json(path.read_text())
    if rules.model != base or rules.tolerance_digest != tolerance.digest:
        raise ValueError(f"{path} is not {base!r}'s rule set for its shipped tolerance")
    if set(rules.family_p90_abs_sigma) != set(tolerance.family_max_abs_sigma):
        raise ValueError(f"{path} does not give a p90 for every family of the tolerance")
    return rules


class BifurcationReport(BaseModel):
    """The bifurcation diagnostic on one row, reported beside its deviations: never gating."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    peak: float = Field(ge=0, description="Largest relative residual difference, lane vs anchor")
    token: int = Field(ge=0)
    block: int = Field(ge=0, description="The block of the peak")
    channel_ratio: float = Field(ge=0, description="At the peak token, the largest of either "
                                                   "side's largest residual channel over that "
                                                   "side's median largest channel at the block")
    channel_block: int = Field(ge=0)
    channel_side: Literal["lane", "anchor"]
    channel: int = Field(ge=0)
    other_fraction: float = Field(ge=0, description="The other side's same channel over it: "
                                                    "near 0 when one engine alone forms it")


def bifurcation_report(lane_hidden: NDArray, anchor_hidden: NDArray,
                       from_depth: float) -> BifurcationReport:
    """Where and how a row's lane and anchor residual streams diverge most.

    ``lane_hidden`` is the lane capture's block outputs ``[layers, tokens, hidden]``,
    ``anchor_hidden`` the anchor's hidden states ``[tokens, layers + 1, hidden]`` (the
    embedding first) over the same tokens. Over the blocks from ``round(from_depth ·
    layers)`` to the second-to-last (the anchor's last state is normed, the lane's is
    not): the peak ``‖lane − anchor‖ / ‖anchor‖`` of any token, and at that token the
    largest one-sided channel ratio, one side's largest channel over that side's median
    (over tokens) largest channel at the block, with the other side's same channel as a
    fraction of it. A massive activation one engine forms and the other does not shows
    as a large ratio with a small fraction.

    Raises
    ------
    ValueError
        When the two do not cover the same tokens and blocks.
    """
    lane = np.asarray(lane_hidden, dtype=np.float32)
    anchor = np.asarray(anchor_hidden, dtype=np.float32)
    if lane.ndim != 3 or anchor.shape != (lane.shape[1], lane.shape[0] + 1, lane.shape[2]):
        raise ValueError("the lane and anchor hidden states do not cover the same tokens and "
                         "blocks")
    blocks = range(round(from_depth * lane.shape[0]), lane.shape[0] - 1)
    if not blocks:
        raise ValueError("no block to read at this depth")
    peak, token, block = 0.0, 0, blocks[0]
    for b in blocks:
        a = anchor[:, b + 1, :].astype(np.float64)
        rel = np.linalg.norm(lane[b].astype(np.float64) - a, axis=-1) \
            / np.maximum(np.linalg.norm(a, axis=-1), 1e-12)
        t = int(rel.argmax())
        if rel[t] > peak:
            peak, token, block = float(rel[t]), t, b
    best = dict(channel_ratio=0.0, channel_block=blocks[0], channel_side="lane", channel=0,
                other_fraction=0.0)
    for b in blocks:
        for side, own, other in (("lane", lane[b], anchor[:, b + 1, :]),
                                 ("anchor", anchor[:, b + 1, :], lane[b])):
            typical = float(np.median(np.abs(own).max(axis=-1)))
            channel = int(np.abs(own[token]).argmax())
            value = float(abs(own[token, channel]))
            if typical > 0 and value > 0 and value / typical > best["channel_ratio"]:
                best = dict(channel_ratio=value / typical, channel_block=b, channel_side=side,
                            channel=channel,
                            other_fraction=abs(float(other[token, channel])) / value)
    return BifurcationReport(peak=peak, token=token, block=block, **best)


def family_values(sample: "TransferSample", tolerance: Tolerance) -> dict[int, dict[str, float]]:
    """Per scored row and family, the row's largest |δ|/σ_cal over the family's continuous
    coordinates."""
    index = check_coordinates(sample.feature_names, tolerance)
    columns: dict[str, list[int]] = {}
    for name, family in tolerance.families.items():
        columns.setdefault(family, []).append(index[name])
    out = {}
    for gid in sample.ids:
        z = np.abs(sample.standardized(gid))
        out[gid] = {f: float(z[cols].max()) for f, cols in columns.items()}
    return out


class TransferSample(BaseModel):
    """A fine-tune's sample: its ruler, its rows (``floor_b`` its own path floor) and
    both lanes' vectors."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    feature_names: tuple[str, ...] = Field(min_length=1)
    sigma_cal: F64
    weights: F64
    rows: tuple[FixtureRow, ...] = Field(min_length=1)
    reference: dict[int, F32]
    candidate: dict[int, F32]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        n = len(self.feature_names)
        for name, array in (("sigma_cal", self.sigma_cal), ("weights", self.weights)):
            if array.shape != (n,) or not (np.isfinite(array).all() and (array > 0).all()):
                raise ValueError(f"{name} must be finite and positive over every feature")
        ids = {r.generation_id for r in self.rows}
        if len(ids) != len(self.rows) or ids != set(self.reference) \
                or ids != set(self.candidate):
            raise ValueError("every sampled row needs exactly one vector from each lane")
        for side in (self.reference, self.candidate):
            for gid, v in side.items():
                if v.shape != (n,) or v.dtype != np.float32 or not np.isfinite(v).all():
                    raise ValueError(f"vector {gid} is not a finite float32 [features]")
        return self

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(sorted(r.generation_id for r in self.rows))

    @property
    def floors(self) -> dict[int, float]:
        return {r.generation_id: r.floor_b for r in self.rows}

    @property
    def digest(self) -> str:
        h = hashlib.sha256(json.dumps([list(self.feature_names),
                                       [r.model_dump(mode="json") for r in self.rows]],
                                      sort_keys=True).encode())
        h.update(self.sigma_cal.tobytes() + self.weights.tobytes())
        for side in (self.reference, self.candidate):
            for gid in sorted(side):
                h.update(gid.to_bytes(8, "little") + side[gid].tobytes())
        return h.hexdigest()

    def standardized(self, gid: int) -> F64:
        """Row ``gid``'s candidate-minus-reference deviation over σ_cal."""
        return (self.candidate[gid].astype(np.float64)
                - self.reference[gid].astype(np.float64)) / self.sigma_cal


class TransferReceipt(BaseModel):
    """The transfer check's verdict for one fine-tune of one shipped lane."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract: Literal["vllm-transfer-receipt/2"] = "vllm-transfer-receipt/2"
    verdict: Literal["pass", "refuse"]
    reasons: tuple[str, ...]
    key: str = Field(min_length=1)
    extends: str = Field(min_length=1)
    base_lane_id: str
    base_tolerance_digest: str = Field(pattern=HEX64)
    lane_id: str
    checkpoint_sha256: str = Field(pattern=HEX64)
    calibration_sha256: str = Field(pattern=HEX64)
    sample_sha256: str = Field(pattern=HEX64)
    max_floor: float | None
    fragile_rows: tuple[int, ...]
    determinism_rows: int
    ordinary_rows: tuple[int, ...] = ()
    rows_over_max: dict[str, tuple[int, ...]] = Field(default_factory=dict)
    worst_max_multiple: dict[str, float] = Field(default_factory=dict)
    ordinary_over_p90: dict[str, int] = Field(default_factory=dict)
    ordinary_over_p99: dict[str, int] = Field(default_factory=dict)
    rules_digest: str = Field(pattern=HEX64)
    weakly_guarded: dict[str, float] = Field(
        description="Families the drift gate guards weakly, with the drift scale it catches")
    family_ordinary_median: dict[str, float] = Field(default_factory=dict)
    family_rows_over_max: dict[str, tuple[int, ...]] = Field(
        default_factory=dict, description="Reported, not gated")
    over_max_multiples: dict[str, dict[int, float]] = Field(
        default_factory=dict, description="Per component, and per family as 'family:<name>', "
                                          "each row over the base maximum with its multiple; "
                                          "reported, not gated")
    diagnostics: dict[int, BifurcationReport] = Field(
        default_factory=dict, description="The bifurcation diagnostic per row measured")
    rows: tuple[RowResult, ...]
    family_report: dict[str, float]
    fixture_digest: str | None = Field(default=None, pattern=HEX64)
    tolerance_digest: str | None = Field(default=None, pattern=HEX64)

    @model_validator(mode="after")
    def _verdict_matches_contents(self) -> Self:
        passed = self.verdict == "pass"
        built = self.fixture_digest is not None
        if passed == bool(self.reasons) or built != (self.tolerance_digest is not None) \
                or (passed and not built):
            raise ValueError("a pass names its fixtures and tolerance and no reason; a "
                             "refusal carries its reasons; fixtures and tolerance come together")
        return self


@dataclass(frozen=True)
class TransferResult:
    receipt: TransferReceipt
    fixtures: FixtureSet | None
    tolerance: Tolerance | None


def path_ruler(replay: F32, incremental: F32) -> tuple[F64, F64]:
    """σ_cal and per-row path floors from the anchor's two paths, ``[rows, features]`` each.

    σ_cal is the replay vectors' per-coordinate standard deviation, floored at the
    5th percentile of its positive values; a floor is the L2 norm of the paths'
    difference over σ_cal.

    Raises
    ------
    ValueError
        On mismatched or nonfinite inputs, fewer than two rows, no varying
        coordinate, or a row whose paths agree exactly (nothing to divide by).
    """
    first, second = np.asarray(replay, np.float64), np.asarray(incremental, np.float64)
    if first.ndim != 2 or first.shape != second.shape or first.shape[0] < 2 \
            or not (np.isfinite(first).all() and np.isfinite(second).all()):
        raise ValueError("two finite [rows, features] arrays of one shape and at least two "
                         "rows are required")
    sigma = first.std(axis=0)
    if not (sigma > 0).any():
        raise ValueError("no coordinate varies over the sample")
    sigma = np.maximum(sigma, float(np.percentile(sigma[sigma > 0], 5)))
    floors = np.linalg.norm((second - first) / sigma, axis=1)
    flat = [i for i, f in enumerate(floors) if not f > 0]
    if flat:
        raise ValueError(f"rows at positions {flat}: the two anchor paths agree exactly, so "
                         "no ratio can be read against their floor")
    return sigma, floors


def ordinary_rows(ids: Collection[int]) -> tuple[int, ...]:
    """:data:`ORDINARY_ROWS` ids at even strides through the sorted ``ids``, position
    ``round((i + 0.5) · n / k − 0.5)``: fixed before anything is measured."""
    ordered, k = sorted(ids), ORDINARY_ROWS
    n = len(ordered)
    return tuple(ordered if n <= k else [ordered[round((i + 0.5) * n / k - 0.5)]
                                         for i in range(k)])


def select_strata(sample: TransferSample, base_tolerance: Tolerance) -> dict[int, str]:
    """Each row's stratum: :func:`ordinary_rows`, then :data:`RULES` over the rest by
    attention distance, largest coverage shift, summed gate-sparsity displacement and
    largest spectral deviation (ties by id), and :data:`UNRANKED` for what is left.

    Raises
    ------
    ValueError
        On a sample outside :data:`SAMPLE_ROWS`, a schema without a coordinate a rule
        reads, or a rule under which every row scores the same (it would rank by id).
    """
    ids = sample.ids
    if not SAMPLE_ROWS[0] <= len(ids) <= SAMPLE_ROWS[1]:
        raise ValueError(f"a transfer sample holds {SAMPLE_ROWS[0]} to {SAMPLE_ROWS[1]} "
                         f"rows, not {len(ids)}")
    index = check_coordinates(sample.feature_names, base_tolerance)
    attention = next((c for c in base_tolerance.components if c.name == "attention"), None)
    groups = [np.asarray([index[n] for n in names], dtype=np.int64) for names in (
        attention.feature_names if attention else (),
        [n for n in index if COVERAGE_COORDINATE.match(n)],
        [n for n in base_tolerance.discrete if GATE_SPARSITY_COORDINATE.match(n)],
        [n for n, f in base_tolerance.families.items() if f == "attn-spectral"])]
    if any(not g.size for g in groups):
        raise ValueError("the schema lacks an attention, coverage, gate-sparsity or "
                         "spectral coordinate to rank by")
    taken = dict.fromkeys(ordinary_rows(ids), ORDINARY_RULE)
    z = {g: np.abs(sample.standardized(g)) for g in ids if g not in taken}
    score = (lambda a: float(np.sqrt(np.sum(sample.weights[groups[0]] * a[groups[0]] ** 2))),
             lambda a: float(a[groups[1]].max()), lambda a: float(a[groups[2]].sum()),
             lambda a: float(a[groups[3]].max()))
    for (rule, count), fn in zip(RULES, score):
        scores = {g: fn(a) for g, a in z.items()}
        if len(set(scores.values())) < 2:
            raise ValueError(f"{rule}: every row scores the same, so it would rank by id alone")
        ranked = [g for g in sorted(scores, key=lambda g: (-scores[g], g)) if g not in taken]
        taken.update(dict.fromkeys(ranked[:count], rule))
    return {g: taken.get(g, UNRANKED) for g in ids}


def build_extension_fixtures(*, key: str, extends: str, checkpoint_sha256: str,
                             calibration_sha256: str, sample: TransferSample,
                             strata: Mapping[int, str], rows: Sequence[RowResult],
                             base_tolerance: Tolerance,
                             fragile: Collection[int]) -> tuple[FixtureSet, Tolerance]:
    """The fine-tune's fixtures (every sampled row with its stratum and vLLM vector) and
    tolerance (the base's structure; ceilings and family maxima read from the sample's
    non-fragile rows, which it names as excluded)."""
    kept = [r for r in rows if r.generation_id not in set(fragile)]
    components = []
    for c in base_tolerance.components:
        ratios = np.asarray([r.component_ratios[c.name] for r in kept])
        components.append(RowComponent(
            name=c.name, feature_names=c.feature_names, max_ratio=float(ratios.max()),
            p90_ratio=float(np.percentile(ratios, 90)),
            p99_ratio=float(np.percentile(ratios, 99)), source=SOURCE_LABEL))
    index = check_coordinates(sample.feature_names, base_tolerance)
    peaks = np.max([np.abs(sample.standardized(r.generation_id)) for r in kept], axis=0)
    family_max: dict[str, float] = {}
    for name, family in base_tolerance.families.items():
        family_max[family] = max(family_max.get(family, 0.0), float(peaks[index[name]]))
    tolerance = base_tolerance.model_copy(update=dict(
        model=key, components=tuple(components), family_max_abs_sigma=family_max,
        excluded_rows=tuple(sorted(fragile)),
        sources={"vllm-transfer/sample": sample.digest,
                 "vllm-transfer/base-tolerance": base_tolerance.digest}))
    ruler = hashlib.sha256(sample.sigma_cal.tobytes() + sample.weights.tobytes()
                           + json.dumps(sorted(sample.floors.items())).encode()).hexdigest()
    fixtures = FixtureSet(
        model=key, lane_id=extension_lane_id(extends, checkpoint_sha256),
        checkpoint_sha256=checkpoint_sha256, calibration_sha256=calibration_sha256,
        ruler_sha256=ruler,
        authority=dict(transfer_sample_sha256=sample.digest, base_lane_id=lane_id(extends),
                       base_tolerance_sha256=base_tolerance.digest),
        feature_names=sample.feature_names, sigma_cal=sample.sigma_cal, weights=sample.weights,
        rows=tuple(r.model_copy(update=dict(selected_by=strata[r.generation_id]))
                   for r in sorted(sample.rows, key=lambda r: r.generation_id)),
        vectors={g: sample.candidate[g] for g in sample.ids})
    return fixtures, Tolerance.model_validate(tolerance.model_dump())


def _preconditions(sample: TransferSample, base_feature_names: Sequence[str],
                   strata: Mapping[int, str], determinism: Sequence[CapturedFixture],
                   extends: str) -> list[str]:
    reasons = []
    if tuple(sample.feature_names) != tuple(base_feature_names):
        reasons.append(f"the sample's feature schema is not the {extends} lane's")
    if not SAMPLE_ROWS[0] <= len(sample.rows) <= SAMPLE_ROWS[1]:
        reasons.append(f"the sample holds {len(sample.rows)} rows; a transfer sample holds "
                       f"{SAMPLE_ROWS[0]} to {SAMPLE_ROWS[1]}")
    if set(strata) != set(sample.ids):
        reasons.append("the strata must name every sampled row and no other")
    elif {g for g, s in strata.items() if s == ORDINARY_RULE} != set(ordinary_rows(sample.ids)):
        reasons.append("the ordinary stratum must be the rows evenly spaced over the sampled "
                       "ids, chosen before any deviation is read")
    by_gid = {c.generation_id: c for c in determinism}
    if len(by_gid) != len(determinism) or not set(by_gid) <= set(sample.ids) \
            or len(by_gid) < DETERMINISM_ROWS:
        reasons.append(f"the determinism check must cover at least {DETERMINISM_ROWS} distinct "
                       "sampled rows")
    for gid, c in sorted(by_gid.items()):
        if gid in sample.candidate and c.first.tobytes() != sample.candidate[gid].tobytes():
            reasons.append(f"row {gid}: the scored vLLM vector is not the determinism "
                           "check's first capture")
        elif c.repeat.tobytes() != c.first.tobytes() or c.batched.tobytes() != c.first.tobytes():
            reasons.append(f"row {gid}: lane disagrees with itself (repeat or batched "
                           "capture differs)")
    return reasons


def check_transfer(*, key: str, extends: str, base_tolerance: Tolerance,
                   base_feature_names: Sequence[str], checkpoint_sha256: str,
                   calibration_sha256: str, sample: TransferSample,
                   strata: Mapping[int, str], determinism: Sequence[CapturedFixture],
                   max_floor: float | None, rules: TransferRules,
                   diagnostics: Mapping[int, BifurcationReport] | None = None
                   ) -> TransferResult:
    """Score a fine-tune's sample against its base's tolerance and rule set.

    ``strata`` must come from :func:`select_strata`; ``determinism`` holds the vLLM
    captures (alone, repeated, batched) of at least :data:`DETERMINISM_ROWS` rows,
    each first capture the vector scored; ``diagnostics`` holds
    :func:`bifurcation_report` per row where it was measured, reported only. Refuses, with every reason
    found: a schema other than the base's, a sample outside :data:`SAMPLE_ROWS`, strata
    whose ordinary rows are not :func:`ordinary_rows`, a lane that disagrees with
    itself, and on the rows at or under ``max_floor``, by ``rules``: more rows over a
    component's max ratio than allowed, an ordinary median over a component's p90, more
    ordinary rows over its p99 than allowed, or a family whose ordinary median exceeds the
    base's family p90. Rows over a maximum are listed with their multiples, not refused.

    Raises
    ------
    ValueError
        When ``extends`` is not shipped, ``key`` is shipped, or the tolerance is
        another model's or gates another stratum: a caller's error, not a verdict.
    """
    if extends not in LANE_MODELS or key in LANE_MODELS:
        raise ValueError(f"an extension takes a new key ({key!r}) and extends a shipped "
                         f"lane ({extends!r})")
    if base_tolerance.model != extends or base_tolerance.median_gate_stratum != ORDINARY_RULE:
        raise ValueError(f"the tolerance given is not {extends!r}'s, gating the "
                         f"{ORDINARY_RULE!r} stratum")
    if rules.model != extends or rules.tolerance_digest != base_tolerance.digest:
        raise ValueError(f"the rule set given is not {extends!r}'s for this tolerance")
    reports = dict(diagnostics or {})
    reasons = _preconditions(sample, base_feature_names, strata, determinism, extends)
    fragile = tuple(g for g, f in sorted(sample.floors.items())
                    if max_floor is not None and f > max_floor)
    ordinary = ordinary_rows(sample.ids)
    rows: list[RowResult] = []
    report: dict[str, float] = {}
    over_max: dict[str, tuple[int, ...]] = {}
    worst: dict[str, float] = {}
    over_p90: dict[str, int] = {}
    over_p99: dict[str, int] = {}
    family_median: dict[str, float] = {}
    family_over: dict[str, tuple[int, ...]] = {}
    multiples: dict[str, dict[int, float]] = {}
    if not reasons and len(fragile) == len(sample.rows):
        reasons.append(f"every sampled row's path floor exceeds {max_floor}")
    elif not reasons:
        rows, report = score_rows(sample.feature_names, sample.sigma_cal, sample.weights,
                                  sample.floors, sample.reference, sample.candidate,
                                  base_tolerance, exclude_from_report=fragile)
        # The row and family gates are the rule set's, below; the shared gates read only
        # the component medians.
        uncapped = [r.model_copy(update=dict(components_within=True, worst_family_excess=0.0))
                    for r in rows]
        no_tail = base_tolerance.model_copy(update=dict(components=tuple(
            c.model_copy(update=dict(p99_ratio=None)) for c in base_tolerance.components)))
        reasons += tolerance_reasons(uncapped, no_tail, strata, exclude=fragile)
        scored = [r for r in rows if r.generation_id not in fragile]

        for c in base_tolerance.components:
            ratio = {r.generation_id: r.component_ratios[c.name] for r in scored}
            over_max[c.name] = tuple(g for g, x in ratio.items() if x > c.max_ratio)
            worst[c.name] = max(ratio.values()) / c.max_ratio
            ordinary_ratios = [ratio[g] for g in ordinary if g in ratio]
            over_p90[c.name] = sum(x > c.p90_ratio for x in ordinary_ratios)
            over_p99[c.name] = sum(x > (c.p99_ratio or np.inf) for x in ordinary_ratios)
            multiples[c.name] = {g: ratio[g] / c.max_ratio for g in over_max[c.name]}
            if len(over_max[c.name]) > rules.component_max_allowance:
                reasons.append(f"{c.name}: rows {list(over_max[c.name])} exceed the recorded "
                               f"max ratio {c.max_ratio:.3g}; at most "
                               f"{rules.component_max_allowance} may")
            if over_p99[c.name] > rules.p99_allowance:
                reasons.append(f"{c.name}: {over_p99[c.name]} of {len(ordinary_ratios)} "
                               f"ordinary rows exceed the recorded p99 {c.p99_ratio:.3g}; "
                               f"at most {rules.p99_allowance} may")
        values = family_values(sample, base_tolerance)
        kept = [g for g in sample.ids if g not in fragile]
        for family, fmax in sorted(base_tolerance.family_max_abs_sigma.items()):
            family_over[family] = tuple(g for g in kept if values[g][family] > fmax)
            ordinary_values = [values[g][family] for g in ordinary if g in values
                               and g not in fragile]
            family_median[family] = float(np.median(ordinary_values)) if ordinary_values \
                else float("inf")
            p90 = rules.family_p90_abs_sigma[family]
            if family_median[family] > p90:
                reasons.append(f"family {family}: the ordinary rows' median |δ|/σ "
                               f"{family_median[family]:.3g} exceeds the base's p90 {p90:.3g}")
            if family_over[family]:
                multiples[f"family:{family}"] = {g: values[g][family] / fmax
                                                  for g in family_over[family]}
    fixtures = tolerance = None
    if rows:
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
        determinism_rows=len({c.generation_id for c in determinism}), ordinary_rows=ordinary,
        rows_over_max=over_max, worst_max_multiple=worst, ordinary_over_p90=over_p90,
        ordinary_over_p99=over_p99, rules_digest=rules.digest,
        weakly_guarded=dict(rules.weakly_guarded), family_ordinary_median=family_median,
        family_rows_over_max={f: g for f, g in family_over.items() if g},
        over_max_multiples=multiples, diagnostics=reports,
        rows=tuple(rows), family_report=report,
        fixture_digest=fixtures and fixtures.digest,
        tolerance_digest=tolerance and tolerance.digest)
    return TransferResult(receipt=receipt, fixtures=fixtures, tolerance=tolerance)

