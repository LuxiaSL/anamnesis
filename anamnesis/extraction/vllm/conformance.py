"""The install check: is this host a lane, and is it the qualified one?

A lane is self-consistency plus provenance: the same tokens give the same
signature every time, under one declared model, engine, arithmetic and host.
Each model ships fixtures (the vectors its qualified lane produced on a set of
rows) and a tolerance (the spread its qualification measured between that lane
and the numeric anchor). The install check captures the fixture rows on this
host and decides one of three tiers:

* **identical** — every fixture vector is byte-identical. The host runs the
  qualified lane itself and keeps its lane id.
* **own-lane** — the vectors differ, but the host agrees with itself (every
  fixture row byte-identical across two single passes and a batched pass),
  every feature is finite, and per component the median fixture ratio is at or
  below the ratio the qualification recorded as its maximum. The host is a lane
  of its own, with its own id: fully usable, and never row-joined with another
  lane's data.
* **refused** — a checkpoint mismatch, a lane that disagrees with itself across
  repeats or batching, a non-finite feature, or a component whose median ratio
  exceeds that maximum: a host that is not a lane, or is obviously broken.

The per-row ceilings, the ordinary rows' median and p99 readings and the family
maxima are reported in the receipt (:attr:`ConformanceReceipt.readings`) as how
far this host sits from the qualified lane; none of them gates. Whether two
lanes agree is an audit run when a claim needs it
(:mod:`anamnesis.extraction.vllm.transfer`), not a condition of using either.

This module is the decision alone. It takes vectors already captured and
never touches an engine, so every rule here is testable without a GPU. The
tolerance it reads is the shipped one; nothing here fits or chooses a number.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

FIXTURE_CONTRACT = "conformance-fixtures/1"
TOLERANCE_CONTRACT = "conformance-tolerance/2"
RECEIPT_CONTRACT = "conformance-receipt/2"

Tier = Literal["identical", "own-lane", "refused"]


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class FixtureRow(BaseModel):
    """One token sequence and where it came from; its vector lives beside it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    generation_id: int = Field(ge=0)
    population: Literal["native", "null"]
    input_ids: tuple[int, ...] = Field(min_length=2)
    prompt_length: int = Field(gt=0)
    end: int
    floor_b: float = Field(gt=0, description="The row's reference noise floor, in σ_cal "
                                              "units, measured under the lane's arithmetic; "
                                              "a row's ratio is its distance d over it")
    selected_by: str = Field(min_length=1, description="The selection rule that took the row")

    @model_validator(mode="after")
    def _span_partitions_ids(self) -> Self:
        if self.end != len(self.input_ids) or not 0 < self.prompt_length < self.end - 1:
            raise ValueError(f"row {self.generation_id}: span does not partition its ids")
        return self


class FixtureSet(BaseModel):
    """A model's fixtures: identity, the standardizer, rows and their vectors."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    contract: Literal["conformance-fixtures/1"] = FIXTURE_CONTRACT
    model: str = Field(min_length=1)
    lane_id: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    calibration_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    ruler_sha256: str = Field(pattern="^[0-9a-f]{64}$",
                              description="Digest of the standardizer and floors the rows carry")
    authority: dict[str, str] = Field(
        default_factory=dict,
        description="What the vectors were produced under: the digests of the records "
                    "and the capture configuration behind them")
    feature_names: tuple[str, ...] = Field(min_length=1)
    sigma_cal: NDArray[np.float64] = Field(
        description="Per-coordinate scale every deviation is standardized by")
    weights: NDArray[np.float64] = Field(
        description="Per-coordinate weight in the row distance d = sqrt(Σ w·(δ/σ_cal)²)")
    rows: tuple[FixtureRow, ...] = Field(min_length=1)
    vectors: dict[int, NDArray[np.float32]]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        n = len(self.feature_names)
        if self.sigma_cal.shape != (n,) or self.weights.shape != (n,):
            raise ValueError("standardizer must cover every feature name")
        if not (self.sigma_cal > 0).all():
            raise ValueError("σ_cal must be positive everywhere")
        ids = [r.generation_id for r in self.rows]
        if len(set(ids)) != len(ids) or set(ids) != set(self.vectors):
            raise ValueError("every fixture row needs exactly one vector")
        for gid, vector in self.vectors.items():
            if vector.shape != (n,) or vector.dtype != np.float32 or not np.isfinite(vector).all():
                raise ValueError(f"fixture vector {gid} is not a finite float32 [features]")
        return self

    @property
    def digest(self) -> str:
        """Content digest: identity, rows and every vector's bytes."""
        h = hashlib.sha256()
        h.update(json.dumps(self.model_dump(mode="json", exclude={"sigma_cal", "weights",
                                                                  "vectors"}),
                            sort_keys=True).encode())
        h.update(self.sigma_cal.tobytes())
        h.update(self.weights.tobytes())
        for gid in sorted(self.vectors):
            h.update(gid.to_bytes(8, "little"))
            h.update(self.vectors[gid].tobytes())
        return h.hexdigest()

    def save(self, directory: Path) -> Path:
        """Write metadata JSON and one npz of arrays; refuse to overwrite."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        meta = self.model_dump(mode="json", exclude={"sigma_cal", "weights", "vectors"})
        meta["digest"] = self.digest
        (directory / "fixtures.json").write_text(json.dumps(meta, indent=2) + "\n")
        gids = sorted(self.vectors)
        np.savez(directory / "vectors.npz", sigma_cal=self.sigma_cal, weights=self.weights,
                 generation_ids=np.asarray(gids, dtype=np.int64),
                 vectors=np.stack([self.vectors[g] for g in gids]))
        return directory

    @classmethod
    def load(cls, directory: Path) -> FixtureSet:
        directory = Path(directory)
        meta = json.loads((directory / "fixtures.json").read_text())
        recorded = meta.pop("digest")
        with np.load(directory / "vectors.npz", allow_pickle=False) as z:
            vectors = {int(g): z["vectors"][i].copy() for i, g in enumerate(z["generation_ids"])}
            loaded = cls(**meta, sigma_cal=z["sigma_cal"].copy(), weights=z["weights"].copy(),
                         vectors=vectors)
        if loaded.digest != recorded:
            raise ValueError("fixture set content differs from its recorded digest")
        return loaded


class RowComponent(BaseModel):
    """A slice of the vector whose row distance has its own recorded ceiling."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    feature_names: tuple[str, ...] = Field(min_length=1)
    max_ratio: float = Field(gt=0, description="The recorded max of d/floor_b; a host's "
                                               "median ratio over the fixtures stays at or "
                                               "below it")
    p90_ratio: float = Field(gt=0, description="The recorded p90 of d/floor_b; reported "
                                               "against the ordinary stratum's median")
    p99_ratio: float | None = Field(
        default=None, gt=0,
        description="The recorded p99 of d/floor_b; ordinary-stratum rows over it are "
                    "reported")
    source: str = Field(min_length=1, description="A provenance label for where the ratios "
                                                  "were read; not a path this package opens")


class Tolerance(BaseModel):
    """The deviations between the qualified lane and the numeric anchor that the
    qualification measured, per row component and per feature family.

    Contract 2: every path floor behind the ratios is the anchor's full-vs-incremental
    disagreement under the lane's arithmetic (the cuBLAS workspace pin and
    deterministic algorithms), the arithmetic every reference the lane is scored
    against was computed in. Contract 1 floors were measured without it, which let
    cuBLAS pick different GEMM algorithms per prompt shape and inflated some floors;
    it is not read."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract: Literal["conformance-tolerance/2"] = TOLERANCE_CONTRACT
    model: str = Field(min_length=1)
    components: tuple[RowComponent, ...] = Field(min_length=1)
    families: dict[str, str] = Field(description="Continuous feature name → its family")
    family_max_abs_sigma: dict[str, float] = Field(
        description="Per family, the largest continuous |δ|/σ_cal the comparison measured")
    discrete: tuple[str, ...] = Field(
        default=(), description="Threshold, count or rank coordinates: crossings reported")
    median_gate_stratum: str | None = Field(
        default=None,
        description="Fixture rows selected by this rule sample the population without "
                    "selection bias; the receipt reports the host's median ratio over them "
                    "against each component's recorded p90")
    excluded_rows: tuple[int, ...] = Field(
        default=(), description="Rows whose reference floor is too fragile to divide by, "
                                "left out when the ceilings and family maxima were computed; "
                                "the install check still captures and scores them")
    sources: dict[str, str] = Field(description="Provenance labels of every record read, "
                                                "each with its sha256; not paths this package "
                                                "opens")

    @model_validator(mode="after")
    def _families_resolve(self) -> Self:
        missing = set(self.families.values()) - set(self.family_max_abs_sigma)
        if missing:
            raise ValueError(f"families without a recorded maximum: {sorted(missing)}")
        if set(self.families) & set(self.discrete):
            raise ValueError("a coordinate is continuous or discrete, not both")
        return self

    @property
    def digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


class HostFingerprint(BaseModel):
    """What makes two installs the same host for the install check."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gpu_name: str
    gpu_uuid: str
    driver: str
    cuda_runtime: str
    torch: str
    vllm: str
    anamnesis: str
    checkpoint_sha256: str
    engine_settings_sha256: str = Field(
        description="The engine settings and the lane source the captures ran under; a "
                    "receipt never outlives a change to either")
    fixture_digest: str
    tolerance_digest: str

    @property
    def digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


class CapturedFixture(BaseModel):
    """One fixture as this host captured it: twice alone, once inside a batch."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    generation_id: int
    first: NDArray[np.float32]
    repeat: NDArray[np.float32]
    batched: NDArray[np.float32]


class RowResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    generation_id: int
    byte_identical: bool
    component_ratios: dict[str, float]
    components_within: bool = Field(description="Every component's ratio at or below its "
                                                "recorded max")
    worst_family: str | None
    worst_family_excess: float = Field(description="max |δ|/σ over the family max")
    discrete_crossings: int


class ConformanceReceipt(BaseModel):
    """The check's tier, cached against the host fingerprint.

    ``reasons`` say why a host is refused; ``readings`` report, for a host whose
    vectors differ, where it sits against every recorded ceiling, and gate nothing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract: Literal["conformance-receipt/2"] = RECEIPT_CONTRACT
    tier: Tier
    lane_id: str | None
    qualified_lane_id: str
    reasons: tuple[str, ...]
    fingerprint: HostFingerprint
    rows: tuple[RowResult, ...]
    component_medians: dict[str, float] = Field(
        default_factory=dict, description="Per component, the median ratio over every "
                                          "fixture row")
    readings: tuple[str, ...] = Field(
        default=(), description="Rows and families past a recorded ceiling, and the "
                                "ordinary stratum's median and tail readings; reported, "
                                "not gated")
    family_report: dict[str, float] = Field(
        description="Per family, the worst |δ|/σ seen on this host over the family max")

    @property
    def digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


def own_lane_id(qualified_lane_id: str, fingerprint: HostFingerprint) -> str:
    """A host's own lane: the fixtures' lane id plus the host's."""
    return f"{qualified_lane_id}+host-{fingerprint.digest[:16]}"


def check_coordinates(feature_names: Sequence[str], tolerance: Tolerance) -> dict[str, int]:
    """Each feature name's index, after refusing a tolerance that names others.

    Raises
    ------
    ValueError
        When a component, family or discrete coordinate of ``tolerance`` is not
        one of ``feature_names``.
    """
    index = {name: i for i, name in enumerate(feature_names)}
    unknown = [n for c in tolerance.components for n in c.feature_names if n not in index]
    unknown += [n for n in (*tolerance.families, *tolerance.discrete) if n not in index]
    if unknown:
        raise ValueError(f"tolerance names coordinates the fixtures lack: {unknown[:5]}")
    return index


def score_rows(
    feature_names: Sequence[str],
    sigma_cal: NDArray[np.float64],
    weights: NDArray[np.float64],
    floors: Mapping[int, float],
    reference: Mapping[int, NDArray[np.float32]],
    candidate: Mapping[int, NDArray[np.float32]],
    tolerance: Tolerance,
    *,
    exclude_from_report: Collection[int] = (),
) -> tuple[list[RowResult], dict[str, float]]:
    """Score each row's deviation of ``candidate`` from ``reference`` under ``tolerance``.

    Per row, in generation-id order: whether the vectors are byte-identical, each
    component's distance d = sqrt(Σ w·(δ/σ_cal)²) over the row's floor, whether every
    component stays within its ``max_ratio``, the family whose largest continuous
    |δ|/σ_cal is furthest over its recorded maximum, and the discrete coordinates that
    changed. Also returns, per family, the worst excess over the rows not in
    ``exclude_from_report``.

    ``reference`` and ``candidate`` must hold the same generation ids, each a float32
    vector over ``feature_names``; ``floors`` holds a positive floor for each.
    """
    index = check_coordinates(feature_names, tolerance)
    component_ix = {c.name: np.asarray([index[n] for n in c.feature_names])
                    for c in tolerance.components}
    members: dict[str, list[int]] = {}
    for name, family in tolerance.families.items():
        members.setdefault(family, []).append(index[name])
    family_ix = {f: np.asarray(ix, dtype=np.int64) for f, ix in members.items()}
    discrete_ix = np.asarray([index[n] for n in tolerance.discrete], dtype=np.int64)
    excluded = set(exclude_from_report)

    rows: list[RowResult] = []
    family_report: dict[str, float] = {f: 0.0 for f in family_ix}
    for gid in sorted(reference):
        fixture = reference[gid]
        vector = candidate[gid]
        identical = vector.tobytes() == fixture.tobytes()
        z = (vector.astype(np.float64) - fixture.astype(np.float64)) / sigma_cal
        ratios = {}
        within = True
        for component in tolerance.components:
            ix = component_ix[component.name]
            distance = float(np.sqrt(np.sum(weights[ix] * z[ix] ** 2)))
            ratios[component.name] = distance / floors[gid]
            within &= ratios[component.name] <= component.max_ratio
        worst_family, worst_excess = None, 0.0
        for family, ix in family_ix.items():
            excess = float(np.abs(z[ix]).max()) / tolerance.family_max_abs_sigma[family] \
                if tolerance.family_max_abs_sigma[family] > 0 \
                else (0.0 if not np.abs(z[ix]).any() else float("inf"))
            if gid not in excluded:
                family_report[family] = max(family_report[family], excess)
            if excess > worst_excess:
                worst_family, worst_excess = family, excess
        crossings = int(np.count_nonzero(vector[discrete_ix] != fixture[discrete_ix])) \
            if discrete_ix.size else 0
        rows.append(RowResult(generation_id=gid, byte_identical=identical,
                              component_ratios=ratios, components_within=within,
                              worst_family=worst_family, worst_family_excess=worst_excess,
                              discrete_crossings=crossings))
    return rows, family_report


def tolerance_reasons(
    rows: Sequence[RowResult],
    tolerance: Tolerance,
    selected_by: Mapping[int, str],
    *,
    exclude: Collection[int] = (),
) -> list[str]:
    """Every way scored ``rows`` sit past a ceiling of ``tolerance``; empty when none does.

    Each row not in ``exclude`` past a component's ceiling or a family's maximum is
    named. When the tolerance names a median-gate stratum, the rows ``selected_by``
    that stratum (``exclude`` left out) are named where a component's median ratio
    exceeds its p90 and, where a p99 is recorded, where a ratio exceeds it. The
    install check reports these; the transfer audit reads the median.
    """
    skip = set(exclude)
    kept = [r for r in rows if r.generation_id not in skip]
    reasons: list[str] = []
    for r in kept:
        if not r.components_within:
            reasons.append(f"row {r.generation_id}: row distance outside the recorded ceiling "
                           f"{r.component_ratios}")
        if r.worst_family_excess > 1:
            reasons.append(f"row {r.generation_id}: family {r.worst_family} at "
                           f"{r.worst_family_excess:.3g}× its recorded maximum")
    if tolerance.median_gate_stratum is not None:
        ordinary = [r for r in kept if selected_by[r.generation_id]
                    == tolerance.median_gate_stratum]
        if not ordinary:
            reasons.append("no fixture row carries the median gate's stratum")
        for component in tolerance.components:
            median = float(np.median([r.component_ratios[component.name] for r in ordinary])) \
                if ordinary else float("inf")
            if median > component.p90_ratio:
                reasons.append(f"{component.name}: median ratio {median:.3g} over the "
                               f"ordinary rows exceeds the recorded p90 "
                               f"{component.p90_ratio:.3g}")
            if component.p99_ratio is None:
                continue
            for r in ordinary:
                if r.component_ratios[component.name] > component.p99_ratio:
                    reasons.append(f"row {r.generation_id}: {component.name} ratio "
                                   f"{r.component_ratios[component.name]:.3g} in the "
                                   f"ordinary stratum exceeds the recorded p99 "
                                   f"{component.p99_ratio:.3g}")
    return reasons


def decide(
    fixtures: FixtureSet,
    tolerance: Tolerance,
    fingerprint: HostFingerprint,
    captured: Sequence[CapturedFixture],
) -> ConformanceReceipt:
    """Decide this host's tier from vectors it captured.

    The checkpoint must be the fixtures', the lane must agree with itself
    (repeat and batched captures byte-identical to the first), and every feature
    must be finite. Then byte-identical vectors are ``identical``; otherwise a
    component whose median ratio over the fixture rows exceeds its recorded max
    refuses, and the host is ``own-lane``. Every refusal names why; every other
    ceiling is reported in ``readings``.
    """
    if tolerance.model != fixtures.model:
        raise ValueError("tolerance and fixtures belong to different models")
    if fingerprint.fixture_digest != fixtures.digest \
            or fingerprint.tolerance_digest != tolerance.digest:
        raise ValueError("fingerprint was taken against other fixtures or tolerance")
    names = fixtures.feature_names
    check_coordinates(names, tolerance)

    reasons: list[str] = []
    if fingerprint.checkpoint_sha256 != fixtures.checkpoint_sha256:
        reasons.append("checkpoint digest differs from the fixtures'")
    by_gid = {c.generation_id: c for c in captured}
    if set(by_gid) != set(fixtures.vectors) or len(by_gid) != len(captured):
        reasons.append("captures must be exactly the fixture rows, each once")
    for c in captured:
        if c.first.shape != (len(names),) or c.first.dtype != np.float32:
            reasons.append(f"row {c.generation_id}: capture is not float32 [features]")
        elif c.repeat.tobytes() != c.first.tobytes() or c.batched.tobytes() != c.first.tobytes():
            reasons.append(f"row {c.generation_id}: lane disagrees with itself "
                           "(repeat or batched capture differs)")
        elif not np.isfinite(c.first).all():
            reasons.append(f"row {c.generation_id}: "
                           f"{int(np.count_nonzero(~np.isfinite(c.first)))} features are "
                           "not finite")
    qualified = fixtures.lane_id
    if reasons:
        return ConformanceReceipt(tier="refused", lane_id=None, qualified_lane_id=qualified,
                                  reasons=tuple(reasons), fingerprint=fingerprint,
                                  rows=(), family_report={})

    rows, family_report = score_rows(
        names, fixtures.sigma_cal, fixtures.weights,
        {r.generation_id: r.floor_b for r in fixtures.rows}, fixtures.vectors,
        {gid: by_gid[gid].first for gid in fixtures.vectors}, tolerance)
    medians = {c.name: float(np.median([r.component_ratios[c.name] for r in rows]))
               for c in tolerance.components}
    if all(r.byte_identical for r in rows):
        return ConformanceReceipt(tier="identical", lane_id=qualified,
                                  qualified_lane_id=qualified, reasons=(),
                                  fingerprint=fingerprint, rows=tuple(rows),
                                  component_medians=medians, family_report=family_report)
    reasons = [f"{c.name}: median ratio {medians[c.name]:.3g} over the fixture rows exceeds "
               f"the recorded max {c.max_ratio:.3g}"
               for c in tolerance.components if medians[c.name] > c.max_ratio]
    readings = tolerance_reasons(rows, tolerance,
                                 {r.generation_id: r.selected_by for r in fixtures.rows})
    return ConformanceReceipt(
        tier="refused" if reasons else "own-lane",
        lane_id=None if reasons else own_lane_id(qualified, fingerprint),
        qualified_lane_id=qualified, reasons=tuple(reasons), fingerprint=fingerprint,
        rows=tuple(rows), component_medians=medians, readings=tuple(readings),
        family_report=family_report)


class ReceiptCache:
    """Receipts on disk, one per host fingerprint, reused only on exact equality and
    only under this receipt contract; a receipt of another contract is decided again."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def _path(self, fingerprint: HostFingerprint) -> Path:
        return self.directory / f"conformance-{fingerprint.digest}.json"

    def load(self, fingerprint: HostFingerprint) -> ConformanceReceipt | None:
        path = self._path(fingerprint)
        if not path.is_file():
            return None
        try:
            if json.loads(path.read_text()).get("contract") != RECEIPT_CONTRACT:
                return None
        except (json.JSONDecodeError, AttributeError):
            return None
        receipt = ConformanceReceipt.model_validate_json(path.read_text())
        if receipt.fingerprint != fingerprint:
            return None
        return receipt

    def store(self, receipt: ConformanceReceipt) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(receipt.fingerprint)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(receipt.model_dump_json(indent=2) + "\n")
        tmp.replace(path)
        return path
