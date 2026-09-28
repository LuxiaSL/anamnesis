"""Install conformance: does this host compute what the lane's fixtures record?

Each model's fixtures and tolerance come from one comparison of the vLLM lane
with the numeric anchor (the hook path), per engine release: the fixture vectors
are what the lane produced on a set of rows, and the tolerance is the largest
deviation from them that comparison found harmless. Install conformance asks a
smaller question that needs neither the anchor nor a second model: does *this*
install reproduce those vectors, and if not exactly, does it stay inside that
tolerance?

The decision has three outcomes:

* **identical** — every fixture vector is byte-identical. The host runs the
  fixtures' lane itself and keeps its lane id.
* **conformant** — the vectors differ, but every row stays within its
  component's recorded row-distance ceiling, every continuous coordinate within
  its family's recorded maximum, and the ordinary (unselected) fixture rows'
  median ratio within the recorded p90 and each of them within the recorded
  p99. The median and tail gates refuse a host that sits near the ceiling on
  many rows even though no single row crosses it. The host is its own lane, with
  its own id: fully usable, and never row-joined with another lane's data.
* **refused** — a checkpoint mismatch, a lane that disagrees with itself
  across repeats or batching, or a deviation outside the tolerance.

This module is the decision alone. It takes vectors already captured and
never touches an engine, so every rule here is testable without a GPU. The
tolerance it applies is the shipped one; nothing here fits or chooses a number.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

FIXTURE_CONTRACT = "conformance-fixtures/1"
TOLERANCE_CONTRACT = "conformance-tolerance/1"
RECEIPT_CONTRACT = "conformance-receipt/1"

Tier = Literal["identical", "conformant", "refused"]


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
                                              "units; a row's ratio is its distance d over it")
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
    max_ratio: float = Field(gt=0, description="Gating: the recorded max of d/floor_b")
    p90_ratio: float = Field(gt=0, description="Gates the ordinary stratum's median ratio")
    p99_ratio: float | None = Field(
        default=None, gt=0,
        description="Tail gate: every ordinary-stratum row's ratio stays at or below it")
    source: str = Field(min_length=1, description="A provenance label for where the ratios "
                                                  "were read; not a path this package opens")


class Tolerance(BaseModel):
    """The largest deviations from the fixtures the anchor comparison found harmless,
    per row component and per feature family."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract: Literal["conformance-tolerance/1"] = TOLERANCE_CONTRACT
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
                    "selection bias; the host's median ratio over them, per component, must "
                    "stay at or below that component's recorded p90")
    excluded_rows: tuple[int, ...] = Field(
        default=(), description="Rows whose reference floor is too fragile to divide by, "
                                "left out when the ceilings and family maxima were computed; "
                                "decide still gates every fixture row, these included")
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
    """What makes two installs the same host for conformance purposes."""

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
    components_within: bool
    worst_family: str | None
    worst_family_excess: float = Field(description="max |δ|/σ over the family max; ≤ 1 passes")
    discrete_crossings: int


class ConformanceReceipt(BaseModel):
    """The check's verdict, cached against the host fingerprint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract: Literal["conformance-receipt/1"] = RECEIPT_CONTRACT
    tier: Tier
    lane_id: str | None
    qualified_lane_id: str
    reasons: tuple[str, ...]
    fingerprint: HostFingerprint
    rows: tuple[RowResult, ...]
    family_report: dict[str, float] = Field(
        description="Per family, the worst |δ|/σ seen on this host over the family max")

    @property
    def digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


def conformant_lane_id(qualified_lane_id: str, fingerprint: HostFingerprint) -> str:
    """A within-tolerance host's own lane: the fixtures' lane id plus the host's."""
    return f"{qualified_lane_id}+host-{fingerprint.digest[:16]}"


def _standardized(delta: NDArray[np.float64], fixtures: FixtureSet) -> NDArray[np.float64]:
    return delta / fixtures.sigma_cal


def decide(
    fixtures: FixtureSet,
    tolerance: Tolerance,
    fingerprint: HostFingerprint,
    captured: Sequence[CapturedFixture],
) -> ConformanceReceipt:
    """Apply the conformance procedure to vectors this host captured.

    The checkpoint must be the fixtures', the lane must agree with itself
    (repeat and batched captures byte-identical to the first), and then the
    deviation from the fixtures decides the tier. Every refusal names why.
    """
    if tolerance.model != fixtures.model:
        raise ValueError("tolerance and fixtures belong to different models")
    if fingerprint.fixture_digest != fixtures.digest \
            or fingerprint.tolerance_digest != tolerance.digest:
        raise ValueError("fingerprint was taken against other fixtures or tolerance")
    names = fixtures.feature_names
    index = {name: i for i, name in enumerate(names)}
    unknown = [n for c in tolerance.components for n in c.feature_names if n not in index]
    unknown += [n for n in (*tolerance.families, *tolerance.discrete) if n not in index]
    if unknown:
        raise ValueError(f"tolerance names coordinates the fixtures lack: {unknown[:5]}")

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
    qualified = fixtures.lane_id
    if reasons:
        return ConformanceReceipt(tier="refused", lane_id=None, qualified_lane_id=qualified,
                                  reasons=tuple(reasons), fingerprint=fingerprint,
                                  rows=(), family_report={})

    component_ix = {c.name: np.asarray([index[n] for n in c.feature_names])
                    for c in tolerance.components}
    members: dict[str, list[int]] = {}
    for name, family in tolerance.families.items():
        members.setdefault(family, []).append(index[name])
    family_ix = {f: np.asarray(ix, dtype=np.int64) for f, ix in members.items()}
    discrete_ix = np.asarray([index[n] for n in tolerance.discrete], dtype=np.int64)

    rows: list[RowResult] = []
    family_report: dict[str, float] = {f: 0.0 for f in family_ix}
    floors = {r.generation_id: r.floor_b for r in fixtures.rows}
    selected_by = {r.generation_id: r.selected_by for r in fixtures.rows}
    for gid in sorted(fixtures.vectors):
        fixture = fixtures.vectors[gid]
        vector = by_gid[gid].first
        identical = vector.tobytes() == fixture.tobytes()
        z = _standardized(vector.astype(np.float64) - fixture.astype(np.float64), fixtures)
        ratios = {}
        within = True
        for component in tolerance.components:
            ix = component_ix[component.name]
            distance = float(np.sqrt(np.sum(fixtures.weights[ix] * z[ix] ** 2)))
            ratios[component.name] = distance / floors[gid]
            within &= ratios[component.name] <= component.max_ratio
        worst_family, worst_excess = None, 0.0
        for family, ix in family_ix.items():
            excess = float(np.abs(z[ix]).max()) / tolerance.family_max_abs_sigma[family] \
                if tolerance.family_max_abs_sigma[family] > 0 \
                else (0.0 if not np.abs(z[ix]).any() else float("inf"))
            family_report[family] = max(family_report[family], excess)
            if excess > worst_excess:
                worst_family, worst_excess = family, excess
        crossings = int(np.count_nonzero(vector[discrete_ix] != fixture[discrete_ix])) \
            if discrete_ix.size else 0
        rows.append(RowResult(generation_id=gid, byte_identical=identical,
                              component_ratios=ratios, components_within=within,
                              worst_family=worst_family, worst_family_excess=worst_excess,
                              discrete_crossings=crossings))

    if all(r.byte_identical for r in rows):
        tier: Tier = "identical"
        lane_id: str | None = qualified
    else:
        for r in rows:
            if not r.components_within:
                reasons.append(f"row {r.generation_id}: row distance outside the recorded ceiling "
                               f"{r.component_ratios}")
            if r.worst_family_excess > 1:
                reasons.append(f"row {r.generation_id}: family {r.worst_family} at "
                               f"{r.worst_family_excess:.3g}× its recorded maximum")
        if tolerance.median_gate_stratum is not None:
            ordinary = [r for r in rows if selected_by[r.generation_id]
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
        tier = "refused" if reasons else "conformant"
        lane_id = None if reasons else conformant_lane_id(qualified, fingerprint)
    return ConformanceReceipt(tier=tier, lane_id=lane_id, qualified_lane_id=qualified,
                              reasons=tuple(reasons), fingerprint=fingerprint,
                              rows=tuple(rows), family_report=family_report)


class ReceiptCache:
    """Receipts on disk, one per host fingerprint, reused only on exact equality."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def _path(self, fingerprint: HostFingerprint) -> Path:
        return self.directory / f"conformance-{fingerprint.digest}.json"

    def load(self, fingerprint: HostFingerprint) -> ConformanceReceipt | None:
        path = self._path(fingerprint)
        if not path.is_file():
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
