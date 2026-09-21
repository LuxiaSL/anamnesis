"""Fail-closed, GPU-independent evidence checks for extraction lane equivalence.

Thresholds are ruled constants, not caller tuning knobs. The caller supplies
immutable cohort/calibration/baseline receipts. This module never fits a ruler,
chooses a condition comparator, or certifies absent coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]
RowKey = tuple[
    str, str, str, int, str
]  # bank namespace, document, cell, replicate, span
FEATURE_SIGMA_FRACTION = 0.1
CONDITION_FRACTION = 0.1
SE_FRACTION = 0.25
REQUIRED_REGIMES = frozenset(
    (
        "longest-prefix",
        "shortest-span",
        "near-floor-B-E",
        "near-floor-F-E",
        "null-forks",
        "E",
        "off-target",
    )
)


class FidelityError(ValueError):
    """Missing, inconsistent, or inadmissible evidence (not a numeric pass)."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise FidelityError(message)


def _digest(value: str, name: str) -> None:
    _require(
        len(value) == 64 and all(c in "0123456789abcdef" for c in value),
        f"{name}: expected lowercase SHA256",
    )


@dataclass(frozen=True)
class ReplayBatch:
    features: F32
    keys: tuple[RowKey, ...]
    feature_names: tuple[str, ...]
    lane_ids: tuple[str, ...]
    input_sha256: tuple[str, ...]  # tokens, positions, model, cache construction
    replay_ids: tuple[str, ...]  # independently constructed full replay invocations
    stack_sha256: str
    calibration_sha256: str
    schema_sha256: str
    channel: str = "probe-free"
    fresh_cache: bool = True

    def validate(self) -> None:
        n, d = len(self.keys), len(self.feature_names)
        _require(n > 0 and d > 0, "empty replay evidence")
        _require(self.features.dtype == np.float32, "features must be float32")
        _require(self.features.shape == (n, d), "feature array shape mismatch")
        _require(bool(np.isfinite(self.features).all()), "nonfinite features")
        _require(len(set(self.keys)) == n, "duplicate row key")
        _require(len(set(self.feature_names)) == d, "duplicate feature name")
        _require(
            all(
                len(x) == n for x in (self.lane_ids, self.input_sha256, self.replay_ids)
            ),
            "per-row receipt length mismatch",
        )
        _require(
            len(set(self.lane_ids)) == 1 and bool(self.lane_ids[0]),
            "G4: mixed or absent lane IDs",
        )
        _require(
            len(set(self.replay_ids)) == n and all(self.replay_ids),
            "reused/empty replay ID",
        )
        _require(self.fresh_cache, "G0: replay did not construct fresh caches")
        _require(
            self.channel == "probe-free", "adapter/excess channel is not certified"
        )
        for name in ("stack_sha256", "calibration_sha256", "schema_sha256"):
            _digest(getattr(self, name), name)
        for digest in self.input_sha256:
            _digest(digest, "input_sha256")


@dataclass(frozen=True)
class Ruler:
    sigma: F64
    weights: F64
    standardizer_sha256: str
    feature_names: tuple[str, ...]
    floor_b: Mapping[RowKey, float]  # measured floor or explicitly typed lower bound
    condition_distance: Mapping[RowKey, float | None]
    condition_pair: Mapping[RowKey, tuple[RowKey, RowKey] | None]
    prefix_lengths: Mapping[RowKey, int]
    reference_keys: tuple[RowKey, ...]
    calibration_sha256: str
    schema_sha256: str
    reference_manifest_sha256: str
    null_contrast_reasons: Mapping[RowKey, str] = field(default_factory=dict)
    floor_kinds: Mapping[RowKey, str] = field(default_factory=dict)
    floor_proof_sha256: Mapping[RowKey, str] = field(default_factory=dict)
    identical_input_zero_pairs: frozenset[RowKey] = frozenset()
    condition_manifest_sha256: str | None = None

    def validate(self) -> None:
        keys = set(self.reference_keys)
        _require(self.identical_input_zero_pairs <= keys, "unknown zero-pair row")
        if self.identical_input_zero_pairs:
            _digest(self.condition_manifest_sha256 or "", "condition manifest")
        _require(
            bool(keys) and len(keys) == len(self.reference_keys),
            "empty/duplicate reference keys",
        )
        d = len(self.feature_names)
        _require(
            self.sigma.shape == self.weights.shape == (d,), "ruler dimension mismatch"
        )
        _require(
            bool(np.isfinite(self.sigma).all() and (self.sigma > 0).all()),
            "invalid sigma",
        )
        _require(
            bool(np.isfinite(self.weights).all() and (self.weights > 0).all()),
            "invalid weights",
        )
        _require(
            bool(np.all(self.weights == self.weights[0])),
            "nonuniform weights not ratified",
        )
        for name in (
            "standardizer_sha256",
            "calibration_sha256",
            "schema_sha256",
            "reference_manifest_sha256",
        ):
            _digest(getattr(self, name), name)
        for name in (
            "floor_b",
            "condition_distance",
            "condition_pair",
            "prefix_lengths",
        ):
            _require(
                set(getattr(self, name)) == keys,
                f"{name}: missing/extra reference rows",
            )
        for name in ("floor_b", "condition_distance"):
            if name == "floor_b":
                _require(
                    all(v is not None for v in self.floor_b.values()),
                    "missing path bound",
                )
            values = np.asarray(
                [v for v in getattr(self, name).values() if v is not None],
                dtype=np.float64,
            )
            _require(
                bool(np.isfinite(values).all() and (values >= 0).all()),
                f"invalid {name}",
            )
        _require(
            all(v >= 1 for v in self.prefix_lengths.values()), "invalid prefix lengths"
        )
        for key, pair in self.condition_pair.items():
            if pair is None:
                _require(
                    self.condition_distance[key] is None
                    and bool(self.null_contrast_reasons.get(key)),
                    "null-by-rule requires an explicit reason and null distance",
                )
                continue
            _require(
                self.condition_distance[key] is not None,
                "paired contrast has null distance",
            )
            _require(
                len(pair) == 2 and all(k in keys for k in pair),
                f"unknown condition pair: {key}",
            )
        _require(
            set(self.null_contrast_reasons)
            == {k for k, p in self.condition_pair.items() if p is None},
            "null-by-rule reasons do not match null contrasts",
        )
        if self.floor_kinds:
            _require(set(self.floor_kinds) == keys, "floor kinds missing/extra rows")
        for key in keys:
            kind = self.floor_kinds.get(key, "measured")
            _require(kind in ("measured", "lower_bound"), "unknown path-bound kind")
            if kind == "lower_bound":
                _digest(self.floor_proof_sha256.get(key, ""), "lower-bound proof")


def _aligned(batch: ReplayBatch, keys: tuple[RowKey, ...]) -> F32:
    _require(set(batch.keys) == set(keys), "missing/extra replay rows")
    lookup = {key: i for i, key in enumerate(batch.keys)}
    return batch.features[[lookup[k] for k in keys]]


def verify_vectors(
    cpu: ReplayBatch, gpu: ReplayBatch, repeat: ReplayBatch, ruler: Ruler
) -> dict:
    """G0/G1/G4. Full-reference rows required, never an intersection join.

    A G0 failure prevents G1 evaluation. No default condition comparator exists:
    in particular anchor A is not silently assigned an invented nonzero delta.
    """
    for batch in (cpu, gpu, repeat):
        batch.validate()
    ruler.validate()
    for batch in (cpu, gpu, repeat):
        _require(batch.feature_names == ruler.feature_names, "feature order mismatch")
        _require(
            batch.calibration_sha256 == ruler.calibration_sha256, "calibration mismatch"
        )
        _require(batch.schema_sha256 == ruler.schema_sha256, "schema mismatch")
    _require(gpu.stack_sha256 == repeat.stack_sha256, "G0: repeat stack differs")
    _require(gpu.lane_ids[0] == repeat.lane_ids[0], "G0: repeat lane differs")
    _require(
        not (set(gpu.replay_ids) & set(repeat.replay_ids)), "G0: same replay reused"
    )
    keys = ruler.reference_keys
    ref, candidate, again = [_aligned(b, keys) for b in (cpu, gpu, repeat)]
    positions = {k: i for i, k in enumerate(keys)}
    for key, pair in ruler.condition_pair.items():
        if pair is None:
            continue
        left, right = pair
        actual = float(
            np.linalg.norm(
                (
                    ref[positions[left]].astype(np.float64)
                    - ref[positions[right]].astype(np.float64)
                )
                * ruler.weights
                / ruler.sigma
            )
        )
        _require(
            np.isclose(actual, ruler.condition_distance[key], rtol=1e-12, atol=1e-12),
            f"condition distance does not reproduce declared pair: {key}",
        )
    inputs = [
        {k: v for k, v in zip(b.keys, b.input_sha256, strict=True)}
        for b in (cpu, gpu, repeat)
    ]
    _require(inputs[0] == inputs[1] == inputs[2], "input/cache/model identity mismatch")
    exact = np.all(candidate.view(np.uint32) == again.view(np.uint32), axis=1)
    result: dict = {
        "G0": bool(exact.all()),
        "G4": True,
        "G1": None,
        "repeat_failures": [
            list(k) for k, ok in zip(keys, exact, strict=True) if not ok
        ],
    }
    if not result["G0"]:
        return result
    for key in ruler.identical_input_zero_pairs:
        pair = ruler.condition_pair[key]
        _require(pair is not None, "zero-pair proof requires endpoints")
        left, right = pair
        _require(
            ruler.condition_distance[key] == 0.0, "nonzero contrast cannot be exempted"
        )
        _require(inputs[0][left] == inputs[0][right], "zero-pair inputs differ")
        for values in (ref, candidate, again):
            _require(
                values[positions[left]].tobytes() == values[positions[right]].tobytes(),
                "zero-pair cancellation is not byte-exact",
            )
    delta = candidate.astype(np.float64) - ref.astype(np.float64)
    distances = np.linalg.norm(delta * ruler.weights / ruler.sigma, axis=1)
    bounds = np.asarray(
        [
            ruler.floor_b[k]
            if ruler.condition_distance[k] is None
            or k in ruler.identical_input_zero_pairs
            else min(ruler.floor_b[k], CONDITION_FRACTION * ruler.condition_distance[k])
            for k in keys
        ],
        dtype=np.float64,
    )
    feature_error = np.abs(delta) / ruler.sigma
    per_feature_pass = np.all(feature_error <= FEATURE_SIGMA_FRACTION, axis=1)
    passed = (distances <= bounds) & per_feature_pass
    condition_ok = np.asarray(
        [
            ruler.condition_distance[k] is None
            or k in ruler.identical_input_zero_pairs
            or distances[i] <= CONDITION_FRACTION * ruler.condition_distance[k]
            for i, k in enumerate(keys)
        ]
    )
    lower = np.asarray(
        [ruler.floor_kinds.get(k, "measured") == "lower_bound" for k in keys]
    )
    path_ok = np.asarray([distances[i] <= ruler.floor_b[k] for i, k in enumerate(keys)])
    failed = (~per_feature_pass) | (~condition_ok) | ((~lower) & (~path_ok))
    pending = lower & (~path_ok) & (~failed)

    def verdict(mask):
        return False if failed[mask].any() else None if pending[mask].any() else True

    prefix = np.asarray([ruler.prefix_lengths[k] for k in keys])
    # Quantile bins preserve equal-length prefixes as one regime. Empty bins
    # are reported, never counted as a tested tercile.
    cuts = np.quantile(prefix, [1 / 3, 2 / 3])
    bins = np.searchsorted(cuts, prefix, side="right")
    groups = []
    for group in range(3):
        mask = bins == group
        groups.append(
            dict(
                tercile=group,
                n=int(mask.sum()),
                passed=verdict(mask) if mask.any() else None,
                max_distance=float(distances[mask].max()) if mask.any() else None,
                max_excess=float((distances - bounds)[mask].max())
                if mask.any()
                else None,
            )
        )
    result.update(
        G1=verdict(np.ones(len(keys), dtype=bool)),
        max_feature_sigma_error=float(feature_error.max()),
        prefix_cutpoints=cuts.tolist(),
        terciles=groups,
        rows=[
            dict(
                key=list(k),
                floor_c=float(distances[i]),
                bound=float(bounds[i]),
                floor_b=float(ruler.floor_b[k])
                if ruler.floor_kinds.get(k, "measured") == "measured"
                else None,
                floor_b_lower_bound=float(ruler.floor_b[k])
                if ruler.floor_kinds.get(k) == "lower_bound"
                else None,
                path_bound_kind=ruler.floor_kinds.get(k, "measured"),
                floor_proof_sha256=ruler.floor_proof_sha256.get(k),
                condition_distance=ruler.condition_distance[k],
                condition_pair=[list(x) for x in ruler.condition_pair[k]]
                if ruler.condition_pair[k] is not None
                else None,
                exact_zero_condition_term_excluded=k
                in ruler.identical_input_zero_pairs,
                contrast_status="null-by-rule"
                if ruler.condition_pair[k] is None
                else "paired",
                contrast_reason=ruler.null_contrast_reasons.get(k),
                feature_sigma_max=float(feature_error[i].max()),
                requires_full_floor=bool(pending[i]),
                passed=False if failed[i] else None if pending[i] else bool(passed[i]),
            )
            for i, k in enumerate(keys)
        ],
    )
    return result


@dataclass(frozen=True)
class DownstreamAnchor:
    point: float
    se: float
    documents: tuple[str, ...]
    clusters: tuple[str, ...]
    artifact_sha256: str
    statistic_source_sha256: str


def verify_downstream(
    cpu_deltas: F64,
    gpu_deltas: F64,
    documents: Sequence[str],
    clusters: Sequence[str],
    anchor: DownstreamAnchor,
    statistic: Callable,
    statistic_source_sha256: str,
    *,
    flag_se_sampling_scale: bool = False,
) -> dict:
    """G2 via the pinned owner's statistic, with exact matched-n/cohort checks.

    `statistic` is genpair.jackknife_coherence_ratio, not a refitted substitute.
    CPU reproduction is a prerequisite; a stale headline cannot be used as a ruler.
    """
    _digest(anchor.artifact_sha256, "anchor artifact")
    _digest(anchor.statistic_source_sha256, "statistic source")
    _require(
        statistic_source_sha256 == anchor.statistic_source_sha256,
        "statistic source mismatch",
    )
    _require(
        tuple(documents) == anchor.documents and tuple(clusters) == anchor.clusters,
        "G2: document/cluster cohort mismatch (matched-n required)",
    )
    _require(len(set(documents)) == len(documents), "G2: duplicate document")
    _require(
        cpu_deltas.shape == gpu_deltas.shape and cpu_deltas.ndim == 2,
        "G2: delta shape mismatch",
    )
    _require(cpu_deltas.shape[0] == len(documents), "G2: member-count mismatch")
    _require(
        bool(np.isfinite(cpu_deltas).all() and np.isfinite(gpu_deltas).all()),
        "G2: nonfinite deltas",
    )
    _require(
        np.isfinite(anchor.point) and np.isfinite(anchor.se) and anchor.se > 0,
        "G2: degenerate anchor",
    )
    reference = statistic(cpu_deltas, clusters)
    candidate = statistic(gpu_deltas, clusters)
    _require(
        np.isclose(reference.point, anchor.point, rtol=1e-10, atol=1e-12)
        and np.isclose(reference.se, anchor.se, rtol=1e-10, atol=1e-12),
        "G2: CPU does not reproduce pinned anchor",
    )
    _require(
        np.isfinite(candidate.point) and np.isfinite(candidate.se),
        "G2: degenerate candidate",
    )
    point_change = abs(candidate.point - reference.point) / anchor.se
    se_change = abs(candidate.se - reference.se) / anchor.se
    n_clusters = len(set(clusters))
    se_warning_scale = 1 / np.sqrt(2 * (n_clusters - 1)) if n_clusters > 1 else None
    return dict(
        G2=bool(max(point_change, se_change) <= SE_FRACTION),
        point_change_in_reference_se=float(point_change),
        se_change_in_reference_se=float(se_change),
        cpu_point=float(reference.point),
        gpu_point=float(candidate.point),
        cpu_se=float(reference.se),
        gpu_se=float(candidate.se),
        n=len(documents),
        anchor_sha256=anchor.artifact_sha256,
        se_sampling_scale_in_reference_se=float(se_warning_scale)
        if flag_se_sampling_scale and se_warning_scale is not None
        else None,
        se_change_warning=bool(se_change > se_warning_scale)
        if flag_se_sampling_scale and se_warning_scale is not None
        else False,
        warning_is="diagnostic only; hard gate remains 0.25 reference SE",
    )


def verify_coverage(
    actual: ReplayBatch,
    required_keys: Sequence[RowKey],
    regime_rows: Mapping[str, Sequence[RowKey]],
) -> dict:
    """G3: all predeclared rows AND each named regime must actually be present."""
    actual.validate()
    _require(set(actual.keys) == set(required_keys), "G3: reference set incomplete")
    _require(bool(regime_rows), "G3: no regime manifest")
    missing = {
        name: [list(k) for k in keys if k not in actual.keys]
        for name, keys in regime_rows.items()
    }
    empty = [name for name, keys in regime_rows.items() if not keys]
    absent = sorted(REQUIRED_REGIMES - set(regime_rows))
    return dict(
        G3=not empty and not absent and not any(missing.values()),
        channel=actual.channel,
        missing=missing,
        unpopulated_regimes=empty,
        absent_regimes=absent,
        adapter_coverage=False,
    )


def verify_sweep_coverage(
    actual: ReplayBatch,
    required_keys: Sequence[RowKey],
    regime_rows: Mapping[str, Sequence[RowKey]],
    sweep: dict,
) -> dict:
    """Ratified 70B G3 replacement; no automatic calibrated-scope extension.

    The caller must supply a sweep recomputed from its pinned arrays, not an
    unchecked saved pass flag. This checks its composition with bank coverage.
    """
    actual.validate()
    _require(set(actual.keys) == set(required_keys), "G3: reference set incomplete")
    required = {
        "near-floor-B-E",
        "near-floor-F-E",
        "E",
        "null-fork-0",
        "null-fork-1",
        "null-fork-2",
    }
    _require(required <= set(regime_rows), "G3: missing bank regime")
    for name in required:
        _require(bool(regime_rows[name]), f"G3: empty regime {name}")
        _require(
            set(regime_rows[name]) <= set(actual.keys), f"G3: unknown rows in {name}"
        )
    _require(sweep["lane_id"] == actual.lane_ids[0], "G3: sweep lane differs from bank")
    lengths = sweep["lengths"]
    _require(
        len(lengths) == 6 and len({r["cache_length"] for r in lengths}) == 6,
        "G3: six distinct lengths required",
    )
    _require(all(r["n"] >= 5 for r in lengths), "G3: five spans per length required")
    _require(max(r["cache_length"] for r in lengths) > 299, "G3: no length extension")
    passed = (
        sweep["coverage_complete"] is True
        and sweep["independent_repeats_exact"] is True
        and sweep["per_feature_pass"] is True
        and all(r["delta_self"] == 0 for r in lengths)
    )
    return dict(
        G3=passed,
        replacement="70B instrument cache-length sweep",
        bank_regime_counts={k: len(regime_rows[k]) for k in sorted(required)},
        lengths=lengths,
        adapter_coverage=False,
        consumer_scope="banked regime only",
        wider_calibrated_scope="not automatically certified; inspect floor(c) trend",
    )


def verify_bank_scope_coverage(
    actual: ReplayBatch,
    required_keys: Sequence[RowKey],
    regime_rows: Mapping[str, Sequence[RowKey]],
    span_bounds: Mapping[RowKey, tuple[int, int]],
    ruling_sha256: str,
) -> dict:
    """Explicitly ratified bank scope; does not certify the sweep envelope."""
    _digest(ruling_sha256, "bank-scope ruling")
    actual.validate()
    keys = set(required_keys)
    _require(set(actual.keys) == keys == set(span_bounds), "G3: incomplete bank scope")
    _require(
        all(
            263 <= start <= 299 and end - start == 128
            for start, end in span_bounds.values()
        ),
        "G3: outside ratified bank geometry",
    )
    required = {
        "near-floor-B-E",
        "near-floor-F-E",
        "E",
        "null-fork-0",
        "null-fork-1",
        "null-fork-2",
    }
    _require(required <= set(regime_rows), "G3: missing bank regime")
    for name in required:
        _require(
            bool(regime_rows[name]) and set(regime_rows[name]) <= keys,
            f"G3: empty/unknown regime {name}",
        )
    return dict(
        G3=True,
        scope="bank cache263–299 / continuation128 only",
        ruling_sha256=ruling_sha256,
        bank_regime_counts={k: len(regime_rows[k]) for k in sorted(required)},
        adapter_coverage=False,
        wider_scope_certified=False,
    )


def verify_memory(
    candidate: Mapping[str, Sequence[int]],
    baseline: Mapping[str, Sequence[int]],
    *,
    cap_bytes: int,
    metric: str,
    baseline_sha256: str,
) -> dict:
    """G5 on a named measurement metric and identical phases/device counts.

    The 135.3 historical cap needs a pinned unit/metric receipt; the caller must
    supply its byte conversion rather than equating allocated/reserved/process.
    Record other metrics separately; they cannot substitute for the gated metric.
    """
    _digest(baseline_sha256, "memory baseline")
    _require(metric in ("allocated", "reserved", "process"), "unknown VRAM metric")
    _require(
        cap_bytes > 0 and bool(baseline) and candidate.keys() == baseline.keys(),
        "G5: missing/mismatched phase measurements",
    )
    phases = {}
    for phase, values in candidate.items():
        ref = baseline[phase]
        _require(len(values) == len(ref) > 0, "G5: device-count mismatch")
        _require(
            all(isinstance(v, int) and v >= 0 for v in (*values, *ref)),
            "G5: invalid peak bytes",
        )
        # Compare each device to both the recorded cap and its matched baseline.
        phases[phase] = all(
            v <= min(b, cap_bytes) for v, b in zip(values, ref, strict=True)
        )
    return dict(
        G5=all(phases.values()),
        metric=metric,
        phases=phases,
        cap_bytes=cap_bytes,
        baseline_sha256=baseline_sha256,
    )


def verify_within_tol(
    vectors: dict, downstream: dict, coverage: dict, memory: dict
) -> dict:
    """Combine receipts; an unexecuted gate can never look like a pass."""
    gates = {
        name: block.get(name)
        for block, names in (
            (vectors, ("G0", "G1", "G4")),
            (downstream, ("G2",)),
            (coverage, ("G3",)),
            (memory, ("G5",)),
        )
        for name in names
    }
    return dict(
        passed=all(value is True for value in gates.values()),
        gates=gates,
        vectors=vectors,
        downstream=downstream,
        coverage=coverage,
        memory=memory,
    )
