"""A synthetic base lane and a passing fine-tune of it, for the transfer check and the
extension-lane guard, without an engine or a model.

The base ships as ``8b`` under a temporary fixture root, with a tolerance over a
twelve-coordinate schema holding every kind of coordinate the transfer check reads.
:func:`sample` is 44 rows whose vLLM vectors sit a small distance from their
fast-lane vectors, so it passes unmodified; :func:`declare_extension` writes a passing
transfer to disk and names its entry in ``ANAMNESIS_VLLM_LANES``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from anamnesis.extraction.vllm import runtime
from anamnesis.extraction.vllm.conformance import (
    CapturedFixture,
    FixtureRow,
    FixtureSet,
    RowComponent,
    Tolerance,
)
from anamnesis.extraction.vllm.envelope import lane_id
from anamnesis.extraction.vllm.extensions import LANES_ENV
from anamnesis.extraction.vllm.transfer import (
    RULES_FILE,
    TransferResult,
    TransferRules,
    TransferSample,
    check_transfer,
    select_strata,
)
from anamnesis.provenance import digest_of_shas, file_sha
from anamnesis.scripts.transfer_vllm import write_transfer

BASE, KEY, PRESET, CHECKPOINT = "8b", "ft-lane", "ft-preset", "c" * 64
ATTENTION = ("spectral_fiedler_L0", "spectral_entropy_L0", "attn_flow_0", "attn_flow_1",
             "cache_cache_coverage_L0", "cache_cache_coverage_L1")
SUBSTRATE = ("res_0", "res_1", "res_2", "res_3", "gate_L0_sparsity_mean", "gate_L0_sparsity_std")
NAMES = ATTENTION + SUBSTRATE
FAMILIES = {"spectral_fiedler_L0": "attn-spectral", "spectral_entropy_L0": "attn-spectral",
            "attn_flow_0": "attn-flow", "attn_flow_1": "attn-flow",
            **{f"res_{i}": "residual" for i in range(4)}}
DISCRETE = ("cache_cache_coverage_L0", "cache_cache_coverage_L1", "gate_L0_sparsity_mean",
            "gate_L0_sparsity_std")


def base_tolerance(residual_max: float = 1.0, **overrides) -> Tolerance:
    """Ceilings 5 (max), 3 (p90), 4 (p99); family maxima 1 (residual ``residual_max``)."""
    fields = dict(model=BASE, components=tuple(
        RowComponent(name=name, feature_names=names, max_ratio=5.0, p90_ratio=3.0,
                     p99_ratio=4.0, source="base record")
        for name, names in (("substrate", SUBSTRATE), ("attention", ATTENTION))),
        families=FAMILIES, discrete=DISCRETE, median_gate_stratum="evenly-spaced",
        family_max_abs_sigma={"attn-spectral": 1.0, "attn-flow": 1.0, "residual": residual_max},
        sources={"base/record": "0" * 64})
    return Tolerance(**{**fields, **overrides})


def base_rules(tolerance: Tolerance | None = None, **overrides) -> TransferRules:
    """The base's rule set: two rows over a component maximum allowed, one ordinary row over
    the p99, family p90s at half the family maxima."""
    tolerance = tolerance or base_tolerance()
    fields = dict(model=BASE, tolerance_digest=tolerance.digest, component_max_allowance=2,
                  p99_allowance=1, diagnostic_from_depth=0.35,
                  family_p90_abs_sigma={f: m / 2 for f, m in
                                        tolerance.family_max_abs_sigma.items()},
                  weakly_guarded={"residual": 2.5},
                  qualification=dict(method="synthetic", false_refusal=0.0))
    return TransferRules(**{**fields, **overrides})


def ship_base(root: Path, monkeypatch, tolerance: Tolerance | None = None) -> Tolerance:
    """Ship a two-row ``8b`` fixture set, ``tolerance`` and its rule set under ``root``."""
    monkeypatch.setattr(runtime, "FIXTURES_ROOT", root)
    tolerance = tolerance or base_tolerance()
    rows = tuple(FixtureRow(generation_id=g, population="native", input_ids=(1, 2, 3, 4),
                            prompt_length=1, end=4, floor_b=1.0, selected_by="evenly-spaced")
                 for g in (1, 2))
    FixtureSet(model=BASE, lane_id=lane_id(BASE), checkpoint_sha256="b" * 64,
               calibration_sha256="d" * 64, ruler_sha256="e" * 64, feature_names=NAMES,
               sigma_cal=np.ones(len(NAMES)), weights=np.ones(len(NAMES)), rows=rows,
               vectors={g: np.zeros(len(NAMES), dtype=np.float32) for g in (1, 2)}
               ).save(root / BASE)
    (root / BASE / "tolerance.json").write_text(tolerance.model_dump_json())
    (root / BASE / RULES_FILE).write_text(base_rules(tolerance).model_dump_json())
    return tolerance


def sample(n_rows: int = 44, seed: int = 0, scale: float = 0.1,
           floors: dict[int, float] | None = None) -> TransferSample:
    """A sample whose vLLM vectors deviate from the fast lane's by about ``scale`` σ."""
    rng = np.random.default_rng(seed)
    ids = list(range(100, 100 + n_rows))
    reference = {g: rng.normal(size=len(NAMES)).astype(np.float32) for g in ids}
    candidate = {g: (reference[g] + rng.uniform(0.2, 1.0) * scale
                     * rng.normal(size=len(NAMES))).astype(np.float32) for g in ids}
    rows = tuple(FixtureRow(generation_id=g, population="native", input_ids=tuple(range(1, 11)),
                            prompt_length=3, end=10, floor_b=(floors or {}).get(g, 1.0),
                            selected_by="sampled") for g in ids)
    return TransferSample(feature_names=NAMES, sigma_cal=np.ones(len(NAMES)),
                          weights=np.ones(len(NAMES)), rows=rows, reference=reference,
                          candidate=candidate)


def with_candidate(s: TransferSample, updates: dict[int, np.ndarray]) -> TransferSample:
    candidate = {**s.candidate, **{g: np.asarray(v, np.float32) for g, v in updates.items()}}
    return s.model_copy(update=dict(candidate=candidate))


def shifted(s: TransferSample, gid: int, name: str, amount: float) -> np.ndarray:
    """Row ``gid``'s reference with one coordinate moved by ``amount`` σ."""
    vector = s.reference[gid].copy()
    vector[NAMES.index(name)] += amount
    return vector


def repeats(s: TransferSample) -> list[CapturedFixture]:
    return [CapturedFixture(generation_id=g, first=v, repeat=v.copy(), batched=v.copy())
            for g, v in sorted(s.candidate.items())]


def run(s: TransferSample, tolerance: Tolerance | None = None, *, strata=None,
        determinism=None, max_floor: float | None = None, key: str = KEY,
        checkpoint: str = CHECKPOINT, calibration: str = "a" * 64,
        rules: TransferRules | None = None, diagnostics: dict | None = None,
        **rule_overrides) -> TransferResult:
    tolerance = tolerance or base_tolerance()
    return check_transfer(
        key=key, extends=BASE, base_tolerance=tolerance, base_feature_names=NAMES,
        checkpoint_sha256=checkpoint, calibration_sha256=calibration, sample=s,
        strata=select_strata(s, tolerance) if strata is None else strata,
        determinism=repeats(s) if determinism is None else determinism, max_floor=max_floor,
        rules=rules or base_rules(tolerance, **rule_overrides), diagnostics=diagnostics)


def registry(path: Path, monkeypatch, row: dict | None = None) -> Path:
    """A registry file holding the fine-tune's preset, named in ``ANAMNESIS_MODELS``."""
    path.write_text(json.dumps({"presets": {PRESET: row or {"extends": BASE,
                                                           "model_id": "someone/fine-tune"}}}))
    monkeypatch.setenv("ANAMNESIS_MODELS", str(path))
    return path


@dataclass
class Declared:
    out: Path
    entry: Path
    receipt: Path
    calib: Path
    result: TransferResult


def declare_extension(tmp_path: Path, monkeypatch, *, key: str = KEY,
                      checkpoint: str = CHECKPOINT, out_name: str = "transfer",
                      audited: TransferSample | None = None) -> Declared:
    """An extension of ``8b`` written to disk and named in ``ANAMNESIS_VLLM_LANES``; its
    audit passes unless ``audited`` supplies another sample."""
    tolerance = base_tolerance() if (tmp_path / "shipped").exists() \
        else ship_base(tmp_path / "shipped", monkeypatch)
    if not (tmp_path / "models.json").exists():
        registry(tmp_path / "models.json", monkeypatch)
    calib = tmp_path / f"calib-{out_name}"
    calib.mkdir()
    for name in runtime.CALIBRATION_FILES:
        (calib / name).write_bytes(f"calibration {name}".encode())
    digest = digest_of_shas({n: file_sha(calib / n) for n in runtime.CALIBRATION_FILES})
    result = run(audited or sample(), tolerance, key=key, checkpoint=checkpoint,
                 calibration=digest)
    written = write_transfer(tmp_path / out_name, result, calib_dir=calib, preset=PRESET)
    monkeypatch.setenv(LANES_ENV, str(written["entry"]))
    return Declared(tmp_path / out_name, written["entry"], written["receipt"], calib, result)


def entry_row(declared: Declared, **changes) -> dict:
    """The declared entry's row with ``changes``, paths made absolute."""
    (row,) = json.loads(declared.entry.read_text())["lanes"].values()
    for field in ("fixtures_dir", "transfer_receipt"):
        row[field] = str(declared.out / row[field])
    return {**row, **changes}


def rewrite_entry(declared: Declared, **changes) -> None:
    declared.entry.write_text(json.dumps({"lanes": {KEY: entry_row(declared, **changes)}}))


def rewrite_receipt(declared: Declared, **changes) -> None:
    """Rewrite the receipt with ``changes`` (unvalidated) and re-pin its digest."""
    receipt = {**json.loads(declared.receipt.read_text()), **changes}
    declared.receipt.write_text(json.dumps(receipt))
    rewrite_entry(declared, transfer_receipt_sha256=file_sha(declared.receipt))
