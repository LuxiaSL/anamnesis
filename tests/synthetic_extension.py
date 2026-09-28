"""A synthetic base lane and a fine-tune of it, so the transfer check and the
extension-lane guard can be exercised end to end without an engine or a model.

The base is shipped as ``8b`` under a temporary fixture root, with a tolerance
over a twelve-coordinate schema that has every kind of coordinate the transfer
check reads: an attention component holding the spectral, flow and coverage
coordinates, a substrate component holding the residual and gate-sparsity ones,
continuous families with recorded maxima, and discrete coverage and gate-sparsity
coordinates. The fine-tune's sample is 44 rows whose vLLM vectors sit a small,
row-varying distance from their fast-lane vectors, well inside the base's
ceilings, so an unmodified sample passes and each test perturbs one thing.

:func:`declare_extension` goes the rest of the way: a registry preset extending
``8b``, a calibration directory, a passing transfer written to disk with the
package's own writer, and ``ANAMNESIS_VLLM_LANES`` naming its entry.
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
from anamnesis.extraction.vllm.extensions import LANES_ENV, calibration_pins
from anamnesis.extraction.vllm.transfer import (
    SampleRow,
    TransferResult,
    TransferSample,
    check_transfer,
    select_strata,
)
from anamnesis.extraction.vllm.transfer_run import write_transfer
from anamnesis.provenance import digest_of_shas, file_sha

BASE = "8b"
KEY = "ft-lane"
PRESET = "ft-preset"
CHECKPOINT = "c" * 64

ATTENTION = ("spectral_fiedler_L0", "spectral_entropy_L0", "attn_flow_0", "attn_flow_1",
             "cache_cache_coverage_L0", "cache_cache_coverage_L1")
SUBSTRATE = ("res_0", "res_1", "res_2", "res_3", "gate_L0_sparsity_mean", "gate_L0_sparsity_std")
NAMES = ATTENTION + SUBSTRATE
FAMILIES = {"spectral_fiedler_L0": "attn-spectral", "spectral_entropy_L0": "attn-spectral",
            "attn_flow_0": "attn-flow", "attn_flow_1": "attn-flow",
            "res_0": "residual", "res_1": "residual", "res_2": "residual", "res_3": "residual"}
DISCRETE = ("cache_cache_coverage_L0", "cache_cache_coverage_L1", "gate_L0_sparsity_mean",
            "gate_L0_sparsity_std")
N_ROWS = 44


def base_tolerance(**overrides) -> Tolerance:
    """The base's recorded tolerance: ceilings 5 (max), 3 (p90), 4 (p99); maxima 1."""
    components = tuple(RowComponent(name=name, feature_names=names, max_ratio=5.0,
                                    p90_ratio=3.0, p99_ratio=4.0, source="base record")
                       for name, names in (("substrate", SUBSTRATE), ("attention", ATTENTION)))
    fields = dict(model=BASE, components=components, families=FAMILIES,
                  family_max_abs_sigma={"attn-spectral": 1.0, "attn-flow": 1.0,
                                        "residual": 1.0},
                  discrete=DISCRETE, median_gate_stratum="evenly-spaced",
                  sources={"base/record": "0" * 64})
    fields.update(overrides)
    return Tolerance(**fields)


def ship_base(root: Path, monkeypatch, tolerance: Tolerance | None = None) -> Tolerance:
    """Ship a base fixture set and ``tolerance`` as ``8b`` under ``root``."""
    monkeypatch.setattr(runtime, "FIXTURES_ROOT", root)
    tolerance = tolerance or base_tolerance()
    rows = tuple(FixtureRow(generation_id=g, population="native", input_ids=(1, 2, 3, 4),
                            prompt_length=1, end=4, floor_b=1.0, selected_by="evenly-spaced")
                 for g in (1, 2))
    fixtures = FixtureSet(model=BASE, lane_id=lane_id(BASE), checkpoint_sha256="b" * 64,
                          calibration_sha256="d" * 64, ruler_sha256="e" * 64,
                          feature_names=NAMES, sigma_cal=np.ones(len(NAMES)),
                          weights=np.ones(len(NAMES)), rows=rows,
                          vectors={g: np.zeros(len(NAMES), dtype=np.float32) for g in (1, 2)})
    target = root / BASE
    fixtures.save(target)
    (target / "tolerance.json").write_text(tolerance.model_dump_json())
    return tolerance


def sample(n_rows: int = N_ROWS, seed: int = 0, scale: float = 0.1,
           floors: dict[int, float] | None = None) -> TransferSample:
    """A sample whose vLLM vectors deviate from the fast lane's by about ``scale`` σ."""
    rng = np.random.default_rng(seed)
    ids = list(range(100, 100 + n_rows))
    reference = {g: rng.normal(size=len(NAMES)).astype(np.float32) for g in ids}
    candidate = {g: (reference[g] + rng.uniform(0.2, 1.0) * scale
                     * rng.normal(size=len(NAMES))).astype(np.float32) for g in ids}
    floors = floors or {}
    rows = tuple(SampleRow(generation_id=g, input_ids=tuple(range(1, 11)), prompt_length=3,
                           end=10, floor_b=floors.get(g, 1.0)) for g in ids)
    return TransferSample(feature_names=NAMES, sigma_cal=np.ones(len(NAMES)),
                          weights=np.ones(len(NAMES)), rows=rows, reference=reference,
                          candidate=candidate)


def with_candidate(base: TransferSample, updates: dict[int, np.ndarray]) -> TransferSample:
    """``base`` with some rows' vLLM vectors replaced."""
    candidate = dict(base.candidate)
    candidate.update({g: np.asarray(v, dtype=np.float32) for g, v in updates.items()})
    return base.model_copy(update=dict(candidate=candidate))


def repeats(s: TransferSample) -> list[CapturedFixture]:
    """Byte-identical repeat and batched captures of every sampled row."""
    return [CapturedFixture(generation_id=g, first=s.candidate[g], repeat=s.candidate[g].copy(),
                            batched=s.candidate[g].copy()) for g in s.ids]


def run(s: TransferSample, tolerance: Tolerance | None = None, *, strata=None,
        determinism=None, max_floor: float | None = None, key: str = KEY,
        checkpoint: str = CHECKPOINT, calibration: str = "a" * 64) -> TransferResult:
    """The transfer decision on ``s`` against the base tolerance."""
    tolerance = tolerance or base_tolerance()
    return check_transfer(
        key=key, extends=BASE, base_tolerance=tolerance, base_feature_names=NAMES,
        checkpoint_sha256=checkpoint, calibration_sha256=calibration, sample=s,
        strata=select_strata(s, tolerance) if strata is None else strata,
        determinism=repeats(s) if determinism is None else determinism, max_floor=max_floor)


def calibration(directory: Path) -> Path:
    """A two-file calibration directory with distinct contents."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in runtime.CALIBRATION_FILES:
        (directory / name).write_bytes(f"calibration {name}".encode())
    return directory


def registry(path: Path, monkeypatch, row: dict | None = None) -> Path:
    """A registry file holding the fine-tune's preset, named in ``ANAMNESIS_MODELS``."""
    row = row if row is not None else {"extends": BASE, "model_id": "someone/fine-tune"}
    path.write_text(json.dumps({"presets": {PRESET: row}}))
    monkeypatch.setenv("ANAMNESIS_MODELS", str(path))
    return path


@dataclass
class Declared:
    """What :func:`declare_extension` wrote."""

    out: Path
    entry: Path
    receipt: Path
    calib: Path
    result: TransferResult


def declare_extension(tmp_path: Path, monkeypatch, *, key: str = KEY,
                      checkpoint: str = CHECKPOINT, out_name: str = "transfer",
                      ship: bool = True) -> Declared:
    """A passing extension of ``8b`` written to disk and named in ``ANAMNESIS_VLLM_LANES``."""
    tolerance = ship_base(tmp_path / "shipped", monkeypatch) if ship else base_tolerance()
    if not (tmp_path / "models.json").exists():
        registry(tmp_path / "models.json", monkeypatch)
    calib = calibration(tmp_path / f"calib-{out_name}")
    digest = digest_of_shas({n: file_sha(calib / n) for n in runtime.CALIBRATION_FILES})
    result = run(sample(), tolerance, key=key, checkpoint=checkpoint, calibration=digest)
    assert result.receipt.verdict == "pass", result.receipt.reasons
    written = write_transfer(tmp_path / out_name, result, calib_dir=calib, preset=PRESET,
                             pins=calibration_pins(calib))
    monkeypatch.setenv(LANES_ENV, str(written["entry"]))
    return Declared(out=tmp_path / out_name, entry=written["entry"],
                    receipt=written["receipt"], calib=calib, result=result)


def rewrite_entry(declared: Declared, **changes) -> dict:
    """Change fields of the declared entry in place; returns the new entry row."""
    document = json.loads(declared.entry.read_text())
    (row,) = document["lanes"].values()
    row.update(changes)
    declared.entry.write_text(json.dumps(document))
    return row


def rewrite_receipt(declared: Declared, **changes) -> None:
    """Rewrite the receipt with ``changes`` (unvalidated) and re-pin its digest."""
    receipt = json.loads(declared.receipt.read_text())
    receipt.update(changes)
    declared.receipt.write_text(json.dumps(receipt))
    rewrite_entry(declared, transfer_receipt_sha256=file_sha(declared.receipt))
