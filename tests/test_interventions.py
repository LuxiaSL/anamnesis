"""An intervention spec: how it travels, and what it refuses.

An intervention is described as data so that the description can travel — written
into a cell's metadata when the cell is generated, read back when it is replayed —
which is what makes a steered generation and its replay provably the same
intervention rather than two similar ones.

So the refusals are the substance. A partial spec is refused rather than completed
by guessing, because a cell banked under a dose it did not receive is a control
mislabelled as a treatment. A write whose position gating did not fire at exactly
the generated span is refused for the same reason: an intervention that quietly did
not happen looks exactly like one that did.

The arming scope is the same concern seen across cells rather than within one. Both
passes walk a roster under one model load, so a handle a failed cell left behind
would stack on the next cell's write and every cell after it would carry a dose
nobody asked for — with nothing in the output to show it. The cases below pin
removal on the ordinary exit and on the raise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from anamnesis.extraction.interventions import (
    INJECTION_METADATA_KEY,
    InjectionSpec,
    armed_interventions,
    PerturbationSpec,
    check_injection_gating,
    injection_fields,
    load_vector,
    resolve_injection,
)


@pytest.fixture()
def bank(tmp_path: Path) -> Path:
    path = tmp_path / "vectors.npz"
    np.savez(path, V3=np.eye(8, dtype=np.float32)[0], loose=np.full(8, 0.5, dtype=np.float32))
    return path


class _Handle:
    """Stands in for an armed write handle, which only its stats are read from here."""

    def __init__(self, **stats: Any) -> None:
        self.stats = stats


def test_load_vector_reads_a_unit_direction(bank: Path) -> None:
    vector = load_vector(bank, "V3")
    assert vector.dtype == np.float32 and float(np.linalg.norm(vector)) == pytest.approx(1.0)


def test_load_vector_names_what_the_bank_holds(bank: Path) -> None:
    """A site name that drifted against the bank is the cheapest failure to diagnose."""
    with pytest.raises(KeyError, match="loose"):
        load_vector(bank, "V9")


def test_load_vector_warns_about_a_non_unit_direction(
    bank: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A drifted norm means the bank is not the one the dose was priced against."""
    with caplog.at_level("WARNING"):
        load_vector(bank, "loose")
    assert "norm" in caplog.text


def test_injection_fields_names_the_keys_once() -> None:
    assert set(injection_fields()) == {
        "inject_npz", "inject_key", "inject_layer", "inject_alpha", "inject_alpha_frac"
    }


def test_a_spec_round_trips_through_its_metadata(bank: Path) -> None:
    spec = InjectionSpec(npz=bank, key="V3", layer=18, alpha=2.5, alpha_frac=0.1)
    again = InjectionSpec.from_mapping(spec.metadata())
    assert again is not None and again.model_dump() == spec.model_dump()


def test_no_bank_means_no_intervention() -> None:
    assert InjectionSpec.from_mapping(injection_fields()) is None
    assert resolve_injection(None, fields={}) is None


def test_a_partial_spec_is_refused(bank: Path) -> None:
    with pytest.raises(SystemExit, match="key, a layer and an absolute alpha"):
        InjectionSpec.from_mapping(injection_fields(bank, "V3", None, None))


def test_a_spec_is_read_from_the_cell_that_was_generated_under_it(
    tmp_path: Path, bank: Path
) -> None:
    run_dir = tmp_path / "cell"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps({
            INJECTION_METADATA_KEY: injection_fields(str(bank), "V3", 18, 2.5, 0.1),
            "generations": [],
        })
    )
    spec = resolve_injection(run_dir, from_metadata=True)
    assert spec is not None and spec.key == "V3" and spec.layer == 18


def test_an_unsteered_cell_refuses_to_be_read_as_steered(tmp_path: Path) -> None:
    run_dir = tmp_path / "cell"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(json.dumps({"generations": []}))
    with pytest.raises(SystemExit, match=INJECTION_METADATA_KEY):
        resolve_injection(run_dir, from_metadata=True)


def test_gating_passes_when_the_write_landed_on_every_generated_position() -> None:
    check_injection_gating(_Handle(saw_cache_position=True, positions=7), 7, "gen_000")


def test_ungated_write_is_refused() -> None:
    """Without a cache position the write's position semantics are unverifiable."""
    with pytest.raises(RuntimeError, match="unverifiable"):
        check_injection_gating(_Handle(saw_cache_position=False, positions=7), 7, "gen_000")


def test_wrong_position_count_is_refused() -> None:
    with pytest.raises(RuntimeError, match="expected 7"):
        check_injection_gating(_Handle(saw_cache_position=True, positions=5), 7, "gen_000")


def test_a_perturbation_carries_its_seed_so_the_pass_is_reproducible() -> None:
    spec = PerturbationSpec(mode="topk", top_k=2, seed=11)
    assert spec.seed == 11 and spec.eps is None


def test_a_perturbation_needs_a_mode() -> None:
    with pytest.raises(ValueError):
        PerturbationSpec(mode="")


class _Armed:
    """Stands in for an attached hook handle: it records that it was removed."""

    def __init__(self) -> None:
        self.removed = 0

    def remove(self) -> None:
        self.removed += 1


def _arming(monkeypatch: pytest.MonkeyPatch) -> tuple[list[tuple[str, Any]], _Armed, _Armed]:
    """Replace both attach calls with recorders, returning the log and the handles."""
    write, perturb = _Armed(), _Armed()
    seen: list[tuple[str, Any]] = []

    def fake_injection(target: Any, spec: Any, label: str) -> Any:
        seen.append(("write", target))
        return None if spec is None else write

    def fake_perturbation(target: Any, fields: Any, label: str) -> Any:
        seen.append(("perturb", target))
        return None if not fields else perturb

    monkeypatch.setattr(
        "anamnesis.extraction.interventions.attach_injection", fake_injection
    )
    monkeypatch.setattr(
        "anamnesis.extraction.interventions.attach_perturbation", fake_perturbation
    )
    return seen, write, perturb


def test_armed_interventions_yields_the_write_handle_and_removes_both(
    monkeypatch: pytest.MonkeyPatch, bank: Path
) -> None:
    seen, write, perturb = _arming(monkeypatch)
    spec = InjectionSpec(npz=bank, key="V3", layer=3, alpha=2.0)
    with armed_interventions(
        "model", "inner", injection=spec, perturbation={"mode": "topk"}, label="w0"
    ) as handle:
        assert handle is write
        assert write.removed == 0 and perturb.removed == 0
    assert write.removed == 1 and perturb.removed == 1
    assert seen == [("write", "model"), ("perturb", "inner")]


def test_armed_interventions_removes_the_write_when_the_cell_raises(
    monkeypatch: pytest.MonkeyPatch, bank: Path
) -> None:
    """The load-bearing case: a roster's next cell must not inherit this one's dose."""
    _, write, perturb = _arming(monkeypatch)
    spec = InjectionSpec(npz=bank, key="V3", layer=3, alpha=2.0)
    with pytest.raises(RuntimeError, match="one generation"):
        with armed_interventions(
            "model", "inner", injection=spec, perturbation={"mode": "topk"}, label="w0"
        ):
            raise RuntimeError("one generation blew up")
    assert write.removed == 1 and perturb.removed == 1


def test_an_unsteered_cell_arms_nothing_and_removes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, write, perturb = _arming(monkeypatch)
    with armed_interventions(
        "model", "inner", injection=None, perturbation=None, label="w0"
    ) as handle:
        assert handle is None
    assert write.removed == 0 and perturb.removed == 0
