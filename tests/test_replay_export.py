"""Replay exports (anamnesis.extraction.fast.replay_export): an export loads back into a row that reduces to the
signed row's bytes, and an export that is not one is refused by name."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from anamnesis.extraction.fast.replay_export import (
    SCHEMA,
    ReplayExportError,
    load_export,
    save_export,
)

# The toy engine rows and lanes of the step-bucket tests.
_spec = importlib.util.spec_from_file_location("_step_buckets_fixtures",
                                               Path(__file__).with_name("test_step_buckets.py"))
sb = importlib.util.module_from_spec(_spec)
sys.modules["_step_buckets_fixtures"] = sb
_spec.loader.exec_module(sb)

pytestmark = pytest.mark.xdist_group("step-buckets")


@pytest.fixture(autouse=True)
def deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def header_for(row, ids, sign_id="r0") -> dict:
    return dict(sign_id=sign_id, request_id=sign_id, token_ids=list(ids), prompt_length=int(row.start),
                start=int(row.start), positions=int(row.positions), weights_sha256=sb.ENGINE["weights"],
                engine_identity_sha256="e" * 64, pins_sha256=None, lever_plan_sha256=None,
                lane_id="engine-stats-b1-toy", replayed_bitwise=True,
                layers=dict(effective=sb.LAYERS, sampled=list(sb.SAMPLED)),
                route_sources=[list(p) for p in row.route_sources], final_sources=int(row.final_sources))


def test_an_export_reduces_to_the_signed_rows_bytes(tmp_path: Path) -> None:
    made = [sb.signed_row(n, 1 + n % sb.MAX_START, 900 + n, 8) for n in (2, 9, 16)]
    want = sb.reduce("cpu", 8, made)
    loaded = []
    for i, (row, ids) in enumerate(made):
        save_export(tmp_path / f"r{i}", row, header_for(row, ids, f"r{i}"))
        header, back = load_export(tmp_path / f"r{i}.safetensors")
        assert header.schema_ == SCHEMA and header.positions == row.positions
        assert isinstance(back.surfaces["committed"], list)
        assert len(back.surfaces["committed"]) == len(row.surfaces["committed"])
        loaded.append((back, list(header.token_ids)))
    got = sb.reduce("cpu", 8, loaded)
    for w, g in zip(want, got, strict=True):
        assert g.features.tobytes() == w.features.tobytes()
        assert g.lane_id == w.lane_id


@pytest.mark.parametrize("edit, message", [
    (lambda h: h.update(replayed_bitwise=False), "did not reproduce"),
    (lambda h: h.update(schema="other-v0"), "schema"),
    (lambda h: h["tensors"]["depth"].update(shape=[1, 1, 1]), "the header names"),
    (lambda h: h["tensors"].pop("gate"), "names no"),
])
def test_an_export_that_is_not_one_is_refused(tmp_path: Path, edit, message: str) -> None:
    row, ids = sb.signed_row(9, 3, 77, 8)
    _, header_path = save_export(tmp_path / "x", row, header_for(row, ids))
    h = json.loads(header_path.read_text())
    edit(h)
    header_path.write_text(json.dumps(h))
    with pytest.raises(ReplayExportError, match=message):
        load_export(tmp_path / "x")


def test_bfloat16_surfaces_keep_their_dtype(tmp_path: Path) -> None:
    row, ids = sb.signed_row(5, 2, 13, 8)
    row.surfaces["depth"] = row.surfaces["depth"].to(torch.bfloat16)
    save_export(tmp_path / "b", row, header_for(row, ids))
    header, back = load_export(tmp_path / "b")
    assert back.surfaces["depth"].dtype == torch.bfloat16 and header.tensors["depth"].dtype == "bfloat16"
    assert torch.equal(back.surfaces["depth"], row.surfaces["depth"])
