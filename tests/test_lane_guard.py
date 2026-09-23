"""Lane guard: which signatures may be joined, and which may not.

A signature carries the identity of the arithmetic that produced it. Two signatures
produced by different arithmetic — a different lane, a different box — are not two
measurements of the same quantity, and the difference between them is not signal. So
the guard's job is to refuse a join, and the hard case is the silent one: a bank that
predates lane tagging is readable, but "untagged" is not a lane identity, and it must
not be allowed to pass as agreement with a tagged bank.

The gate therefore has one asymmetry worth stating. All-untagged input returns `None`
and is permitted: that is the historical corpus, read as itself. Mixed input raises,
including the mixed case that looks most like nothing — one tagged row beside one
untagged row. Conflicting calibration or schema digests inside a single lane raise for
the same reason: a lane id that two different rulers share is not one lane.

Pure mappings in, a verdict out. The rest of this file is the readers that call
it, because a guard nothing reaches is decoration. There are four, and they all
turn a directory of banked vectors into one matrix: `signature_io.load_run4`,
`battery.floors.load_signature_matrix`, `audit_lib.load_merged_signature_matrix`
and `steering.readouts.floor_z`. The last three reach the gate through one
function, `lane_guard.gate_banked_signatures`, and the identity of that function
is asserted here: two copies of a check are two things that can disagree, which is
how a reader ends up ungated while looking gated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anamnesis.analysis import audit_lib
from anamnesis.analysis.battery import floors
from anamnesis.analysis.gauntlet.signature_io import load_run4
from anamnesis.analysis.lane_guard import (
    MixedLaneError,
    gate_banked_signatures,
    lane_is_drawn,
    require_single_lane,
    sidecar_path,
)
from anamnesis.steering import readouts
from anamnesis.synthetic_bank import SYNTHETIC_LANE


def test_legacy_inputs_are_not_invented_cpu_certification():
    assert require_single_lane([{}, {}]) is None
    assert (
        require_single_lane(
            [{"lane_id": "gpu"}, {"extraction_lane": {"lane_id": "gpu"}}]
        )
        == "gpu"
    )


@pytest.mark.parametrize(
    "metadata",
    [
        [{"lane_id": "gpu"}, {}],
        [{"lane_id": "gpu"}, {"lane_id": "cpu"}],
        [{"lane_id": "gpu", "extraction_lane": {"lane_id": "other"}}],
        [{"lane_id": ""}],
        [
            {"lane_id": "gpu", "extraction_lane": {"calibration_sha256": "a"}},
            {"lane_id": "gpu", "extraction_lane": {"calibration_sha256": "b"}},
        ],
    ],
)
def test_mixed_or_conflicting_identity_rejected(metadata):
    with pytest.raises(MixedLaneError):
        require_single_lane(metadata)


def write_row(folder: Path, i: int, lane: str | None = None,
              feature_key: str = "features_tier1") -> None:
    """One generation's (npz, json) pair, tagged with a lane or deliberately not."""
    folder.mkdir(exist_ok=True)
    np.savez(
        folder / f"gen_{i:03d}.npz",
        **{feature_key: np.array([1.0, 2.0], dtype=np.float32)},
        feature_names=np.array(["a", "b"]),
    )
    meta = dict(
        generation_id=i,
        topic="topic",
        topic_idx=0,
        mode="linear",
        mode_idx=0,
        num_generated_tokens=128,
    )
    if lane is not None:
        meta["lane_id"] = lane
    (folder / f"gen_{i:03d}.json").write_text(json.dumps(meta))


def test_loader_preserves_and_enforces_lane_identity(tmp_path):
    write_row(tmp_path, 0, "gpu")
    write_row(tmp_path, 1, "gpu")
    data = load_run4(tmp_path, core_only=False)
    assert data.lane_id == "gpu" and data.n_samples == 2
    write_row(tmp_path, 1, "other")
    with pytest.raises(MixedLaneError):
        load_run4(tmp_path, core_only=False)


def test_loader_legacy_behavior_remains_available(tmp_path):
    write_row(tmp_path, 0)
    assert load_run4(tmp_path, core_only=False).lane_id is None


def test_tagged_dataset_cannot_silently_skip_missing_metadata(tmp_path):
    write_row(tmp_path, 0, "gpu")
    write_row(tmp_path, 1, "gpu")
    (tmp_path / "gen_001.json").unlink()
    with pytest.raises(MixedLaneError, match="without lane metadata"):
        load_run4(tmp_path, core_only=False)


# ── The gate over files ───────────────────────────────────────────────────────
FEATURES = ("res_norm_L4_mean", "attn_entropy_L8_mean")


def write_bank(
    root: Path, lanes: dict[int, str | None], *, subdir: str = "signatures"
) -> Path:
    """A battery-shaped bank: one npz per generation, a sidecar each, metadata.json.

    A lane of ``None`` writes the sidecar without a lane id, which is what an
    untagged historical bank looks like.
    """
    sig_dir = root / subdir
    sig_dir.mkdir(parents=True, exist_ok=True)
    generations = []
    for gid, lane in lanes.items():
        np.savez(
            sig_dir / f"gen_{gid:03d}.npz",
            features=np.arange(len(FEATURES), dtype=np.float32) + gid,
            feature_names=np.array(FEATURES),
        )
        record = {
            "generation_id": gid,
            "mode": "linear",
            "mode_idx": 0,
            "topic": "topic",
            "topic_idx": gid % 2,
            "prompt_length": 40 + gid,
            "num_generated_tokens": 100 + gid,
        }
        if lane is not None:
            record["lane_id"] = lane
        (sig_dir / f"gen_{gid:03d}.json").write_text(json.dumps(record))
        generations.append(record)
    (root / "metadata.json").write_text(json.dumps({"generations": generations}))
    return sig_dir


def paths_of(sig_dir: Path) -> list[Path]:
    return sorted(sig_dir.glob("gen_*.npz"))


def test_the_gate_reads_the_sidecar_beside_each_vector(tmp_path):
    sig_dir = write_bank(tmp_path, {0: "gpu", 1: "gpu"})
    assert sidecar_path(sig_dir / "gen_000.npz") == sig_dir / "gen_000.json"
    assert gate_banked_signatures(paths_of(sig_dir)) == "gpu"
    assert gate_banked_signatures([]) is None


def test_an_untagged_bank_stays_readable_and_is_not_certified(tmp_path):
    sig_dir = write_bank(tmp_path, {0: None, 1: None})
    assert gate_banked_signatures(paths_of(sig_dir)) is None
    (sig_dir / "gen_001.json").unlink()
    assert gate_banked_signatures(paths_of(sig_dir)) is None, (
        "an absent sidecar in an untagged bank contaminates no lane"
    )


@pytest.mark.parametrize(
    "second_lane, sidecar",
    [
        ("other", None),        # a second lane, declared
        ("gpu", "absent"),      # a row whose lane nothing states
        ("gpu", "{"),           # a sidecar that does not parse
        ("gpu", "[]"),          # a sidecar that parses to something that is not a mapping
    ],
    ids=["mixed-lanes", "sidecar-absent", "sidecar-unparseable", "sidecar-not-a-mapping"],
)
def test_a_tagged_bank_refuses_every_row_whose_lane_it_cannot_read(
    tmp_path, second_lane, sidecar
):
    sig_dir = write_bank(tmp_path, {0: "gpu", 1: second_lane})
    if sidecar == "absent":
        (sig_dir / "gen_001.json").unlink()
    elif sidecar is not None:
        (sig_dir / "gen_001.json").write_text(sidecar)
    with pytest.raises(MixedLaneError):
        gate_banked_signatures(paths_of(sig_dir))


def test_the_bypass_has_to_be_named_and_reports_no_lane(tmp_path, caplog):
    sig_dir = write_bank(tmp_path, {0: "gpu", 1: "other"})
    with caplog.at_level("ERROR"):
        assert gate_banked_signatures(paths_of(sig_dir), allow_mixed_lanes=True) is None
    assert "across lanes anyway" in caplog.text, "a bypass that leaves no trace is a hole"


# ── One gate, every reader ────────────────────────────────────────────────────
def test_every_banked_reader_resolves_the_same_gate():
    """The three same-shaped readers hold the gate itself, not a copy of its rule.

    The two matrix loaders once shared a name and neither reached the guard; the
    identity check is what keeps a second implementation from appearing beside
    this one and drifting from it.
    """
    assert floors.gate_banked_signatures is gate_banked_signatures
    assert audit_lib.gate_banked_signatures is gate_banked_signatures
    assert readouts.load_signature_matrix is floors.load_signature_matrix
    assert not hasattr(audit_lib, "load_signature_matrix"), (
        "two readers with different return types under one name is how the "
        "ungated one stayed invisible"
    )


def test_floors_reader_gates_its_rows(tmp_path):
    sig_dir = write_bank(tmp_path, {0: "gpu", 1: "gpu"})
    X, names, gen_ids = floors.load_signature_matrix(sig_dir)
    assert X.shape == (2, len(FEATURES)) and names == list(FEATURES) and gen_ids == [0, 1]
    write_bank(tmp_path, {1: "other"})
    with pytest.raises(MixedLaneError):
        floors.load_signature_matrix(sig_dir)
    assert floors.load_signature_matrix(sig_dir, allow_mixed_lanes=True)[0].shape == (
        2,
        len(FEATURES),
    )


def test_merged_reader_gates_across_runs_and_stamps_the_lane(tmp_path):
    write_bank(tmp_path / "run_a", {0: "gpu"}, subdir="signatures_v3")
    write_bank(tmp_path / "run_b", {1: "gpu"}, subdir="signatures_v3")
    merged = audit_lib.load_merged_signature_matrix(["run_a", "run_b"], tmp_path)
    assert merged.X.shape == (2, len(FEATURES)) and merged.lane_id == "gpu"
    write_bank(tmp_path / "run_b", {1: "other"}, subdir="signatures_v3")
    with pytest.raises(MixedLaneError):
        audit_lib.load_merged_signature_matrix(["run_a", "run_b"], tmp_path)
    assert (
        audit_lib.load_merged_signature_matrix(
            ["run_a", "run_b"], tmp_path, allow_mixed_lanes=True
        ).lane_id
        is None
    ), "a bank with no single lane has no lane to stamp"


def test_steering_readout_gates_the_bank_it_normalizes(tmp_path):
    sig_dir = write_bank(tmp_path, {0: "gpu", 1: "gpu"})
    median = np.zeros(len(FEATURES))
    scale = np.ones(len(FEATURES))
    assert readouts.floor_z(sig_dir, median, scale).shape == (2, len(FEATURES))
    write_bank(tmp_path, {1: "other"})
    with pytest.raises(MixedLaneError):
        readouts.floor_z(sig_dir, median, scale)


def test_a_drawn_bank_is_readable_from_its_lane_alone():
    """The lane is the only place a drawn bank differs from a measured one.

    A reading over drawn vectors has to say so, and nothing in a vector's shape or
    values carries that — so the bank the fixture writer produces names itself, and
    the check that reads it is the one every reader uses.
    """
    assert lane_is_drawn(SYNTHETIC_LANE)
    assert not lane_is_drawn("cuda-h100-torch271")
    # An untagged bank is unknown provenance, which the gate above handles as its own
    # case; claiming it drawn would be inventing a fact about it.
    assert not lane_is_drawn(None)
