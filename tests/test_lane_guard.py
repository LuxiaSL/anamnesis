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

Pure mappings in, a verdict out. The second half of this file is the loader that
calls it — `signature_io.load_run4`, the only place in the package where a lane
identity is enforced over files rather than over dictionaries, which is where the
guard either does its job or is decoration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anamnesis.analysis.gauntlet.signature_io import load_run4
from anamnesis.analysis.lane_guard import MixedLaneError, require_single_lane


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


def test_loader_refuses_untagged_addon_on_gpu_lane(tmp_path):
    base, addon = tmp_path / "base", tmp_path / "addon"
    write_row(base, 0, "gpu")
    write_row(addon, 0, feature_key="features_tier2")
    with pytest.raises(MixedLaneError):
        load_run4(base, core_only=False, addon_dirs=[addon])


def test_loader_legacy_behavior_remains_available(tmp_path):
    write_row(tmp_path, 0)
    assert load_run4(tmp_path, core_only=False).lane_id is None


def test_tagged_dataset_cannot_silently_skip_missing_metadata(tmp_path):
    write_row(tmp_path, 0, "gpu")
    write_row(tmp_path, 1, "gpu")
    (tmp_path / "gen_001.json").unlink()
    with pytest.raises(MixedLaneError, match="without lane metadata"):
        load_run4(tmp_path, core_only=False)
