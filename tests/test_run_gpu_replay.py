"""The replay CLI's qualification boundary, asserted on the argument parser.

`run_gpu_replay.py` is qualified for one configuration: dense 3B/8B Llama, one full
teacher-forced pass per span, the complete battery, one device. That is what the lane
tests and the equivalence suite cover, so it is what the entry point accepts without
ceremony — a covered command line runs.

The boundary is the other half, and it is enforced by absence: adapters, activation
interventions and batched submission each change what a forward pass is, and none of
them has an argument here. A command line naming one is rejected rather than
reinterpreted as the covered case, which is the failure mode that matters — a caller
who asks for an adapter and silently gets a plain pass has banked a mislabelled bank.
These tests pin that refusal so a later flag cannot widen the scope without also
widening the qualification.

The metadata reader is tested for the one thing that is not shape: generation labels
from the source run (mode, topic and their indices) survive into the replayed bank, so
a replayed signature is still attributable to the condition it came from, and a
duplicate generation id is refused rather than silently resolved to the last copy.

No model, no device: argparse and a JSON file.
"""

from __future__ import annotations

import json

import pytest

from anamnesis.scripts.run_gpu_replay import parser, read_generation_metadata

BASE = [
    "--model",
    "3b",
    "--model-path",
    "/model",
    "--calib-dir",
    "/calibration",
    "--manifest",
    "/manifest.json",
    "--output",
    "/new-output",
]


def test_the_qualified_configuration_needs_no_acknowledgement():
    args = parser().parse_args(BASE)
    assert args.model == "3b"
    assert args.gen_ids is None


@pytest.mark.parametrize(
    "extra",
    [
        ["--adapter-path", "/adapter"],
        ["--inject-npz", "/write.npz"],
        ["--batch-size", "2"],
    ],
)
def test_unqualified_execution_modes_are_not_accepted(extra):
    with pytest.raises(SystemExit):
        parser().parse_args(BASE + extra)


def test_source_metadata_labels_are_preserved(tmp_path):
    path = tmp_path / "metadata.json"
    record = dict(
        generation_id=7, mode="linear", topic="topic", mode_idx=0, topic_idx=2
    )
    path.write_text(json.dumps(dict(generations=[record])))
    assert read_generation_metadata(path) == {7: record}
    path.write_text(json.dumps(dict(generations=[record, record])))
    with pytest.raises(ValueError, match="duplicate"):
        read_generation_metadata(path)
