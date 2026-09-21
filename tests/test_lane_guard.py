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

Pure mappings in, a verdict out; the loaders that call it are the analysis layer's.
"""

from __future__ import annotations

import pytest

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
