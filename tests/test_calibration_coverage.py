"""Calibration coverage is the rows a fit filled, not the width of the table.

A fit allocates a row for every position it might reach and leaves the ones it did
not reach — or reached no more than
:data:`anamnesis.extraction.calibration.POSITION_COUNT_FLOOR` times — at exact zeros.
Subtracting a zero row corrects nothing, so a pass that reads the table's width as
its coverage computes uncorrected features at those positions under the corrected
features' names. The cases below pin
:func:`anamnesis.extraction.calibration.positions_calibrated` on a zero-tailed
table, and pin that each place which refuses an uncalibrated position — the fast
lane's span resolution, the lane's own span check, and the per-layer basis fit —
measures coverage with it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from test_fast_lane_equivalence import ANCHOR_LAYER_PLAN, tiny_loaded
from anamnesis.config import FeaturePipelineConfig
from anamnesis.extraction.calibration import (
    POSITION_COUNT_FLOOR,
    POSITION_COUNTS_KEY,
    POSITIONAL_MEANS_KEY,
    POSITIONAL_MEANS_NAME,
    load_position_counts,
    positions_calibrated,
)
from anamnesis.extraction.calibration_fit import fit_per_layer_basis
from anamnesis.extraction.fast.features import GpuFeatureLane
from anamnesis.extraction.fast.runtime import LaneSpan, resolve_lane_spans

LAYERS = 4
WIDTH = 16
FILLED = 10
UNITS = 32


def zero_tailed_means(width: int = WIDTH, filled: int = FILLED) -> np.ndarray:
    """A ``[layer, position, unit]`` table whose rows past ``filled`` are zeros."""
    means = np.random.default_rng(5).normal(0, 0.01, size=(LAYERS, width, UNITS))
    means[:, filled:] = 0.0
    return means.astype(np.float32)


def counts_for(width: int = WIDTH, filled: int = FILLED) -> np.ndarray:
    counts = np.zeros((LAYERS, width), dtype=np.int64)
    counts[:, :filled] = POSITION_COUNT_FLOOR + 1
    return counts


# ── the measure ────────────────────────────────────────────────────────────────


def test_a_full_table_is_covered_to_its_width() -> None:
    assert positions_calibrated(zero_tailed_means(filled=WIDTH)) == WIDTH


def test_a_zero_tail_shortens_coverage_to_the_last_filled_row() -> None:
    means = zero_tailed_means()
    assert means.shape[1] == WIDTH
    assert positions_calibrated(means) == FILLED


def test_counts_decide_coverage_when_given() -> None:
    assert positions_calibrated(zero_tailed_means(), counts_for()) == FILLED


def test_a_row_counted_at_the_floor_is_not_filled() -> None:
    """The fit leaves such a row at zeros, so a nonzero count is not coverage."""
    counts = counts_for()
    counts[:, FILLED : FILLED + 3] = POSITION_COUNT_FLOOR
    assert positions_calibrated(zero_tailed_means(), counts) == FILLED


def test_a_position_is_filled_only_when_every_layer_is() -> None:
    means = zero_tailed_means()
    means[2, FILLED - 1] = 0.0
    assert positions_calibrated(means) == FILLED - 1
    counts = counts_for()
    counts[1, FILLED - 1] = 0
    assert positions_calibrated(zero_tailed_means(), counts) == FILLED - 1


def test_an_interior_gap_does_not_shorten_coverage() -> None:
    means = zero_tailed_means()
    means[:, 3] = 0.0
    assert positions_calibrated(means) == FILLED


def test_an_empty_table_covers_nothing() -> None:
    assert positions_calibrated(np.zeros((LAYERS, WIDTH, UNITS), dtype=np.float32)) == 0
    assert positions_calibrated(
        np.zeros((LAYERS, WIDTH, UNITS), dtype=np.float32),
        np.zeros((LAYERS, WIDTH), dtype=np.int64),
    ) == 0


def test_counts_of_another_shape_are_refused() -> None:
    with pytest.raises(ValueError, match="do not match"):
        positions_calibrated(zero_tailed_means(), counts_for(width=WIDTH + 1))
    with pytest.raises(ValueError, match="layer, position, unit"):
        positions_calibrated(np.zeros((WIDTH, UNITS), dtype=np.float32))


def test_counts_are_read_from_the_archive_a_fit_writes(tmp_path: Path) -> None:
    np.savez(
        tmp_path / POSITIONAL_MEANS_NAME,
        **{POSITIONAL_MEANS_KEY: zero_tailed_means(), POSITION_COUNTS_KEY: counts_for()},
    )
    counts = load_position_counts(tmp_path)
    assert counts is not None and counts.dtype == np.int64
    assert np.array_equal(counts, counts_for())


def test_counts_are_none_where_the_archive_has_none(tmp_path: Path) -> None:
    assert load_position_counts(tmp_path) is None
    np.savez(tmp_path / POSITIONAL_MEANS_NAME, **{POSITIONAL_MEANS_KEY: zero_tailed_means()})
    assert load_position_counts(tmp_path) is None


# ── the sites that refuse an uncalibrated position ─────────────────────────────


def test_span_resolution_refuses_a_span_inside_the_width_but_past_the_filled_rows() -> None:
    """A span ending inside the zero tail fits the table and is still uncalibrated."""
    tokens = list(range(1, WIDTH))
    entries = {"0": {"input_ids": tokens, "prompt_length": 4}}
    span = LaneSpan(gen_id=0, input_ids=tokens, prompt_length=4)
    assert span.end - 2 < WIDTH, "the span fits the table's width"

    means = zero_tailed_means()
    resolve_lane_spans(entries, [0], positions_calibrated=WIDTH)
    with pytest.raises(ValueError, match="outside supported span/calibration"):
        resolve_lane_spans(entries, [0], positions_calibrated=positions_calibrated(means))


def _lane(positional_means: np.ndarray) -> GpuFeatureLane:
    families = FeaturePipelineConfig(
        include_core_blocks=True,
        enable_residual_trajectory=True,
        enable_attention_flow=True,
        enable_gate_features=True,
        enable_per_head=True,
        enable_value_geometry=True,
        enable_qk_geometry=True,
        enable_kv_cka=True,
        trajectory_layers=[8, 16, 20, 24, 28],
        contrastive_layers=[8, 16, 20, 24, 28],
    )
    return GpuFeatureLane(
        ANCHOR_LAYER_PLAN,
        families,
        [],
        positional_means,
        np.zeros((5, UNITS), np.float32),
        np.zeros(UNITS, np.float32),
        device="cpu",
        calibration_sha256="a" * 64,
    )


def test_the_lane_measures_its_coverage_by_filled_rows() -> None:
    lane = _lane(zero_tailed_means())
    assert lane.positions_calibrated == FILLED


def test_the_lane_refuses_a_span_reaching_the_zero_tail() -> None:
    loaded = tiny_loaded()
    lane = _lane(zero_tailed_means())
    tokens = [1 + (i % 60) for i in range(WIDTH - 2)]
    assert len(tokens) - 2 < WIDTH, "the table's width would have admitted this span"
    with torch.no_grad(), pytest.raises(ValueError, match="does not cover span"):
        lane.replay_span(loaded, tokens, 4, len(tokens))


def test_the_basis_fit_drops_samples_in_the_zero_tail() -> None:
    """A zero row subtracts nothing, so a sample there would be fitted raw."""
    means = zero_tailed_means()
    rng = np.random.default_rng(9)
    inside = [(1, position, rng.normal(size=UNITS).astype(np.float32)) for position in range(FILLED)]
    outside = [
        (1, position, np.full(UNITS, 1e6, dtype=np.float32))
        for position in range(FILLED, WIDTH)
    ]
    with_tail = fit_per_layer_basis(inside + outside, means, [1], n_components=2)
    without = fit_per_layer_basis(inside, means, [1], n_components=2)
    assert np.array_equal(with_tail[1]["components"], without[1]["components"])
    assert np.array_equal(with_tail[1]["mean"], without[1]["mean"])
