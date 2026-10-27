"""The CPU path refuses a position its positional means do not fill (RULING §57.1/§60).

A calibration's means fill the positions its fit reached; past them the rows are
zeros, and the old correction clamped to the table's width and subtracted those
zeros, so a row longer than the calibration came out uncorrected under the
corrected feature names. Every means-indexing site now refuses the first uncovered
position, and :func:`truncate_to_calibration` is the one, recorded way to compute a
row that runs past reach.
"""

from __future__ import annotations

import numpy as np
import pytest

from anamnesis.config import ExtractionConfig
from anamnesis.extraction.calibration import PositionsUncovered
from anamnesis.extraction.state_extractor import (
    RawGenerationData,
    extract_all_features,
    truncate_to_calibration,
)
from anamnesis.extraction.state_extractor_reference import (
    extract_all_features as reference_extract_all,
)

LAYERS, HIDDEN, HEADS, KV, HEAD_DIM, VOCAB = 4, 16, 4, 2, 8, 64
PROMPT, STEPS = 6, 10


def _config(**blocks: bool) -> ExtractionConfig:
    on = dict(enable_norms_and_output_stats=True, enable_attention_and_deltas=True,
              enable_cache_and_keys=True, enable_residual_pca=False, enable_knnlm_baseline=False)
    on.update(blocks)
    return ExtractionConfig(sampled_layers=[0, 1, 2, 3], pca_layers=[1, 2],
                            early_layer_cutoff=1, late_layer_cutoff=3, **on)


def _row(steps: int = STEPS, reach: int | None = None, width: int = PROMPT + STEPS + 8,
         seed: int = 0) -> RawGenerationData:
    """A synthetic row whose means fill positions 0..reach-1 of a ``width``-wide table."""
    rng = np.random.default_rng(seed)
    means = (rng.normal(size=(LAYERS + 1, width, HIDDEN)) * 0.01).astype(np.float32)
    if reach is not None:
        means[:, reach:, :] = 0.0
    attentions = []
    for t in range(steps):
        a = rng.random((LAYERS, HEADS, PROMPT + t + 1)).astype(np.float32)
        attentions.append(a / a.sum(axis=2, keepdims=True))
    return RawGenerationData(
        hidden_states=[rng.normal(size=(LAYERS + 1, HIDDEN)).astype(np.float32) for _ in range(steps)],
        attentions=attentions,
        logits=[rng.normal(size=VOCAB).astype(np.float32) for _ in range(steps)],
        chosen_token_ids=rng.integers(0, VOCAB, size=steps).astype(np.float32),
        pre_rope_keys={l: [rng.normal(size=(KV, HEAD_DIM)).astype(np.float32) for _ in range(steps)]
                       for l in range(LAYERS)},
        prompt_length=PROMPT,
        positional_means=means,
    )


def test_a_row_inside_its_calibration_extracts() -> None:
    extract_all_features(_row(reach=PROMPT + STEPS), _config())


@pytest.mark.parametrize("blocks, what", [
    (dict(enable_attention_and_deltas=False, enable_cache_and_keys=False), "the activation-norm block"),
    (dict(enable_norms_and_output_stats=False, enable_cache_and_keys=False), "the residual-delta block"),
])
def test_each_block_refuses_one_position_past_reach(blocks: dict[str, bool], what: str) -> None:
    # The last generated position is PROMPT + STEPS - 1; a reach one short of it leaves
    # exactly that position uncovered, inside the table's width.
    row = _row(reach=PROMPT + STEPS - 1)
    with pytest.raises(PositionsUncovered, match=rf"{what} reads position {PROMPT + STEPS - 1}, "
                                                 rf"and the positional means fill positions 0\.\.{PROMPT + STEPS - 2}"):
        extract_all_features(row, _config(**blocks))


def test_the_reference_extractor_refuses_the_same_row() -> None:
    with pytest.raises(PositionsUncovered):
        reference_extract_all(_row(reach=PROMPT + STEPS - 1), _config())


def test_truncation_is_explicit_recorded_and_equals_the_shorter_row() -> None:
    reach = PROMPT + STEPS - 3
    row = _row(reach=reach)
    cut, record = truncate_to_calibration(row)
    assert record.as_dict() == dict(prompt_length=PROMPT, reach=reach, original_steps=STEPS,
                                    kept_steps=STEPS - 3, dropped_steps=3)
    assert len(row.hidden_states) == STEPS  # the input is not modified
    assert all(len(v) == STEPS - 3 for v in cut.pre_rope_keys.values())
    shorter = _row(reach=reach)  # the same draws, built for comparison and then cut by hand
    shorter.hidden_states = shorter.hidden_states[:STEPS - 3]
    shorter.attentions = shorter.attentions[:STEPS - 3]
    shorter.logits = shorter.logits[:STEPS - 3]
    shorter.chosen_token_ids = shorter.chosen_token_ids[:STEPS - 3]
    shorter.pre_rope_keys = {k: v[:STEPS - 3] for k, v in shorter.pre_rope_keys.items()}
    a = extract_all_features(cut, _config())
    b = extract_all_features(shorter, _config())
    assert a.feature_names == b.feature_names
    assert np.array_equal(a.features, b.features)


def test_truncating_a_row_that_fits_changes_nothing() -> None:
    row = _row(reach=PROMPT + STEPS)
    cut, record = truncate_to_calibration(row)
    assert cut is row and record.dropped_steps == 0


def test_truncation_refuses_when_too_little_is_left_or_there_is_nothing_to_cut_by() -> None:
    with pytest.raises(ValueError, match="recalibrate instead of truncating"):
        truncate_to_calibration(_row(reach=PROMPT + 1))
    row = _row()
    row.positional_means = None
    with pytest.raises(ValueError, match="needs the positional means"):
        truncate_to_calibration(row)
