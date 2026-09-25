"""A pass refuses a position its calibration does not fill, before computing anything there.

The correction clamps a position to the table's width, and a row no fit reached is
zeros, so a position past the filled rows is corrected by nothing and its feature is
an uncorrected quantity under the corrected name — with no error anywhere. The refusal
is made where a pass knows how far it will read: per banked sequence on the replay
path (the fast lane has made it since its span check read filled rows), and from the
longest prompt and the token budget before ``run_extraction`` samples. The anchor's
clamp is unchanged, so no banked feature moves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from synthetic_runtime import loaded_tiny_model
from test_fail_closed import (
    TINY_IDS,
    TINY_PLAN,
    TINY_PROMPT_LENGTH,
    tiny_extraction,
    tiny_families,
    write_manifest,
)
from anamnesis.extraction.calibration import (
    POSITION_COUNT_FLOOR,
    PositionsUncovered,
    require_positions_covered,
)
from anamnesis.extraction.replay.cell import ReplaySurface, cell_shortfall, replay_cell

LAST_READ = len(TINY_IDS) - 2
"""The last position a replay of the tiny sequence reads a state at."""


def _means(depth: int, width: int, units: int, filled: int) -> np.ndarray:
    means = np.zeros((depth, width, units), dtype=np.float32)
    means[:, :filled] = np.random.default_rng(3).normal(0, 0.01, size=(depth, filled, units))
    return means


# ── the rule ───────────────────────────────────────────────────────────────────


def test_a_position_inside_the_filled_rows_passes() -> None:
    require_positions_covered(_means(3, 20, 4, filled=12), 11, what="a pass")


def test_the_first_unfilled_row_is_refused_and_named() -> None:
    with pytest.raises(PositionsUncovered, match=r"reads position 12, and the positional means fill positions 0\.\.11"):
        require_positions_covered(_means(3, 20, 4, filled=12), 12, what="a pass")


def test_counts_decide_the_filled_rows_when_given() -> None:
    counts = np.zeros((3, 20), dtype=np.int64)
    counts[:, :8] = POSITION_COUNT_FLOOR + 1
    with pytest.raises(PositionsUncovered, match=r"0\.\.7"):
        require_positions_covered(_means(3, 20, 4, filled=12), 9, what="a pass", counts=counts)


def test_no_means_is_nothing_to_check() -> None:
    require_positions_covered(None, 10_000, what="an uncorrected pass")


# ── the replay path ────────────────────────────────────────────────────────────


def _replay(tmp_path: Path, filled: int):
    loaded, _ = loaded_tiny_model(TINY_PLAN)
    config = loaded.model.config
    means = _means(config.num_hidden_layers + 1, 32, config.hidden_size, filled)
    manifest = write_manifest(tmp_path / "run", [0, 1])
    surface = ReplaySurface(loaded=loaded, extraction=tiny_extraction(), families=tiny_families())
    result = replay_cell(
        surface, (means, None, None), tmp_path / "run", manifest,
        signatures_subdir="signatures", save_raw=False, label="t",
    )
    return result, cell_shortfall(result, manifest, command="test", label="t")


def test_a_covered_cell_replays_every_generation(tmp_path: Path) -> None:
    result, shortfall = _replay(tmp_path, filled=LAST_READ + 1)
    assert shortfall.ok, shortfall
    assert TINY_PROMPT_LENGTH < LAST_READ


def test_a_cell_reaching_past_the_filled_rows_is_short_and_says_why(tmp_path: Path) -> None:
    result, shortfall = _replay(tmp_path, filled=LAST_READ)
    assert not shortfall.ok
    assert set(result.failed) == {0, 1}
    for message in result.failed.values():
        assert f"reads position {LAST_READ}" in message
        assert "--required-through" in message
    assert not list((tmp_path / "run" / "signatures").glob("gen_*.npz")), "nothing computed there"


# ── the extraction path ────────────────────────────────────────────────────────


def test_extraction_refuses_before_sampling_when_its_budget_reaches_past_the_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The longest prompt plus the budget is known before a token is sampled, so the
    refusal comes then — not after a pass whose tail features are uncorrected."""
    import json

    from test_loaded_model_seam import TINY_ROW, save_tiny_checkpoint
    from test_onboarding import save_tiny_tokenizer
    from anamnesis.extraction import generation_runner
    from anamnesis.extraction.calibration import POSITIONAL_MEANS_KEY, POSITIONAL_MEANS_NAME
    from anamnesis.scripts import run_extraction

    checkpoint = tmp_path / "checkpoint"
    loaded = save_tiny_checkpoint(checkpoint)
    save_tiny_tokenizer(checkpoint, loaded.model.config.vocab_size)
    outputs = tmp_path / "outputs"
    calib = outputs / "calibration" / "tiny"
    calib.mkdir(parents=True)
    np.savez(
        calib / POSITIONAL_MEANS_NAME,
        **{POSITIONAL_MEANS_KEY: _means(4, 64, 32, filled=10)},
    )
    row = {**TINY_ROW, "model_id": str(checkpoint), "calibration_dir": "calibration/tiny"}
    registry = tmp_path / "models.json"
    registry.write_text(json.dumps({"presets": {"tiny-llama": row}}))
    monkeypatch.setenv("ANAMNESIS_MODELS", str(registry))
    monkeypatch.setenv("ANAMNESIS_OUTPUTS", str(outputs))

    def sampled(*args, **kwargs):
        raise AssertionError("a generation was sampled against a calibration that cannot correct it")

    monkeypatch.setattr(generation_runner, "run_experiment", sampled)
    with pytest.raises(SystemExit, match=r"positional means fill positions 0\.\.9"):
        run_extraction.main(
            ["--model", "tiny-llama", "--run-name", "short", "--smoke-test", "--no-pca"]
        )
