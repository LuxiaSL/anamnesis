"""A calibration build that is asked to cover a position, and refuses to be written short.

Three properties are pinned here, each a way a calibration could look complete and
not be:

* **The gate reads filled rows.** A table wide enough to hold a position but whose
  generations never reached it has zeros there, and subtracting zeros corrects
  nothing. :func:`anamnesis.extraction.calibration_fit.require_coverage` measures by
  :func:`anamnesis.extraction.calibration.positions_calibrated`, and a fit that falls
  short writes neither artifact.
* **A stop token can be suppressed.** An instruct checkpoint ends where its answer
  ends; ``suppress_eos`` is what lets a generation run to its budget, so a late
  position gets data at all. It is exercised against a real decoder with random
  weights, with every token a stop token, so the difference is the whole budget.
* **Every write leaves a receipt** naming the settings, the coverage reached and the
  bytes written.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_calibration_fit import narrow_preset, synthetic_states
from test_fast_lane_equivalence import tiny_loaded
from anamnesis.extraction.calibration import (
    PCA_MODEL_NAME,
    POSITIONAL_MEANS_NAME,
)
from anamnesis.extraction.calibration_fit import (
    PROMPT_HEADROOM,
    CalibrationBuildReceipt,
    CalibrationFitError,
    CoverageShortfall,
    build_receipt,
    build_receipt_path,
    fit_calibration,
    generate_prompt_states,
    generation_settings,
    prompts_digest,
    require_coverage,
    write_build_receipt,
    write_calibration,
)
from anamnesis.provenance import digest_of_shas, file_sha
from anamnesis.scripts import run_calibration

UNITS = 12
PROMPT_LENGTH = 4
STEPS = 6
REACHED = PROMPT_LENGTH + STEPS
"""Positions 0..9 are filled by every prompt; the table is wider than that."""


def _fit(max_positions: int | None = None, existing_means: np.ndarray | None = None):
    preset = narrow_preset("3b", UNITS)
    settings = generation_settings(preset, max_new_tokens=STEPS)
    fit = fit_calibration(
        synthetic_states(
            "3b", n_prompts=8, prompt_length=PROMPT_LENGTH, n_steps=STEPS, units=UNITS
        ),
        preset=preset,
        settings=settings,
        n_components=3,
        pooled=True,
        existing_means=existing_means,
        max_positions=max_positions,
    )
    return fit, settings


# ── the table and its coverage ─────────────────────────────────────────────────


def test_the_table_width_defaults_to_the_budget_plus_headroom_and_can_be_set() -> None:
    fit, _ = _fit()
    assert fit.positional_means.shape[1] == STEPS + PROMPT_HEADROOM
    wide, _ = _fit(max_positions=64)
    assert wide.positional_means.shape[1] == 64
    with pytest.raises(CalibrationFitError, match="not a table width"):
        _fit(max_positions=0)


def test_coverage_is_the_filled_rows_not_the_width() -> None:
    fit, _ = _fit(max_positions=64)
    assert fit.positions_calibrated == REACHED
    assert fit.positional_means.shape[1] == 64


def test_a_fit_reaching_the_required_position_passes_the_gate() -> None:
    fit, _ = _fit(max_positions=64)
    require_coverage(fit, REACHED - 1)


def test_a_fit_short_of_the_required_position_is_refused_and_writes_nothing(
    tmp_path: Path,
) -> None:
    fit, _ = _fit(max_positions=64)
    means_path, basis_path = tmp_path / POSITIONAL_MEANS_NAME, tmp_path / PCA_MODEL_NAME
    with pytest.raises(CoverageShortfall, match=f"position {REACHED} is required"):
        write_calibration(fit, means_path, basis_path, required_through=REACHED)
    assert not means_path.exists() and not basis_path.exists()


def test_a_required_position_outside_the_table_names_the_width() -> None:
    fit, _ = _fit()
    with pytest.raises(CoverageShortfall, match="raise max_positions"):
        require_coverage(fit, STEPS + PROMPT_HEADROOM)
    with pytest.raises(CoverageShortfall, match="not a position"):
        require_coverage(fit, -1)


def test_reused_means_are_measured_off_the_table() -> None:
    """A reused table has no counts in this pass, so its rows are what is read."""
    first, _ = _fit(max_positions=64)
    again, _ = _fit(existing_means=first.positional_means)
    assert again.means_refitted is False
    assert again.positions_calibrated == REACHED


# ── the receipt ────────────────────────────────────────────────────────────────


def test_a_written_fit_leaves_a_receipt_naming_its_bytes(tmp_path: Path) -> None:
    fit, settings = _fit(max_positions=64)
    means_path, basis_path = tmp_path / POSITIONAL_MEANS_NAME, tmp_path / PCA_MODEL_NAME
    write_calibration(fit, means_path, basis_path, required_through=REACHED - 1)
    prompts = ("one", "two")
    receipt = build_receipt(
        fit,
        model="3b",
        model_id="some/checkpoint",
        prompts=prompts,
        settings=settings,
        suppress_eos=True,
        chat_template=False,
        pooled=True,
        n_components=3,
        means_path=means_path,
        basis_path=basis_path,
        required_through=REACHED - 1,
    )
    target = write_build_receipt(receipt, basis_path)
    assert target == build_receipt_path(basis_path) == tmp_path / "pca_model.build.json"

    read = CalibrationBuildReceipt.model_validate(json.loads(target.read_text()))
    assert read == receipt
    files = {POSITIONAL_MEANS_NAME: file_sha(means_path), PCA_MODEL_NAME: file_sha(basis_path)}
    assert read.files == files and read.calibration_sha256 == digest_of_shas(files)
    assert read.prompt_set_sha256 == prompts_digest(prompts) and read.n_prompts == 2
    assert read.suppress_eos is True and read.max_new_tokens == STEPS
    assert read.chat_template is False
    assert read.coverage.positions_calibrated == REACHED
    assert read.coverage.table_positions == 64
    assert read.coverage.trailing_zero_rows == 64 - REACHED
    assert read.coverage.measured_by == "pos_counts"
    assert read.coverage.required_through == REACHED - 1


def test_the_prompt_digest_is_order_sensitive() -> None:
    assert prompts_digest(["a", "b"]) != prompts_digest(["b", "a"])


# ── the stop token ─────────────────────────────────────────────────────────────


class _BareTokenizer:
    """Stands in for a base model's tokenizer: no chat template, a fixed prompt."""

    chat_template = None

    def __call__(self, text: str, return_tensors: str) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([[1, 2, 3, 4]])}


@pytest.mark.parametrize("suppress_eos", [False, True])
def test_suppressing_the_stop_token_runs_each_generation_to_its_budget(
    suppress_eos: bool,
) -> None:
    loaded = tiny_loaded()
    loaded.tokenizer = _BareTokenizer()
    loaded.disable_hooks()
    budget = 5
    settings = generation_settings("3b", max_new_tokens=budget).model_copy(
        update={"eos_token_ids": list(range(64))}
    )
    (states,) = list(
        generate_prompt_states(loaded, ["prompt"], settings, suppress_eos=suppress_eos)
    )
    # The prefill yields the first token, so a budget of n leaves n - 1 decode steps.
    assert len(states.steps) == (budget - 1 if suppress_eos else 0)


# ── the command ────────────────────────────────────────────────────────────────


def test_the_command_refuses_a_required_position_outside_its_table_before_loading(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="--max-positions"):
        run_calibration.main(
            [
                "--model", "8b", "--out-dir", str(tmp_path),
                "--max-positions", "100", "--required-through", "100",
            ]
        )


def test_the_dry_run_reports_the_coverage_settings(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    run_calibration.main(
        [
            "--model", "8b", "--out-dir", str(tmp_path), "--dry-run",
            "--suppress-eos", "--max-positions", "4096", "--required-through", "4095",
        ]
    )
    printed = capsys.readouterr().out
    assert "stop tokens honoured: no, suppressed" in printed
    assert "means table: 4096 positions" in printed
    assert "required through position: 4095" in printed
