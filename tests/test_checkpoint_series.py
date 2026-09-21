"""The checkpoint series: its schema, its enumeration, and the drift it refuses.

Replaying a series through one model load is a performance argument, and the tests are
about the correctness conditions that argument rests on:

  * the series document has **one home**, so a series written by hand and one enumerated
    from a training directory are the same object and round-trip through the same schema;
  * enumeration is in step order with ``final`` last, which is the order a readout plots;
  * a **full-weight** checkpoint cannot be swapped into a loaded base, so it is refused by
    name with the command that can replay it;
  * more than one checkpoint **requires** pristine restore, because merge-and-unmerge
    drift accumulates along the series and reads as a training effect;
  * the pristine snapshot and restore are exact on the wrapped modules they cover.

The replay itself needs a model and adapters, so the loop is covered by its own
skip-with-a-reason; what runs here is everything up to it.

CPU only; no model, no adapters, no device.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from anamnesis.extraction.replay.checkpoint_series import (
    ADAPTER_CONFIG_NAME,
    CheckpointSeries,
    CheckpointSpec,
    pristine_snapshot,
    replay_series,
    require_pristine_restore,
    restore_pristine,
    series_from_adapter_dir,
)


def write_adapter_checkpoints(root: Path, steps: list[int], *, final: bool = True,
                              adapter: bool = True) -> Path:
    for step in steps:
        directory = root / f"checkpoint-{step:04d}"
        directory.mkdir(parents=True)
        if adapter:
            (directory / ADAPTER_CONFIG_NAME).write_text("{}")
    if final:
        directory = root / "final"
        directory.mkdir(parents=True)
        if adapter:
            (directory / ADAPTER_CONFIG_NAME).write_text("{}")
    return root


def test_the_series_round_trips_through_one_schema(tmp_path: Path) -> None:
    series = CheckpointSeries(
        checkpoints=[
            CheckpointSpec(label="arm-step-75", adapter_path=tmp_path / "a", run_dir=tmp_path / "r1"),
            CheckpointSpec(label="arm-final", adapter_path=tmp_path / "b", run_dir=tmp_path / "r2"),
        ]
    )
    path = series.write(tmp_path / "cells.json")
    document = json.loads(path.read_text())
    assert list(document) == ["checkpoints"]
    assert [row["label"] for row in document["checkpoints"]] == series.labels
    assert CheckpointSeries.load(path).labels == series.labels


def test_a_series_needs_at_least_one_checkpoint() -> None:
    with pytest.raises(ValueError):
        CheckpointSeries(checkpoints=[])


def test_enumeration_is_in_step_order_with_final_last(tmp_path: Path) -> None:
    write_adapter_checkpoints(tmp_path / "ckpts", [150, 25, 75])
    series = series_from_adapter_dir(
        tmp_path / "ckpts", arm="cat_dpo", run_root=tmp_path / "runs"
    )
    assert series.labels == [
        "cat_dpo-checkpoint-0025",
        "cat_dpo-checkpoint-0075",
        "cat_dpo-checkpoint-0150",
        "cat_dpo-final",
    ]
    assert [c.run_dir.name for c in series.checkpoints] == [
        "step-0025", "step-0075", "step-0150", "final",
    ]
    assert all(c.run_dir.parent.name == "cat_dpo" for c in series.checkpoints)


def test_final_can_be_left_out(tmp_path: Path) -> None:
    write_adapter_checkpoints(tmp_path / "ckpts", [10, 20])
    series = series_from_adapter_dir(
        tmp_path / "ckpts", arm="arm", run_root=tmp_path / "runs", include_final=False
    )
    assert series.labels == ["arm-checkpoint-0010", "arm-checkpoint-0020"]


def test_a_full_weight_checkpoint_is_refused_with_the_command_that_can_read_it(
    tmp_path: Path,
) -> None:
    write_adapter_checkpoints(tmp_path / "ckpts", [10, 20], adapter=False)
    with pytest.raises(SystemExit, match="FULL-WEIGHT"):
        series_from_adapter_dir(tmp_path / "ckpts", arm="arm", run_root=tmp_path / "runs")
    with pytest.raises(SystemExit, match="run_replay.py"):
        series_from_adapter_dir(tmp_path / "ckpts", arm="arm", run_root=tmp_path / "runs")


def test_an_empty_checkpoint_directory_is_refused(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no checkpoint"):
        series_from_adapter_dir(tmp_path / "empty", arm="arm", run_root=tmp_path / "runs")


def test_more_than_one_checkpoint_requires_the_pristine_restore() -> None:
    require_pristine_restore(1, False)
    require_pristine_restore(5, True)
    with pytest.raises(SystemExit, match="drifts along the"):
        require_pristine_restore(2, False)


class Wrapped(nn.Module):
    """A module shaped like an adapter-wrapped one: a base layer under a wrapper."""

    def __init__(self) -> None:
        super().__init__()
        self.base_layer = nn.Linear(4, 4, bias=False)


class Holder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wrapped = Wrapped()
        self.plain = nn.Linear(4, 4, bias=False)


def test_the_snapshot_covers_the_wrapped_modules_and_restores_them_exactly() -> None:
    model = Holder()
    snapshot = pristine_snapshot(model)
    assert list(snapshot) == ["wrapped"], "only wrapped modules have a base layer to restore"
    original = snapshot["wrapped"].clone()

    with torch.no_grad():
        model.wrapped.base_layer.weight.add_(1.0)
    assert not torch.allclose(model.wrapped.base_layer.weight, original)

    restore_pristine(model, snapshot)
    assert torch.allclose(model.wrapped.base_layer.weight, original), (
        "a merge is undone by restoring the snapshot, not by unmerging twice"
    )


def test_the_series_replay_refuses_before_it_touches_a_model(tmp_path: Path) -> None:
    """The refusal is checked first, so a drifting series never starts."""
    series = CheckpointSeries(
        checkpoints=[
            CheckpointSpec(label="a", adapter_path=tmp_path / "a", run_dir=tmp_path / "r1"),
            CheckpointSpec(label="b", adapter_path=tmp_path / "b", run_dir=tmp_path / "r2"),
        ]
    )
    with pytest.raises(SystemExit, match="drifts along the"):
        replay_series(
            surface=None,  # type: ignore[arg-type]
            series=series,
            calibration=(None, None, None),
            manifest_path=tmp_path / "manifest.json",
            pristine_restore=False,
        )
