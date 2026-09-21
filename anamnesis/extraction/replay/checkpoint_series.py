"""Replaying one manifest through a series of checkpoints, on one model load.

A training run leaves a trail of checkpoints, and the question worth asking of it is
what changed *between* them — so the same banked token sequences are replayed through
each one and the signatures compared along the series. The naive way to do that pays
the base model's load time once per checkpoint per worker, which at seventeen
gigabytes is most of the wall clock.

For an adapter series it is avoidable. Each worker loads the base model **once**,
wraps it, and walks its checkpoint list:

    load the adapter -> select it -> restore pristine weights -> merge -> replay ->
    unmerge -> delete the adapter -> next

Merging bakes ``W' = W + s·B·A`` into the wrapped layer's own weight tensor while the
module objects stay alive, which is why the capture hooks — registered on those
modules at load time — stay attached across the whole series. That is the property
the whole arrangement rests on.

**Pristine restore is required, not advised.** Merge followed by unmerge is not
exactly the identity in finite precision, so a long series accumulates drift that
looks like a training effect and is arithmetic. Snapshotting the wrapped weights once
and restoring before each merge removes it, and a series of more than one checkpoint
refuses to run without it.

**A full-weight checkpoint is not an adapter** and cannot be swapped this way. The
series builder refuses a directory whose checkpoints carry no adapter configuration,
and says which path to use instead: a full-weight series is replayed one model load
per checkpoint, which is the cost of what it is.

The wrapper is ``peft``'s, imported at the point of use rather than at module
import: the schema, the enumeration and the pristine-restore refusal are all
readable and testable where it is not installed, and only the two functions that
touch a wrapped model need it.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from anamnesis.extraction.replay.cell import CellResult, ReplaySurface, replay_cell

logger = logging.getLogger(__name__)

ADAPTER_CONFIG_NAME = "adapter_config.json"
"""The file that tells an adapter checkpoint from a full-weight one."""

CHECKPOINT_GLOB = "checkpoint-*"
FINAL_NAME = "final"
CHECKPOINT_STEP_RE = re.compile(r"checkpoint-(\d+)")

SERIES_KEY = "checkpoints"
"""The key the series document carries its rows under."""


class CheckpointSpec(BaseModel):
    """One checkpoint of a series: what to load, and where its signatures go."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(min_length=1, description="How this checkpoint is named in readouts")
    adapter_path: Path = Field(description="Directory holding the adapter weights")
    run_dir: Path = Field(description="Where this checkpoint's signatures are written")


class CheckpointSeries(BaseModel):
    """A series, in replay order, as one banked document.

    The document is the interface between the builder and the replay: one schema, one
    home, so a series written by hand and one enumerated from a training directory are
    the same object.
    """

    model_config = ConfigDict(extra="forbid")

    checkpoints: list[CheckpointSpec] = Field(min_length=1)

    @property
    def labels(self) -> list[str]:
        return [checkpoint.label for checkpoint in self.checkpoints]

    def document(self) -> dict[str, Any]:
        """The series as JSON: relative order preserved, paths as strings."""
        return {
            SERIES_KEY: [
                {
                    "label": checkpoint.label,
                    "adapter_path": str(checkpoint.adapter_path),
                    "run_dir": str(checkpoint.run_dir),
                }
                for checkpoint in self.checkpoints
            ]
        }

    def write(self, path: Path) -> Path:
        """Bank the series document; return where it landed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.document(), indent=1), encoding="utf-8")
        logger.info(f"wrote {len(self.checkpoints)} checkpoints -> {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> CheckpointSeries:
        """Read a banked series document."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(checkpoints=[CheckpointSpec(**row) for row in payload[SERIES_KEY]])


def series_from_adapter_dir(
    checkpoint_dir: Path,
    *,
    arm: str,
    run_root: Path,
    include_final: bool = True,
) -> CheckpointSeries:
    """Enumerate a training directory's adapter checkpoints, in step order.

    ``final`` sorts last because it is the end of the series rather than a step
    number, and each checkpoint is pointed at a per-step run directory under the
    cohort root, so the series and the replays it produces share one layout.

    Raises
    ------
    SystemExit
        When the directory holds no checkpoints, or when one of them is full-weight.
        The second names the path to use instead, because a full-weight checkpoint in
        an adapter series is a configuration mistake with a working alternative and
        not a missing feature.
    """
    checkpoint_dir = Path(checkpoint_dir)
    steps = sorted(
        (d for d in checkpoint_dir.glob(CHECKPOINT_GLOB) if d.is_dir()),
        key=lambda d: int(CHECKPOINT_STEP_RE.search(d.name).group(1)),  # type: ignore[union-attr]
    )
    if include_final and (checkpoint_dir / FINAL_NAME).is_dir():
        steps.append(checkpoint_dir / FINAL_NAME)
    if not steps:
        raise SystemExit(f"no {CHECKPOINT_GLOB} or {FINAL_NAME} directory under {checkpoint_dir}")

    specs: list[CheckpointSpec] = []
    for step in steps:
        if not (step / ADAPTER_CONFIG_NAME).exists():
            raise SystemExit(
                f"{step} has no {ADAPTER_CONFIG_NAME} — this is a FULL-WEIGHT checkpoint. "
                f"Replay it with the one-load-per-checkpoint path (run_replay.py "
                f"--model-path=<checkpoint>); the swap driver handles adapters only."
            )
        specs.append(
            CheckpointSpec(
                label=f"{arm}-{step.name}",
                adapter_path=step,
                run_dir=Path(run_root) / arm / step.name.replace("checkpoint-", "step-"),
            )
        )
    return CheckpointSeries(checkpoints=specs)


def require_pristine_restore(n_checkpoints: int, pristine_restore: bool) -> None:
    """Refuse a multi-checkpoint swap that would accumulate merge drift.

    Raises
    ------
    SystemExit
        When more than one checkpoint would be merged without restoring the pristine
        weights between them. A drift that grows along the series is indistinguishable
        from the effect the series is measuring, which is why this is a refusal rather
        than a warning.
    """
    if n_checkpoints > 1 and not pristine_restore:
        raise SystemExit(
            f"refusing {n_checkpoints} checkpoints without pristine restore: repeated "
            f"merge and unmerge without restoring the base weights drifts along the "
            f"swap sequence, and the drift reads as a training effect. Keep the restore "
            f"(it is the default), or run one checkpoint per job."
        )


def pristine_snapshot(wrapped_model: Any) -> dict[str, Any]:
    """Clone every wrapped layer's base weight, so a merge can be undone exactly."""
    import torch

    with torch.no_grad():
        return {
            name: module.base_layer.weight.detach().clone()
            for name, module in wrapped_model.named_modules()
            if hasattr(module, "base_layer")
        }


def restore_pristine(wrapped_model: Any, snapshot: dict[str, Any]) -> None:
    """Write the snapshot back, in place, before the next merge."""
    import torch

    with torch.no_grad():
        for name, module in wrapped_model.named_modules():
            if hasattr(module, "base_layer") and name in snapshot:
                module.base_layer.weight.copy_(snapshot[name])


def replay_series(
    surface: ReplaySurface,
    series: CheckpointSeries,
    calibration: tuple[Any, Any, Any],
    manifest_path: Path,
    *,
    gen_ids: Sequence[int] | None = None,
    signatures_subdir: str = "signatures_v3",
    raw_subdir: str = "raw_tensors_v3",
    save_raw: bool = True,
    resume: bool = True,
    pristine_restore: bool = True,
    label: str = "w",
) -> list[CellResult]:
    """Walk the series on one loaded model, replaying the manifest through each.

    ``surface`` arrives already wrapped for adapters: the first checkpoint's adapter
    is what the wrapper was built from, which is why it is loaded by the caller and
    the rest are loaded here. Every checkpoint's result comes back, so a partial
    series is visible as a partial series rather than as a shorter one.
    """
    require_pristine_restore(len(series.checkpoints), pristine_restore)
    wrapped = surface.loaded.model
    lora = wrapped.base_model
    snapshot = pristine_snapshot(wrapped) if pristine_restore else None
    if snapshot is not None:
        logger.info(f"[{label}] pristine snapshot over {len(snapshot)} wrapped modules")

    results: list[CellResult] = []
    started = time.time()
    for index, checkpoint in enumerate(series.checkpoints):
        adapter_name = f"ck{index}"
        if index > 0:
            wrapped.load_adapter(str(checkpoint.adapter_path), adapter_name=adapter_name)
        wrapped.set_adapter(adapter_name)
        if snapshot is not None:
            restore_pristine(wrapped, snapshot)
        lora.merge_adapter()
        try:
            results.append(
                replay_cell(
                    surface,
                    calibration,
                    Path(checkpoint.run_dir),
                    Path(manifest_path),
                    gen_ids=gen_ids,
                    signatures_subdir=signatures_subdir,
                    raw_subdir=raw_subdir,
                    save_raw=save_raw,
                    resume=resume,
                    label=f"{label}-{checkpoint.label}",
                )
            )
        finally:
            lora.unmerge_adapter()
            if len(series.checkpoints) > 1:
                try:
                    wrapped.delete_adapter(adapter_name)
                except (KeyError, ValueError, AttributeError) as exc:
                    logger.warning(f"[{label}] adapter slot {adapter_name} not released: {exc}")
        logger.info(
            f"[{label}] checkpoint {index + 1}/{len(series.checkpoints)} "
            f"({checkpoint.label}) done, {time.time() - started:.0f}s elapsed"
        )
    logger.info(
        f"[{label}] all {len(series.checkpoints)} checkpoints done in "
        f"{time.time() - started:.0f}s"
    )
    return results


def wrap_with_first_adapter(surface: ReplaySurface, series: CheckpointSeries) -> ReplaySurface:
    """Wrap a loaded base model with the series' first adapter, in place.

    The wrapper is built once and the remaining adapters are loaded into it, because
    wrapping is what creates the modules the merge writes into — and the hooks were
    placed on the plain modules that become their base layers, so they survive.
    """
    from peft import PeftModel

    wrapped = PeftModel.from_pretrained(
        surface.loaded.model, str(series.checkpoints[0].adapter_path), adapter_name="ck0"
    )
    surface.loaded.model = wrapped
    return surface


__all__ = [
    "ADAPTER_CONFIG_NAME",
    "CheckpointSeries",
    "CheckpointSpec",
    "SERIES_KEY",
    "pristine_snapshot",
    "replay_series",
    "require_pristine_restore",
    "restore_pristine",
    "series_from_adapter_dir",
    "wrap_with_first_adapter",
]
