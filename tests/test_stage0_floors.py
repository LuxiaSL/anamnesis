"""The Stage-0 command: its two stages, and the pairing it refuses to guess at.

The replay stage plans and fans out, and the law stage computes. What is tested here is the
command's own obligations:

  * ``replays --dry-run`` banks the plan and stops, so the manifest and the index exist
    before any device is touched — a plan is an artifact, not a side effect;
  * the faithfulness legs come as a pair: a replay corpus with no index cannot be split into
    its within-device and cross-device components, so naming one without the other is
    refused rather than silently reduced to one floor;
  * the law stage writes its reports and its table and returns zero.

The fan-out itself spawns replay workers, which needs devices and a checkpoint, so it is the
one path here without a test — its partition arithmetic is ``orchestration.launch``'s, tested
there, and its per-device shares are the plan's, tested in ``test_battery_stage0.py``.

CPU only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anamnesis.analysis.battery.stage0 import (
    INDEX_NAME,
    MANIFEST_NAME,
    N_REPLAYS,
    N_TOPICS,
    SEEDS_PER_CLASS,
    TOPICS_PER_STRATUM,
)
from anamnesis.scripts.stage0_floors import main

FEATURE_NAMES = ["attn_prompt_mass_L8", "gate_sparsity_L16"]


def write_floor_run(root: Path) -> Path:
    """A floor run with the manifest layout the continuations are selected from."""
    root.mkdir(parents=True, exist_ok=True)
    entries = {}
    for topic in range(N_TOPICS):
        gid = (topic // TOPICS_PER_STRATUM * N_TOPICS + topic) * SEEDS_PER_CLASS
        entries[str(gid)] = {"input_ids": [1, 2, 3, 4], "prompt_length": 2}
    (root / MANIFEST_NAME).write_text(
        json.dumps({"entries": entries, "n_ok": len(entries), "n_flagged": 0, "flagged": []})
    )
    return root


def write_floor_signatures(root: Path, *, n_topics: int = 4, n_seeds: int = 4) -> tuple[Path, Path]:
    sig_dir = root / "signatures_v3"
    sig_dir.mkdir(parents=True)
    rng = np.random.default_rng(11)
    generations = []
    gid = 0
    for topic in range(n_topics):
        centre = rng.standard_normal(len(FEATURE_NAMES))
        for _ in range(n_seeds):
            np.savez(
                sig_dir / f"gen_{gid:03d}.npz",
                feature_names=np.array(FEATURE_NAMES),
                features=(centre + 0.05 * rng.standard_normal(len(FEATURE_NAMES))).astype(
                    np.float32
                ),
            )
            generations.append(
                {
                    "generation_id": gid,
                    "mode": "floor",
                    "topic_idx": topic,
                    "mode_idx": 0,
                    "num_generated_tokens": 50,
                }
            )
            gid += 1
    (root / "metadata.json").write_text(json.dumps({"generations": generations}))
    return sig_dir, root / "metadata.json"


def test_the_dry_run_banks_the_plan_and_touches_no_device(tmp_path: Path) -> None:
    floor_run = write_floor_run(tmp_path / "floor")
    out_dir = tmp_path / "faithfulness"
    assert (
        main(
            [
                "replays",
                "--model", "3b",
                "--model-path", "/nonexistent/checkpoint",
                "--floor-run-dir", str(floor_run),
                "--calib-dir", str(tmp_path / "calib"),
                "--out-dir", str(out_dir),
                "--pinned-gpu", "0",
                "--spread-gpus", "1,2,3",
                "--dry-run",
            ]
        )
        == 0
    )
    manifest = json.loads((out_dir / MANIFEST_NAME).read_text())
    assert manifest["n_ok"] == N_TOPICS * N_REPLAYS
    index = json.loads((out_dir / INDEX_NAME).read_text())
    assert len(index) == N_TOPICS * N_REPLAYS
    assert {row["component"] for row in index} == {"within", "cross"}


def test_the_faithfulness_legs_come_as_a_pair(tmp_path: Path) -> None:
    sig_dir, metadata = write_floor_signatures(tmp_path / "floor")
    with pytest.raises(SystemExit, match="both --faith-sig-dir and --faith-index"):
        main(
            [
                "law",
                "--model", "3b",
                "--n-layers", "28",
                "--floor-sig-dir", str(sig_dir),
                "--floor-metadata", str(metadata),
                "--out-dir", str(tmp_path / "floors"),
                "--faith-sig-dir", str(sig_dir),
            ]
        )


def test_the_law_stage_writes_its_reports_and_its_table(tmp_path: Path) -> None:
    sig_dir, metadata = write_floor_signatures(tmp_path / "floor")
    out_dir = tmp_path / "floors"
    assert (
        main(
            [
                "law",
                "--model", "3b",
                "--n-layers", "28",
                "--floor-sig-dir", str(sig_dir),
                "--floor-metadata", str(metadata),
                "--out-dir", str(out_dir),
            ]
        )
        == 0
    )
    assert (out_dir / "floors_stochastic_3b.json").is_file()
    table = (out_dir / "law_table_3b.md").read_text()
    assert table.startswith("# Stage-0 law table — 3b")
    assert "| whole_vector |" in table
