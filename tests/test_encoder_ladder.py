"""The extraction ladder: matching, the three rungs, and the reading over them.

The ladder's whole output is a comparison between a hand-built projection and the raw
state, so the tests plant both cases and check that the reading names them differently:

  * a difference the hand features **carry** reads as numbers rather than as a catch;
  * a difference only the raw state carries — hand features at the floor, raw encoder well
    above it — reads as the projection having missed it;
  * neither carrying it reads as the hand features not being the limitation.

Matching is the other half. Only generations present in both arms enter, the two arms are
stacked in one order so a row and its partner are the same continuation, and an arm pair
with nothing in common is refused rather than compared on whatever each has.

Also pinned: a raw capture missing a surface yields no row instead of a zero row, and the
banked signature is read from the npz rather than from the metadata beside it.

CPU only; the readouts are the real ones at small epoch counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anamnesis.analysis.encoder_ladder import (
    CATCH,
    MISS,
    SEE_NUMBERS,
    Arm,
    LadderRung,
    RungPair,
    hand_vector,
    load_arm,
    raw_vector,
    read_ladder,
    run_ladder,
)

N_TOPICS = 4
N_PER_TOPIC = 3
WIDTH = 6
STEPS = 5
PROMPT_LENGTH = 4


def source_metadata(n: int) -> dict[int, dict[str, object]]:
    return {
        gid: {
            "generation_id": gid,
            "topic_idx": gid % N_TOPICS,
            "prompt_length": PROMPT_LENGTH,
            "num_generated_tokens": 40 + gid % 5,
        }
        for gid in range(n)
    }


def write_arm(
    root: Path,
    *,
    n: int,
    raw_shift: float,
    hand_shift: float,
    seed: int,
    skip_raw: set[int] | None = None,
    skip_hand: set[int] | None = None,
    drop_surface: bool = False,
) -> tuple[Path, Path]:
    """One arm: raw captures and banked signatures, each shifted as the test asks.

    The two shifts are independent so a test can plant a difference the raw state carries
    and the hand features do not, which is the case the ladder exists to adjudicate.
    """
    raw_dir = root / "raw"
    sig_dir = root / "run" / "signatures_v3"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for gid in range(n):
        if skip_raw is None or gid not in skip_raw:
            arrays: dict[str, np.ndarray] = {
                "actual_lengths": (PROMPT_LENGTH + 1 + np.arange(STEPS)).astype(np.int32),
                "prompt_length": np.array(PROMPT_LENGTH),
                "hidden_states": (
                    raw_shift + rng.standard_normal((STEPS, 3, WIDTH))
                ).astype(np.float32),
                "saved_layers_hs": np.array([-1, 0, 1], dtype=np.int32),
            }
            if not drop_surface:
                attention = rng.random((STEPS, 2, 2, 12)).astype(np.float32)
                arrays["attentions"] = attention
                arrays["saved_layers_attn"] = np.array([0, 1], dtype=np.int32)
            np.savez(raw_dir / f"gen_{gid:03d}.npz", **arrays)
        if skip_hand is None or gid not in skip_hand:
            np.savez(
                sig_dir / f"gen_{gid:03d}.npz",
                features=(hand_shift + rng.standard_normal(WIDTH)).astype(np.float32),
                feature_names=np.array([f"f{i}" for i in range(WIDTH)]),
            )
    return raw_dir, root / "run"


def test_the_reading_names_the_three_cases_apart() -> None:
    at_floor = RungPair(logit=LadderRung(test=0.50, std=0.05), deep=LadderRung(test=0.52, std=0.05))
    carrying = RungPair(logit=LadderRung(test=0.90, std=0.03), deep=LadderRung(test=0.92, std=0.03))
    raw_high = RungPair(logit=LadderRung(test=0.60, std=0.04), deep=LadderRung(test=0.85, std=0.04))
    raw_floor = RungPair(logit=LadderRung(test=0.51, std=0.04), deep=LadderRung(test=0.53, std=0.04))

    assert read_ladder(at_floor, raw_high) == CATCH
    assert read_ladder(at_floor, raw_floor) == MISS
    assert read_ladder(carrying, raw_high) == SEE_NUMBERS, (
        "a hand side above the floor bar is not the clean case either way"
    )
    assert carrying.best == pytest.approx(0.92)


def test_only_generations_present_in_both_arms_are_compared(tmp_path: Path) -> None:
    metadata = source_metadata(N_TOPICS * N_PER_TOPIC)
    positive_raw, positive_run = write_arm(
        tmp_path / "pos", n=12, raw_shift=1.0, hand_shift=1.0, seed=1, skip_raw={0, 1}
    )
    negative_raw, negative_run = write_arm(
        tmp_path / "neg", n=12, raw_shift=0.0, hand_shift=0.0, seed=2, skip_hand={11}
    )
    result = run_ladder(
        Arm(label="steered", raw_dir=positive_raw, run_dir=positive_run, y=1),
        Arm(label="control", raw_dir=negative_raw, run_dir=negative_run, y=0),
        metadata,
        n_seeds=1,
        n_splits=3,
        deep_epochs=20,
    )
    assert result.n_matched_pairs == 9, "two missing raw captures and one missing signature"
    assert result.n_topics == N_TOPICS
    assert result.n_hand_features == WIDTH
    assert result.n_raw_features > WIDTH, "the raw vector is the sampled surfaces, flattened"
    assert result.surfaces_raw == ["residual", "attention"]


def test_arms_with_nothing_in_common_are_refused(tmp_path: Path) -> None:
    metadata = source_metadata(4)
    positive_raw, positive_run = write_arm(
        tmp_path / "pos", n=4, raw_shift=1.0, hand_shift=1.0, seed=1, skip_raw={0, 1, 2, 3}
    )
    negative_raw, negative_run = write_arm(
        tmp_path / "neg", n=4, raw_shift=0.0, hand_shift=0.0, seed=2
    )
    with pytest.raises(ValueError, match="share no generation"):
        run_ladder(
            Arm(label="steered", raw_dir=positive_raw, run_dir=positive_run, y=1),
            Arm(label="control", raw_dir=negative_raw, run_dir=negative_run, y=0),
            metadata,
            n_seeds=1,
            n_splits=2,
            deep_epochs=5,
        )


def test_a_difference_the_hand_features_carry_is_not_read_as_a_catch(tmp_path: Path) -> None:
    metadata = source_metadata(12)
    positive_raw, positive_run = write_arm(
        tmp_path / "pos", n=12, raw_shift=6.0, hand_shift=6.0, seed=1
    )
    negative_raw, negative_run = write_arm(
        tmp_path / "neg", n=12, raw_shift=0.0, hand_shift=0.0, seed=2
    )
    result = run_ladder(
        Arm(label="steered", raw_dir=positive_raw, run_dir=positive_run, y=1),
        Arm(label="control", raw_dir=negative_raw, run_dir=negative_run, y=0),
        metadata,
        n_seeds=1,
        n_splits=3,
        deep_epochs=40,
    )
    assert result.hand_features.best > 0.8, "the hand projection carries the planted shift"
    assert result.reading != CATCH
    assert result.chance == 0.5
    assert len(result.notes) == 3


def test_a_capture_missing_a_surface_yields_no_row(tmp_path: Path) -> None:
    write_arm(tmp_path / "arm", n=2, raw_shift=0.0, hand_shift=0.0, seed=1, drop_surface=True)
    assert raw_vector(tmp_path / "arm" / "raw" / "gen_000.npz") is None, (
        "an absent surface is a missing row, not a row of zeros"
    )
    assert raw_vector(
        tmp_path / "arm" / "raw" / "gen_000.npz", surfaces=("residual",)
    ) is not None


def test_a_capture_with_no_positions_yields_no_row(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    np.savez(raw_dir / "gen_000.npz", actual_lengths=np.array([]),
             prompt_length=np.array(PROMPT_LENGTH))
    assert raw_vector(raw_dir / "gen_000.npz") is None
    np.savez(raw_dir / "gen_001.npz", nothing=np.array([1]))
    assert raw_vector(raw_dir / "gen_001.npz") is None


def test_the_signature_is_read_from_the_npz_not_the_metadata(tmp_path: Path) -> None:
    _raw_dir, run_dir = write_arm(tmp_path / "arm", n=2, raw_shift=0.0, hand_shift=0.0, seed=1)
    assert hand_vector(run_dir, 0) is not None
    assert hand_vector(run_dir, 99) is None, "a generation with no banked signature has none"
    (run_dir / "signatures_v3" / "gen_001.json").write_text(json.dumps({"features": [1, 2, 3]}))
    np.savez(run_dir / "signatures_v3" / "gen_001.npz", something_else=np.zeros(3))
    assert hand_vector(run_dir, 1) is None, (
        "the metadata's fields are not the vector; the vector lives in the npz"
    )


def test_an_arm_only_loads_generations_the_source_run_records(tmp_path: Path) -> None:
    raw_dir, run_dir = write_arm(tmp_path / "arm", n=6, raw_shift=0.0, hand_shift=0.0, seed=1)
    rows = load_arm(
        Arm(label="arm", raw_dir=raw_dir, run_dir=run_dir, y=1), source_metadata(3)
    )
    assert rows.gen_ids == [0, 1, 2], "a generation with no source record cannot be placed"
    assert set(rows.covariates[0]) == {float(PROMPT_LENGTH), 40.0}
    assert rows.topic[1] == "1"
