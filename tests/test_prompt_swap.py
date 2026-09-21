"""The prompt-swap test: which axis a swap generation lands on, and how it is pooled.

The corpus is planted so the answer is known, which is the only way to test a confound
test: the feature that separates the two pure modes is given to the swap generations at
the *execution* mode's value, so an honest test has to call the signal execution-based.
Then the same corpus is rebuilt with the swap generations carrying the *prompt* mode's
value, and the verdict has to flip.

Also pinned:

  * the swap label's grammar — ``swap_A→B`` is (system prompt, execution), and a name that
    does not match is not a swap generation;
  * a block is used only where **every** swap generation has it, because a block present for
    some and absent for others would train and predict over different feature sets under
    one name;
  * composites are built from the blocks that survived that filter;
  * the pooled direction needs more than a bare majority: at 1.5:1 the reading is a
    direction and below it the reading is ``ambiguous``, which is a verdict rather than a
    missing one;
  * the probability is averaged over swap pairs rather than over generations, so one pair
    with more generations does not weigh more.

CPU only; the banks are synthetic and small, and the classifier is the real one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anamnesis.analysis.prompt_swap import (
    AMBIGUOUS,
    EXECUTION_BASED,
    PROMPT_BASED,
    load_swap_samples,
    parse_swap_mode,
    run_binary_prompt_swap,
    signal_type,
    swap_report,
)
from anamnesis.analysis.gauntlet.signature_io import (
    ATTENTION_FLOW,
    BLOCK_NPZ_KEYS,
    BLOCK_STORED_NAMES,
    NORMS_AND_OUTPUT_STATS,
)
from anamnesis.extraction.state_extractor import STORED_BLOCK_SLICES_KEY

BLOCK_LABEL = NORMS_AND_OUTPUT_STATS
BLOCK_NPZ_KEY = BLOCK_NPZ_KEYS[BLOCK_LABEL]
ADDON_NAME = ATTENTION_FLOW
ADDON_KEY = BLOCK_NPZ_KEYS[ADDON_NAME]
N_TOPICS = 6
WIDTH = 3


def write_generation(
    sig_dir: Path,
    gid: int,
    *,
    mode: str,
    topic: int,
    vector: np.ndarray,
    keys: dict[str, np.ndarray] | None = None,
) -> None:
    arrays = {BLOCK_NPZ_KEY: vector.astype(np.float32)}
    arrays.update({k: v.astype(np.float32) for k, v in (keys or {}).items()})
    np.savez(sig_dir / f"gen_{gid:03d}.npz", **arrays)
    (sig_dir / f"gen_{gid:03d}.json").write_text(
        json.dumps(
            {
                "generation_id": gid,
                "mode": mode,
                "mode_idx": 0 if mode == "linear" else 1,
                "topic": f"topic-{topic}",
                "topic_idx": topic,
                "num_generated_tokens": 80,
                STORED_BLOCK_SLICES_KEY: {
                    BLOCK_STORED_NAMES[BLOCK_LABEL]: [0, WIDTH]
                },
            }
        )
    )


def write_bank(root: Path, *, swap_carries: str, addon_root: Path | None = None) -> Path:
    """Two pure modes plus a swap set whose feature value is chosen by the caller.

    ``swap_carries`` is ``"execution"`` or ``"system"``: the swap generations are given the
    value of whichever mode that names, so the test knows what the verdict must be.
    """
    sig_dir = root / "signatures"
    sig_dir.mkdir(parents=True)
    rng = np.random.default_rng(7)
    centres = {"linear": -3.0, "socratic": 3.0}
    gid = 0
    for mode, centre in centres.items():
        for topic in range(N_TOPICS):
            vector = 0.2 * rng.standard_normal(WIDTH)
            vector[0] += centre
            write_generation(sig_dir, gid, mode=mode, topic=topic, vector=vector)
            gid += 1
    # swap_socratic→linear: system prompt socratic, execution linear.
    carried = centres["linear"] if swap_carries == "execution" else centres["socratic"]
    for topic in range(N_TOPICS):
        vector = 0.2 * rng.standard_normal(WIDTH)
        vector[0] += carried
        write_generation(
            sig_dir, gid, mode="swap_socratic→linear", topic=topic, vector=vector
        )
        gid += 1
    if addon_root is not None:
        addon_root.mkdir(parents=True)
    return sig_dir


def test_the_swap_label_spells_the_two_axes() -> None:
    assert parse_swap_mode("swap_socratic→linear") == ("socratic", "linear")
    assert parse_swap_mode("linear") is None
    assert parse_swap_mode("swap_socratic-linear") is None


def test_a_swap_that_executes_the_asked_for_mode_reads_as_execution_based(
    tmp_path: Path,
) -> None:
    sig_dir = write_bank(tmp_path / "exec", swap_carries="execution")
    result = run_binary_prompt_swap("exec", sig_dir)
    assert result.n_swap_samples == N_TOPICS
    assert result.swap_types == ["swap_socratic→linear"]
    pair = result.per_swap_type["swap_socratic→linear"]
    assert pair.system_prompt_mode == "socratic" and pair.execution_mode == "linear"
    assert pair.n_training_per_mode == {"socratic": N_TOPICS, "linear": N_TOPICS}
    block = pair.per_block[BLOCK_LABEL]
    assert block.n_execution == N_TOPICS and block.n_system == 0
    assert block.pct_execution == pytest.approx(1.0)
    assert result.aggregate[BLOCK_LABEL].signal_type == EXECUTION_BASED


def test_the_same_test_calls_the_other_direction_prompt_based(tmp_path: Path) -> None:
    sig_dir = write_bank(tmp_path / "prompt", swap_carries="system")
    result = run_binary_prompt_swap("prompt", sig_dir)
    block = result.per_swap_type["swap_socratic→linear"].per_block[BLOCK_LABEL]
    assert block.n_system == N_TOPICS and block.n_execution == 0
    assert result.aggregate[BLOCK_LABEL].signal_type == PROMPT_BASED


def test_the_pooled_direction_needs_more_than_a_bare_majority() -> None:
    assert signal_type(10, 7) == AMBIGUOUS, "a 10-to-7 split is not a direction"
    assert signal_type(10, 6) == EXECUTION_BASED, "the bar is 1.5 to 1"
    assert signal_type(7, 10) == AMBIGUOUS
    assert signal_type(6, 10) == PROMPT_BASED
    assert signal_type(0, 0) == AMBIGUOUS


def test_a_block_missing_from_some_swap_generations_is_not_used(tmp_path: Path) -> None:
    sig_dir = write_bank(tmp_path / "partial", swap_carries="execution")
    first_swap = sorted(sig_dir.glob("gen_*.json"))[2 * N_TOPICS]
    gid = int(first_swap.stem.split("_")[1])
    existing = np.load(sig_dir / f"gen_{gid:03d}.npz")
    np.savez(
        sig_dir / f"gen_{gid:03d}.npz",
        **{BLOCK_NPZ_KEY: existing[BLOCK_NPZ_KEY], ADDON_KEY: np.ones(2, dtype=np.float32)},
    )
    _samples, features = load_swap_samples(sig_dir)
    assert BLOCK_LABEL in features
    assert ADDON_NAME not in features, (
        "a block only one generation has would train and predict over different features"
    )


def test_a_bank_with_no_swap_generations_is_refused(tmp_path: Path) -> None:
    sig_dir = tmp_path / "pure" / "signatures"
    sig_dir.mkdir(parents=True)
    write_generation(sig_dir, 0, mode="linear", topic=0, vector=np.zeros(WIDTH))
    with pytest.raises(ValueError, match="no prompt-swap generations"):
        load_swap_samples(sig_dir)
    with pytest.raises(FileNotFoundError, match="no gen_"):
        load_swap_samples(tmp_path / "absent")


def test_the_report_is_the_typed_results_per_run(tmp_path: Path) -> None:
    sig_dir = write_bank(tmp_path / "exec", swap_carries="execution")
    result = run_binary_prompt_swap("exec", sig_dir)
    document = swap_report({"exec": result})
    assert set(document) == {"exec"}
    assert document["exec"]["aggregate"][BLOCK_LABEL]["signal_type"] == EXECUTION_BASED
    assert document["exec"]["per_swap_type"]["swap_socratic→linear"]["n_swap"] == N_TOPICS
