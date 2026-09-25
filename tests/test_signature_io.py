"""Reading signatures off disk: the join rules, and one read of a real bank.

Everything the gauntlet concludes rests on this loader putting the right rows in
the right order with the right labels, so the tests here are about the joins
rather than about the arithmetic:

  * block discovery from npz contents, and the unions built over them — built
    only where every member a union names is present, so that a label never
    reports a narrower feature set than it claims;
  * the core-only filter, which is what makes "one repetition per topic-mode
    pair" a property of the loaded matrix rather than of the caller's care;
  * the text half, which is the only difference between the two loaded types.

The last test reads the banked ``8b_fat_01`` signatures if this machine has them
and skips with a named reason if it does not, because CI has no banked data and
a loader that has only ever seen synthetic npz files has not been exercised.

CPU only; no model, no GPU, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anamnesis.analysis.gauntlet.signature_io import (
    ALL_CORE,
    ALL_FAMILIES,
    ALL_LABELS,
    ATTENTION_AND_CACHE,
    ATTENTION_AND_CACHE_WITH_FAMILIES,
    ATTENTION_AND_DELTAS,
    ATTENTION_FLOW,
    AnalysisData,
    CACHE_AND_KEYS,
    CORE_BLOCKS,
    EVERYTHING,
    FAMILY_BLOCKS,
    GATE_FEATURES,
    NORMS_AND_OUTPUT_STATS,
    NPZ_KEY_PREFIX,
    RESIDUAL_PCA,
    RESIDUAL_TRAJECTORY,
    Run4Data,
    SampleMeta,
    BLOCK_STORED_NAMES,
    BLOCK_UNIONS,
    BLOCK_NPZ_KEYS,
    check_data_quality,
    default_signature_dir,
    load_analysis_data,
    load_run4,
)
from anamnesis.config.paths import LEGACY_DATA_ENV, outputs_root
from anamnesis.extraction.state_extractor import (
    STORED_ATTENTION_AND_DELTAS,
    STORED_BLOCK_SLICES_KEY,
    STORED_CACHE_AND_KEYS,
    STORED_NORMS_AND_OUTPUT_STATS,
    STORED_RESIDUAL_PCA,
)

BANKED_RUN = "8b_fat_01"
BANKED_SUBDIR = "signatures_v3"


def write_gen(
    folder: Path,
    index: int,
    *,
    mode: str = "linear",
    mode_idx: int = 0,
    topic: str = "topic_a",
    topic_idx: int = 0,
    blocks: dict[str, int] | None = None,
    text: str = "some generated text",
    tokens: int = 128,
    lane: str | None = None,
) -> None:
    """One synthetic (npz, json) generation pair, with per-block widths."""
    folder.mkdir(parents=True, exist_ok=True)
    blocks = blocks or {NORMS_AND_OUTPUT_STATS: 3, ATTENTION_AND_DELTAS: 2}
    arrays: dict[str, np.ndarray] = {}
    names: list[str] = []
    slices: dict[str, list[int]] = {}
    cursor = 0
    for block, width in blocks.items():
        arrays[BLOCK_NPZ_KEYS[block]] = np.arange(width, dtype=np.float32) + float(index)
        slices[BLOCK_STORED_NAMES[block]] = [cursor, cursor + width]
        names.extend(f"{block}_f{i}" for i in range(width))
        cursor += width
    np.savez(folder / f"gen_{index:03d}.npz", feature_names=np.array(names), **arrays)
    meta: dict[str, object] = {
        "generation_id": index,
        "topic": topic,
        "topic_idx": topic_idx,
        "mode": mode,
        "mode_idx": mode_idx,
        "num_generated_tokens": tokens,
        "generated_text": text,
        "system_prompt": f"system for {mode}",
        "user_prompt": f"user for {topic}",
        STORED_BLOCK_SLICES_KEY: slices,
    }
    if lane is not None:
        meta["lane_id"] = lane
    (folder / f"gen_{index:03d}.json").write_text(json.dumps(meta))


@pytest.fixture
def two_mode_run(tmp_path: Path) -> Path:
    """Two modes × two shared topics, plus a second repetition of one pair."""
    folder = tmp_path / "signatures"
    index = 0
    for mode_idx, mode in enumerate(["linear", "socratic"]):
        for topic_idx, topic in enumerate(["topic_a", "topic_b"]):
            for _ in range(2 if (mode == "linear" and topic == "topic_a") else 1):
                write_gen(
                    folder, index, mode=mode, mode_idx=mode_idx,
                    topic=topic, topic_idx=topic_idx, tokens=100 + index,
                )
                index += 1
    return folder


def test_blocks_are_discovered_not_declared(tmp_path: Path) -> None:
    folder = tmp_path / "sig"
    write_gen(folder, 0, blocks={ATTENTION_AND_DELTAS: 4, GATE_FEATURES: 6})
    data = load_run4(folder, core_only=False)
    assert set(data.block_features) == {ATTENTION_AND_DELTAS, GATE_FEATURES}
    assert data.block_features[GATE_FEATURES].shape == (1, 6)
    assert data.all_features.shape == (1, 10)
    # all_features concatenates the core blocks before the engineered families.
    assert list(CORE_BLOCKS)[:2] == [NORMS_AND_OUTPUT_STATS, ATTENTION_AND_DELTAS]
    assert GATE_FEATURES in FAMILY_BLOCKS
    assert set(BLOCK_UNIONS) >= {ALL_CORE, EVERYTHING}


def test_a_union_missing_a_member_is_not_built(tmp_path: Path) -> None:
    """A label names what is in it, so a short union is absent rather than narrow.

    One core block and one family are present here. Every union defined over
    them names at least one block this corpus does not hold, so none is built —
    the alternative is `every_block` reported at the width of one block.
    """
    folder = tmp_path / "sig"
    write_gen(folder, 0, blocks={ATTENTION_AND_DELTAS: 4, GATE_FEATURES: 6})
    data = load_run4(folder, core_only=False)
    assert data.group_features == {}
    assert not data.has_block(ALL_CORE)
    assert not data.has_block(ATTENTION_AND_CACHE)
    assert data.has_block(ATTENTION_AND_DELTAS)


def test_a_union_whose_members_are_all_present_is_built_at_its_full_width(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "sig"
    write_gen(folder, 0, blocks={
        NORMS_AND_OUTPUT_STATS: 3, ATTENTION_AND_DELTAS: 4,
        CACHE_AND_KEYS: 5, RESIDUAL_PCA: 6,
    })
    data = load_run4(folder, core_only=False)
    assert data.group_features[ALL_CORE].shape == (1, 18)
    assert data.group_features[ATTENTION_AND_CACHE].shape == (1, 9)
    # Every union reported holds exactly the blocks its definition names.
    for label, members in BLOCK_UNIONS.items():
        if label in data.group_features:
            width = sum(data.block_features[m].shape[1] for m in members)
            assert data.group_features[label].shape[1] == width, label


def test_the_family_union_is_spelled_by_the_members_it_holds() -> None:
    """A union label is a claim about membership, so the membership is pinned here.

    Adding a family to ``FAMILY_BLOCKS`` or removing one fails this until the union is
    respelled and the membership it had is entered in the legacy table of
    `anamnesis/analysis/gauntlet/schemas/compat.py`, so that a results file reporting
    the old membership never reads under the new label.
    """
    assert FAMILY_BLOCKS == [RESIDUAL_TRAJECTORY, ATTENTION_FLOW, GATE_FEATURES]
    assert ALL_FAMILIES == "trajectory_flow_and_gate"
    assert BLOCK_UNIONS[ALL_FAMILIES] == FAMILY_BLOCKS
    assert ATTENTION_AND_CACHE_WITH_FAMILIES == f"{ATTENTION_AND_CACHE}+{ALL_FAMILIES}"
    assert BLOCK_UNIONS[ATTENTION_AND_CACHE_WITH_FAMILIES] == [
        *BLOCK_UNIONS[ATTENTION_AND_CACHE], *FAMILY_BLOCKS,
    ]


def test_npz_keys_come_from_the_stored_block_names() -> None:
    """The frozen strings live in one place, and this is what says so.

    A banked npz keys the four core blocks by names that predate the labels, and
    every signature ever written indexes into its vector with exactly them.
    """
    assert BLOCK_STORED_NAMES[NORMS_AND_OUTPUT_STATS] == STORED_NORMS_AND_OUTPUT_STATS
    assert BLOCK_STORED_NAMES[ATTENTION_AND_DELTAS] == STORED_ATTENTION_AND_DELTAS
    assert BLOCK_STORED_NAMES[CACHE_AND_KEYS] == STORED_CACHE_AND_KEYS
    assert BLOCK_STORED_NAMES[RESIDUAL_PCA] == STORED_RESIDUAL_PCA
    for block in FAMILY_BLOCKS:
        assert BLOCK_STORED_NAMES[block] == block, "a family is stored under its own label"
    assert BLOCK_NPZ_KEYS == {
        label: NPZ_KEY_PREFIX + stored for label, stored in BLOCK_STORED_NAMES.items()
    }
    assert set(BLOCK_NPZ_KEYS) | set(BLOCK_UNIONS) == set(ALL_LABELS)


def test_core_only_keeps_one_repetition_of_each_shared_pair(two_mode_run: Path) -> None:
    full = load_run4(two_mode_run, core_only=False)
    core = load_run4(two_mode_run, core_only=True)
    assert full.n_samples == 5
    assert core.n_samples == 4
    pairs = [(s.mode, s.topic) for s in core.samples]
    assert len(set(pairs)) == 4
    # Rows are ordered by (mode_idx, topic_idx), which is what lets two loads of
    # the same directory be compared row for row.
    assert [s.mode_idx for s in core.samples] == sorted(s.mode_idx for s in core.samples)
    assert core.unique_modes == ["linear", "socratic"]
    assert core.unique_topics == ["topic_a", "topic_b"]
    assert core.mode_mask("linear").sum() == 2
    assert core.topic_mask("topic_b").sum() == 2
    assert isinstance(core.samples[0], SampleMeta)
    assert isinstance(core, Run4Data)


def test_core_only_excludes_swap_modes_from_the_shared_intersection(tmp_path: Path) -> None:
    folder = tmp_path / "sig"
    write_gen(folder, 0, mode="linear", mode_idx=0, topic="topic_a", topic_idx=0)
    write_gen(folder, 1, mode="socratic", mode_idx=1, topic="topic_a", topic_idx=0)
    # A swap mode covering no shared topic would empty the intersection if it
    # were allowed into it.
    write_gen(folder, 2, mode="swap_socratic→linear", mode_idx=2,
              topic="topic_z", topic_idx=9)
    core = load_run4(folder, core_only=True)
    assert core.n_samples == 2
    assert "swap_socratic→linear" not in core.unique_modes


def test_core_only_says_what_it_set_aside_and_why(
    two_mode_run: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A count of samples read does not say how many were on disk, so the narrowing is logged."""
    write_gen(two_mode_run, 5, mode="socratic", mode_idx=1, topic="topic_c", topic_idx=2)
    write_gen(two_mode_run, 6, mode="swap_socratic→linear", mode_idx=2,
              topic="topic_a", topic_idx=0)
    with caplog.at_level("INFO", logger="anamnesis.analysis.gauntlet.signature_io"):
        core = load_run4(two_mode_run, core_only=True)
    assert core.n_samples == 4
    assert (
        "4 of 7 generations read; set aside 1 further repetitions, 1 on topics some mode "
        "lacks and 1 prompt-swap generations"
    ) in caplog.text


def test_a_balanced_bank_logs_no_narrowing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    folder = tmp_path / "sig"
    write_gen(folder, 0, mode="linear", mode_idx=0, topic="topic_a", topic_idx=0)
    write_gen(folder, 1, mode="socratic", mode_idx=1, topic="topic_a", topic_idx=0)
    with caplog.at_level("INFO", logger="anamnesis.analysis.gauntlet.signature_io"):
        load_run4(folder, core_only=True)
    assert "Balanced core set" not in caplog.text


def test_mode_filter_rejects_a_filter_that_selects_nothing(two_mode_run: Path) -> None:
    only = load_run4(two_mode_run, core_only=False, mode_filter=["socratic"])
    assert only.unique_modes == ["socratic"]
    with pytest.raises(ValueError, match="mode_filter"):
        load_run4(two_mode_run, core_only=False, mode_filter=["dialectical"])


def test_unknown_block_label_names_what_is_available(two_mode_run: Path) -> None:
    data = load_run4(two_mode_run, core_only=True)
    with pytest.raises(KeyError, match="Unknown block/group"):
        data.get_block("no_such_block")
    assert data.get_block(NORMS_AND_OUTPUT_STATS).shape[0] == data.n_samples


def test_missing_directory_and_empty_directory_are_distinguished(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_run4(tmp_path / "absent", core_only=False)
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No .npz files"):
        load_run4(tmp_path / "empty", core_only=False)


def test_analysis_data_carries_text_and_delegates_the_rest(two_mode_run: Path) -> None:
    data = load_analysis_data(two_mode_run, run_name="synthetic", core_only=True)
    assert isinstance(data, AnalysisData)
    assert data.run_name == "synthetic"
    assert data.generated_texts is not None
    assert len(data.generated_texts) == data.n_samples
    assert data.system_prompts is not None and data.user_prompts is not None
    assert data.generation_lengths is not None
    assert data.generation_lengths.dtype == np.int64
    # The delegating accessors are the whole reason sections take this type.
    assert data.n_samples == data.run4.n_samples
    assert list(data.modes) == list(data.run4.modes)
    assert list(data.topics) == list(data.run4.topics)
    assert data.unique_modes == data.run4.unique_modes
    assert data.unique_topics == data.run4.unique_topics
    assert data.get_block(NORMS_AND_OUTPUT_STATS).shape == data.run4.get_block(NORMS_AND_OUTPUT_STATS).shape
    assert data.mode_mask("linear").sum() == 2
    assert data.topic_mask("topic_a").sum() == 2


def test_text_loading_is_skippable_and_missing_json_reads_as_empty(two_mode_run: Path) -> None:
    lean = load_analysis_data(two_mode_run, run_name="lean", core_only=True, load_text=False)
    assert lean.generated_texts is None
    assert lean.generation_lengths is None

    stems = sorted(p.stem for p in two_mode_run.glob("gen_*.npz"))
    data = load_analysis_data(two_mode_run, run_name="holes", core_only=True)
    assert data.generated_texts is not None
    (two_mode_run / f"{stems[0]}.json").unlink()
    # The npz is still there, so the row is gone from the matrix too: a row with
    # no metadata has no mode and cannot be loaded at all.
    fewer = load_analysis_data(two_mode_run, run_name="holes", core_only=False)
    assert fewer.n_samples < 5


def test_quality_report_counts_what_a_reader_would_check(two_mode_run: Path) -> None:
    data = load_run4(two_mode_run, core_only=True)
    report = check_data_quality(data)
    assert report["n_samples"] == 4
    assert report["n_modes"] == 2
    assert report["samples_per_mode"] == {"linear": 2, "socratic": 2}
    assert report["nan_counts"][NORMS_AND_OUTPUT_STATS] == 0
    assert report["inf_counts"][NORMS_AND_OUTPUT_STATS] == 0
    assert report["block_dims"][NORMS_AND_OUTPUT_STATS] == 3
    # This corpus holds two of the four core blocks, so no union over them is
    # built and there is nothing for the group table to report.
    assert report["group_dims"] == {}


def test_default_signature_dir_follows_the_legacy_root_at_call_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(LEGACY_DATA_ENV, str(tmp_path))
    first = default_signature_dir()
    assert first == tmp_path / "outputs" / "runs" / "run4_format_controlled" / "signatures"
    monkeypatch.setenv(LEGACY_DATA_ENV, str(tmp_path / "elsewhere"))
    assert default_signature_dir() != first, "the root is read when asked, not at import"


def test_reads_the_banked_8b_signatures() -> None:
    """The read smoke: a real bank through the ported loader, or a named skip.

    CI has no banked data, so absence is a skip rather than a failure. What this
    catches that synthetic npz files cannot: the real feature-name arrays, the
    real block_slices, the real repetition structure the core filter reduces, and
    the lane field a legacy bank does not carry.
    """
    sig_dir = outputs_root() / "runs" / BANKED_RUN / BANKED_SUBDIR
    if not sig_dir.is_dir():
        pytest.skip(f"banked signatures absent: {sig_dir}")
    data = load_run4(sig_dir, core_only=False)
    assert data.n_samples > 0
    assert data.block_features, "a banked directory discovered no blocks"
    for block, matrix in data.block_features.items():
        assert matrix.shape[0] == data.n_samples, f"{block} row count disagrees with the samples"
        assert matrix.ndim == 2 and matrix.shape[1] > 0
    assert len(data.modes) == data.n_samples
    assert len(data.topics) == data.n_samples
    assert data.mode_indices.shape == (data.n_samples,)
    assert data.topic_indices.shape == (data.n_samples,)
    assert set(data.unique_modes), "no mode labels came back"
    # An untagged historical bank reads as itself: None is "no lane recorded",
    # never an invented certification.
    assert data.lane_id is None or isinstance(data.lane_id, str)
    report = check_data_quality(data)
    assert report["nan_counts"], "quality report found no blocks to check"
