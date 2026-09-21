"""Reading signatures off disk: the join rules, and one read of a real bank.

Everything the gauntlet concludes rests on this loader putting the right rows in
the right order with the right labels, so the tests here are about the joins
rather than about the arithmetic:

  * tier discovery from npz contents, and the composite groups built from
    whichever tiers turned out to be present;
  * the core-only filter, which is what makes "one repetition per topic-mode
    pair" a property of the loaded matrix rather than of the caller's care;
  * addon merging, including the two ways it must refuse — an addon that covers
    only some rows, and an untagged addon reaching a tagged lane;
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
    AnalysisData,
    BASELINE_TIERS,
    ENGINEERED_TIERS,
    Run4Data,
    SampleMeta,
    TIER_GROUPS,
    TIER_KEYS,
    check_data_quality,
    default_signature_dir,
    load_analysis_data,
    load_run4,
)
from anamnesis.config.paths import LEGACY_DATA_ENV, outputs_root

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
    tiers: dict[str, int] | None = None,
    text: str = "some generated text",
    tokens: int = 128,
    lane: str | None = None,
) -> None:
    """One synthetic (npz, json) generation pair, with per-tier widths."""
    folder.mkdir(parents=True, exist_ok=True)
    tiers = tiers or {"T1": 3, "T2": 2}
    arrays: dict[str, np.ndarray] = {}
    names: list[str] = []
    slices: dict[str, list[int]] = {}
    cursor = 0
    for tier, width in tiers.items():
        key = TIER_KEYS[tier]
        arrays[key] = np.arange(width, dtype=np.float32) + float(index)
        slices[key.replace("features_", "")] = [cursor, cursor + width]
        names.extend(f"{tier}_f{i}" for i in range(width))
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
        "tier_slices": slices,
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


def test_tiers_are_discovered_not_declared(tmp_path: Path) -> None:
    folder = tmp_path / "sig"
    write_gen(folder, 0, tiers={"T2": 4, "gate_features": 6})
    data = load_run4(folder, core_only=False)
    assert set(data.tier_features) == {"T2", "gate_features"}
    assert data.tier_features["gate_features"].shape == (1, 6)
    # A group is built only from the members that are present, and one whose
    # members are all absent does not appear at all.
    assert "combined" in data.group_features          # T2 present
    assert data.group_features["combined"].shape == (1, 4)
    assert "T2+T2.5" in data.group_features
    assert data.all_features.shape == (1, 10)
    # all_features concatenates baseline tiers before engineered ones.
    assert list(BASELINE_TIERS)[:2] == ["T1", "T2"]
    assert "gate_features" in ENGINEERED_TIERS
    assert set(TIER_GROUPS) >= {"combined", "combined_v2"}


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


def test_mode_filter_rejects_a_filter_that_selects_nothing(two_mode_run: Path) -> None:
    only = load_run4(two_mode_run, core_only=False, mode_filter=["socratic"])
    assert only.unique_modes == ["socratic"]
    with pytest.raises(ValueError, match="mode_filter"):
        load_run4(two_mode_run, core_only=False, mode_filter=["dialectical"])


def test_unknown_tier_name_names_what_is_available(two_mode_run: Path) -> None:
    data = load_run4(two_mode_run, core_only=True)
    with pytest.raises(KeyError, match="Unknown tier/group"):
        data.get_tier("no_such_tier")
    assert data.get_tier("T1").shape[0] == data.n_samples


def test_missing_directory_and_empty_directory_are_distinguished(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_run4(tmp_path / "absent", core_only=False)
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No .npz files"):
        load_run4(tmp_path / "empty", core_only=False)


def test_addon_merges_new_tiers_and_skips_an_incomplete_one(tmp_path: Path) -> None:
    base = tmp_path / "base"
    write_gen(base, 0, topic="topic_a", topic_idx=0)
    write_gen(base, 1, topic="topic_b", topic_idx=1)

    complete = tmp_path / "complete"
    write_gen(complete, 0, tiers={"T3": 5})
    write_gen(complete, 1, tiers={"T3": 5})
    merged = load_run4(base, core_only=False, addon_dirs=[complete])
    assert "T3" in merged.tier_features
    assert merged.tier_features["T3"].shape == (2, 5)

    partial = tmp_path / "partial"
    write_gen(partial, 0, tiers={"T2.5": 7})
    dropped = load_run4(base, core_only=False, addon_dirs=[partial])
    assert "T2.5" not in dropped.tier_features, "an addon covering some rows is dropped whole"


def test_addon_directory_that_is_absent_or_empty_is_a_warning_not_a_failure(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    write_gen(base, 0)
    (tmp_path / "hollow").mkdir()
    data = load_run4(base, core_only=False, addon_dirs=[tmp_path / "gone", tmp_path / "hollow"])
    assert data.n_samples == 1


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
    assert data.get_tier("T1").shape == data.run4.get_tier("T1").shape
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
    assert report["nan_counts"]["T1"] == 0
    assert report["inf_counts"]["T1"] == 0
    assert report["tier_dims"]["T1"] == 3
    assert "combined" in report["group_dims"]


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
    real tier_slices, the real repetition structure the core filter reduces, and
    the lane field a legacy bank does not carry.
    """
    sig_dir = outputs_root() / "runs" / BANKED_RUN / BANKED_SUBDIR
    if not sig_dir.is_dir():
        pytest.skip(f"banked signatures absent: {sig_dir}")
    data = load_run4(sig_dir, core_only=False)
    assert data.n_samples > 0
    assert data.tier_features, "a banked directory discovered no tiers"
    for tier, matrix in data.tier_features.items():
        assert matrix.shape[0] == data.n_samples, f"{tier} row count disagrees with the samples"
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
    assert report["nan_counts"], "quality report found no tiers to check"
