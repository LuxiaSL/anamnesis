"""Stage 0: the protocol that collects the floors, and the law table it produces.

The floors themselves are tested in ``test_battery.py``. What is tested here is the
protocol around them, which is where a floor gets attributed to the wrong thing:

  * the continuation selection is arithmetic over the topic index, so the generation
    ids it asks the floor corpus for are checked by value rather than by shape, and a
    corpus missing one of them is an error rather than a smaller set;
  * the stratified plan puts exactly four replays of each continuation on the pinned
    device and spreads the rest, and the index row for every replay carries the device
    and the component — which is the only thing separating replay determinism from
    operational jitter, since the two are the same arithmetic on different pairs;
  * a pinned device that also appears among the spread devices is refused, because the
    cross-device component would then be measured within one device;
  * the synthetic manifest's entries are the continuation's own banked tokens, ten
    times over, so a replay command sees an ordinary manifest;
  * the law table's PLAN column is the conservative reading — the larger of the
    σ-based and MAD-based n — and an exactly-zero floor has no n at all and says so;
  * the faithfulness floors are standardized on the stochastic corpus's scale, which
    ``compute_stage0_law`` enforces by computing the stochastic pass first.

CPU only; no model, no device, no banked data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anamnesis.analysis.battery.floors import FloorCell, FloorReport, FloorType, LawParams
from anamnesis.analysis.battery.stage0 import (
    CROSS,
    N_PINNED,
    N_REPLAYS,
    N_TOPICS,
    SEEDS_PER_CLASS,
    TOPICS_PER_STRATUM,
    WITHIN,
    compute_stage0_law,
    floor_gid,
    law_table_md,
    plan_stratified_replays,
    select_continuations,
    whole_vector_reading,
)

FEATURE_NAMES = [
    "attn_prompt_mass_L8",
    "attn_recency_bias_L16",
    "gate_sparsity_L16",
    "resid_velocity_L24",
]


def floor_manifest() -> dict[str, object]:
    """A floor run's manifest, laid out the way the floor corpus lays its ids out."""
    entries = {}
    for topic in range(N_TOPICS):
        for seed in range(SEEDS_PER_CLASS):
            gid = (topic // TOPICS_PER_STRATUM * N_TOPICS + topic) * SEEDS_PER_CLASS + seed
            entries[str(gid)] = {
                "input_ids": [1, 2, 3, 4, 5 + topic],
                "prompt_length": 3,
                "generation_id": gid,
            }
    return {"entries": entries, "n_ok": len(entries), "n_flagged": 0, "flagged": []}


def test_the_floor_gid_layout_is_arithmetic_over_the_topic_index() -> None:
    assert floor_gid(0) == 0
    assert floor_gid(4) == (0 * N_TOPICS + 4) * SEEDS_PER_CLASS
    assert floor_gid(5) == (1 * N_TOPICS + 5) * SEEDS_PER_CLASS
    assert floor_gid(19) == (3 * N_TOPICS + 19) * SEEDS_PER_CLASS
    assert floor_gid(7, seed_idx=3) == floor_gid(7) + 3
    with pytest.raises(ValueError, match="topic_idx"):
        floor_gid(N_TOPICS)


def test_one_continuation_per_topic_is_selected_by_id_not_by_position() -> None:
    selected = select_continuations(floor_manifest())
    assert sorted(selected) == list(range(N_TOPICS))
    for topic, entry in selected.items():
        assert entry["generation_id"] == floor_gid(topic)


def test_a_floor_corpus_missing_a_continuation_is_an_error() -> None:
    manifest = floor_manifest()
    del manifest["entries"][str(floor_gid(11))]  # type: ignore[index]
    with pytest.raises(KeyError, match="topic 11"):
        select_continuations(manifest)


def test_the_plan_pins_four_replays_and_spreads_the_rest() -> None:
    plan = plan_stratified_replays(
        select_continuations(floor_manifest()),
        pinned_device="0",
        spread_devices=["1", "2", "3"],
    )
    assert len(plan.instances) == N_TOPICS * N_REPLAYS
    assert plan.n_continuations == N_TOPICS
    for continuation in range(N_TOPICS):
        mine = [i for i in plan.instances if i.continuation_id == continuation]
        pinned = [i for i in mine if i.component == WITHIN]
        spread = [i for i in mine if i.component == CROSS]
        assert len(pinned) == N_PINNED
        assert {i.device for i in pinned} == {"0"}
        assert len(spread) == N_REPLAYS - N_PINNED
        assert {i.device for i in spread} == {"1", "2", "3"}
        assert [i.gen_id for i in mine] == [
            continuation * N_REPLAYS + r for r in range(N_REPLAYS)
        ]


def test_every_replay_carries_its_device_and_component_into_the_index() -> None:
    plan = plan_stratified_replays(
        select_continuations(floor_manifest()), pinned_device="0", spread_devices=["1"]
    )
    rows = [i.index_row() for i in plan.instances]
    assert {row["sig"] for row in rows} == {
        f"gen_{i.gen_id:03d}" for i in plan.instances
    }
    assert {row["device"] for row in rows} == {"gpu0", "gpu1"}
    assert {row["component"] for row in rows} == {WITHIN, CROSS}
    by_device = plan.gen_ids_by_device()
    assert sorted(by_device) == ["0", "1"]
    assert len(by_device["0"]) == N_TOPICS * N_PINNED


def test_the_pinned_device_may_not_also_be_a_spread_device() -> None:
    continuations = select_continuations(floor_manifest())
    with pytest.raises(ValueError, match="both pinned and spread"):
        plan_stratified_replays(
            continuations, pinned_device="1", spread_devices=["1", "2"]
        )
    with pytest.raises(ValueError, match="both components"):
        plan_stratified_replays(
            continuations, pinned_device="0", spread_devices=["1"],
            n_replays=4, n_pinned=4,
        )


def test_the_synthetic_manifest_repeats_one_continuation_ten_times(tmp_path: Path) -> None:
    continuations = select_continuations(floor_manifest())
    plan = plan_stratified_replays(
        continuations, pinned_device="0", spread_devices=["1", "2", "3"]
    )
    manifest_path, index_path = plan.write(tmp_path / "faith")
    written = json.loads(manifest_path.read_text())
    assert written["n_ok"] == N_TOPICS * N_REPLAYS
    assert written["n_flagged"] == 0 and written["flagged"] == []
    source = continuations[7]
    for replay in range(N_REPLAYS):
        entry = written["entries"][str(7 * N_REPLAYS + replay)]
        assert entry["input_ids"] == source["input_ids"]
        assert entry["prompt_length"] == source["prompt_length"]
    assert len(json.loads(index_path.read_text())) == N_TOPICS * N_REPLAYS


def cell(name: str, *, exact_zero: bool = False) -> FloorCell:
    if exact_zero:
        return FloorCell(
            cell=name, n_features=4, n_pairs=12, model="3b",
            floor_type=FloorType.faithfulness_within_device,
            median=0.0, std=0.0, mad=0.0, q10=0.0, q90=0.0,
            effect_d=float("inf"), effect_d_robust=float("inf"),
            n_min_by_alpha={str(a): 0 for a in LawParams().alpha_grid},
            n_min_by_alpha_rank={str(a): 0 for a in LawParams().alpha_grid},
            n_min_by_alpha_robust={"0.05": 0},
            exact_zero=True,
            law=LawParams(),
        )
    return FloorCell(
        cell=name, n_features=4, n_pairs=12, model="3b", floor_type=FloorType.stochastic,
        median=0.5, std=0.25, mad=0.4, q10=0.2, q90=0.9,
        effect_d=2.0, effect_d_robust=1.25,
        n_min_by_alpha={str(a): 7 + 4 * i for i, a in enumerate(LawParams().alpha_grid)},
        n_min_by_alpha_rank={str(a): 8 + 4 * i for i, a in enumerate(LawParams().alpha_grid)},
        n_min_by_alpha_robust={"0.05": 13},
        exact_zero=False,
        law=LawParams(),
    )


def report(cells: list[FloorCell], floor_type: FloorType = FloorType.stochastic) -> FloorReport:
    return FloorReport(
        model="3b", floor_type=floor_type, n_gens=20, n_pairs_total=12,
        corpus="synthetic", law=LawParams(), cells=cells,
    )


def test_the_law_table_plans_on_the_more_conservative_of_the_two_n() -> None:
    table = law_table_md([report([cell("whole_vector")])], "3b")
    row = [line for line in table.splitlines() if line.startswith("| whole_vector")][0]
    columns = [c.strip() for c in row.split("|")]
    assert columns[-2] == "13", "the PLAN column takes the larger of the two n at 0.05"
    assert "n_min@alpha=0.05" in table
    assert "2x PLAN" in table


def test_an_exactly_zero_floor_has_no_n_and_the_table_says_so() -> None:
    table = law_table_md(
        [report([cell("whole_vector", exact_zero=True)], FloorType.faithfulness_within_device)],
        "3b",
    )
    row = [line for line in table.splitlines() if line.startswith("| whole_vector")][0]
    assert "0 (EXACT)" in row and "EXACT*" in row
    assert "bitwise ZERO" in table


def test_the_whole_vector_reading_carries_its_n_and_its_law() -> None:
    line = whole_vector_reading(report([cell("whole_vector")]))
    assert "median=0.5000" in line and "n=12 pairs" in line and "law k=2.0" in line
    assert "floor=stochastic" in line
    exact = whole_vector_reading(
        report([cell("whole_vector", exact_zero=True)], FloorType.faithfulness_within_device)
    )
    assert "EXACT ZERO" in exact
    with pytest.raises(ValueError, match="no whole_vector cell"):
        whole_vector_reading(report([cell("attention")]))


def write_floor_bank(root: Path, *, n_topics: int = 4, n_seeds: int = 4) -> tuple[Path, Path]:
    """A floor corpus: several seeds per topic, so pair deltas exist within a class."""
    sig_dir = root / "signatures_v3"
    sig_dir.mkdir(parents=True)
    rng = np.random.default_rng(3)
    generations = []
    gid = 0
    for topic in range(n_topics):
        centre = rng.standard_normal(len(FEATURE_NAMES))
        for _seed in range(n_seeds):
            np.savez(
                sig_dir / f"gen_{gid:03d}.npz",
                feature_names=np.array(FEATURE_NAMES),
                features=(centre + 0.1 * rng.standard_normal(len(FEATURE_NAMES))).astype(
                    np.float32
                ),
            )
            generations.append(
                {
                    "generation_id": gid,
                    "mode": "floor",
                    "topic_idx": topic,
                    "mode_idx": 0,
                    "num_generated_tokens": 64,
                }
            )
            gid += 1
    (root / "metadata.json").write_text(json.dumps({"generations": generations}))
    return sig_dir, root / "metadata.json"


def write_faithfulness_bank(root: Path, *, n_continuations: int = 3) -> tuple[Path, Path]:
    """Replays of a few continuations: identical within a device, jittered across."""
    sig_dir = root / "signatures_v3"
    sig_dir.mkdir(parents=True)
    rng = np.random.default_rng(5)
    index = []
    for continuation in range(n_continuations):
        base = rng.standard_normal(len(FEATURE_NAMES))
        for replay in range(4):
            gid = continuation * N_REPLAYS + replay
            jitter = 0.0 if replay < 2 else 1e-3 * rng.standard_normal(len(FEATURE_NAMES))
            np.savez(
                sig_dir / f"gen_{gid:03d}.npz",
                feature_names=np.array(FEATURE_NAMES),
                features=(base + jitter).astype(np.float32),
            )
            index.append(
                {
                    "sig": f"gen_{gid:03d}",
                    "continuation_id": continuation,
                    "replay_idx": replay,
                    "device": "gpu0" if replay < 2 else f"gpu{replay}",
                    "component": WITHIN if replay < 2 else CROSS,
                }
            )
    index_path = root / "replay_index.json"
    index_path.write_text(json.dumps(index))
    return sig_dir, index_path


def test_the_law_pass_banks_every_report_and_the_table(tmp_path: Path) -> None:
    floor_sig, floor_meta = write_floor_bank(tmp_path / "floor")
    faith_sig, faith_index = write_faithfulness_bank(tmp_path / "faith")
    result = compute_stage0_law(
        model="3b",
        n_layers=28,
        floor_sig_dir=floor_sig,
        floor_metadata=floor_meta,
        out_dir=tmp_path / "floors",
        faith_sig_dir=faith_sig,
        faith_index=faith_index,
    )
    assert [r.floor_type for r in result.reports] == [
        FloorType.stochastic,
        FloorType.faithfulness_within_device,
        FloorType.faithfulness_cross_device,
    ]
    assert result.stochastic.floor_type is FloorType.stochastic
    for path in result.report_paths:
        assert path.is_file()
    assert result.law_table_path.is_file()
    table = result.law_table_path.read_text()
    for report_ in result.reports:
        assert report_.floor_type.value in table


def test_the_faithfulness_floors_sit_below_the_stochastic_one(tmp_path: Path) -> None:
    """The point of the two passes: a replay drifts less than a resample does."""
    floor_sig, floor_meta = write_floor_bank(tmp_path / "floor")
    faith_sig, faith_index = write_faithfulness_bank(tmp_path / "faith")
    result = compute_stage0_law(
        model="3b", n_layers=28,
        floor_sig_dir=floor_sig, floor_metadata=floor_meta,
        out_dir=tmp_path / "floors",
        faith_sig_dir=faith_sig, faith_index=faith_index,
    )
    whole = {
        report_.floor_type: next(c for c in report_.cells if c.cell == "whole_vector")
        for report_ in result.reports
    }
    assert (
        whole[FloorType.faithfulness_within_device].median
        < whole[FloorType.stochastic].median
    )
    assert whole[FloorType.faithfulness_within_device].median == pytest.approx(0.0, abs=1e-6)


def test_the_law_pass_runs_without_a_faithfulness_corpus(tmp_path: Path) -> None:
    floor_sig, floor_meta = write_floor_bank(tmp_path / "floor")
    result = compute_stage0_law(
        model="3b", n_layers=28,
        floor_sig_dir=floor_sig, floor_metadata=floor_meta, out_dir=tmp_path / "floors",
    )
    assert [r.floor_type for r in result.reports] == [FloorType.stochastic]
