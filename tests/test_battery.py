"""The battery: metrology for the visibility map.

The battery does not classify anything. It answers a prior question — how big a
difference would have to be before it counted, and how many samples it would take
to see one — and then refuses to emit a number that does not carry the answer.
These tests cover the parts that make that refusal real:

  * the stamp gate, which raises rather than warns: an unstamped row is a bug in
    its caller, caught where it is written rather than found in a record file;
  * the law, ``n_min`` per α, which is floored at the smallest n where a
    permutation test can resolve that α at all — power arithmetic alone answers
    n=2 for a large effect, and no test resolves α=0.05 with six relabelings;
  * BH-FDR and the permutation p-value with its ``+1``, so a p of exactly zero is
    unreportable;
  * the paired-delta construction, which pairs only within a prompt class — the
    unit of analysis is the pair, never a raw signature position;
  * the cell masks, which partition a feature name list at four granularities
    through the shared taxonomy;
  * the channel split, which separates the fixed-token part of a deformation from
    the part that only happens because the tokens changed.

Three Wave-1 modules (``decomp``, ``dissoc``, and the report rollup) are typed
containers whose compute functions raise ``NotImplementedError``; their contracts
are tested, and the unimplemented calls are asserted to say so rather than to
return something plausible.

CPU only; no banked data, no model, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anamnesis.analysis.battery import (
    ALPHA_GRID,
    Arm,
    BatteryCell,
    BatteryManifest,
    CellType,
    FloorCell,
    FloorReport,
    FloorType,
    LawParams,
    ResultStamp,
    StampedValue,
    bh_fdr,
    compute_stochastic_floors,
    permutation_pvalue,
)
from anamnesis.analysis.battery.channel import ChannelSplit, decompose_channel
from anamnesis.analysis.battery.decomp import CellVerdict, decompose
from anamnesis.analysis.battery.deltas import (
    ConditionCorpus,
    cross_condition_deltas,
    load_floor_scale,
    location_dispersion,
    within_condition_deltas,
)
from anamnesis.analysis.battery.dissoc import DissociationRow, dissociation_row
from anamnesis.analysis.battery.floors import (
    build_cells,
    floor_cell_from_deltas,
    load_class_labels,
    load_signature_matrix,
    min_n_permutation,
    n_min_for,
    pair_deltas_by_class,
    robust_scale,
)
from anamnesis.analysis.battery.gates import (
    GateError,
    reject_blind_judge_defense,
    require_gated_outcome,
    require_stamp,
)
from anamnesis.analysis.battery.magnitude import decomposed_magnitude
from anamnesis.analysis.battery.manifest import MODEL_META, ModelMeta
from anamnesis.analysis.battery.report import CellResult, VisibilityMap
from anamnesis.analysis.battery.text_decode import BYTE_DEC, byte_decoder, maybe_decode

#: A tiny feature-name set that the shared taxonomy classifies into several cells.
FEATURE_NAMES = [
    "attn_prompt_mass_L8",
    "attn_recency_bias_L16",
    "gate_sparsity_L16",
    "resid_velocity_L24",
]


def stamp(n: int = 24) -> ResultStamp:
    return ResultStamp(n=n, M="3b", law="stage0-3b n_min=24 @alpha=0.0025 k=2",
                       floor_type=FloorType.stochastic)


# ── The stamp gate ────────────────────────────────────────────────────────────
def test_an_unstamped_row_is_refused_not_warned_about() -> None:
    with pytest.raises(GateError, match="unstamped row"):
        require_stamp({"cell": "whole_vector", "value": 0.4}, context="A1 readout")
    with pytest.raises(GateError, match="stamp missing"):
        require_stamp({"cell": "whole_vector", "stamp": {"n": 24, "M": "3b"}})
    require_stamp({"cell": "whole_vector", "stamp": stamp().model_dump()})


def test_a_verdict_cannot_ship_without_the_gate_that_licenses_it() -> None:
    with pytest.raises(GateError, match="12d violation"):
        require_gated_outcome({"outcome": "CARRIES"}, "outcome", ["q_own_tail", "bh_family"])
    require_gated_outcome(
        {"outcome": "CARRIES", "q_own_tail": 0.01, "bh_family": "A1"},
        "outcome", ["q_own_tail", "bh_family"],
    )
    require_gated_outcome({"no_outcome_here": 1}, "outcome", ["q_own_tail"])


def test_a_blind_judge_failure_cannot_be_read_as_class_membership() -> None:
    clean = [{"status": "MEMBER", "row": "socratic-3B", "judge_gap": 0.12,
              "hardening": "2AFC hardened"}]
    reject_blind_judge_defense(clean)
    # A small gap is not quotable, so it needs no hardening annotation.
    reject_blind_judge_defense([{"status": "MEMBER", "judge_gap": 0.02}])
    reject_blind_judge_defense([{"status": "NON-MEMBER", "judge_gap": 0.5}])
    with pytest.raises(GateError, match="12g codicil"):
        reject_blind_judge_defense([
            {"status": "MEMBER", "row": "x", "model": "3b", "judge_gap": 0.2,
             "hardening": "blind-k-way judge could not tell them apart"},
        ])


# ── Statistics ────────────────────────────────────────────────────────────────
def test_bh_fdr_adjusts_upward_and_keeps_the_p_order() -> None:
    reject, adjusted = bh_fdr([0.001, 0.02, 0.4, 0.9], alpha=0.05)
    assert adjusted.shape == (4,)
    assert np.all(adjusted >= np.array([0.001, 0.02, 0.4, 0.9]) - 1e-12)
    assert np.all(np.diff(adjusted) >= -1e-12), "monotone in the sorted p-values"
    assert reject.tolist() == [True, True, False, False]
    empty_reject, empty_adj = bh_fdr([])
    assert empty_reject.shape == (0,) and empty_adj.shape == (0,)


def test_a_permutation_p_value_carries_the_plus_one() -> None:
    null = np.zeros(99)
    assert permutation_pvalue(1.0, null) == pytest.approx(1 / 100)
    assert permutation_pvalue(-1.0, null) == pytest.approx(100 / 100)
    assert permutation_pvalue(-1.0, null, alternative="less") == pytest.approx(1 / 100)
    assert permutation_pvalue(5.0, np.array([-1.0, 0.0, 1.0]), alternative="two-sided") \
        == pytest.approx(1 / 4)
    with pytest.raises(ValueError, match="empty null"):
        permutation_pvalue(1.0, np.array([]))
    with pytest.raises(ValueError, match="unknown alternative"):
        permutation_pvalue(1.0, null, alternative="sideways")


def test_a_stamped_value_is_frozen_and_can_point_at_its_artifact() -> None:
    value = StampedValue(value=0.42, stamp=stamp(), raw_artifact="outputs/battery/x.json")
    assert value.value == 0.42
    assert value.stamp.floor_type is FloorType.stochastic
    with pytest.raises(Exception):
        value.value = 0.5  # type: ignore[misc]


# ── The law ───────────────────────────────────────────────────────────────────
def test_the_law_is_floored_at_what_a_permutation_test_can_resolve() -> None:
    # Power arithmetic alone answers n=2 for a huge effect; the permutation
    # resolution floor is what stops that being reported as a plan.
    assert n_min_for(effect_d=50.0, alpha=0.05, power=0.9) == min_n_permutation(0.05)
    assert min_n_permutation(0.05) > 2
    assert min_n_permutation(1e-4) > min_n_permutation(0.05), "a stricter α needs more n"
    # A degenerate effect is unpowerable rather than cheap.
    assert n_min_for(effect_d=0.0, alpha=0.05, power=0.9) == 10**9
    # n_min falls as the effect grows and rises as α tightens.
    assert n_min_for(0.5, 0.05, 0.9) > n_min_for(1.0, 0.05, 0.9)
    assert n_min_for(0.5, 1e-4, 0.9) >= n_min_for(0.5, 0.05, 0.9)
    assert ALPHA_GRID == (0.05, 0.01, 1e-3, 1e-4)


def test_a_floor_cell_carries_an_n_min_row_for_every_alpha() -> None:
    rng = np.random.default_rng(3)
    deltas = np.abs(rng.standard_normal(500)).astype(np.float32) + 1.0
    cell = floor_cell_from_deltas(
        cell="whole_vector", deltas=deltas, n_features=4, model="3b",
        floor_type=FloorType.stochastic, law=LawParams(),
    )
    assert isinstance(cell, FloorCell)
    assert cell.n_pairs == 500
    assert cell.median > 0 and cell.mad > 0
    for grid in (cell.n_min_by_alpha, cell.n_min_by_alpha_rank, cell.n_min_by_alpha_robust):
        assert len(grid) == len(ALPHA_GRID)
        assert all(isinstance(v, int) and v > 0 for v in grid.values())
    # The rank test pays for its robustness in samples.
    for key in cell.n_min_by_alpha:
        assert cell.n_min_by_alpha_rank[key] >= cell.n_min_by_alpha[key]
    assert cell.exact_zero is False


def test_an_exactly_zero_floor_is_recorded_as_such() -> None:
    cell = floor_cell_from_deltas(
        cell="whole_vector", deltas=np.zeros(64, dtype=np.float32), n_features=4,
        model="8b", floor_type=FloorType.faithfulness, law=LawParams(),
    )
    assert cell.exact_zero is True, "bitwise-deterministic replay has no floor to clear"


def test_the_law_params_document_their_own_reading() -> None:
    law = LawParams()
    assert law.k == 2.0 and law.power == 0.9
    assert "location shift" in law.shift_reading


# ── Cells and paired deltas ───────────────────────────────────────────────────
def test_cells_partition_a_name_list_at_four_granularities() -> None:
    cells = build_cells(FEATURE_NAMES, n_layers=32)
    assert cells["whole_vector"].all()
    assert any(k.startswith("family:") for k in cells)
    assert any(k.startswith("source:") for k in cells)
    assert any(k.startswith("source_band:") for k in cells)
    for key, mask in cells.items():
        assert mask.shape == (len(FEATURE_NAMES),), key
        assert mask.dtype == bool
        assert mask.any(), f"{key} claims no features"


def write_bank(root: Path, per_gen: dict[int, np.ndarray], *, modes: dict[int, str]) -> Path:
    """A battery-shaped bank: signatures/ plus metadata.json."""
    sig_dir = root / "signatures"
    sig_dir.mkdir(parents=True)
    gens = []
    for gid, vec in per_gen.items():
        np.savez(sig_dir / f"gen_{gid:03d}.npz",
                 feature_names=np.array(FEATURE_NAMES),
                 features=vec.astype(np.float32))
        gens.append({"generation_id": gid, "mode": modes[gid], "topic_idx": gid % 2,
                     "mode_idx": 0, "num_generated_tokens": 100})
    (root / "metadata.json").write_text(json.dumps({"generations": gens}))
    return sig_dir


@pytest.fixture
def bank(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(17)
    per_gen = {gid: rng.standard_normal(len(FEATURE_NAMES)) for gid in range(8)}
    modes = {gid: "linear" if gid % 2 == 0 else "socratic" for gid in range(8)}
    sig_dir = write_bank(tmp_path / "cond_a", per_gen, modes=modes)
    return sig_dir, tmp_path / "cond_a" / "metadata.json"


def test_signature_loading_and_robust_scaling(bank: tuple[Path, Path]) -> None:
    sig_dir, metadata_path = bank
    X, names, gen_ids = load_signature_matrix(sig_dir)
    assert X.shape == (8, len(FEATURE_NAMES))
    assert names == FEATURE_NAMES
    assert gen_ids == list(range(8))
    med, scale = robust_scale(X)
    assert med.shape == scale.shape == (len(FEATURE_NAMES),)
    assert np.all(scale > 0), "a zero scale would divide by zero downstream"
    labels = load_class_labels(metadata_path)
    assert set(labels) == set(range(8))
    with pytest.raises(FileNotFoundError, match="no gen_"):
        load_signature_matrix(sig_dir.parent)


def test_pairs_are_drawn_within_a_prompt_class_only(bank: tuple[Path, Path]) -> None:
    sig_dir, metadata_path = bank
    X, names, gen_ids = load_signature_matrix(sig_dir)
    med, scale = robust_scale(X)
    Z = ((X - med) / scale).astype(np.float32)
    labels = load_class_labels(metadata_path)
    cells = build_cells(names, n_layers=32)
    deltas = pair_deltas_by_class(Z, gen_ids, labels, cells)
    # Two classes of four gens each: C(4,2) = 6 pairs per class, 12 in total —
    # never the 28 an all-pairs construction would give.
    assert len(deltas["whole_vector"]) == 12
    assert all(len(v) == 12 for v in deltas.values())
    assert np.all(deltas["whole_vector"] >= 0), "deltas are |Δz|"


def test_cross_and_within_condition_deltas_and_the_mover_spreader_axis(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(23)
    modes = {gid: "linear" if gid % 2 == 0 else "socratic" for gid in range(8)}
    base = {gid: rng.standard_normal(len(FEATURE_NAMES)) for gid in range(8)}
    a_dir = write_bank(tmp_path / "a", base, modes=modes)
    # Condition b is the same cloud moved along every feature and spread wider.
    shifted = {gid: base[gid] * 3.0 + 10.0 for gid in range(8)}
    b_dir = write_bank(tmp_path / "b", shifted, modes=modes)

    med, scale = load_floor_scale(a_dir)
    a = ConditionCorpus(a_dir, tmp_path / "a" / "metadata.json", med, scale, "native")
    b = ConditionCorpus(b_dir, tmp_path / "b" / "metadata.json", med, scale, "dosed")
    cells = build_cells(FEATURE_NAMES, n_layers=32)

    within = within_condition_deltas(a, cells)
    assert len(within["whole_vector"]) == 12
    cross = cross_condition_deltas(a, b, cells)
    # Same-class cross pairs: 4×4 per class, two classes.
    assert len(cross["whole_vector"]) == 32
    capped = cross_condition_deltas(a, b, cells, max_pairs_per_class=3, seed=0)
    assert len(capped["whole_vector"]) == 6

    split = location_dispersion(a, b, cells)
    assert split["whole_vector"]["centroid_shift"] > 0, "b moved"
    assert split["whole_vector"]["dispersion_ratio"] > 1.0, "b is wider"
    assert split["whole_vector"]["n_a"] == 8 and split["whole_vector"]["n_b"] == 8


def test_a_condition_with_a_different_feature_set_is_refused(tmp_path: Path) -> None:
    modes = {0: "linear", 1: "linear"}
    a_dir = write_bank(tmp_path / "a", {0: np.zeros(4), 1: np.ones(4)}, modes=modes)
    med, scale = load_floor_scale(a_dir)
    with pytest.raises(ValueError, match="must share the extraction feature set"):
        ConditionCorpus(a_dir, tmp_path / "a" / "metadata.json", med[:2], scale[:2], "short")


def test_decomposed_magnitude_reports_shift_dispersion_and_their_nulls(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(29)
    modes = {gid: "linear" if gid % 2 == 0 else "socratic" for gid in range(8)}
    base = {gid: rng.standard_normal(len(FEATURE_NAMES)) for gid in range(8)}
    a_dir = write_bank(tmp_path / "a", base, modes=modes)
    b_dir = write_bank(tmp_path / "b", {g: base[g] + 8.0 for g in base}, modes=modes)
    med, scale = load_floor_scale(a_dir)
    a = ConditionCorpus(a_dir, tmp_path / "a" / "metadata.json", med, scale, "native")
    b = ConditionCorpus(b_dir, tmp_path / "b" / "metadata.json", med, scale, "dosed")
    cells = {"whole_vector": np.ones(len(FEATURE_NAMES), dtype=bool)}
    out = decomposed_magnitude(a, b, cells, n_perm=40, seed=0)["whole_vector"]
    assert out["centroid_shift"] > 0
    assert 0.0 < out["p_shift"] <= 1.0
    assert out["n_a"] == 8 and out["n_b"] == 8 and out["n_perm"] == 40
    assert out["p_shift"] <= 0.05, "a large planted shift should beat its own null"


def test_a_condition_pair_with_no_shared_prompt_class_is_refused(tmp_path: Path) -> None:
    a_dir = write_bank(tmp_path / "a", {0: np.zeros(4), 1: np.ones(4)},
                       modes={0: "linear", 1: "linear"})
    b_dir = write_bank(tmp_path / "b", {0: np.zeros(4), 1: np.ones(4)},
                       modes={0: "socratic", 1: "socratic"})
    med, scale = load_floor_scale(a_dir)
    a = ConditionCorpus(a_dir, tmp_path / "a" / "metadata.json", med, scale, "a")
    b = ConditionCorpus(b_dir, tmp_path / "b" / "metadata.json", med, scale, "b")
    cells = {"whole_vector": np.ones(4, dtype=bool)}
    with pytest.raises(ValueError, match="no shared prompt classes"):
        cross_condition_deltas(a, b, cells)


def test_stochastic_floors_compile_a_report_over_a_bank(bank: tuple[Path, Path]) -> None:
    sig_dir, metadata_path = bank
    report = compute_stochastic_floors(
        sig_dir=sig_dir, metadata_path=metadata_path, model="3b", n_layers=32,
    )
    assert isinstance(report, FloorReport)
    assert report.floor_type is FloorType.stochastic
    assert report.model == "3b"
    assert report.cells, "no cells in the floor report"
    assert any(c.cell == "whole_vector" for c in report.cells)
    out = sig_dir.parent / "floors" / "report.json"
    report.save(out)
    assert json.loads(out.read_text())["model"] == "3b"


# ── Channel split ─────────────────────────────────────────────────────────────
def test_the_channel_split_separates_fixed_token_from_token_mediated() -> None:
    mask = np.ones(4, dtype=bool)
    # The free-gen shift is exactly the direct part: nothing is token-mediated.
    aligned = decompose_channel(np.array([1.0, 1.0, 1.0, 1.0]),
                                np.array([1.0, 1.0, 1.0, 1.0]), mask)
    assert aligned["fraction_direct"] == pytest.approx(1.0)
    assert aligned["cos_direct_freegen"] == pytest.approx(1.0)
    assert aligned["token_mediated_rms_z"] == pytest.approx(0.0)

    # No fixed-token deformation at all: the whole effect rode on the tokens.
    mediated = decompose_channel(np.zeros(4), np.array([2.0, 0.0, 0.0, 0.0]), mask)
    assert mediated["fraction_direct"] == pytest.approx(0.0)
    assert mediated["direct_at_floor"] is True, "a zero direct part sits at the replay floor"
    assert mediated["cos_direct_freegen"] == 0.0

    partial = decompose_channel(np.array([1.0, 0.0, 0.0, 0.0]),
                                np.array([2.0, 0.0, 0.0, 0.0]), mask)
    assert 0.0 < partial["fraction_direct"] < 1.0
    assert partial["direct_at_floor"] is False


def test_the_channel_container_carries_stamps_beside_the_split() -> None:
    split = ChannelSplit(
        cell_id="A5_activation_write|3b|free_gen|alpha=0.3",
        direct=StampedValue(value=0.1, stamp=stamp()),
        token_mediated=StampedValue(value=0.4, stamp=stamp()),
        direct_at_floor=False,
    )
    assert split.direct.stamp.M == "3b"
    assert "A5" in split.cell_id


# ── The manifest ──────────────────────────────────────────────────────────────
def test_the_manifest_refuses_a_duplicate_cell() -> None:
    manifest = BatteryManifest()
    cell = BatteryCell(arm=Arm.A1_sampling, model="3b", cell_type=CellType.free_gen,
                       floor_type=FloorType.stochastic, dose="T=0.9",
                       confirmatory_cells=["source:output", "whole_vector"])
    manifest.add(cell)
    with pytest.raises(ValueError, match="duplicate battery cell"):
        manifest.add(cell)
    assert cell.cell_id() == "A1_sampling|3b|free_gen|T=0.9"
    assert manifest.by_arm(Arm.A1_sampling) == [cell]
    assert manifest.by_model("3b") == [cell]
    assert manifest.by_model("8b") == []


def test_only_pre_registered_confirmatory_cells_inflate_the_family_size() -> None:
    manifest = BatteryManifest()
    manifest.add(BatteryCell(arm=Arm.A1_sampling, model="3b", cell_type=CellType.free_gen,
                             floor_type=FloorType.stochastic, dose="T=0.9",
                             confirmatory_cells=["source:output", "whole_vector"]))
    manifest.add(BatteryCell(arm=Arm.A3_processing_strategy, model="8b",
                             cell_type=CellType.free_gen,
                             floor_type=FloorType.stochastic, dose="socratic"))
    assert manifest.confirmatory_m() == 2, "an exploratory-only cell contributes zero"


def test_every_onboarded_model_declares_its_layers_and_its_floor_bank() -> None:
    assert set(MODEL_META) >= {"3b", "8b", "qwen-7b", "olmo2-7b", "gemma3-27b", "dsv2-lite"}
    for key, meta in MODEL_META.items():
        assert isinstance(meta, ModelMeta)
        assert meta.label == key
        assert meta.n_layers > 0
        assert meta.stage0_dir.startswith("vmb_stage0_")
        assert meta.native_temperature > 0


# ── Wave-1 containers ─────────────────────────────────────────────────────────
def test_the_unimplemented_wave_one_readouts_say_so() -> None:
    report = FloorReport(model="3b", floor_type=FloorType.stochastic, n_gens=8,
                         n_pairs_total=12, corpus="outputs/battery/vmb_stage0_3b",
                         law=LawParams(), cells=[])
    with pytest.raises(NotImplementedError, match="family decomposition"):
        decompose(deltas=None, floor=report)
    with pytest.raises(NotImplementedError, match="dissociation"):
        dissociation_row("A1_sampling|3b|free_gen|T=0.9", None, None)


def test_the_visibility_map_rolls_up_stamped_cells(tmp_path: Path) -> None:
    verdict = CellVerdict(cell="source:attention",
                          effect=StampedValue(value=0.7, stamp=stamp()),
                          passes_ruler=True, ruler_k=2.0, confirmatory=True)
    row = DissociationRow(cell_id="A1_sampling|3b|free_gen|T=0.9",
                          token_kl=StampedValue(value=0.0, stamp=stamp()),
                          signature_effect=StampedValue(value=0.7, stamp=stamp()),
                          direction="signature-only")
    result = CellResult(
        cell=BatteryCell(arm=Arm.A1_sampling, model="3b", cell_type=CellType.free_gen,
                         floor_type=FloorType.stochastic, dose="T=0.9"),
        verdicts=[verdict], dissociation=row,
        raw_artifacts=["outputs/battery/a1_3b/deltas.json"],
    )
    out = tmp_path / "map.json"
    VisibilityMap(cells=[result]).save(out)
    payload = json.loads(out.read_text())
    assert payload["cells"][0]["verdicts"][0]["cell"] == "source:attention"
    assert payload["cells"][0]["raw_artifacts"], "a claim ships with its raw artifact"


# ── Banked text ───────────────────────────────────────────────────────────────
def test_byte_bpe_text_is_decoded_only_when_it_needs_to_be() -> None:
    assert maybe_decode("plain text stays plain") == "plain text stays plain"
    assert maybe_decode("ĠhelloĊworld") == " hello\nworld"
    assert maybe_decode(maybe_decode("ĠhelloĊworld")) == " hello\nworld", "idempotent"
    assert byte_decoder() == BYTE_DEC
    assert len(BYTE_DEC) == 256, "the map covers every byte"
