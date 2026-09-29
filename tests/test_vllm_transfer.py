"""The transfer check's decision, one test per gate, on synthetic vectors.

An unmodified sample (``synthetic_extension``) passes and yields fixtures and a
tolerance under the extension's lane id; each test then breaks one thing. The
device-side runner's refusals before any device work, and the anchor's incremental
path against the one-forward replay on the tiny model, close the file.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError

from anamnesis.extraction.replay.cached import replay_extract_incremental
from anamnesis.extraction.replay.extract import replay_extract
from anamnesis.extraction.vllm import runtime
from anamnesis.extraction.vllm.conformance import (
    CapturedFixture,
    FixtureSet,
    HostFingerprint,
    RowComponent,
    Tolerance,
    decide,
)
from anamnesis.extraction.vllm.envelope import extension_lane_id, lane_id
from anamnesis.extraction.vllm.transfer import (
    BASE_MAX_FLOOR,
    ORDINARY_RULE,
    RULES,
    RULES_FILE,
    UNRANKED,
    TransferReceipt,
    BifurcationReport,
    TransferRules,
    bifurcation_report,
    load_transfer_rules,
    ordinary_rows,
    path_ruler,
    select_strata,
)
from anamnesis.scripts.transfer_vllm import main as transfer_main
from anamnesis.scripts.transfer_vllm import run_transfer
from synthetic_extension import (
    ATTENTION,
    BASE,
    CHECKPOINT,
    KEY,
    NAMES,
    PRESET,
    SUBSTRATE,
    base_rules,
    base_tolerance,
    registry,
    repeats,
    run,
    sample,
    ship_base,
    shifted,
    with_candidate,
)
from synthetic_runtime import HookPlan, loaded_tiny_model


def ordinary(s):
    return list(ordinary_rows(s.ids))


def reasons_of(result):
    return " | ".join(result.receipt.reasons)


# --- the pass ------------------------------------------------------------------------


def test_a_sample_inside_the_base_regime_passes_with_fixtures_and_a_tolerance(tmp_path):
    s = sample()
    result = run(s)
    receipt, fixtures, tolerance = result.receipt, result.fixtures, result.tolerance
    assert receipt.verdict == "pass" and receipt.reasons == ()
    assert receipt.lane_id == fixtures.lane_id == extension_lane_id(BASE, CHECKPOINT) \
        != lane_id(BASE)
    assert (receipt.fixture_digest, receipt.tolerance_digest) == (fixtures.digest,
                                                                  tolerance.digest)
    assert fixtures.model == tolerance.model == KEY
    assert all(fixtures.vectors[g].tobytes() == s.candidate[g].tobytes() for g in s.ids)
    assert [r.generation_id for r in fixtures.rows if r.selected_by == ORDINARY_RULE] \
        == ordinary(s) == list(receipt.ordinary_rows)
    assert tolerance.families == base_tolerance().families
    ratios = [r.component_ratios["attention"] for r in receipt.rows]
    component = next(c for c in tolerance.components if c.name == "attention")
    assert component.max_ratio == pytest.approx(max(ratios))
    fixtures.save(tmp_path / "f")
    assert FixtureSet.load(tmp_path / "f").digest == fixtures.digest
    fp = HostFingerprint(gpu_name="g", gpu_uuid="u", driver="d", cuda_runtime="c", torch="t",
                         vllm="v", anamnesis="x", checkpoint_sha256=CHECKPOINT,
                         engine_settings_sha256="e" * 64, fixture_digest=fixtures.digest,
                         tolerance_digest=tolerance.digest)
    captured = [CapturedFixture(generation_id=g, first=v, repeat=v, batched=v)
                for g, v in fixtures.vectors.items()]
    assert decide(fixtures, tolerance, fp, captured).tier == "identical"


def _ratio_sample(ratios, seed):
    """A sample whose substrate ratio on row i is ``ratios[i]`` (floors are 1)."""
    s = sample(seed=seed, scale=0.01)
    residual = [NAMES.index(f"res_{i}") for i in range(4)]
    updates = {}
    for g, ratio in zip(s.ids, ratios):
        updates[g] = s.candidate[g].copy()
        updates[g][residual] = s.reference[g][residual] + np.float32(ratio / 2.0)
    return with_candidate(s, updates)


def base_from_draws(population):
    base = base_tolerance(residual_max=100.0)
    substrate = RowComponent(name="substrate", feature_names=SUBSTRATE,
                             max_ratio=float(population.max()),
                             p90_ratio=float(np.percentile(population, 90)),
                             p99_ratio=float(np.percentile(population, 99)), source="draws")
    return base.model_copy(update=dict(components=(substrate, base.components[1])))


def test_samples_drawn_from_the_base_ratio_distribution_mostly_pass():
    """The base's ceilings come from 190 draws; fine-tunes drawing from the same
    distribution are refused only in single-digit percent of draws."""
    tolerance = base_from_draws(np.random.default_rng(11).lognormal(0.0, 0.35, size=190))
    verdicts = [run(_ratio_sample(np.random.default_rng(1000 + seed).lognormal(0.0, 0.35, 44),
                                  seed), tolerance).receipt.verdict for seed in range(60)]
    assert verdicts.count("pass") / len(verdicts) >= 0.9


# --- each gate ------------------------------------------------------------------------


LOOSE = base_tolerance(residual_max=100.0)
"""A base tolerance whose residual family never binds, so row gates read alone."""


@pytest.mark.parametrize("amounts,verdict,expected", [
    ((6.0, 6.0), "pass", ""),
    ((6.0, 6.0, 6.0), "refuse", "substrate: rows"),
    ((25.0,), "pass", ""),
])
def test_rows_over_the_base_max_ratio_are_counted_and_capped(amounts, verdict, expected):
    s = sample()
    others = [g for g in s.ids if g not in ordinary(s)]
    s = with_candidate(s, {g: shifted(s, g, "res_0", a) for g, a in zip(others, amounts)})
    result = run(s, LOOSE)
    assert result.receipt.verdict == verdict and expected in reasons_of(result)
    assert result.receipt.rows_over_max["substrate"] == tuple(others[:len(amounts)])
    assert result.receipt.worst_max_multiple["substrate"] == pytest.approx(max(amounts) / 5.0,
                                                                          rel=0.05)


@pytest.mark.parametrize("count,verdict", [(1, "pass"), (2, "refuse")])
def test_ordinary_rows_over_the_base_p99_are_counted(count, verdict):
    s = sample()
    rows = ordinary(s)[:count]
    result = run(with_candidate(s, {g: shifted(s, g, "res_2", 4.5) for g in rows}), LOOSE)
    assert result.receipt.verdict == verdict
    assert result.receipt.ordinary_over_p99["substrate"] == count
    assert result.receipt.ordinary_over_p90["substrate"] == count
    if count == 2:
        assert "substrate: 2 of 16 ordinary rows exceed the recorded p99" in reasons_of(result)


def test_the_ordinary_median_over_the_base_p90_refuses():
    s = sample()
    s = with_candidate(s, {g: shifted(s, g, "res_0", 3.4) for g in ordinary(s)})
    assert "substrate: median ratio 3.4" in reasons_of(run(s, LOOSE))


def test_a_row_over_a_family_maximum_is_reported_not_refused():
    """Eleven per-row maxima read from about 190 rows refuse most in-regime samples, so
    a single row over one is listed; the drift gate does the gating."""
    s = sample()
    gid = s.ids[5]
    result = run(with_candidate(s, {gid: shifted(s, gid, "attn_flow_0", 1.5)}))
    assert result.receipt.verdict == "pass", result.receipt.reasons
    assert result.receipt.family_rows_over_max == {"attn-flow": (gid,)}


def test_a_far_outlier_is_listed_with_its_multiple_and_its_diagnostic_not_refused():
    """A single row's extremity is reported whatever its size: the row is listed with its
    multiple in its component and family, beside the diagnostic, and the sample passes."""
    s = sample()
    gid = [g for g in s.ids if g not in ordinary(s)][0]
    s = with_candidate(s, {gid: shifted(s, gid, "attn_flow_0", 30.0)})
    report = BifurcationReport(peak=1.2, token=2, block=5, channel_ratio=9.5, channel_block=4,
                               channel_side="anchor", channel=3, other_fraction=0.03)
    result = run(s, diagnostics={gid: report})
    assert result.receipt.verdict == "pass", result.receipt.reasons
    assert result.receipt.over_max_multiples["family:attn-flow"][gid] > 20
    assert gid in result.receipt.over_max_multiples["attention"]
    assert result.receipt.diagnostics[gid] == report


def test_a_family_drifting_across_the_ordinary_rows_refuses():
    """The drift gate: the ordinary rows' median family deviation over the base's p90
    refuses even when no row reaches the family maximum."""
    s = sample()
    s = with_candidate(s, {g: shifted(s, g, "attn_flow_0", 0.8) for g in ordinary(s)})
    result = run(s)
    assert "family attn-flow: the ordinary rows' median" in reasons_of(result)
    assert result.receipt.family_ordinary_median["attn-flow"] > 0.5


def test_a_receipt_names_its_rule_set_and_weakly_guarded_families():
    receipt = run(sample()).receipt
    assert receipt.rules_digest == base_rules().digest
    assert receipt.weakly_guarded == {"residual": 2.5}


def test_the_rule_set_must_be_the_base_s_for_its_tolerance():
    with pytest.raises(ValueError, match="rule set given"):
        run(sample(), rules=base_rules(tolerance_digest="f" * 64))


# --- the bifurcation diagnostic -----------------------------------------------------------


def _hidden(layers=10, tokens=6, width=8, seed=0):
    rng = np.random.default_rng(seed)
    anchor = rng.normal(size=(tokens, layers + 1, width)).astype(np.float32) + 3.0
    lane = np.transpose(anchor[:, 1:, :], (1, 0, 2)).copy()
    return lane, anchor


def test_the_diagnostic_finds_the_peak_and_a_one_sided_massive_channel():
    """Block 5 (anchor state 6) of token 2: the anchor's channel 3 at 40× the row's usual
    largest channel, the lane's same channel ordinary."""
    lane, anchor = _hidden()
    assert bifurcation_report(lane, anchor, 0.35).peak == 0.0
    anchor[2, 6, 3] = 40.0 * np.median(np.abs(anchor[:, 6, :]).max(axis=-1))
    report = bifurcation_report(lane, anchor, 0.35)
    assert (report.token, report.block, report.channel_side, report.channel,
            report.channel_block) == (2, 5, "anchor", 3, 5)
    assert report.channel_ratio == pytest.approx(40.0, rel=0.05) and report.other_fraction < 0.1


def test_a_channel_both_sides_share_reads_as_symmetric():
    lane, anchor = _hidden()
    big = 40.0 * np.median(np.abs(anchor[:, 6, :]).max(axis=-1))
    anchor[2, 6, 3], lane[5, 2, 3] = big, 1.8 * big
    report = bifurcation_report(lane, anchor, 0.35)
    assert report.channel_side == "lane" and report.other_fraction > 0.5


def test_the_diagnostic_skips_early_blocks_and_the_normed_last_state():
    lane, anchor = _hidden()
    lane[1, 0] *= 5.0
    lane[9, 0] *= 5.0
    assert bifurcation_report(lane, anchor, 0.35).peak == 0.0


def test_the_diagnostic_refuses_misaligned_states():
    lane, anchor = _hidden()
    with pytest.raises(ValueError, match="same tokens"):
        bifurcation_report(lane[:, :-1], anchor, 0.35)


# --- the shipped rule sets ----------------------------------------------------------------


@pytest.mark.parametrize("model", ["3b", "8b", "70b"])
def test_every_shipped_base_carries_a_rule_set_for_its_tolerance(model):
    _, tolerance = runtime.load_fixtures(model)
    rules = load_transfer_rules(model, tolerance)
    assert rules.qualification["false_refusal"] < 0.1
    assert rules.component_max_allowance == 2 and rules.p99_allowance == 1
    assert set(rules.weakly_guarded) <= set(tolerance.family_max_abs_sigma)


def test_a_base_without_a_rule_set_is_refused_by_name(tmp_path, monkeypatch):
    tolerance = ship_base(tmp_path, monkeypatch)
    (tmp_path / BASE / RULES_FILE).unlink()
    with pytest.raises(ValueError, match="no transfer rules are qualified for '8b'"):
        load_transfer_rules(BASE, tolerance)


def _swap(strata):
    ordinary_row = next(g for g, v in strata.items() if v == ORDINARY_RULE)
    ranked = next(g for g, v in strata.items() if v != ORDINARY_RULE)
    return {**strata, ordinary_row: strata[ranked], ranked: ORDINARY_RULE}


def _determinism_broken(s):
    captured = repeats(s)
    wrong = captured[3].batched.copy()
    wrong[0] += 1e-3
    captured[3] = captured[3].model_copy(update=dict(batched=wrong))
    return captured


def _first_not_scored(s):
    captured = repeats(s)
    other = captured[0].first + np.float32(1.0)
    captured[0] = CapturedFixture(generation_id=captured[0].generation_id, first=other,
                                  repeat=other, batched=other)
    return captured


@pytest.mark.parametrize("case,expected", [
    ("disagrees", "lane disagrees with itself"),
    ("few", "at least 16 distinct"),
    ("first", "not the determinism check's first capture"),
    ("stride", "ordinary stratum must be the rows evenly spaced"),
    ("cover", "strata must name every sampled row"),
    ("small", "the sample holds 43 rows"),
])
def test_each_precondition_refuses(case, expected):
    s = sample(n_rows=43) if case == "small" else sample()
    strata = {g: ORDINARY_RULE for g in s.ids} if case == "small" \
        else select_strata(s, base_tolerance())
    determinism = {"disagrees": _determinism_broken, "first": _first_not_scored,
                   "few": lambda s: repeats(s)[:15]}.get(case, repeats)(s)
    if case == "stride":
        strata = _swap(strata)
    if case == "cover":
        strata.pop(s.ids[0])
    result = run(s, strata=strata, determinism=determinism)
    assert result.receipt.verdict == "refuse" and expected in reasons_of(result)


def test_a_schema_other_than_the_base_s_refuses_before_scoring():
    from anamnesis.extraction.vllm.transfer import check_transfer

    s = sample()
    result = check_transfer(key=KEY, extends=BASE, base_tolerance=base_tolerance(),
                            base_feature_names=NAMES[::-1], checkpoint_sha256=CHECKPOINT,
                            calibration_sha256="a" * 64, sample=s,
                            strata=select_strata(s, base_tolerance()), determinism=repeats(s),
                            max_floor=None, rules=base_rules())
    assert "feature schema" in reasons_of(result) and result.receipt.rows == ()


@pytest.mark.parametrize("key,tolerance", [("70b", None),
                                           (KEY, base_tolerance(median_gate_stratum=None))])
def test_caller_errors_raise_rather_than_refuse(key, tolerance):
    with pytest.raises(ValueError):
        run(sample(), tolerance, key=key)


def test_a_receipt_cannot_pass_without_its_outputs_or_refuse_without_reasons():
    receipt = run(sample()).receipt
    for change in ({"fixture_digest": None}, {"verdict": "refuse"}):
        with pytest.raises(ValidationError):
            TransferReceipt.model_validate({**receipt.model_dump(), **change})


# --- fragile rows ------------------------------------------------------------------------


def test_a_row_over_the_floor_limit_is_named_and_left_out_of_the_scoring():
    gid = 107
    s = sample(floors={gid: 10.0})
    s = with_candidate(s, {gid: shifted(s, gid, "attn_flow_1", 80.0)})
    strata = select_strata(s, base_tolerance())
    screened = run(s, strata=strata, max_floor=2.39)
    assert screened.receipt.verdict == "pass", screened.receipt.reasons
    assert screened.receipt.fragile_rows == screened.tolerance.excluded_rows == (gid,)
    assert gid in screened.fixtures.vectors and screened.receipt.family_report["attn-flow"] <= 1
    unscreened = run(s, strata=strata, max_floor=None).receipt
    assert unscreened.fragile_rows == () and gid in unscreened.over_max_multiples["family:attn-flow"]


def test_the_base_floor_limits_name_exactly_the_rows_the_shipped_tolerances_exclude():
    for model, limit in BASE_MAX_FLOOR.items():
        fixtures, tolerance = runtime.load_fixtures(model)
        assert tuple(sorted(r.generation_id for r in fixtures.rows if limit is not None
                            and r.floor_b > limit)) == tolerance.excluded_rows, model


# --- the ruler and the strata ------------------------------------------------------------


def test_the_path_ruler_standardizes_by_the_replay_spread():
    replay = np.asarray([[0.0, 1.0, 5.0], [2.0, 1.0, 5.0], [4.0, 1.0, 5.0]], dtype=np.float32)
    sigma, floors = path_ruler(replay, replay + np.asarray([1.0, 0, 0], dtype=np.float32))
    spread = np.std([0.0, 2.0, 4.0])
    assert sigma[:2] == pytest.approx([spread, spread])
    assert floors == pytest.approx(np.full(3, 1.0 / spread))
    with pytest.raises(ValueError, match="agree exactly"):
        path_ruler(replay[:2], replay[:2] + np.asarray([[1.0, 0, 0], [0, 0, 0]], np.float32))


def test_the_ordinary_rows_are_id_strided_and_the_rest_ranked():
    s = sample(n_rows=50)
    strata = select_strata(s, base_tolerance())
    ids = sorted(s.ids)
    expected = [ids[round((i + 0.5) * 50 / 16 - 0.5)] for i in range(16)]
    assert [g for g, v in strata.items() if v == ORDINARY_RULE] == expected == ordinary(s)
    for rule, count in RULES:
        assert list(strata.values()).count(rule) == count
    assert list(strata.values()).count(UNRANKED) == 50 - 16 - 28
    pool = [g for g in ids if g not in expected]
    top = max(pool, key=lambda g: (np.linalg.norm(s.standardized(g)[:len(ATTENTION)]), -g))
    assert strata[top] == "largest-attention-shift"


def test_the_ordinary_stratum_does_not_depend_on_the_deviations():
    s = sample()
    ids = list(s.ids)
    order = np.random.default_rng(7).permutation(len(ids))
    deviations = [s.candidate[g] - s.reference[g] for g in ids]
    permuted = with_candidate(s, {g: s.reference[g] + deviations[j] for g, j in zip(ids, order)})
    boosted = with_candidate(s, {g: shifted(s, g, "attn_flow_0", 0.9) for g in ordinary(s)})
    before = select_strata(s, base_tolerance())
    for other in (permuted, boosted):
        after = select_strata(other, base_tolerance())
        assert [g for g, v in after.items() if v == ORDINARY_RULE] == ordinary(s)
    assert select_strata(permuted, base_tolerance()) != before


def test_the_strata_refuse_a_tie_everywhere_and_a_sample_out_of_range():
    s = sample()
    gate = [NAMES.index("gate_L0_sparsity_mean"), NAMES.index("gate_L0_sparsity_std")]
    unmoved = {g: np.where(np.isin(np.arange(len(NAMES)), gate), s.reference[g], s.candidate[g])
               for g in s.ids}
    with pytest.raises(ValueError, match="rank by id alone"):
        select_strata(with_candidate(s, unmoved), base_tolerance())
    with pytest.raises(ValueError, match="44 to 60 rows"):
        select_strata(sample(n_rows=61), base_tolerance())


# --- the runner, before any device work ---------------------------------------------------


def _entries(n):
    return {str(100 + i): dict(input_ids=list(range(1, 11)), prompt_length=3, n_gen=7)
            for i in range(n)}


@pytest.mark.parametrize("key,preset_row,n,existing,raises,match", [
    (KEY, {"extends": "3b", "model_id": "x"}, 44, False, ValueError, "does not extend '8b'"),
    ("8b", None, 44, False, ValueError, "already a lane"),
    (KEY, None, 10, False, ValueError, "44 to 60 distinct rows"),
    (KEY, None, 44, True, FileExistsError, "exists"),
])
def test_the_runner_refuses_before_device_work(tmp_path, monkeypatch, key, preset_row, n,
                                               existing, raises, match):
    ship_base(tmp_path / "shipped", monkeypatch)
    registry(tmp_path / "models.json", monkeypatch, row=preset_row)
    if existing:
        (tmp_path / "out").mkdir()
    with pytest.raises(raises, match=match):
        run_transfer(key=key, extends=BASE, preset=PRESET, model_path=tmp_path / "m",
                     calib_dir=tmp_path / "c", entries=_entries(n),
                     ids=list(range(100, 100 + n)), out=tmp_path / "out",
                     work_dir=tmp_path / "work", device="cpu")


def test_the_command_reports_a_refusal_before_device_work_as_exit_two(tmp_path, monkeypatch,
                                                                     capsys):
    ship_base(tmp_path / "shipped", monkeypatch)
    registry(tmp_path / "models.json", monkeypatch)
    manifest = tmp_path / "replay_manifest.json"
    manifest.write_text(json.dumps(dict(entries=_entries(10), n_ok=10, n_flagged=0, flagged=[])))
    assert transfer_main(["--key", KEY, "--extends", BASE, "--preset", PRESET, "--model-path",
                          str(tmp_path / "m"), "--calib-dir", str(tmp_path / "c"),
                          "--manifest", str(manifest), "--out", str(tmp_path / "out")]) == 2
    assert "44 to 60 distinct rows" in capsys.readouterr().err


# --- the anchor's incremental path ---------------------------------------------------------


FULL_PLAN = HookPlan(key_layers=[0, 1, 2], value_layers=[0, 1, 2], query_layers=[0, 1, 2],
                     attn_output_layers=[0, 1, 2], gate_layers=[0, 1, 2])


def test_the_incremental_path_aligns_with_the_one_forward_replay():
    loaded, _ = loaded_tiny_model(FULL_PLAN)
    ids = list(range(1, 13))
    full = replay_extract(loaded, ids, prompt_length=5)
    step = replay_extract_incremental(loaded, ids, prompt_length=5)
    assert step.chosen_token_ids.tolist() == full.chosen_token_ids.tolist()
    assert step.prompt_length == full.prompt_length and len(step.hidden_states) == 6
    for a, b in [*zip(step.hidden_states, full.hidden_states),
                 *zip(step.attentions, full.attentions)]:
        np.testing.assert_allclose(a, b, atol=1e-5)
    for surface in ("pre_rope_keys", "v_proj_values", "queries", "gate_activations",
                    "attn_outputs"):
        mine, theirs = getattr(step, surface), getattr(full, surface)
        assert sorted(mine) == sorted(theirs)
        for layer in mine:
            np.testing.assert_allclose(np.stack(mine[layer]), np.stack(theirs[layer]),
                                       atol=1e-5)
    assert loaded.hook_state.pre_rope_keys == {}
    with pytest.raises(ValueError, match="fewer than two generated tokens"):
        replay_extract_incremental(loaded, list(range(1, 7)), prompt_length=5)
