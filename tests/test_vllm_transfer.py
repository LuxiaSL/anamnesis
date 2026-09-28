"""The transfer check's decision: a fine-tune's deviation, scored with its base's tolerance.

:func:`anamnesis.extraction.vllm.transfer.check_transfer` is pure, so every gate is
exercised here on synthetic vectors (``synthetic_extension``): an unmodified sample
passes and yields fixtures and a tolerance under the extension's lane id; each
gate is then broken on its own — a row over its component ceiling, a family over
its maximum, the ordinary stratum's median over the p90, one ordinary row over the
p99, a lane that disagrees with itself, a schema or sample size other than the
base's — and refuses with its reason. A row whose path floor exceeds the base's
limit is named and left out of the scoring, and the base limit table is held to
the shipped tolerances it describes.

What needs a device is producing the vectors:
:func:`anamnesis.extraction.vllm.transfer_run.run_transfer`, whose refusals before
any device work are pinned at the end, and the incremental anchor path, pinned
against the one-forward replay on the tiny model from ``synthetic_runtime``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError

from anamnesis.extraction.replay.cached import replay_extract_incremental
from anamnesis.extraction.replay.extract import replay_extract
from anamnesis.extraction.vllm import runtime
from anamnesis.extraction.vllm.conformance import CapturedFixture, FixtureSet, Tolerance
from anamnesis.extraction.vllm.envelope import lane_id
from anamnesis.extraction.vllm.transfer import (
    BASE_MAX_FLOOR,
    ORDINARY_RULE,
    RULES,
    UNRANKED,
    TransferReceipt,
    extension_lane_id,
    ordinary_rows,
    path_ruler,
    select_strata,
)
from anamnesis.extraction.vllm.transfer_run import run_transfer
from synthetic_extension import (
    ATTENTION,
    BASE,
    CHECKPOINT,
    KEY,
    NAMES,
    PRESET,
    base_tolerance,
    registry,
    repeats,
    run,
    sample,
    ship_base,
    with_candidate,
)
from synthetic_runtime import HookPlan, loaded_tiny_model

ORDINARY_OF = lambda strata: [g for g, s in strata.items() if s == ORDINARY_RULE]  # noqa: E731


def _shift(s, gid, index, amount):
    vector = s.reference[gid].copy()
    vector[index] += amount
    return vector


# --- the pass --------------------------------------------------------------------


def test_a_sample_inside_the_base_regime_passes_with_fixtures_and_a_tolerance():
    s = sample()
    result = run(s)
    receipt = result.receipt
    assert receipt.verdict == "pass" and receipt.reasons == ()
    assert receipt.lane_id == extension_lane_id(BASE, CHECKPOINT) != lane_id(BASE)
    assert receipt.base_lane_id == lane_id(BASE)
    assert receipt.base_tolerance_digest == base_tolerance().digest
    assert receipt.sample_sha256 == s.digest and receipt.determinism_rows == len(s.rows)
    fixtures, tolerance = result.fixtures, result.tolerance
    assert receipt.fixture_digest == fixtures.digest
    assert receipt.tolerance_digest == tolerance.digest
    assert fixtures.model == KEY and tolerance.model == KEY
    assert fixtures.lane_id == receipt.lane_id and fixtures.checkpoint_sha256 == CHECKPOINT
    assert set(fixtures.vectors) == set(s.ids)
    assert all(fixtures.vectors[g].tobytes() == s.candidate[g].tobytes() for g in s.ids)
    assert tolerance.families == base_tolerance().families
    assert tolerance.discrete == base_tolerance().discrete
    assert tolerance.median_gate_stratum == ORDINARY_RULE
    assert [r.generation_id for r in fixtures.rows if r.selected_by == ORDINARY_RULE] \
        == list(ordinary_rows(s.ids)) == list(receipt.ordinary_rows)
    by_name = {c.name: c for c in tolerance.components}
    ratios = [r.component_ratios["attention"] for r in receipt.rows]
    assert by_name["attention"].max_ratio == pytest.approx(max(ratios))
    assert by_name["attention"].p90_ratio == pytest.approx(np.percentile(ratios, 90))


def test_the_extension_fixtures_and_tolerance_round_trip_through_their_contracts(tmp_path):
    result = run(sample())
    result.fixtures.save(tmp_path / "f")
    assert FixtureSet.load(tmp_path / "f").digest == result.fixtures.digest
    again = Tolerance.model_validate_json(result.tolerance.model_dump_json())
    assert again.digest == result.tolerance.digest


def test_a_host_reproducing_the_extension_fixtures_is_identical_under_its_own_tolerance():
    """The fixtures a pass produces are what a host's install check reads."""
    from anamnesis.extraction.vllm.conformance import HostFingerprint, decide

    result = run(sample())
    fixtures, tolerance = result.fixtures, result.tolerance
    fp = HostFingerprint(gpu_name="g", gpu_uuid="u", driver="d", cuda_runtime="c", torch="t",
                         vllm="v", anamnesis="x", checkpoint_sha256=CHECKPOINT,
                         engine_settings_sha256="e" * 64, fixture_digest=fixtures.digest,
                         tolerance_digest=tolerance.digest)
    captured = [CapturedFixture(generation_id=g, first=v, repeat=v, batched=v)
                for g, v in fixtures.vectors.items()]
    receipt = decide(fixtures, tolerance, fp, captured)
    assert receipt.tier == "identical" and receipt.lane_id == result.receipt.lane_id


# --- each gate refuses on its own ----------------------------------------------------


def test_a_row_over_its_component_ceiling_refuses():
    s = sample()
    gid = s.ids[0]
    s = with_candidate(s, {gid: _shift(s, gid, NAMES.index("res_0"), 6.0)})
    result = run(s, base_tolerance(**LOOSE_RESIDUAL))
    assert result.receipt.verdict == "refuse" and result.fixtures is None
    assert any(f"row {gid}: row distance outside" in r for r in result.receipt.reasons)


def test_a_family_over_its_recorded_maximum_refuses():
    s = sample()
    gid = s.ids[5]
    s = with_candidate(s, {gid: _shift(s, gid, NAMES.index("attn_flow_0"), 1.5)})
    reasons = run(s).receipt.reasons
    assert any(f"row {gid}: family attn-flow" in r for r in reasons)


LOOSE_RESIDUAL = dict(family_max_abs_sigma={"attn-spectral": 1.0, "attn-flow": 1.0,
                                             "residual": 100.0})
"""A base tolerance whose residual family never binds, so a residual deviation is
read by the row gates alone."""


def test_the_ordinary_median_over_the_base_p90_refuses():
    s = sample()
    tolerance = base_tolerance(**LOOSE_RESIDUAL)
    strata = select_strata(s, tolerance)
    ordinary = ORDINARY_OF(strata)
    s = with_candidate(s, {g: _shift(s, g, NAMES.index("res_0"), 3.4) for g in ordinary})
    reasons = run(s, tolerance, strata=strata).receipt.reasons
    assert any("substrate: median ratio 3.4" in r and "p90" in r for r in reasons)
    assert not any("outside the recorded ceiling" in r for r in reasons)


def test_one_ordinary_row_over_the_base_p99_passes_and_is_counted():
    """An in-regime fine-tune puts about one row in a hundred over the base's p99."""
    s = sample()
    tolerance = base_tolerance(**LOOSE_RESIDUAL)
    strata = select_strata(s, tolerance)
    gid = ORDINARY_OF(strata)[0]
    result = run(with_candidate(s, {gid: _shift(s, gid, NAMES.index("res_2"), 4.5)}),
                 tolerance, strata=strata)
    assert result.receipt.verdict == "pass", result.receipt.reasons
    assert result.receipt.ordinary_over_p99 == {"substrate": 1, "attention": 0}
    assert result.receipt.ordinary_over_p90["substrate"] == 1


def test_two_ordinary_rows_over_the_base_p99_refuse_though_under_the_ceiling():
    s = sample()
    tolerance = base_tolerance(**LOOSE_RESIDUAL)
    strata = select_strata(s, tolerance)
    first, second = ORDINARY_OF(strata)[:2]
    shifted = {g: _shift(s, g, NAMES.index("res_2"), 4.5) for g in (first, second)}
    receipt = run(with_candidate(s, shifted), tolerance, strata=strata).receipt
    assert receipt.verdict == "refuse"
    assert any("substrate: 2 of 16 ordinary rows exceed the recorded p99" in r
               for r in receipt.reasons)
    assert not any("outside the recorded ceiling" in r or "median" in r
                   for r in receipt.reasons)
    assert receipt.ordinary_over_p99["substrate"] == 2


def test_a_lane_that_disagrees_with_itself_refuses():
    s = sample()
    captured = repeats(s)
    wrong = captured[3].batched.copy()
    wrong[0] += 1e-3
    captured[3] = captured[3].model_copy(update=dict(batched=wrong))
    reasons = run(s, determinism=captured).receipt.reasons
    assert any("lane disagrees with itself" in r for r in reasons)


def test_too_few_determinism_rows_refuse():
    s = sample()
    reasons = run(s, determinism=repeats(s)[:15]).receipt.reasons
    assert any("at least 16" in r for r in reasons)


def test_the_scored_vector_must_be_the_determinism_first_capture():
    s = sample()
    captured = repeats(s)
    other = captured[0].first + np.float32(1.0)
    captured[0] = CapturedFixture(generation_id=captured[0].generation_id, first=other,
                                  repeat=other, batched=other)
    reasons = run(s, determinism=captured).receipt.reasons
    assert any("not the determinism check's first capture" in r for r in reasons)


def test_a_schema_other_than_the_base_s_refuses():
    from anamnesis.extraction.vllm.transfer import check_transfer

    s = sample()
    result = check_transfer(key=KEY, extends=BASE, base_tolerance=base_tolerance(),
                            base_feature_names=NAMES[::-1], checkpoint_sha256=CHECKPOINT,
                            calibration_sha256="a" * 64, sample=s,
                            strata=select_strata(s, base_tolerance()), determinism=repeats(s),
                            max_floor=None)
    assert any("feature schema" in r for r in result.receipt.reasons)
    assert result.receipt.rows == ()


def test_a_sample_below_the_minimum_refuses():
    s = sample(n_rows=43)
    strata = {g: ORDINARY_RULE for g in s.ids}
    reasons = run(s, strata=strata).receipt.reasons
    assert any("the sample holds 43 rows" in r for r in reasons)


def test_strata_whose_ordinary_rows_are_not_the_id_stride_refuse():
    s = sample()
    strata = select_strata(s, base_tolerance())
    ordinary = ORDINARY_OF(strata)
    ranked = next(g for g, v in strata.items() if v != ORDINARY_RULE)
    strata[ordinary[0]], strata[ranked] = strata[ranked], ORDINARY_RULE
    reasons = run(s, strata=strata).receipt.reasons
    assert any("ordinary stratum must be the rows evenly spaced" in r for r in reasons)


def test_strata_must_cover_the_sample():
    s = sample()
    strata = select_strata(s, base_tolerance())
    strata.pop(s.ids[0])
    assert any("strata must name every sampled row" in r
               for r in run(s, strata=strata).receipt.reasons)


# --- fragile rows ----------------------------------------------------------------


def test_a_row_over_the_floor_limit_is_named_and_left_out_of_the_scoring():
    base = sample()
    gid = base.ids[7]
    s = sample(floors={gid: 10.0})
    vector = s.reference[gid].copy()
    vector[NAMES.index("attn_flow_1")] += 80.0
    s = with_candidate(s, {gid: vector})
    tolerance = base_tolerance()
    strata = select_strata(s, tolerance)
    scored = run(s, tolerance, strata=strata, max_floor=2.39)
    assert scored.receipt.verdict == "pass", scored.receipt.reasons
    assert scored.receipt.fragile_rows == (gid,)
    assert scored.tolerance.excluded_rows == (gid,)
    assert gid in scored.fixtures.vectors
    assert scored.receipt.family_report["attn-flow"] <= 1.0
    unscreened = run(s, tolerance, strata=strata, max_floor=None)
    assert unscreened.receipt.verdict == "refuse"
    assert any(f"row {gid}" in r for r in unscreened.receipt.reasons)


def test_the_base_floor_limits_name_exactly_the_rows_the_shipped_tolerances_exclude():
    for model, limit in BASE_MAX_FLOOR.items():
        fixtures, tolerance = runtime.load_fixtures(model)
        over = tuple(sorted(r.generation_id for r in fixtures.rows
                            if limit is not None and r.floor_b > limit))
        assert over == tolerance.excluded_rows, model


# --- the ruler and the strata ----------------------------------------------------


def test_the_path_ruler_standardizes_by_the_replay_spread_with_a_floor():
    replay = np.asarray([[0.0, 1.0, 5.0], [2.0, 1.0, 5.0], [4.0, 1.0, 5.0]], dtype=np.float32)
    incremental = replay + np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    sigma, floors = path_ruler(replay, incremental)
    spread = np.std([0.0, 2.0, 4.0])
    assert sigma[0] == pytest.approx(spread) and sigma[1] == pytest.approx(spread)
    assert floors == pytest.approx(np.full(3, 1.0 / spread))


def test_a_row_whose_paths_agree_exactly_has_no_floor_to_divide_by():
    replay = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    incremental = replay.copy()
    incremental[0, 0] += 1.0
    with pytest.raises(ValueError, match="agree exactly"):
        path_ruler(replay, incremental)


def test_the_ordinary_rows_are_evenly_spaced_ids_and_the_rest_are_ranked():
    s = sample(n_rows=50)
    strata = select_strata(s, base_tolerance())
    ids = sorted(s.ids)
    expected = [ids[round((i + 0.5) * 50 / 16 - 0.5)] for i in range(16)]
    assert ORDINARY_OF(strata) == expected == list(ordinary_rows(s.ids))
    for rule, count in RULES:
        assert sum(v == rule for v in strata.values()) == count
    assert sum(v == UNRANKED for v in strata.values()) == 50 - 16 - sum(c for _, c in RULES)
    pool = [g for g in s.ids if g not in expected]
    top = max(pool, key=lambda g: (np.linalg.norm(s.standardized(g)[:len(ATTENTION)]), -g))
    assert strata[top] == "largest-attention-shift"


def test_the_ordinary_stratum_does_not_depend_on_the_deviations():
    """Permuting which row carries which deviation changes the ranked labels, never
    which rows are ordinary."""
    s = sample()
    ids = list(s.ids)
    deviations = [s.candidate[g] - s.reference[g] for g in ids]
    rng = np.random.default_rng(7)
    order = rng.permutation(len(ids))
    permuted = with_candidate(s, {g: s.reference[g] + deviations[j]
                                  for g, j in zip(ids, order)})
    before, after = select_strata(s, base_tolerance()), select_strata(permuted, base_tolerance())
    assert ORDINARY_OF(before) == ORDINARY_OF(after)
    assert before != after
    boosted = with_candidate(s, {g: _shift(s, g, NAMES.index("attn_flow_0"), 0.9)
                                 for g in ORDINARY_OF(before)})
    assert ORDINARY_OF(select_strata(boosted, base_tolerance())) == ORDINARY_OF(before)


def _ratio_sample(ratios: np.ndarray, seed: int):
    """A sample whose substrate ratio on row i is ``ratios[i]`` (floors are 1)."""
    s = sample(seed=seed, scale=0.01)
    residual = [NAMES.index(n) for n in ("res_0", "res_1", "res_2", "res_3")]
    updates = {}
    for g, ratio in zip(s.ids, ratios):
        vector = s.candidate[g].copy()
        vector[residual] = s.reference[g][residual] + np.float32(ratio / 2.0)
        updates[g] = vector
    return with_candidate(s, updates)


def test_a_sample_drawn_from_the_base_ratio_distribution_passes():
    """The base's ceilings are read from 190 draws of a ratio distribution, as a
    qualification reads them from its rows; a fine-tune drawing its rows from the same
    distribution passes, and does so across most draws."""
    rng = np.random.default_rng(11)
    population = rng.lognormal(mean=0.0, sigma=0.35, size=190)
    from anamnesis.extraction.vllm.conformance import RowComponent
    from synthetic_extension import SUBSTRATE

    base = base_tolerance(**LOOSE_RESIDUAL)
    substrate = RowComponent(name="substrate", feature_names=SUBSTRATE,
                             max_ratio=float(population.max()),
                             p90_ratio=float(np.percentile(population, 90)),
                             p99_ratio=float(np.percentile(population, 99)), source="draws")
    tolerance = base.model_copy(update=dict(components=(substrate, base.components[1])))
    verdicts = []
    for seed in range(60):
        draw = np.random.default_rng(1000 + seed).lognormal(0.0, 0.35, size=44)
        verdicts.append(run(_ratio_sample(draw, seed), tolerance).receipt.verdict)
    assert verdicts[0] == "pass"
    assert verdicts.count("pass") / len(verdicts) >= 0.7


def test_a_rule_under_which_every_row_ties_is_refused():
    """No gate-sparsity coordinate moves on any row, so that rule has nothing to rank."""
    s = sample()
    gate = [NAMES.index("gate_L0_sparsity_mean"), NAMES.index("gate_L0_sparsity_std")]
    unmoved = {}
    for g in s.ids:
        unmoved[g] = s.candidate[g].copy()
        unmoved[g][gate] = s.reference[g][gate]
    with pytest.raises(ValueError, match="rank by id alone"):
        select_strata(with_candidate(s, unmoved), base_tolerance())


def test_a_sample_outside_the_size_range_cannot_be_stratified():
    with pytest.raises(ValueError, match="44 to 60 rows"):
        select_strata(sample(n_rows=61), base_tolerance())


# --- the receipt and the caller's errors -----------------------------------------------


def test_a_receipt_cannot_pass_without_naming_what_it_produced_or_refuse_without_reasons():
    receipt = run(sample()).receipt
    with pytest.raises(ValidationError):
        TransferReceipt.model_validate({**receipt.model_dump(), "fixture_digest": None})
    with pytest.raises(ValidationError):
        TransferReceipt.model_validate({**receipt.model_dump(), "verdict": "refuse"})


def test_an_extension_key_that_is_a_shipped_lane_is_a_caller_error():
    with pytest.raises(ValueError, match="shipped lane"):
        run(sample(), key="70b")


# --- the device-side runner's refusals before any device work ------------------------


def _entries(n):
    return {str(100 + i): dict(input_ids=list(range(1, 11)), prompt_length=3, n_gen=7)
            for i in range(n)}


def test_the_runner_refuses_a_preset_that_does_not_extend_the_base(tmp_path, monkeypatch):
    ship_base(tmp_path / "shipped", monkeypatch)
    registry(tmp_path / "models.json", monkeypatch, row={"extends": "3b", "model_id": "x"})
    with pytest.raises(ValueError, match="does not extend '8b'"):
        run_transfer(key=KEY, extends=BASE, preset=PRESET, model_path=tmp_path / "m",
                     calib_dir=tmp_path / "c", entries=_entries(44), ids=list(range(100, 144)),
                     out=tmp_path / "out", work_dir=tmp_path / "work", device="cpu")


def test_the_runner_refuses_an_existing_key_and_a_small_sample(tmp_path, monkeypatch):
    ship_base(tmp_path / "shipped", monkeypatch)
    registry(tmp_path / "models.json", monkeypatch)
    common = dict(extends=BASE, preset=PRESET, model_path=tmp_path / "m",
                  calib_dir=tmp_path / "c", out=tmp_path / "out", work_dir=tmp_path / "work",
                  device="cpu")
    with pytest.raises(ValueError, match="already a lane"):
        run_transfer(key="8b", entries=_entries(44), ids=list(range(100, 144)), **common)
    with pytest.raises(ValueError, match="44 to 60 distinct rows"):
        run_transfer(key=KEY, entries=_entries(10), ids=list(range(100, 110)), **common)
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        run_transfer(key=KEY, entries=_entries(44), ids=list(range(100, 144)), **common)


def test_the_command_reports_a_refusal_before_device_work_as_exit_two(tmp_path, monkeypatch,
                                                                     capsys):
    from anamnesis.scripts import transfer_vllm

    ship_base(tmp_path / "shipped", monkeypatch)
    registry(tmp_path / "models.json", monkeypatch)
    manifest = tmp_path / "replay_manifest.json"
    manifest.write_text(json.dumps(dict(entries=_entries(10), n_ok=10, n_flagged=0,
                                        flagged=[])))
    argv = ["--key", KEY, "--extends", BASE, "--preset", PRESET, "--model-path",
            str(tmp_path / "m"), "--calib-dir", str(tmp_path / "c"), "--manifest",
            str(manifest), "--out", str(tmp_path / "out")]
    assert transfer_vllm.main(argv) == 2
    assert "44 to 60 distinct rows" in capsys.readouterr().err


# --- the anchor's incremental path -------------------------------------------------


FULL_PLAN = HookPlan(key_layers=[0, 1, 2], value_layers=[0, 1, 2], query_layers=[0, 1, 2],
                     attn_output_layers=[0, 1, 2], gate_layers=[0, 1, 2])


def test_the_incremental_path_aligns_with_the_one_forward_replay():
    loaded, model = loaded_tiny_model(FULL_PLAN)
    ids = list(range(1, 13))
    full = replay_extract(loaded, ids, prompt_length=5)
    step = replay_extract_incremental(loaded, ids, prompt_length=5)
    assert len(step.hidden_states) == len(full.hidden_states) == 6
    assert step.chosen_token_ids.tolist() == full.chosen_token_ids.tolist()
    assert step.prompt_length == full.prompt_length
    for a, b in zip(step.hidden_states, full.hidden_states):
        np.testing.assert_allclose(a, b, atol=1e-5)
    for a, b in zip(step.attentions, full.attentions):
        assert a.shape == b.shape
        np.testing.assert_allclose(a, b, atol=1e-5)
    for surface in ("pre_rope_keys", "v_proj_values", "queries", "gate_activations",
                    "attn_outputs"):
        mine, theirs = getattr(step, surface), getattr(full, surface)
        assert sorted(mine) == sorted(theirs)
        for layer in mine:
            assert len(mine[layer]) == len(theirs[layer]) == 6
            np.testing.assert_allclose(mine[layer][2], theirs[layer][2], atol=1e-5)
    assert loaded.hook_state.pre_rope_keys == {}


def test_the_incremental_path_refuses_a_span_without_two_generated_tokens():
    loaded, _ = loaded_tiny_model(FULL_PLAN)
    with pytest.raises(ValueError, match="fewer than two generated tokens"):
        replay_extract_incremental(loaded, list(range(1, 7)), prompt_length=5)
