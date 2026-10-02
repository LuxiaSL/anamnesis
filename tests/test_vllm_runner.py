"""capture_groups against a stand-in engine: no vLLM import, no model, no device.

:func:`anamnesis.extraction.vllm.runner.capture_groups` drives an engine the caller
built through four calls (``generate``, ``_add_request``, ``_run_engine`` and the
output processor's request states). The stand-in engine here implements those on
top of the stand-in model runner from ``test_vllm_capture``, schedules each prompt
whole or in chunks of a chosen size, and can be told to change its output when
hooked or to return outputs out of submission order. That is enough to exercise
every refusal the runner makes before, during and after a group, the files it
writes, and the schedule proof :func:`~anamnesis.extraction.vllm.runner.validate_schedule`
derives from the recorded trace. Whether a real engine schedules this way is proved
from its recorded trace by ``validate_schedule`` on every real capture.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
import torch

from anamnesis.extraction.vllm.envelope import request_groups
from anamnesis.extraction.vllm.receipts import SUBSTRATE_FIELDS, capture_receipt
from anamnesis.extraction.vllm.runner import (
    NoninterferenceError,
    capture_groups,
    generation_record,
    validate_schedule,
    write_json,
)
from anamnesis.provenance import file_sha
from test_vllm_capture import Runner


class FakeEngine:
    def __init__(self, cap=None, changed=False, reverse=False):
        self.runner = Runner({})
        self.queue = []
        self.counter = 0
        self.cap = cap
        self.changed = changed
        self.reverse = reverse
        self.generate_calls = self.run_calls = 0
        self.llm_engine = SimpleNamespace(
            output_processor=SimpleNamespace(request_states={}),
            has_unfinished_requests=lambda: bool(self.queue),
        )

    def _add_request(self, prompt, sampling):
        external = str(self.counter)
        self.counter += 1
        internal = "internal-" + external
        ids = prompt["prompt_token_ids"]
        self.llm_engine.output_processor.request_states[internal] = SimpleNamespace(
            external_req_id=external
        )
        self.runner.specs[internal] = dict(input_ids=ids, start=0, end=len(ids))
        self.runner.requests[internal] = SimpleNamespace(
            prompt_token_ids=ids, num_computed_tokens=0
        )
        self.runner.num_prompt_logprobs[internal] = 0
        self.queue.append(internal)
        return internal

    def generate(self, prompts, sampling, use_tqdm):
        self.generate_calls += 1
        for p in prompts:
            self._add_request(p, sampling)
        return self._run_engine(use_tqdm=False)

    def _run_engine(self, use_tqdm):
        self.run_calls += 1
        active = list(self.queue)
        while active:
            counts = {
                r: min(
                    self.cap or 1024,
                    len(self.runner.specs[r]["input_ids"])
                    - self.runner.requests[r].num_computed_tokens,
                )
                for r in active
            }
            self.runner.step(counts)
            active = [
                r
                for r in active
                if self.runner.requests[r].num_computed_tokens
                < len(self.runner.specs[r]["input_ids"])
            ]
        hooked = bool(self.runner.model._forward_pre_hooks)
        outputs = []
        for r in self.queue:
            ids = self.runner.specs[r]["input_ids"]
            external = self.llm_engine.output_processor.request_states[
                r
            ].external_req_id
            lp = SimpleNamespace(logprob=-0.5, rank=1, decoded_token=None)
            token = 9 if self.changed and hooked else 7
            completion = SimpleNamespace(
                token_ids=[token],
                logprobs=[{token: lp}],
                finish_reason="length",
                stop_reason=None,
            )
            outputs.append(
                SimpleNamespace(
                    request_id=external,
                    prompt_token_ids=ids,
                    outputs=[completion],
                    finished=True,
                    prompt_logprobs=[None] + [{x: lp} for x in ids[1:]],
                )
            )
            del self.llm_engine.output_processor.request_states[r]
        self.queue = []
        return (
            list(reversed(outputs))
            if self.reverse
            else sorted(outputs, key=lambda x: int(x.request_id))
        )


@pytest.fixture
def plan():
    rows = [
        dict(generation_id=i, input_ids=list(range(1, n + 1)), prompt_length=1, end=n)
        for i, n in [(0, 4), (1, 5), (10000, 6)]
    ]
    condition = dict(condition_id="synthetic", max_num_seqs=2)
    groups = request_groups([r["generation_id"] for r in rows], 2)
    sampling = SimpleNamespace(
        max_tokens=1, n=1, prompt_logprobs=0, logprobs=0, temperature=0.0
    )
    return rows, groups, condition, sampling


def execute(engine, plan, path, rounding=True):
    rows, groups, condition, sampling = plan
    return capture_groups(
        engine, engine.runner, rows, groups, condition, [0, 2], sampling, dict, path,
        attention_rounding=rounding,
    )


def test_groups_controls_fillers_scheduling_and_native_receipts(plan, tmp_path):
    engine = FakeEngine()
    result = execute(engine, plan, tmp_path)
    assert engine.generate_calls == 2 and engine.run_calls == 4
    assert [r["generation_id"] for r in result["rows"]] == [0, 1, 10000]
    assert len(list(tmp_path.glob("row-*.pt"))) == 3
    assert [g["actual_first_step_concurrency"] for g in result["groups"]] == [2, 2]
    assert all(g["full_prompt_coverage"] for g in result["groups"])
    schedule = json.loads((tmp_path / "group-0001.schedule.json").read_text())
    assert len(schedule["controls"]) == len(schedule["observed"]) == 2
    assert sum(x["retain"] for x in schedule["internal_requests"].values()) == 1
    assert schedule["hook_noninterference"]
    for r in result["rows"]:
        assert r["capture_contract"] == "lane-capture/1" and "repeat_exact" not in r
        raw = tmp_path / f"row-{r['generation_id']:05d}.pt"
        assert file_sha(raw) == r["raw_sha256"]
        assert capture_receipt(torch.load(raw, weights_only=True)) == r["substrate"]
        assert file_sha(tmp_path / r["schedule_file"]) == r["schedule_sha256"]
        assert json.loads(raw.with_suffix(".json").read_text()) == r
    assert not engine.llm_engine.has_unfinished_requests()
    assert not engine.runner.model._forward_pre_hooks


def test_every_group_prefills_each_prompt_whole_in_one_step(plan, tmp_path):
    execute(FakeEngine(), plan, tmp_path)
    trace = json.loads((tmp_path / "group-0000.schedule.json").read_text())["trace"]
    assert len(trace) == 1 and [r["count"] for r in trace[0]["requests"]] == [4, 5]


def test_attention_capture_threads_through_the_producer(plan, tmp_path):
    execute(FakeEngine(), plan, tmp_path)
    raw = torch.load(tmp_path / "row-00000.pt", map_location="cpu", weights_only=True)
    assert set(raw) == SUBSTRATE_FIELDS
    for key in ("attn_span_rows", "attn_entropy_rows", "attn_head_ent",
                "attn_head_sink", "attn_head_prompt", "attn_head_recency"):
        assert raw[key], key
    schedule = json.loads((tmp_path / "group-0000.schedule.json").read_text())
    assert schedule["attention_rounding"] is True
    assert all("row_sum_worst" in step for step in schedule["trace"])


def test_the_rounding_switch_is_recorded(plan, tmp_path):
    execute(FakeEngine(), plan, tmp_path, rounding=False)
    schedule = json.loads((tmp_path / "group-0000.schedule.json").read_text())
    assert schedule["attention_rounding"] is False


def test_fail_noninterference_keeps_failure_and_raw(plan, tmp_path):
    with pytest.raises(RuntimeError, match="changed the generation"):
        execute(FakeEngine(changed=True), plan, tmp_path)
    assert (tmp_path / "group-0000.failed.json").exists()
    receipt = json.loads((tmp_path / "row-00000.json").read_text())
    assert receipt["hook_noninterference"] is False and "repeat_exact" not in receipt
    assert (tmp_path / "row-00000.pt").exists()


@pytest.mark.parametrize("cap", [1, 2])
def test_actual_chunking_refused_and_recorded(plan, tmp_path, cap):
    with pytest.raises(ValueError, match="prefilled in chunks"):
        execute(FakeEngine(cap=cap), plan, tmp_path)
    failure = json.loads((tmp_path / "group-0000.failed.json").read_text())
    assert failure["exception"] == "ValueError"
    assert "prefilled in chunks" in failure["message"]
    assert failure["trace"]
    assert not list(tmp_path.glob("row-*"))


def test_output_order_refused(plan, tmp_path):
    with pytest.raises(ValueError, match="submission order"):
        execute(FakeEngine(reverse=True), plan, tmp_path)


def test_first_step_capacity_requires_actual_concurrency():
    requests = {"a": dict(end=2), "b": dict(end=2)}
    trace = [
        dict(step=0, total_tokens=2,
             requests=[dict(request_id="a", start=0, count=2, offset=0)]),
        dict(step=1, total_tokens=2,
             requests=[dict(request_id="b", start=0, count=2, offset=0)]),
    ]
    with pytest.raises(ValueError, match="concurrency was not realized"):
        validate_schedule(trace, requests, 2)


def test_a_prompt_split_across_steps_is_refused():
    requests = {"a": dict(end=2)}
    trace = [
        dict(step=0, total_tokens=1,
             requests=[dict(request_id="a", start=0, count=1, offset=0)]),
        dict(step=1, total_tokens=1,
             requests=[dict(request_id="a", start=1, count=1, offset=0)]),
    ]
    with pytest.raises(ValueError, match="prefilled in chunks"):
        validate_schedule(trace, requests, 1)


def test_schedule_proof_of_a_whole_prompt_step():
    requests = {"a": dict(end=3), "b": dict(end=2)}
    trace = [dict(step=0, total_tokens=5, requests=[
        dict(request_id="a", start=0, count=3, offset=0),
        dict(request_id="b", start=0, count=2, offset=3)])]
    assert validate_schedule(trace, requests, 2) == dict(
        actual_first_step_concurrency=2, max_concurrency=2, max_request_chunk=3,
        one_token_chunks=0, full_prompt_coverage=True)
    with pytest.raises(ValueError, match="packed total"):
        validate_schedule([dict(trace[0], total_tokens=6)], requests, 2)
    with pytest.raises(ValueError, match="omitted"):
        validate_schedule(trace, dict(requests, c=dict(end=2)), 2)


def test_duplicate_target_rejected_before_any_execution(plan, tmp_path):
    rows, groups, condition, sampling = plan
    groups[-1][-1]["retain"] = True
    engine = FakeEngine()
    with pytest.raises(ValueError, match="exactly once"):
        execute(engine, plan, tmp_path)
    assert engine.generate_calls == 0


def test_groups_must_realize_the_declared_capacity(plan, tmp_path):
    rows, groups, condition, sampling = plan
    engine = FakeEngine()
    with pytest.raises(ValueError, match="declared capacity"):
        execute(engine, (rows, request_groups([0, 1, 10000], 1), condition, sampling),
                tmp_path)
    with pytest.raises(ValueError, match="invalid condition capacity"):
        execute(engine, (rows, groups, dict(condition, max_num_seqs=3), sampling),
                tmp_path)
    assert engine.generate_calls == 0


def test_sampling_must_be_one_greedy_token_with_logprobs(plan, tmp_path):
    rows, groups, condition, sampling = plan
    engine = FakeEngine()
    for change, fragment in [(dict(max_tokens=2), "one output token"),
                             (dict(temperature=0.7), "deterministic"),
                             (dict(prompt_logprobs=None), "deterministic")]:
        wrong = SimpleNamespace(**dict(vars(sampling), **change))
        with pytest.raises(ValueError, match=fragment):
            execute(engine, (rows, groups, condition, wrong), tmp_path)
    assert engine.generate_calls == 0


def test_output_collision_refused_before_execution(plan, tmp_path):
    (tmp_path / "row-00000.json").write_text("{}")
    engine = FakeEngine()
    with pytest.raises(FileExistsError):
        execute(engine, plan, tmp_path)
    assert engine.generate_calls == 0


def test_hooked_external_id_map_refused(plan, tmp_path):
    engine = FakeEngine()
    original = engine._run_engine

    def wrong_ids(**kwargs):
        hooked = bool(engine.runner.model._forward_pre_hooks)
        outputs = original(**kwargs)
        if hooked:
            for result in outputs:
                result.request_id = str(int(result.request_id) + 100)
        return outputs

    engine._run_engine = wrong_ids
    with pytest.raises(ValueError, match="exact queued occurrence mapping"):
        execute(engine, plan, tmp_path)


def _output(ids, tokens=(7,)):
    lp = SimpleNamespace(logprob=-0.25, rank=1, decoded_token="x")
    completion = SimpleNamespace(token_ids=list(tokens),
                                 logprobs=[{t: lp} for t in tokens],
                                 finish_reason="length", stop_reason=None)
    return SimpleNamespace(prompt_token_ids=ids, outputs=[completion], finished=True,
                           prompt_logprobs=[None] + [{x: lp} for x in ids[1:]])


def test_generation_record_is_comparable_json():
    record = generation_record([_output([3, 4])], [3, 4])
    assert record == dict(
        prompt_token_ids=[3, 4], token_ids=[7],
        prompt_logprobs=[None, {"4": dict(logprob=-0.25, rank=1, decoded_token="x")}],
        logprobs=[{"7": dict(logprob=-0.25, rank=1, decoded_token="x")}],
        finish_reason="length", stop_reason=None)


def test_generation_record_refusals():
    with pytest.raises(ValueError, match="exactly one"):
        generation_record([], [3, 4])
    with pytest.raises(ValueError, match="identity/completion"):
        generation_record([_output([3, 4])], [3, 5])
    with pytest.raises(ValueError, match="lengths differ"):
        generation_record([_output([3, 4], tokens=(7, 8))], [3, 4])
    missing = _output([3, 4])
    missing.outputs[0].logprobs = None
    with pytest.raises(ValueError, match="logprobs absent"):
        generation_record([missing], [3, 4])


def test_write_json_is_sorted_and_refuses_nan(tmp_path):
    path = tmp_path / "x.json"
    write_json(path, {"b": 1, "a": [1, 2]})
    assert path.read_text() == '{\n  "a": [\n    1,\n    2\n  ],\n  "b": 1\n}\n'
    with pytest.raises(ValueError):
        write_json(path, {"a": math.nan})


def run_with(engine, plan, path, checks=None, publish=None):
    rows, groups, condition, sampling = plan
    return capture_groups(
        engine, engine.runner, rows, groups, condition, [0, 2], sampling, dict, path,
        attention_rounding=True, checks=checks, publish=publish,
    )


def test_the_disk_path_records_no_cadence_status(plan, tmp_path):
    result = execute(FakeEngine(), plan, tmp_path)
    for r in result["rows"]:
        assert r["hook_noninterference"] is True and "noninterference" not in r
    assert all("noninterference" not in g and "hook_noninterference" not in g
               for g in result["groups"])
    schedule = json.loads((tmp_path / "group-0000.schedule.json").read_text())
    assert "noninterference" not in schedule


def test_a_check_plan_must_name_every_group_before_anything_runs(plan, tmp_path):
    engine = FakeEngine()
    for checks in ([True], [True, 1], [True, True, True]):
        with pytest.raises(ValueError, match="check plan"):
            run_with(engine, plan, tmp_path, checks=checks)
    assert engine.generate_calls == engine.run_calls == 0


def test_a_publisher_takes_every_retained_row_instead_of_a_file(plan, tmp_path):
    taken = []
    result = run_with(FakeEngine(), plan, tmp_path, checks=[True, False],
                      publish=lambda record, capture: taken.append((dict(record), capture)))
    assert [r["generation_id"] for r, _ in taken] == [0, 1, 10000]
    assert not list(tmp_path.glob("row-*"))
    for record, capture in taken:
        assert set(capture) == SUBSTRATE_FIELDS
        assert "substrate" not in record and "raw_sha256" not in record
        assert file_sha(tmp_path / record["schedule_file"]) == record["schedule_sha256"]
    assert [r["noninterference"] for r, _ in taken] == ["checked", "checked", "not_checked"]
    assert [g["noninterference"] for g in result["groups"]] == ["checked", "not_checked"]
    assert [g["retained_rows"] for g in result["groups"]] == [2, 1]


def test_an_unchecked_group_runs_only_hooked_and_claims_nothing(plan, tmp_path):
    engine = FakeEngine(changed=True)
    taken = []
    run_with(engine, plan, tmp_path, checks=[False, False],
             publish=lambda record, capture: taken.append(record))
    assert engine.generate_calls == 0 and engine.run_calls == 2
    assert all("hook_noninterference" not in r for r in taken)
    assert not list(tmp_path.glob("*.control.json"))
    schedule = json.loads((tmp_path / "group-0000.schedule.json").read_text())
    assert schedule["noninterference"] == "not_checked" and schedule["controls"] is None
    assert "hook_noninterference" not in schedule


def test_a_failed_check_publishes_nothing_of_its_group(plan, tmp_path):
    taken = []
    with pytest.raises(NoninterferenceError, match="changed the generation"):
        run_with(FakeEngine(changed=True), plan, tmp_path, checks=[True, True],
                 publish=lambda record, capture: taken.append(record))
    assert taken == []
    assert (tmp_path / "group-0000.failed.json").exists()
    schedule = json.loads((tmp_path / "group-0000.schedule.json").read_text())
    assert schedule["hook_noninterference"] is False


def test_the_disk_paths_failed_check_is_a_noninterference_error(plan, tmp_path):
    with pytest.raises(NoninterferenceError):
        execute(FakeEngine(changed=True), plan, tmp_path)
