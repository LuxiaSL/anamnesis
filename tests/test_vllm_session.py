"""The resident session: planning, the cadence, the release rule, and a whole session on the CPU.

:class:`anamnesis.extraction.vllm.session.LaneSession` runs two child processes, one of
which builds a vLLM engine. Everything above the engine is exercised here without one:
the engine child's loop and the readout child's loop are the module's own, run in
threads against the stand-in engine from ``test_vllm_runner`` and a stand-in reduction,
with rows handed between them through real shared-memory segments
(:mod:`anamnesis.extraction.vllm.handoff`). The same rows through the disk path
(:func:`anamnesis.extraction.vllm.runner.capture_groups` writing torch files) must give
the same vectors and the same content receipts, which is the property the acceptance on
a real engine checks on real rows.

What needs a device and the engine: the children as processes (``LaneSession.open``
admits a lane, reads this host's install-check receipt and builds a real engine).
"""

from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from anamnesis.extraction.vllm import runtime, session
from anamnesis.extraction.vllm.envelope import CONDITIONS, request_groups
from anamnesis.extraction.vllm.receipts import capture_receipt
from anamnesis.extraction.vllm.runner import capture_groups
from test_vllm_runner import FakeEngine

NAMES = ["hidden", "logits", "keys0", "chosen"]


# ── planning ────────────────────────────────────────────────────────────────


def test_a_short_set_runs_one_request_at_a_time_without_fillers():
    condition, groups = session.plan_groups([5, 1, 3])
    assert condition == "full-b1-order0"
    assert [[(m["generation_id"], m["retain"]) for m in g] for g in groups] == [
        [(1, True)], [(3, True)], [(5, True)]]


def test_a_full_set_runs_in_batches_of_eight_filled_from_its_own_rows():
    condition, groups = session.plan_groups(list(range(12)))
    assert condition == "full-b8-order0"
    assert groups == request_groups(list(range(12)), 8)
    assert sum(m["retain"] for g in groups for m in g) == 12
    assert session.plan_groups(list(range(8)))[0] == "full-b8-order0"


@pytest.mark.parametrize("ids", [[], [1, 1], [1.0, 2]])
def test_a_capture_needs_unique_integer_ids(ids):
    with pytest.raises(ValueError):
        session.plan_groups(ids)


def test_the_capacity_is_the_engine_conditions():
    assert session.CAPACITY == CONDITIONS[session.ENGINE_CONDITION]["max_num_seqs"] == 8
    assert CONDITIONS[session.SHORT_CONDITION]["max_num_seqs"] == 1


# ── the cadence ─────────────────────────────────────────────────────────────


def run(cadence: session.Cadence, condition: str, n: int) -> list[bool]:
    checks = cadence.plan(condition, n)
    for checked in checks:
        cadence.advance(condition, checked)
    return checks


def test_every_group_is_checked_until_both_conditions_and_eight_groups_passed():
    cadence = session.Cadence()
    assert run(cadence, "full-b8-order0", 20) == [True] * 20
    assert run(cadence, "full-b1-order0", 1) == [True]
    after = run(cadence, "full-b8-order0", 40)
    assert [21 + i for i, c in enumerate(after) if c] == [32, 48]


def test_after_the_warm_up_every_sixteenth_group_by_index_is_checked():
    cadence = session.Cadence()
    first = run(cadence, "full-b1-order0", 3) + run(cadence, "full-b8-order0", 61)
    assert [i for i, c in enumerate(first) if c] == list(range(8)) + [16, 32, 48]


def test_the_plan_is_the_same_however_the_groups_are_split_into_calls():
    one, many = session.Cadence(), session.Cadence()
    whole = run(one, "full-b1-order0", 1) + run(one, "full-b8-order0", 63)
    split = run(many, "full-b1-order0", 1)
    for size in (3, 1, 16, 7, 36):
        split += run(many, "full-b8-order0", size)
    assert whole == split


def test_planning_does_not_advance_the_session():
    cadence = session.Cadence()
    assert cadence.plan("full-b8-order0", 4) == cadence.plan("full-b8-order0", 4)
    assert cadence.next_index == 0
    with pytest.raises(ValueError):
        cadence.plan("full-b2-order0", 1)


def test_unverified_rows_are_those_after_the_last_passing_check():
    ledger = [dict(index=i, call=1 + i // 2, generation_ids=[10 * i, 10 * i + 1])
              for i in range(4)]
    assert session.unverified_rows(ledger, 1) == [
        dict(call=2, generation_id=20), dict(call=2, generation_id=21),
        dict(call=2, generation_id=30), dict(call=2, generation_id=31)]
    assert session.unverified_rows(ledger, 3) == []


# ── the release rule and the wire ───────────────────────────────────────────


def test_vectors_cross_the_pipe_bit_for_bit():
    vector = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
    vector[:3] = [np.float32(1e-45), -0.0, np.float32(3.4e38)]
    back = session.decode_array(session.encode_array(vector))
    assert back.dtype == np.float32 and back.tobytes() == vector.tobytes()


def handle_and_receipt():
    handle = dict(generation_id=4, receipt_id="t-000001-4", tensors=[
        dict(field="hidden", layer=None, shape=[2, 3, 4], dtype="torch.bfloat16"),
        dict(field="keys", layer=2, shape=[3, 1, 4], dtype="torch.bfloat16")])
    receipt = dict(generation_id=4, receipt_id="t-000001-4", substrate=dict(tensors={
        "hidden": dict(shape=[2, 3, 4], dtype="torch.bfloat16", sha256="a"),
        "keys/2": dict(shape=[3, 1, 4], dtype="torch.bfloat16", sha256="b")}, sha256="c"))
    return handle, receipt


def test_a_row_is_released_with_the_receipt_of_its_own_hand_off():
    session.check_release(*handle_and_receipt())


@pytest.mark.parametrize("change", [
    lambda h, r: r.update(receipt_id="t-000002-4"),
    lambda h, r: r.update(generation_id=5),
    lambda h, r: r["substrate"]["tensors"]["keys/2"].update(shape=[3, 2, 4]),
    lambda h, r: r["substrate"]["tensors"]["hidden"].update(dtype="torch.float16"),
    lambda h, r: r["substrate"]["tensors"].pop("keys/2"),
])
def test_a_row_whose_receipt_is_about_another_hand_off_is_not_released(change):
    handle, receipt = handle_and_receipt()
    change(handle, receipt)
    with pytest.raises(session.LaneSessionError):
        session.check_release(handle, receipt)


def test_a_session_is_built_only_by_open():
    with pytest.raises(TypeError, match="only by LaneSession.open"):
        session.LaneSession(object())


def test_the_child_command_line_is_checked(capsys):
    assert session.main(["engine", "spec.json"]) == 2
    assert "usage" in capsys.readouterr().err


# ── a whole session on the CPU ──────────────────────────────────────────────


REFUSE: dict[str, str | None] = dict(reason=None)
"""Set a reason here and the stand-in reduction refuses with it."""


def fake_reduce(lane, capture, *, start, end, model):
    """A stand-in reduction: a few sums over the substrate, in float32."""
    if REFUSE["reason"]:
        raise ValueError(REFUSE["reason"])
    values = [capture["hidden"].float().sum(), capture["logits"].float().sum(),
              capture["keys"][0].float().sum(), capture["chosen"].float().sum()]
    return SimpleNamespace(features=np.asarray([float(v) for v in values], dtype=np.float32))


class ThreadChild(session._Child):
    """A session child whose loop runs in a thread of this process."""

    def __init__(self, step, target, spec):
        self.step = step
        read_fd, write_fd = os.pipe()
        self.fd, self._buf, self.ready = read_fd, b"", {}
        self.requests: queue.Queue = queue.Queue()
        self.returncode = None
        reply = session._reply_writer(write_fd)

        def body():
            self.returncode = target(spec, reply, iter(self.requests.get, None))

        self.proc = SimpleNamespace(wait=lambda timeout=None: self.thread.join(timeout),
                                    returncode=None)
        self.thread = threading.Thread(target=body, daemon=True)
        self.thread.start()

    def alive(self):
        return self.thread.is_alive()

    def send(self, message):
        if not self.alive():
            raise session.LaneSessionError(f"the {self.step} thread is gone")
        self.requests.put(json.dumps(dict(message)) + "\n")

    def close(self, timeout=30.0):
        if self.alive():
            self.requests.put(json.dumps(dict(op="shutdown")) + "\n")
            self.thread.join(timeout)


@pytest.fixture
def lane(tmp_path, monkeypatch):
    """A session whose engine child drives the stand-in engine and whose readout child
    reduces with the stand-in reduction, both through real segments."""
    engine = FakeEngine()
    monkeypatch.setattr(runtime, "build_engine", lambda model, path, condition: dict(
        llm=engine, runner=engine.runner, tokens_prompt=dict, sampled_layers=[0, 2],
        sampling=SimpleNamespace(max_tokens=1, n=1, prompt_logprobs=0, logprobs=0,
                                 temperature=0.0),
        condition=CONDITIONS[condition], record=dict(lane_id="lane")))
    monkeypatch.setattr(runtime, "prepare_readout_process", lambda: None)
    monkeypatch.setattr(runtime, "readout_lane", lambda model, calib, names: SimpleNamespace(
        config=SimpleNamespace(enable_knnlm_baseline=True)))
    monkeypatch.setattr(runtime, "READOUT_DEVICE", "cpu")
    import anamnesis.extraction.vllm.readout as readout

    monkeypatch.setattr(readout, "reduce_capture", fake_reduce)
    segments = tmp_path / "shm"
    segments.mkdir()
    engine_child = ThreadChild("engine", session._engine_main, dict(
        model="8b", model_path="unused", segments=str(segments), max_in_flight=2,
        token="tok"))
    engine_child.wait_ready(30)
    readout_child = ThreadChild("readout", session._readout_main, dict(
        model="8b", calib_dir="unused", segments=str(segments), feature_names=NAMES))
    readout_child.wait_ready(30)
    work = tmp_path / "work"
    work.mkdir()
    lane_session = session.LaneSession(
        session.LaneSession._OPENING, model="8b",
        receipt=SimpleNamespace(lane_id="lane", qualified_lane_id="lane", tier="identical",
                                digest="d"),
        fixtures=SimpleNamespace(feature_names=NAMES, calibration_sha256="c"),
        work_dir=work, segments=segments, schema_inputs=None, engine=engine_child,
        readout=readout_child, idle_timeout=60)
    lane_session._schemas = {steps: True for steps in range(1, 64)}
    lane_session.engine = engine
    yield lane_session
    lane_session.close()


def rows(ids, length=6):
    return [dict(generation_id=g, input_ids=[1 + (g + k) % 9 for k in range(length + g % 3)],
                 prompt_length=2) for g in ids]


def disk_path(tmp_path, chosen, condition_id):
    """The same rows captured to disk, loaded back and reduced the same way."""
    engine = FakeEngine()
    checked = [dict(r, end=len(r["input_ids"])) for r in chosen]
    groups = request_groups([r["generation_id"] for r in checked],
                            CONDITIONS[condition_id]["max_num_seqs"])
    out = tmp_path / f"disk-{condition_id}"
    capture_groups(engine, engine.runner, checked, groups, CONDITIONS[condition_id], [0, 2],
                   SimpleNamespace(max_tokens=1, n=1, prompt_logprobs=0, logprobs=0,
                                   temperature=0.0), dict, out, attention_rounding=True)
    vectors, receipts, knnlm = {}, {}, {}
    for r in checked:
        gid = r["generation_id"]
        raw = torch.load(out / f"row-{gid:05d}.pt", weights_only=True)
        vectors[gid] = fake_reduce(None, raw, start=0, end=0, model="").features
        receipts[gid] = json.loads((out / f"row-{gid:05d}.json").read_text())
        knnlm[gid] = raw["hidden"][-1][-1].float().numpy()
    return vectors, receipts, knnlm


@pytest.mark.parametrize("ids,condition", [([3, 1, 2], "full-b1-order0"),
                                           (list(range(11)), "full-b8-order0")])
def test_a_session_gives_the_disk_paths_vectors_and_receipts(lane, tmp_path, ids, condition):
    result = lane.capture(rows(ids))
    assert result.condition_id == condition
    vectors, receipts, knnlm = disk_path(tmp_path, rows(ids), condition)
    assert set(result.vectors) == set(ids)
    for gid in ids:
        assert result.vectors[gid].tobytes() == vectors[gid].tobytes()
        assert result.receipts[gid]["substrate"] == receipts[gid]["substrate"]
        assert result.receipts[gid]["condition_id"] == condition
        assert result.receipts[gid]["handoff"] == "memory"
        assert "raw_sha256" not in result.receipts[gid]
        on_disk = json.loads((result.directory / f"row-{gid:05d}.json").read_text())
        assert on_disk == result.receipts[gid]
        assert set(result.timing[gid]) >= {"capture", "handoff_write", "handoff_open",
                                           "to_device", "receipt", "reduce"}
        assert result.knnlm[gid].tobytes() == knnlm[gid].tobytes()
    assert not list(lane.segments.iterdir())
    assert not list(result.directory.glob("*.pt"))


def test_a_session_keeps_one_engine_across_captures(lane):
    first = lane.capture(rows([0, 1]))
    second = lane.capture(rows([0, 1]))
    assert first.vectors[0].tobytes() == second.vectors[0].tobytes()
    assert first.receipts[1]["substrate"] == second.receipts[1]["substrate"]
    assert first.receipts[1]["receipt_id"] != second.receipts[1]["receipt_id"]
    assert [g["session_index"] for g in first.groups + second.groups] == [0, 1, 2, 3]


def test_an_unchecked_group_runs_once_and_claims_nothing(lane, monkeypatch):
    monkeypatch.setattr(session, "WARMUP_GROUPS", 1)
    monkeypatch.setattr(session, "CHECK_EVERY", 4)
    lane.capture(rows([0]))
    calls = lane.engine.generate_calls
    result = lane.capture(rows(range(24)))
    assert [g["noninterference"] for g in result.groups] == ["checked", "not_checked",
                                                             "not_checked"]
    assert lane.engine.generate_calls == calls + 1
    for gid, receipt in result.receipts.items():
        group = gid // 8
        assert receipt["noninterference"] == ("checked" if group == 0 else "not_checked")
        assert ("hook_noninterference" in receipt) is (group == 0)
    assert not (result.directory / "group-0001.control.json").exists()
    schedule = json.loads((result.directory / "group-0001.schedule.json").read_text())
    assert "hook_noninterference" not in schedule and schedule["controls"] is None


def test_a_failed_check_stops_the_session_and_names_the_unverified_rows(lane, monkeypatch):
    monkeypatch.setattr(session, "WARMUP_GROUPS", 2)
    monkeypatch.setattr(session, "CHECK_EVERY", 4)
    lane.capture(rows([0, 1, 2]))            # groups 0-2, b1, checked
    lane.capture(rows(range(8)))             # group 3, b8, checked: the warm-up is over
    lane.capture(rows(range(16)))            # group 4 checked, group 5 not
    lane.engine.changed = True
    lane.capture(rows(range(100, 108)))      # group 6, not checked: nothing can see it
    with pytest.raises(session.NoninterferenceFailure) as failure:
        lane.capture(rows(range(200, 216)))  # group 7 not checked, group 8 checked: fails
    unverified = failure.value.unverified
    assert unverified == ([dict(call=3, generation_id=g) for g in range(8, 16)]
                          + [dict(call=4, generation_id=g) for g in range(100, 108)])
    record = json.loads(failure.value.record.read_text())
    assert record["failed_session_index"] == 8 and record["last_passing_session_index"] == 4
    assert record["unreleased_rows"] == list(range(200, 216))
    assert lane.stopped
    with pytest.raises(session.LaneSessionError, match="stopped"):
        lane.capture(rows([0]))


def test_a_failed_check_publishes_nothing_of_its_group(lane, tmp_path):
    lane.engine.changed = True
    with pytest.raises(session.NoninterferenceFailure) as failure:
        lane.capture(rows([0, 1]))
    assert failure.value.unverified == []
    assert not list(lane.segments.glob("*.seg"))


def test_a_refused_row_leaves_the_session_running(lane):
    with pytest.raises(ValueError, match="two generated tokens"):
        lane.capture([dict(generation_id=1, input_ids=[1, 2], prompt_length=1)])
    with pytest.raises(ValueError, match="token count"):
        lane.capture([dict(generation_id=1, input_ids=[1, 2, 3, 4], prompt_length=1, end=3)])
    assert lane.stopped is None
    assert set(lane.capture(rows([1])).vectors) == {1}


def test_a_readout_refusal_stops_the_session(lane, monkeypatch):
    monkeypatch.setitem(REFUSE, "reason", "the readout produced a nonfinite feature")
    with pytest.raises(session.LaneSessionError, match="nonfinite"):
        lane.capture(rows([0, 1]))
    assert lane.stopped


def test_the_extraction_lane_record_names_the_condition_and_the_receipt(lane):
    chosen = rows([0, 1])
    result = lane.capture(chosen)
    record = lane.extraction_lane(chosen[0], result.receipts[0])
    assert record["condition_id"] == "full-b1-order0" and record["resident"] is True
    assert record["capture_receipt_sha256"] == result.receipts[0]["substrate"]["sha256"]
    assert record["noninterference"] == "checked"
    assert record["span_end"] == len(chosen[0]["input_ids"])
    assert record["feature_schema_sha256"] == runtime.feature_schema_sha256(NAMES)


def test_closing_removes_the_shared_memory_directory(lane):
    lane.capture(rows([0]))
    lane.close()
    assert not Path(lane.segments).exists()
    assert lane.stopped == "closed"
