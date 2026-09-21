"""Fan-out, multicell partitioning, and run assembly — with no device in sight.

Everything a launcher decides is arithmetic over lists plus process bookkeeping, so
all of it is testable on a machine with no accelerator: the subprocess constructor
is a parameter, and a fake one records the argv and environment each worker would
have been given.

Two claims here matter more than the rest.

**The partition is a function of the roster.** Which worker runs which item is
``index % workers``, so a plan is reproducible and a resumed pass re-derives the
same assignment. That is what makes it legitimate for the assignment to be
arbitrary: an output does not depend on it.

**A roster is fanned out through one arrangement.** The generation and the replay
side both hand their roster to :func:`anamnesis.orchestration.launch.fan_out_roster`,
so there is one answer to where a job file lands, what a dry run prints, and which
fields of a cell are the launcher's business — target, items, and payload it carries
without reading.

**Assembly writes the manifest through the manifest's own module.** The frozen
record holds two launchers that each format ``replay_manifest.json`` from their own
dict literal, with a copy of the schema each. Assembly here calls
:mod:`anamnesis.extraction.replay.manifest` instead, and the case below pins the
result against the literal bytes those copies produce — because a schema module is
only the single home if adopting it left the file alone.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from anamnesis.extraction.replay.manifest import load_replay_manifest
from anamnesis.orchestration.gpu import (
    THREAD_LIMIT_ENV,
    VISIBLE_DEVICES_ENV,
    resolve_physical_gpus,
)
from anamnesis.orchestration.launch import (
    Cell,
    LaunchPlan,
    assemble_run,
    fan_out_roster,
    launch,
    plan_multicell,
    round_robin,
    write_worker_inputs,
)


class _FakeProcess:
    """A process that already exited, so a launch can be waited on instantly."""

    def __init__(self, argv: list[str], env: dict[str, str], returncode: int) -> None:
        self.argv = argv
        self.env = env
        self.returncode = returncode

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def poll(self) -> int:
        return self.returncode


class _Spawner:
    """Records every spawn instead of making one."""

    def __init__(self, returncodes: dict[int, int] | None = None) -> None:
        self.calls: list[_FakeProcess] = []
        self.returncodes = returncodes or {}

    def __call__(self, argv: list[str], env: dict[str, str], stdout: Any, stderr: Any) -> Any:
        process = _FakeProcess(list(argv), dict(env), self.returncodes.get(len(self.calls), 0))
        self.calls.append(process)
        return process


def _plan(tmp_path: Path, devices: tuple[str, ...] = ("0", "1"), per_device: int = 2) -> LaunchPlan:
    return LaunchPlan(
        devices=devices, workers_per_device=per_device, log_dir=tmp_path / "logs"
    )


def test_round_robin_is_order_stable() -> None:
    assert round_robin([1, 2, 3, 4, 5], 2) == [[1, 3, 5], [2, 4]]


def test_round_robin_refuses_zero_buckets() -> None:
    """Zero buckets would drop the work silently, so it is not a partition."""
    with pytest.raises(ValueError, match="positive"):
        round_robin([1, 2], 0)


def test_plan_counts_and_devices(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    assert plan.n_workers == 4
    assert [plan.device_for(w) for w in range(4)] == ["0", "1", "0", "1"]


def test_plan_resolve_maps_slots_through_the_assignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slot is an index into what the scheduler gave the job, not a physical device."""
    monkeypatch.setenv(VISIBLE_DEVICES_ENV, "5,6,7")
    plan = LaunchPlan.resolve("0,2", 1, tmp_path / "logs")
    assert plan.devices == ("5", "7")


def test_plan_resolve_refuses_slots_beyond_the_assignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(VISIBLE_DEVICES_ENV, "5,6")
    with pytest.raises(ValueError, match="exceeds the scheduler assignment"):
        LaunchPlan.resolve("0,1,2", 1, tmp_path / "logs")


def test_slot_zero_is_a_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first slot of an assignment, and of a bare box: the boundary the check keeps."""
    monkeypatch.setenv(VISIBLE_DEVICES_ENV, "3")
    assert resolve_physical_gpus(["0"]) == ["3"]
    monkeypatch.delenv(VISIBLE_DEVICES_ENV)
    assert resolve_physical_gpus(["0", "1"]) == ["0", "1"]


@pytest.mark.parametrize("assignment", ["4,5,6", None])
def test_a_negative_slot_is_refused_with_or_without_an_assignment(
    assignment: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``-1`` would index the last device of the assignment.

    A fleet asked for slots ``0,-1`` against a two-device assignment would put two
    workers on one device and report two devices, which reads as a throughput result
    rather than as a mistake. The refusal does not depend on the job carrying an
    assignment, because the typo is the same on bare metal.
    """
    if assignment is None:
        monkeypatch.delenv(VISIBLE_DEVICES_ENV, raising=False)
    else:
        monkeypatch.setenv(VISIBLE_DEVICES_ENV, assignment)
    with pytest.raises(ValueError, match="negative"):
        resolve_physical_gpus(["0", "-1"])


def test_a_slot_that_is_not_an_index_is_refused_on_a_bare_box(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(VISIBLE_DEVICES_ENV, raising=False)
    with pytest.raises(ValueError, match="not an integer"):
        resolve_physical_gpus(["cuda:0"])


def test_launch_confines_each_worker_and_pins_its_threads(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    spawner = _Spawner()
    result = launch(plan, lambda w: ["echo", str(w)], stem="unit", popen=spawner)

    assert result.ok and result.n_workers == 4
    devices = [process.env[VISIBLE_DEVICES_ENV] for process in spawner.calls]
    assert devices == ["0", "1", "0", "1"]
    for process in spawner.calls:
        for name in THREAD_LIMIT_ENV:
            assert process.env[name] == "1", "an unpinned worker can change a reduction's order"
    assert all(path.exists() for path in result.log_paths.values())


def test_launch_waits_for_every_worker_even_after_one_fails(tmp_path: Path) -> None:
    """Returning while siblings hold devices leaves the machine unreasonable-about."""
    plan = _plan(tmp_path, devices=("0",), per_device=3)
    spawner = _Spawner(returncodes={1: 7})
    result = launch(plan, lambda w: ["echo", str(w)], stem="unit", popen=spawner)
    assert len(spawner.calls) == 3
    assert result.failed == (1,)
    assert not result.ok


def test_launch_result_raises_with_the_log_to_read(tmp_path: Path) -> None:
    plan = _plan(tmp_path, devices=("0",), per_device=2)
    result = launch(
        plan, lambda w: ["echo", str(w)], stem="unit", popen=_Spawner(returncodes={0: 2})
    )
    with pytest.raises(SystemExit) as exc:
        result.raise_on_failure()
    assert "unit_w0_gpu0.log" in str(exc.value)


def test_launch_spawns_only_the_named_workers(tmp_path: Path) -> None:
    """A worker with an empty share is absent, not present and idle."""
    plan = _plan(tmp_path)
    spawner = _Spawner()
    launch(plan, lambda w: ["echo", str(w)], stem="unit", workers=[0, 2], popen=spawner)
    assert [p.argv[1] for p in spawner.calls] == ["0", "2"]


def test_plan_multicell_gives_each_worker_its_slice_of_every_cell() -> None:
    cells = [
        Cell(target={"run_dir": "/runs/a"}, payload={"inject_key": "V3"}),
        Cell(target={"run_dir": "/runs/b"}, payload={}),
    ]
    items = {id(cells[0]): [0, 1, 2], id(cells[1]): [10]}
    jobs = plan_multicell(cells, lambda c: items[id(c)], 2, items_key="gen_ids")

    assert jobs[0] == [
        {"run_dir": "/runs/a", "inject_key": "V3", "gen_ids": [0, 2]},
        {"run_dir": "/runs/b", "gen_ids": [10]},
    ]
    assert jobs[1] == [{"run_dir": "/runs/a", "inject_key": "V3", "gen_ids": [1]}]


def test_plan_multicell_omits_a_worker_with_no_share() -> None:
    cell = Cell(target={"run_dir": "/runs/a"})
    jobs = plan_multicell(cell and [cell], lambda c: [0], 3, items_key="gen_ids")
    assert set(jobs) == {0}


def test_write_worker_inputs_names_files_by_worker(tmp_path: Path) -> None:
    paths = write_worker_inputs(tmp_path / "jobs", "specs", {0: [{"a": 1}], 3: [{"b": 2}]})
    assert paths[0].name == "specs_w0.json" and paths[3].name == "specs_w3.json"
    assert json.loads(paths[3].read_text()) == [{"b": 2}]


def _record(gen_id: int, prompt_length: int, n_gen: int) -> dict[str, Any]:
    return {
        "generation_id": gen_id,
        "prompt_set": "UNIT",
        "topic": "tides",
        "topic_idx": 0,
        "mode": "linear",
        "mode_idx": 0,
        "system_prompt": "",
        "user_prompt": "Write about: tides",
        "seed": 1234,
        "repetition": 0,
        "condition": "standard",
        "generated_text": "text",
        "num_generated_tokens": n_gen,
        "prompt_length": prompt_length,
        "input_ids": list(range(prompt_length + n_gen)),
    }


def _bank(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    run_dir = tmp_path / "run"
    rec_dir = run_dir / "gen_records"
    rec_dir.mkdir(parents=True)
    for record in records:
        (rec_dir / f"gen_{record['generation_id']:03d}.json").write_text(json.dumps(record))
    return run_dir


def test_assemble_writes_metadata_and_a_loadable_manifest(tmp_path: Path) -> None:
    run_dir = _bank(tmp_path, [_record(0, 4, 3), _record(1, 5, 2)])
    result = assemble_run(run_dir, {"model": {"model_id": "unit"}})

    assert result.n_generations == 2 and not result.flagged
    metadata = json.loads(result.metadata_path.read_text())
    assert metadata["total_generations"] == 2
    assert metadata["model"] == {"model_id": "unit"}
    assert all("input_ids" not in g for g in metadata["generations"]), (
        "the ids belong to the manifest; duplicating them makes two files answer one question"
    )
    manifest = load_replay_manifest(result.manifest_path)
    assert manifest.gen_ids() == (0, 1)
    assert manifest.entry(1).prompt_length == 5 and manifest.entry(1).n_gen == 2


def test_assembled_manifest_is_byte_identical_to_the_inline_launcher_form(
    tmp_path: Path,
) -> None:
    """The consolidation's proof: routing through the schema module changed no byte.

    The two donor launchers each wrote this file from a dict literal. The literal is
    reproduced here, and the assembled file must match it exactly — key order,
    separators and all — because a manifest is read by banked tooling that indexes
    it directly, and because "one home for the schema" is a claim about the file and
    not only about the code.
    """
    records = [_record(0, 4, 3), _record(1, 5, 2), _record(2, 6, 1)]
    run_dir = _bank(tmp_path, records)
    result = assemble_run(run_dir, {})

    entries = {
        str(r["generation_id"]): {
            "input_ids": r["input_ids"],
            "prompt_length": r["prompt_length"],
            "n_gen": len(r["input_ids"]) - r["prompt_length"],
        }
        for r in records
    }
    donor_bytes = json.dumps(
        {"entries": entries, "n_ok": len(entries), "n_flagged": 0, "flagged": []}
    )
    assert result.manifest_path.read_text() == donor_bytes


def test_assemble_flags_a_record_that_cannot_be_replayed(tmp_path: Path) -> None:
    """A generation that produced nothing is named, not written as a manifest row.

    An unreplayable row written into the manifest becomes a replay failure with
    nothing saying why, which is what the record's inline form does. Flagging it keeps
    the manifest's own promise instead: it says how many of a run's generations it
    covers, so a replay over it cannot quietly be a replay over a subset.
    """
    run_dir = _bank(tmp_path, [_record(0, 4, 3), _record(1, 5, 0)])
    result = assemble_run(run_dir, {})

    assert result.n_generations == 1
    assert [row.gen_id for row in result.flagged] == [1]
    manifest = load_replay_manifest(result.manifest_path)
    assert manifest.gen_ids() == (0,)
    assert manifest.n_flagged == 1
    with pytest.raises(KeyError, match="flagged"):
        manifest.entry(1)


def test_assemble_reads_records_in_generation_order(tmp_path: Path) -> None:
    """Ten sorts before nine as text, and a metadata list is read in order."""
    run_dir = _bank(tmp_path, [_record(i, 4, 2) for i in (0, 9, 10, 2)])
    result = assemble_run(run_dir, {})
    metadata = json.loads(result.metadata_path.read_text())
    assert [g["generation_id"] for g in metadata["generations"]] == [0, 2, 9, 10]


def test_launch_defaults_to_the_real_subprocess_constructor() -> None:
    """The injectable spawner is for tests; the default must be the real one."""
    assert launch.__defaults__ is None
    assert launch.__kwdefaults__["popen"] is subprocess.Popen


_ROSTER = [
    {
        "run_dir": "/runs/cell_a",
        "manifest": "/runs/cell_a/replay_manifest.json",
        "gen_ids": [0, 1, 2],
        "inject_key": "V3",
    },
    {
        "run_dir": "/runs/cell_b",
        "manifest": "/runs/cell_b/replay_manifest.json",
        "gen_ids": [7],
    },
]


def _fan_out(tmp_path: Path, spawner: Any, **overrides: Any) -> LaunchPlan:
    plan = _plan(tmp_path, devices=("0",), per_device=2)
    kwargs: dict[str, Any] = dict(
        target_fields=("run_dir", "manifest"),
        item_fields=("gen_ids",),
        items_key="gen_ids",
        jobs_dir=tmp_path / "jobs",
        command_for=lambda worker, jobs_file: ["python", "-m", "x", str(jobs_file)],
        stem="unit",
        popen=spawner,
    )
    kwargs.update(overrides)
    fan_out_roster(plan, _ROSTER, lambda row: list(row["gen_ids"]), **kwargs)
    return plan


def test_fan_out_roster_gives_each_worker_one_job_per_cell_it_has_a_share_of(
    tmp_path: Path,
) -> None:
    """The load-once arrangement: a worker's jobs travel in one file it reads itself."""
    spawner = _Spawner()
    _fan_out(tmp_path, spawner)
    assert len(spawner.calls) == 2

    first = json.loads((tmp_path / "jobs" / "jobs_w0.json").read_text())
    second = json.loads((tmp_path / "jobs" / "jobs_w1.json").read_text())
    assert [job["run_dir"] for job in first] == ["/runs/cell_a", "/runs/cell_b"]
    assert [job["gen_ids"] for job in first] == [[0, 2], [7]]
    assert [job["gen_ids"] for job in second] == [[1]]
    assert spawner.calls[0].argv[-1] == str(tmp_path / "jobs" / "jobs_w0.json")


def test_fan_out_roster_carries_a_payload_through_and_drops_the_item_fields(
    tmp_path: Path,
) -> None:
    """What an intervention means is the worker's business; where the work is, is not."""
    _fan_out(tmp_path, _Spawner())
    jobs = json.loads((tmp_path / "jobs" / "jobs_w0.json").read_text())
    assert jobs[0]["inject_key"] == "V3", "the payload rides through untouched"
    assert jobs[0]["gen_ids"] == [0, 2], "the worker gets its own slice, not the cell's"
    assert "inject_key" not in jobs[1], "a cell that named no write carries none"


def test_fan_out_roster_dry_run_partitions_without_spawning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spawner = _Spawner()
    _fan_out(tmp_path, spawner, dry_run=True)
    assert not spawner.calls
    assert not (tmp_path / "jobs").exists(), "a dry run writes no job files either"
    printed = capsys.readouterr().out
    assert "worker 0 (0): 2 cells" in printed
    assert "worker 1 (0): 1 cells" in printed


def test_fan_out_roster_raises_when_a_worker_fails(tmp_path: Path) -> None:
    """A launcher that returned zero here would report a complete pass over a gap."""
    with pytest.raises(SystemExit, match="workers failed"):
        _fan_out(tmp_path, _Spawner(returncodes={1: 9}))


def test_fan_out_roster_inherits_a_shortfall_status(tmp_path: Path) -> None:
    from anamnesis.shortfall import EXIT_SHORT

    with pytest.raises(SystemExit) as raised:
        _fan_out(tmp_path, _Spawner(returncodes={0: EXIT_SHORT, 1: EXIT_SHORT}))
    assert raised.value.code == EXIT_SHORT


def test_fan_out_roster_defaults_to_the_real_subprocess_constructor() -> None:
    """The fake above is a test's substitution, not the production path."""
    assert fan_out_roster.__kwdefaults__["popen"] is subprocess.Popen
