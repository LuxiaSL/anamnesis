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
from anamnesis.orchestration.gpu import THREAD_LIMIT_ENV, VISIBLE_DEVICES_ENV
from anamnesis.orchestration.launch import (
    Cell,
    LaunchPlan,
    assemble_run,
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
