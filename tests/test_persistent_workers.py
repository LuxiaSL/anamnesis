"""The file queue: dispatch, collect, resume, drain — all on a CPU.

The queue's whole value is what it does when things go wrong, so that is what
these cases are about: a job survives a worker that dies, a stale marker from a
crashed fleet does not poison a fresh one, a missing result times out loudly rather
than being read as a short pass, and a handler that raises kills its worker while
leaving its job queued.

A trivial arithmetic handler stands in for the model-bound one. What the queue
carries is not its concern — that is the whole point of the handler being a
parameter — so nothing here needs a device. The legs that do need one are the
replay handler's, and they are checked by the parity gate against a real cell.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from anamnesis.orchestration.workers import PersistentWorker, WorkerFleet, dispatch


def _square_handler(job: dict) -> dict[str, np.ndarray]:
    values = np.asarray(job["x"], dtype=np.float64)
    return {"y": values * values, "job_id": np.array([job["id"]])}


def _run_worker_thread(work_dir: Path, worker_id: int) -> tuple[threading.Thread, list[int]]:
    completed: list[int] = []
    worker = PersistentWorker(
        work_dir=work_dir, worker_id=worker_id, handler=_square_handler, poll_s=0.01
    )
    thread = threading.Thread(target=lambda: completed.append(worker.run()), daemon=True)
    thread.start()
    return thread, completed


def test_dispatch_collect_roundtrip(tmp_path: Path) -> None:
    fleet = WorkerFleet(work_dir=tmp_path, worker_ids=[0, 1])
    threads = [_run_worker_thread(tmp_path, w) for w in (0, 1)]
    fleet.wait_ready(timeout_s=10)

    expected = [
        fleet.submit(w, f"{i:03d}", {"id": i, "x": [i, i + 1]})
        for i, w in enumerate([0, 1, 0, 1])
    ]
    got = fleet.collect(expected, timeout_s=10)
    assert len(got) == 4
    for arrays in got.values():
        i = int(arrays["job_id"][0])
        assert np.array_equal(arrays["y"], np.array([i * i, (i + 1) ** 2], dtype=np.float64))
    assert not any(p.exists() for p in expected), "a consumed result is unlinked"

    fleet.stop(timeout_s=5)
    for thread, completed in threads:
        thread.join(timeout=5)
        assert completed and completed[0] == 2


def test_job_files_survive_until_result(tmp_path: Path) -> None:
    """Resume: a job is unlinked only after its result lands, so a death is recoverable."""
    fleet = WorkerFleet(work_dir=tmp_path, worker_ids=[0])
    result_path = fleet.submit(0, "007", {"id": 7, "x": [3]})
    job_file = tmp_path / "jobs" / "w0" / "job_007.json"
    assert job_file.exists() and not result_path.exists()

    fleet.clear_stale_state()
    assert job_file.exists(), "clearing stale state must keep pending jobs; they are the resume"

    thread, _ = _run_worker_thread(tmp_path, 0)
    got = fleet.collect([result_path], timeout_s=10)
    assert np.array_equal(got[result_path]["y"], np.array([9.0]))
    assert not job_file.exists(), "the job is consumed once its result exists"
    (tmp_path / "STOP").touch()
    thread.join(timeout=5)


def test_clear_stale_state_removes_stop_ready_results(tmp_path: Path) -> None:
    """What a crashed fleet leaves behind would otherwise lie to the next one."""
    (tmp_path / "STOP").touch()
    (tmp_path / "ready").mkdir()
    (tmp_path / "ready" / "w0").touch()
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "w0_job_zzz.npz").touch()
    WorkerFleet(work_dir=tmp_path, worker_ids=[0]).clear_stale_state()
    assert not (tmp_path / "STOP").exists()
    assert not (tmp_path / "ready" / "w0").exists()
    assert not (tmp_path / "results" / "w0_job_zzz.npz").exists()


def test_stop_drains_idle_workers(tmp_path: Path) -> None:
    fleet = WorkerFleet(work_dir=tmp_path, worker_ids=[0])
    thread, completed = _run_worker_thread(tmp_path, 0)
    fleet.wait_ready(timeout_s=10)
    fleet.stop(timeout_s=5)
    thread.join(timeout=5)
    assert completed and completed[0] == 0


def test_on_stop_runs_after_drain(tmp_path: Path) -> None:
    """The cleanup hook is how a worker detaches what it armed, so it must always run."""
    hits: list[str] = []
    worker = PersistentWorker(
        work_dir=tmp_path, worker_id=3, handler=_square_handler,
        poll_s=0.01, on_stop=lambda: hits.append("cleanup"),
    )
    (tmp_path / "STOP").touch()
    assert worker.run() == 0
    assert hits == ["cleanup"]


def test_collect_timeout_fails_loud(tmp_path: Path) -> None:
    fleet = WorkerFleet(work_dir=tmp_path, worker_ids=[0])
    ghost = tmp_path / "results" / "w0_job_never.npz"
    with pytest.raises(SystemExit, match="timed out"):
        fleet.collect([ghost], timeout_s=0.2, poll_s=0.05)


def test_handler_failure_leaves_job_for_retry(tmp_path: Path) -> None:
    """A raising handler kills its worker loudly and keeps the job queued."""

    def bad_handler(job: dict) -> dict[str, np.ndarray]:
        raise RuntimeError("boom")

    fleet = WorkerFleet(work_dir=tmp_path, worker_ids=[0])
    fleet.submit(0, "013", {"id": 13, "x": [1]})
    worker = PersistentWorker(
        work_dir=tmp_path, worker_id=0, handler=bad_handler, poll_s=0.01
    )
    with pytest.raises(RuntimeError, match="boom"):
        worker.run()
    assert (tmp_path / "jobs" / "w0" / "job_013.json").exists(), (
        "a failed job stays queued for a respawned fleet"
    )


def test_dispatch_sums_what_workers_reported(tmp_path: Path) -> None:
    """The driver's total is the workers' own counts, not a re-derivation."""

    def counting_handler(job: dict) -> dict[str, np.ndarray]:
        return {"n_done": np.array([int(job["n"])])}

    fleet = WorkerFleet(work_dir=tmp_path, worker_ids=[0, 1])
    workers = [
        PersistentWorker(work_dir=tmp_path, worker_id=w, handler=counting_handler, poll_s=0.01)
        for w in (0, 1)
    ]
    threads = [threading.Thread(target=w.run, daemon=True) for w in workers]
    for thread in threads:
        thread.start()
    fleet.wait_ready(timeout_s=10)
    try:
        assert dispatch(fleet, [{"n": 3}, {"n": 4}, {"n": 5}], timeout_s=10) == 12
    finally:
        fleet.stop(timeout_s=5)
    for thread in threads:
        thread.join(timeout=5)
