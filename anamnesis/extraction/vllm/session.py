"""A resident vLLM lane: one engine process and one readout process kept open across captures.

:func:`anamnesis.extraction.vllm.runtime.replay_bank` starts a fresh engine process and a
fresh readout process for every chunk of a bank, and hands each row between them on disk.
That is right for a bank. A caller that captures a few rows at a time, for as long as it
runs, pays an engine build per call and a gigabyte-per-row disk round trip. A
:class:`LaneSession` is the same lane held open:

* **The same two processes.** :meth:`LaneSession.open` admits the lane exactly as a replay
  does (the lane key, its fixtures and calibration, this host's install-check receipt,
  which must not be ``refused``) and starts both children with
  :func:`anamnesis.extraction.vllm.runtime.child_environment`. The engine child builds its
  engine with :func:`anamnesis.extraction.vllm.runtime.build_engine`, the lane's own
  startup; the readout child fixes its arithmetic with
  :func:`anamnesis.extraction.vllm.runtime.prepare_readout_process` and builds
  :func:`anamnesis.extraction.vllm.runtime.readout_lane`. No constructor takes an engine
  or a readout lane from a caller, so a caller cannot build either half itself.
* **The hand-off in memory.** Each retained row's substrate goes from the engine to the
  readout through shared host memory (:mod:`anamnesis.extraction.vllm.handoff`), and the
  readout reduces it while the engine captures the next group. The content receipt (the
  sha256 of every tensor's native bytes, as the disk path records it) is computed by the
  engine on a background thread over the bytes it handed off, and written beside the
  group's schedule record; :meth:`LaneSession.capture` releases no row's vector before its
  receipt is written and names the handle the readout took.
* **The non-interference check on a cadence.** Every group is checked (run once unhooked,
  once hooked, and compared) from the session's start until a group under each declared
  condition and the session's first :data:`WARMUP_GROUPS` groups have all passed; from
  then on every :data:`CHECK_EVERY`-th group by its index in the session. A group's record
  says ``checked`` or ``not_checked``, and an unchecked group never claims
  ``hook_noninterference``. A failed check stops the session and names every row released
  since the last passing check as unverified (:class:`NoninterferenceFailure`). Banks,
  transfers and install checks are not sessions and check every group.
* **One engine, built for batches of eight.** A capture of eight rows or more runs as
  ``full-b8-order0`` groups of eight (a short final group filled with other rows of the
  same capture, never read back, as :func:`anamnesis.extraction.vllm.envelope.request_groups`
  fills one). A capture of fewer rows runs one request at a time, as ``full-b1-order0``;
  each group's recorded schedule proves the concurrency it ran at, and the install check
  certifies the two conditions byte-identical on this host. Every receipt names the
  condition its row ran under.

A single-GPU lane only: a tensor-parallel lane is refused.

The children are this module run as ``python -m anamnesis.extraction.vllm.session
{engine,readout} SPEC --reply-fd N``: requests arrive as JSON lines on stdin, answers
leave as JSON lines on a dedicated pipe, and the children's own stdout and stderr (the
engine's log) pass through to the caller's.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import selectors
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from anamnesis.extraction.vllm.envelope import CONDITIONS

ENGINE_CONDITION = "full-b8-order0"
"""The condition the session's engine is built for: batches of eight."""

SHORT_CONDITION = "full-b1-order0"
"""The condition a capture of fewer rows than one batch holds runs under."""

CAPACITY = int(CONDITIONS[ENGINE_CONDITION]["max_num_seqs"])
"""Rows per batch under :data:`ENGINE_CONDITION`."""

WARMUP_GROUPS = 8
"""Every group is checked until at least this many groups of the session have passed."""

CHECK_EVERY = 16
"""After the warm-up, the groups whose session index is a multiple of this are checked."""

STARTUP_TIMEOUT_S = 3600.0
"""How long a child may take to become ready: a 70B engine build is minutes."""

IDLE_TIMEOUT_S = 1800.0
"""How long a capture may go without any answer from either child."""

STEP_MODULE = "anamnesis.extraction.vllm.session"
"""The module each child runs."""


class LaneSessionError(RuntimeError):
    """The session refused, or one of its processes failed; the session is stopped."""


class NoninterferenceFailure(LaneSessionError):
    """A checked group's hooked run differed from its unhooked run.

    ``unverified`` lists every row released since the last passing check, as
    ``{"call": n, "generation_id": g}``: rows to recapture or drop. ``record`` is the
    failure record written into the session's work directory.
    """

    def __init__(self, message: str, *, unverified: list[dict[str, int]],
                 record: Path) -> None:
        super().__init__(message)
        self.unverified = unverified
        self.record = record


# ── planning: pure, no engine, no device ─────────────────────────────────────


def plan_groups(generation_ids: Sequence[int]) -> tuple[str, list[list[dict[str, Any]]]]:
    """The condition and engine batches a capture of ``generation_ids`` runs as.

    Fewer than :data:`CAPACITY` rows run one at a time under :data:`SHORT_CONDITION`;
    otherwise :func:`anamnesis.extraction.vllm.envelope.request_groups` at
    :data:`CAPACITY` under :data:`ENGINE_CONDITION`.

    Raises
    ------
    ValueError
        On no ids, duplicates or non-integer ids.
    """
    from anamnesis.extraction.vllm.envelope import request_groups

    ids = list(generation_ids)
    if not ids or len(set(ids)) != len(ids) or any(type(i) is not int for i in ids):
        raise ValueError("a capture needs unique integer generation ids")
    if len(ids) < CAPACITY:
        return SHORT_CONDITION, request_groups(ids, 1)
    return ENGINE_CONDITION, request_groups(ids, CAPACITY)


@dataclass
class Cadence:
    """Which of a session's groups run the non-interference check.

    ``next_index`` is the session index the next group gets. Every group is checked
    while the warm-up lasts: until a group under each condition in :data:`CONDITIONS`
    has passed and the session's first :data:`WARMUP_GROUPS` groups have passed. After
    it, a group is checked when its index is a multiple of :data:`CHECK_EVERY`. The
    plan is a function of the indices and conditions alone, never chosen.
    """

    next_index: int = 0
    passed_conditions: set[str] = field(default_factory=set)
    passed_groups: int = 0

    def warm(self) -> bool:
        return (self.passed_groups < WARMUP_GROUPS
                or not set(CONDITIONS) <= self.passed_conditions)

    def plan(self, condition_id: str, n_groups: int) -> list[bool]:
        """The checks for the next ``n_groups`` groups under ``condition_id``, given
        that every checked group before them passes (a failure stops the session)."""
        if condition_id not in CONDITIONS or n_groups < 1:
            raise ValueError("a declared condition and at least one group are required")
        trial = Cadence(self.next_index, set(self.passed_conditions), self.passed_groups)
        checks = []
        for _ in range(n_groups):
            checked = trial.warm() or trial.next_index % CHECK_EVERY == 0
            trial.advance(condition_id, checked)
            checks.append(checked)
        return checks

    def advance(self, condition_id: str, checked: bool) -> None:
        """Record one group that ran and, when checked, passed."""
        if checked:
            self.passed_conditions.add(condition_id)
            self.passed_groups += 1
        self.next_index += 1


def unverified_rows(ledger: Sequence[Mapping[str, Any]],
                    last_pass: int) -> list[dict[str, int]]:
    """Every released row of a group after session index ``last_pass``.

    ``ledger`` holds one entry per released group: ``index``, ``call`` and
    ``generation_ids``.
    """
    return [dict(call=int(g["call"]), generation_id=int(gid))
            for g in ledger if int(g["index"]) > last_pass
            for gid in g["generation_ids"]]


def encode_array(array: np.ndarray) -> str:
    """A float32 vector as base64 of its little-endian bytes: exact, and JSON-safe."""
    return base64.b64encode(np.ascontiguousarray(array, dtype="<f4").tobytes()).decode()


def decode_array(text: str) -> np.ndarray:
    """The inverse of :func:`encode_array`, as native float32."""
    return np.frombuffer(base64.b64decode(text), dtype="<f4").astype(np.float32)


def check_release(handle: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    """Refuse to release a row whose receipt is not about the handle the readout took.

    The receipt must name the handle's generation id and receipt id and list exactly the
    handle's tensors, with their shapes and dtypes.

    Raises
    ------
    LaneSessionError
        Naming what differs.
    """
    gid = handle["generation_id"]
    if receipt.get("generation_id") != gid or receipt.get("receipt_id") != handle["receipt_id"]:
        raise LaneSessionError(f"row {gid}: its receipt is not the one its hand-off named")
    named = {(e["field"] if e["layer"] is None else f"{e['field']}/{e['layer']}"):
             (list(e["shape"]), e["dtype"]) for e in handle["tensors"]}
    receipted = {name: (list(t["shape"]), t["dtype"])
                 for name, t in receipt["substrate"]["tensors"].items()}
    if named != receipted:
        raise LaneSessionError(f"row {gid}: its receipt describes other tensors than the "
                               "hand-off it was reduced from")


# ── the children ─────────────────────────────────────────────────────────────


def _reply_writer(fd: int):  # type: ignore[no-untyped-def]
    stream = os.fdopen(fd, "w", buffering=1)
    lock = threading.Lock()

    def reply(payload: Mapping[str, Any]) -> None:
        line = json.dumps(dict(payload), default=str, allow_nan=False) + "\n"
        with lock:
            stream.write(line)
            stream.flush()
    return reply


def _wait_for_room(directory: Path, limit: int, timeout: float) -> float:
    """Block while ``limit`` segments wait for the readout; return the seconds waited."""
    from anamnesis.extraction.vllm.handoff import outstanding

    began = time.perf_counter()
    while outstanding(directory) >= limit:
        if time.perf_counter() - began > timeout:
            raise RuntimeError(f"the readout took no segment for {timeout:.0f} s")
        time.sleep(0.005)
    return time.perf_counter() - began


def _device_record() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return dict(device=None, device_uuid=None)
    return dict(device=torch.cuda.get_device_name(0),
                device_uuid=str(torch.cuda.get_device_properties(0).uuid))


def _engine_main(spec: Mapping[str, Any], reply,  # type: ignore[no-untyped-def]
                 requests: Iterable[str] | None = None) -> int:
    """The engine child: one engine, one capture request at a time, read from
    ``requests`` (this process's stdin unless given)."""
    from anamnesis.extraction.vllm import handoff
    from anamnesis.extraction.vllm.receipts import capture_receipt
    from anamnesis.extraction.vllm.runner import (
        NoninterferenceError,
        capture_groups,
        write_json,
    )
    from anamnesis.extraction.vllm.runtime import ATTENTION_ROUNDING, build_engine

    engine = build_engine(str(spec["model"]), spec["model_path"], ENGINE_CONDITION)
    segments = Path(spec["segments"])
    limit, token = int(spec["max_in_flight"]), str(spec["token"])
    reply(dict(ready=True, record=dict(
        engine["record"], resident=True, engine_pid=os.getpid(),
        device_uuid=_device_record()["device_uuid"])))
    receipts = ThreadPoolExecutor(max_workers=1, thread_name_prefix="capture-receipt")

    def receipt_job(record: dict[str, Any], view: dict[str, Any], handle: dict[str, Any],
                    out: Path, condition_id: str) -> None:
        began = time.perf_counter()
        substrate = capture_receipt(handoff.substrate(view))
        handoff.release(view)
        receipt = dict(record, receipt_id=handle["receipt_id"], substrate=substrate,
                       condition_id=condition_id, handoff="memory")
        write_json(out / f"row-{record['generation_id']:05d}.json", receipt)
        reply(dict(event="receipt", generation_id=record["generation_id"], receipt=receipt,
                   receipt_s=time.perf_counter() - began))

    for line in (sys.stdin if requests is None else requests):
        message = json.loads(line)
        if message.get("op") == "shutdown":
            break
        began = time.perf_counter()
        pending = []
        try:
            if message.get("op") != "capture":
                raise ValueError(f"unknown op {message.get('op')!r}")
            condition_id = str(message["condition_id"])
            out = Path(message["out"])
            out.mkdir(parents=True, exist_ok=False)
            call = int(message["call"])

            def publish(record: dict[str, Any], capture: dict[str, Any]) -> None:
                waited = _wait_for_room(segments, limit, IDLE_TIMEOUT_S)
                start = time.perf_counter()
                gid = int(record["generation_id"])
                handle, view = handoff.publish(capture, segments, generation_id=gid,
                                               receipt_id=f"{token}-{call:06d}-{gid}")
                written = time.perf_counter() - start
                reply(dict(event="row", handle=handle, handoff_write_s=written,
                           handoff_wait_s=waited))
                pending.append(receipts.submit(receipt_job, record, view, handle, out,
                                               condition_id))

            result = capture_groups(
                engine["llm"], engine["runner"], message["rows"], message["groups"],
                CONDITIONS[condition_id], engine["sampled_layers"], engine["sampling"],
                engine["tokens_prompt"], out, attention_rounding=ATTENTION_ROUNDING,
                checks=message["checks"], publish=publish)
            for job in pending:
                job.result()
            reply(dict(event="done", ok=True, groups=result["groups"],
                       seconds=time.perf_counter() - began))
        except NoninterferenceError as exc:
            for job in pending:
                job.exception()
            failed = sorted(out.glob("group-*.failed.json"))
            reply(dict(event="done", ok=False, noninterference_failed=True,
                       error=f"{type(exc).__name__}: {exc}",
                       failed_group=(json.loads(failed[-1].read_text())["group_index"]
                                     if failed else None)))
            return 4
        except Exception as exc:  # noqa: BLE001 - every failure is answered, by name
            for job in pending:
                job.exception()
            reply(dict(event="done", ok=False, noninterference_failed=False,
                       error=f"{type(exc).__name__}: {exc}"))
            return 3
    receipts.shutdown(wait=True)
    return 0


def _readout_main(spec: Mapping[str, Any], reply,  # type: ignore[no-untyped-def]
                  requests: Iterable[str] | None = None) -> int:
    """The readout child: one readout lane, one row at a time, read from ``requests``
    (this process's stdin unless given)."""
    import torch

    from anamnesis.extraction.vllm import handoff
    from anamnesis.extraction.vllm.readout import reduce_capture
    from anamnesis.extraction.vllm.runtime import (
        READOUT_DEVICE,
        prepare_readout_process,
        readout_lane,
    )

    prepare_readout_process()
    lane = readout_lane(str(spec["model"]), Path(spec["calib_dir"]),
                        list(spec["feature_names"]))
    segments = Path(spec["segments"])
    reply(dict(ready=True, record=dict(
        _device_record(), identity=getattr(lane, "identity", None), resident=True,
        readout_pid=os.getpid())))

    def on_device(value):  # type: ignore[no-untyped-def]
        return ({k: on_device(v) for k, v in value.items()}
                if isinstance(value, dict) else value.to(READOUT_DEVICE))

    for line in (sys.stdin if requests is None else requests):
        message = json.loads(line)
        if message.get("op") == "shutdown":
            return 0
        gid = None
        try:
            if message.get("op") != "reduce":
                raise ValueError(f"unknown op {message.get('op')!r}")
            row = message["row"]
            gid = int(row["generation_id"])
            handle = message["handle"]
            if Path(handle["path"]).parent != segments:
                raise handoff.HandoffError("the handle names a segment outside this session")
            began = time.perf_counter()
            mapped = handoff.open_segment(handle, generation_id=gid)
            handoff.unlink(handle)
            opened = time.perf_counter()
            raw = handoff.substrate(mapped)
            knnlm = (raw["hidden"][-1][-1].float().numpy().copy()
                     if lane.config.enable_knnlm_baseline else None)
            with torch.inference_mode():
                on_card = on_device(raw)
                del raw
                handoff.release(mapped)
                moved = time.perf_counter()
                readout = reduce_capture(lane, on_card, start=int(row["prompt_length"]),
                                         end=int(row["end"]), model=str(spec["model"]))
            del on_card
            reduced = time.perf_counter()
            reply(dict(ok=True, generation_id=gid,
                       vector=encode_array(np.asarray(readout.features, dtype=np.float32)),
                       knnlm=None if knnlm is None else encode_array(knnlm),
                       open_s=opened - began, to_device_s=moved - opened,
                       reduce_s=reduced - moved))
        except Exception as exc:  # noqa: BLE001 - every failure is answered, by name
            reply(dict(ok=False, generation_id=gid, error=f"{type(exc).__name__}: {exc}"))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m anamnesis.extraction.vllm.session {engine,readout} SPEC --reply-fd N``."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 4 or args[0] not in ("engine", "readout") or args[2] != "--reply-fd":
        print(f"usage: python -m {STEP_MODULE} {{engine,readout}} SPEC.json --reply-fd N",
              file=sys.stderr)
        return 2
    reply = _reply_writer(int(args[3]))
    spec = json.loads(Path(args[1]).read_text())
    try:
        return (_engine_main if args[0] == "engine" else _readout_main)(spec, reply)
    except Exception as exc:  # noqa: BLE001 - a setup failure is answered, then raised
        reply(dict(ready=False, ok=False, error=f"{type(exc).__name__}: {exc}"))
        raise


# ── the parent ───────────────────────────────────────────────────────────────


class _Child:
    """One child process: JSON requests on its stdin, JSON answers on a dedicated pipe."""

    def __init__(self, step: str, spec: Mapping[str, Any], spec_path: Path,
                 env: Mapping[str, str]) -> None:
        self.step = step
        spec_path.write_text(json.dumps(dict(spec), indent=2, default=str) + "\n")
        read_fd, write_fd = os.pipe()
        self.fd = read_fd
        self._buf = b""
        self.proc = subprocess.Popen(
            [sys.executable, "-u", "-m", STEP_MODULE, step, str(spec_path),
             "--reply-fd", str(write_fd)],
            stdin=subprocess.PIPE, env=dict(env), pass_fds=(write_fd,))
        os.close(write_fd)
        self.ready: dict[str, Any] = {}

    def alive(self) -> bool:
        return self.proc.poll() is None

    def lines(self) -> list[dict[str, Any]]:
        """Every complete answer readable now (call when the pipe is readable)."""
        chunk = os.read(self.fd, 1 << 20)
        if not chunk:
            self.proc.wait(timeout=60)
            raise LaneSessionError(f"the {self.step} process closed its answers (exit "
                                   f"{self.proc.returncode}; its error is in the log above)")
        self._buf += chunk
        out = []
        while b"\n" in self._buf:
            line, _, self._buf = self._buf.partition(b"\n")
            out.append(dict(json.loads(line)))
        return out

    def wait_ready(self, timeout: float) -> dict[str, Any]:
        selector = selectors.DefaultSelector()
        selector.register(self.fd, selectors.EVENT_READ)
        deadline = time.monotonic() + timeout
        try:
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise LaneSessionError(f"the {self.step} process did not start in "
                                           f"{timeout:.0f} s")
                if selector.select(min(left, 5.0)):
                    for reply in self.lines():
                        if not reply.get("ready"):
                            raise LaneSessionError(f"the {self.step} process did not start: "
                                                   f"{reply.get('error', reply)}")
                        self.ready = reply
                        return reply
                elif not self.alive():
                    raise LaneSessionError(f"the {self.step} process exited "
                                           f"{self.proc.returncode} before it was ready")
        finally:
            selector.close()

    def send(self, message: Mapping[str, Any]) -> None:
        if not self.alive():
            raise LaneSessionError(f"the {self.step} process is gone (exit "
                                   f"{self.proc.returncode})")
        assert self.proc.stdin is not None
        self.proc.stdin.write((json.dumps(dict(message), default=str) + "\n").encode())
        self.proc.stdin.flush()

    def close(self, timeout: float = 120.0) -> None:
        if self.alive():
            try:
                assert self.proc.stdin is not None
                self.proc.stdin.write(b'{"op": "shutdown"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=timeout)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()
                self.proc.wait(timeout=60)
        try:
            os.close(self.fd)
        except OSError:
            pass


@dataclass
class SessionCapture:
    """What one :meth:`LaneSession.capture` released.

    ``vectors`` and ``knnlm`` are by generation id, in the lane's feature order
    (``knnlm`` is empty unless the lane's configuration enables it). ``receipts`` holds
    each row's capture receipt as written beside the call's schedule records in
    ``directory``; ``groups`` each group's evidence, with its ``session_index`` and its
    ``noninterference`` status. ``timing`` holds per-row seconds: ``capture`` (the
    group's engine time, less its hand-off writes and waits, per retained row),
    ``handoff_write``, ``handoff_wait`` (the engine waiting for the readout to take
    segments), ``handoff_open``, ``to_device``, ``receipt`` (off the critical path) and
    ``reduce``.
    """

    call: int
    condition_id: str
    vectors: dict[int, np.ndarray]
    knnlm: dict[int, np.ndarray]
    receipts: dict[int, dict[str, Any]]
    groups: list[dict[str, Any]]
    timing: dict[int, dict[str, float]]
    wall_s: float
    directory: Path


class LaneSession:
    """A vLLM lane held open: build with :meth:`open`, capture with :meth:`capture`,
    and :meth:`close` (or use as a context manager)."""

    _OPENING = object()

    def __init__(self, _token: object, **state: Any) -> None:
        if _token is not LaneSession._OPENING:
            raise TypeError("a LaneSession is built only by LaneSession.open")
        self.model: str = state["model"]
        self.lane_receipt = state["receipt"]
        self.fixtures = state["fixtures"]
        self.work_dir: Path = state["work_dir"]
        self.segments: Path = state["segments"]
        self._schema_inputs = state["schema_inputs"]
        self._engine: _Child = state["engine"]
        self._readout: _Child = state["readout"]
        self.idle_timeout = float(state["idle_timeout"])
        self.cadence = Cadence()
        self._ledger: list[dict[str, Any]] = []
        self._last_pass = -1
        self._schemas: dict[int, Any] = {}
        self.calls = 0
        self.stopped: str | None = None

    # construction ---------------------------------------------------------
    @classmethod
    def open(cls, model: str, model_path: Path, *, work_dir: Path, cache_dir: Path,
             calib_dir: Path | None = None, handoff_root: Path = Path("/dev/shm"),
             max_in_flight: int = 8, startup_timeout: float = STARTUP_TIMEOUT_S,
             idle_timeout: float = IDLE_TIMEOUT_S) -> LaneSession:
        """Admit ``model``'s lane on this host and start its two processes.

        ``work_dir`` must not exist: it receives the session record and, per call, the
        schedule records and capture receipts. ``handoff_root`` is a shared-memory
        directory; the session's segments live in a private directory under it, at most
        ``max_in_flight`` rows waiting for the readout at a time (about a gigabyte each
        for the largest model). ``cache_dir`` holds the install-check receipts.

        Raises
        ------
        LaneSessionError
            Naming the refusal: an unknown or tensor-parallel lane, a calibration other
            than the fixtures', no usable install-check receipt for this host, or a
            child that did not start. Nothing falls back to another lane.
        """
        from anamnesis.config import resolve_preset
        from anamnesis.extraction.calibration import read_pca_basis, resolve_pca_model
        from anamnesis.extraction.replay_config import native_replay_configs
        from anamnesis.extraction.vllm import extensions
        from anamnesis.extraction.vllm.envelope import lane_preset, lane_tensor_parallel_size
        from anamnesis.extraction.vllm.hub import fetch_calibration, verify_calibration
        from anamnesis.extraction.vllm.runtime import (
            child_environment,
            load_fixtures,
            require_fixture_calibration,
            usable_receipt,
        )

        work_dir, model_path = Path(work_dir), Path(model_path)
        if not 1 <= int(max_in_flight) <= 64:
            raise LaneSessionError(f"max_in_flight must be 1..64, got {max_in_flight}")
        try:
            if model not in extensions.lane_keys():
                raise ValueError(f"{model!r} is not a vLLM lane here (lanes: "
                                 f"{', '.join(extensions.lane_keys())})")
            if lane_tensor_parallel_size(model) > 1:
                raise ValueError(f"{model!r} is a tensor-parallel lane; a session holds a "
                                 "single-GPU engine")
            fixtures, _ = load_fixtures(model)
            if extensions.declared_lane(model) is not None:
                calib = extensions.verify_calibration(model, calib_dir)
            else:
                if calib_dir is not None:
                    verify_calibration(model, calib_dir)
                calib = Path(calib_dir) if calib_dir is not None else fetch_calibration(model)
            require_fixture_calibration(fixtures, calib)
            receipt = usable_receipt(model, model_path, Path(cache_dir))
            if not Path(handoff_root).is_dir():
                raise ValueError(f"the hand-off root {handoff_root} is not a directory")
            pca_path = resolve_pca_model(Path(calib))
            if pca_path is None:
                raise ValueError(f"no PCA basis in {calib}")
            components, _ = read_pca_basis(pca_path).float32_arrays()
            preset = resolve_preset(lane_preset(model))
            extraction, families = native_replay_configs(preset)
            work_dir.mkdir(parents=True, exist_ok=False)
        except (ValueError, RuntimeError, OSError, ImportError) as exc:
            raise LaneSessionError(f"vLLM lane {model!r} refused: {exc}") from exc

        token = secrets.token_hex(6)
        segments = Path(handoff_root) / f"anamnesis-lane-{os.getpid()}-{token}"
        segments.mkdir(mode=0o700)
        engine = readout = None
        try:
            engine = _Child("engine", dict(
                model=model, model_path=str(model_path), segments=str(segments),
                max_in_flight=int(max_in_flight), token=token),
                work_dir / "engine.spec.json", child_environment("capture", model=model))
            engine.wait_ready(startup_timeout)
            readout = _Child("readout", dict(
                model=model, calib_dir=str(calib), segments=str(segments),
                feature_names=list(fixtures.feature_names)),
                work_dir / "readout.spec.json", child_environment("reduce"))
            readout.wait_ready(startup_timeout)
        except BaseException:
            for child in (readout, engine):
                if child is not None:
                    child.close()
            shutil.rmtree(segments, ignore_errors=True)
            raise
        session = cls(cls._OPENING, model=model, receipt=receipt, fixtures=fixtures,
                      work_dir=work_dir, segments=segments,
                      schema_inputs=(extraction, families, components, int(preset.num_layers)),
                      engine=engine, readout=readout, idle_timeout=idle_timeout)
        (work_dir / "session.json").write_text(json.dumps(dict(
            model=model, lane_id=receipt.lane_id, qualified_lane_id=receipt.qualified_lane_id,
            conformance_tier=receipt.tier, conformance_receipt_sha256=receipt.digest,
            fixture_digest=fixtures.digest, calibration_sha256=fixtures.calibration_sha256,
            engine_condition=ENGINE_CONDITION, short_condition=SHORT_CONDITION,
            cadence=dict(warmup_groups=WARMUP_GROUPS, check_every=CHECK_EVERY,
                         conditions=sorted(CONDITIONS)),
            handoff="memory", max_in_flight=int(max_in_flight),
            engine=engine.ready.get("record"), readout=readout.ready.get("record")),
            indent=2, default=str) + "\n")
        return session

    def __enter__(self) -> LaneSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Shut both processes down and remove the session's shared-memory directory."""
        self._readout.close()
        self._engine.close()
        shutil.rmtree(self.segments, ignore_errors=True)
        if self.stopped is None:
            self.stopped = "closed"

    # identity -------------------------------------------------------------
    @property
    def lane_id(self) -> str:
        return str(self.lane_receipt.lane_id)

    @property
    def feature_names(self) -> list[str]:
        return list(self.fixtures.feature_names)

    def extraction_lane(self, row: Mapping[str, Any], receipt: Mapping[str, Any]) -> dict[str, Any]:
        """The ``extraction_lane`` record a bank stamps on ``row``, as
        :func:`anamnesis.extraction.vllm.runtime.replay_bank` writes it, naming the
        condition the row ran under, its capture receipt and its check status."""
        from anamnesis.extraction.vllm.runtime import feature_schema_sha256

        r = self.lane_receipt
        input_ids = np.asarray([int(x) for x in row["input_ids"]], dtype="<i8")
        return dict(
            lane_id=r.lane_id, engine="vllm", qualified_lane_id=r.qualified_lane_id,
            conformance_tier=r.tier, conformance_receipt_sha256=r.digest,
            condition_id=receipt["condition_id"], span_start=int(row["prompt_length"]),
            span_end=len(input_ids),
            input_tokens_sha256=hashlib.sha256(input_ids.tobytes()).hexdigest(),
            feature_schema_sha256=feature_schema_sha256(self.feature_names),
            calibration_sha256=self.fixtures.calibration_sha256, certified=False,
            resident=True, capture_receipt_sha256=receipt["substrate"]["sha256"],
            noninterference=receipt["noninterference"])

    # rows -----------------------------------------------------------------
    def check_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """``row`` as the engine takes it, or a refusal naming why the lane cannot take it.

        A row has ``generation_id``, ``input_ids`` and ``prompt_length`` (and ``end``,
        which must be the token count when given). Refused: a span without a prompt and
        two generated tokens, a span past the lane's context, and a span whose schema is
        not the fixtures'.

        Raises
        ------
        ValueError
            Naming the refusal.
        """
        from anamnesis.extraction.fast.schema import resolve_gpu_schema
        from anamnesis.extraction.vllm.runtime import replay_rows

        gid = int(row["generation_id"])
        checked = replay_rows({str(gid): dict(input_ids=list(row["input_ids"]),
                                              prompt_length=int(row["prompt_length"]))},
                              [gid])[0]
        if "end" in row and int(row["end"]) != checked["end"]:
            raise ValueError(f"generation {gid}: end {row['end']} is not its token count "
                             f"{checked['end']}")
        steps = checked["end"] - checked["prompt_length"] - 1
        if steps not in self._schemas:
            extraction, families, components, num_layers = self._schema_inputs
            schema = resolve_gpu_schema(num_layers, steps, extraction, families, components)
            if tuple(schema.feature_names) != tuple(self.feature_names):
                raise ValueError(f"generation {gid}: a {steps}-step span resolves to a schema "
                                 "other than the fixtures'; the vLLM lane does not cover it")
            self._schemas[steps] = True
        return checked

    # the capture ----------------------------------------------------------
    def capture(self, rows: Sequence[Mapping[str, Any]]) -> SessionCapture:
        """Capture and reduce ``rows``; return their vectors and receipts.

        Raises
        ------
        ValueError
            When a row is refused (:meth:`check_row`) or ids repeat; the session goes on.
        NoninterferenceFailure
            When a checked group failed; the session is stopped.
        LaneSessionError
            When a process failed or a release check refused; the session is stopped.
        """
        if self.stopped is not None:
            raise LaneSessionError(f"this session is stopped ({self.stopped})")
        checked_rows = [self.check_row(r) for r in rows]
        ids = [r["generation_id"] for r in checked_rows]
        condition_id, groups = plan_groups(ids)
        checks = self.cadence.plan(condition_id, len(groups))
        base = self.cadence.next_index
        self.calls += 1
        call = self.calls
        out = self.work_dir / f"call-{call:06d}"
        began = time.perf_counter()
        try:
            result = self._run(call, out, checked_rows, condition_id, groups, checks)
        except NoninterferenceFailure:
            raise
        except BaseException as exc:
            self._stop(f"{type(exc).__name__}: {exc}")
            raise
        for local, checked in enumerate(checks):
            self.cadence.advance(condition_id, checked)
            if checked:
                self._last_pass = base + local
        retained = {local: [m["generation_id"] for m in group if m["retain"]]
                    for local, group in enumerate(groups)}
        self._ledger += [dict(index=base + local, call=call, generation_ids=gids)
                         for local, gids in retained.items()]
        engine_groups = result["groups"]
        for record in engine_groups:
            record["session_index"] = base + int(record["group_index"])
        timing = result["timing"]
        for record in engine_groups:
            members = retained[int(record["group_index"])]
            writes = sum(timing[g]["handoff_write"] + timing[g]["handoff_wait"]
                         for g in members)
            for g in members:
                timing[g]["capture"] = (float(record["seconds"]) - writes) / len(members)
        return SessionCapture(call=call, condition_id=condition_id, vectors=result["vectors"],
                              knnlm=result["knnlm"], receipts=result["receipts"],
                              groups=engine_groups, timing=timing,
                              wall_s=time.perf_counter() - began, directory=out)

    def _stop(self, reason: str) -> None:
        self.stopped = reason
        self.close()
        self.stopped = reason

    def _run(self, call: int, out: Path, rows: list[dict[str, Any]], condition_id: str,
             groups: list[list[dict[str, Any]]], checks: list[bool]) -> dict[str, Any]:
        by_id = {r["generation_id"]: r for r in rows}
        self._engine.send(dict(op="capture", call=call, condition_id=condition_id, rows=rows,
                               groups=groups, checks=checks, out=str(out)))
        handles: dict[int, dict[str, Any]] = {}
        receipts: dict[int, dict[str, Any]] = {}
        vectors: dict[int, np.ndarray] = {}
        knnlm: dict[int, np.ndarray] = {}
        timing: dict[int, dict[str, float]] = {g: {} for g in by_id}
        errors: list[str] = []
        done: dict[str, Any] | None = None
        # The readout is given one row at a time: a handle is tens of kilobytes, and a
        # readout blocked answering while this process blocks feeding it would deadlock.
        waiting: deque[int] = deque()
        busy: list[int] = []

        def feed() -> None:
            if not busy and waiting:
                gid = waiting.popleft()
                row = by_id[gid]
                self._readout.send(dict(op="reduce", handle=handles[gid], row=dict(
                    generation_id=gid, prompt_length=row["prompt_length"], end=row["end"])))
                busy.append(gid)

        selector = selectors.DefaultSelector()
        selector.register(self._engine.fd, selectors.EVENT_READ, self._engine)
        selector.register(self._readout.fd, selectors.EVENT_READ, self._readout)
        last = time.monotonic()
        try:
            while done is None or len(vectors) + sum(
                    1 for e in errors if e.startswith("row")) < len(handles):
                ready = selector.select(5.0)
                if not ready:
                    if time.monotonic() - last > self.idle_timeout:
                        raise LaneSessionError(f"no answer from the session's processes in "
                                               f"{self.idle_timeout:.0f} s")
                    for child in (self._engine, self._readout):
                        if not child.alive() and not (child is self._engine and done):
                            raise LaneSessionError(f"the {child.step} process exited "
                                                   f"{child.proc.returncode} mid-capture")
                    continue
                last = time.monotonic()
                for key, _ in ready:
                    child = key.data
                    for reply in child.lines():
                        if child is self._engine:
                            done = self._on_engine(reply, by_id, handles, receipts, timing,
                                                   done, waiting)
                        else:
                            if reply.get("generation_id") not in busy:
                                raise LaneSessionError(
                                    f"the readout answered for row {reply.get('generation_id')}"
                                    ", which it was not given")
                            busy.clear()
                            self._on_readout(reply, handles, vectors, knnlm, timing, errors)
                        feed()
        finally:
            selector.close()
        if not done.get("ok"):
            self._fail(call, done, groups, checks)
        if errors:
            raise LaneSessionError("the readout refused: " + "; ".join(errors[:3]))
        if set(vectors) != set(by_id) or set(receipts) != set(by_id):
            raise LaneSessionError(f"call {call}: rows without a vector or a receipt: "
                                   f"{sorted(set(by_id) - (set(vectors) & set(receipts)))}")
        for gid in by_id:
            check_release(handles[gid], receipts[gid])
        return dict(vectors=vectors, knnlm=knnlm, receipts=receipts,
                    groups=done["groups"], timing=timing)

    def _on_engine(self, reply: Mapping[str, Any], by_id: Mapping[int, dict[str, Any]],
                   handles: dict[int, dict[str, Any]], receipts: dict[int, dict[str, Any]],
                   timing: dict[int, dict[str, float]], done: dict[str, Any] | None,
                   waiting: deque[int]) -> dict[str, Any] | None:
        event = reply.get("event")
        if event == "row":
            handle = reply["handle"]
            gid = handle["generation_id"]
            if gid not in by_id or gid in handles:
                raise LaneSessionError(f"the engine published row {gid}, which this call "
                                       "did not ask for or already has")
            handles[gid] = handle
            timing[gid].update(handoff_write=float(reply["handoff_write_s"]),
                               handoff_wait=float(reply["handoff_wait_s"]))
            waiting.append(gid)
            return done
        if event == "receipt":
            gid = int(reply["generation_id"])
            receipts[gid] = dict(reply["receipt"])
            timing[gid]["receipt"] = float(reply["receipt_s"])
            return done
        if event == "done":
            return dict(reply)
        if "error" in reply:
            return dict(reply, ok=False)
        raise LaneSessionError(f"the engine answered {reply!r}")

    @staticmethod
    def _on_readout(reply: Mapping[str, Any], handles: Mapping[int, dict[str, Any]],
                    vectors: dict[int, np.ndarray], knnlm: dict[int, np.ndarray],
                    timing: dict[int, dict[str, float]], errors: list[str]) -> None:
        gid = reply.get("generation_id")
        if not reply.get("ok"):
            errors.append(f"row {gid}: {reply.get('error')}")
            return
        gid = int(gid)
        if gid not in handles or gid in vectors:
            raise LaneSessionError(f"the readout answered for row {gid}, which it was not "
                                   "given or already answered")
        vectors[gid] = decode_array(reply["vector"])
        if reply.get("knnlm") is not None:
            knnlm[gid] = decode_array(reply["knnlm"])
        timing[gid].update(handoff_open=float(reply["open_s"]),
                           to_device=float(reply["to_device_s"]),
                           reduce=float(reply["reduce_s"]))

    def _fail(self, call: int, done: Mapping[str, Any], groups: list[list[dict[str, Any]]],
              checks: list[bool]) -> None:
        if not done.get("noninterference_failed"):
            raise LaneSessionError(f"call {call}: the capture failed: {done.get('error')}")
        failed = done.get("failed_group")
        base = self.cadence.next_index
        passed = [base + i for i, c in enumerate(checks)
                  if c and failed is not None and i < int(failed)]
        last_pass = max(passed, default=self._last_pass)
        unverified = unverified_rows(self._ledger, last_pass)
        record = self.work_dir / "noninterference-failure.json"
        record.write_text(json.dumps(dict(
            call=call, failed_group=failed,
            failed_session_index=None if failed is None else base + int(failed),
            last_passing_session_index=last_pass, error=done.get("error"),
            unverified_released_rows=unverified,
            unreleased_rows=sorted(m["generation_id"] for g in groups for m in g
                                   if m["retain"])), indent=2) + "\n")
        self._stop(f"non-interference failed in call {call}")
        raise NoninterferenceFailure(
            f"call {call}: a checked group's hooks changed its generation; the session is "
            f"stopped and {len(unverified)} released rows are unverified (see {record})",
            unverified=unverified, record=record)


if __name__ == "__main__":
    raise SystemExit(main())
