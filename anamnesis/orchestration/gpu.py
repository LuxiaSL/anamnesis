"""Which device a worker gets, and the environment it is handed.

Three concerns, all of them about the boundary between a launcher and the
subprocesses it spawns:

* **Slot resolution.** ``--gpus`` names *logical* slots, not physical devices.
  A scheduler communicates its assignment by exporting
  ``CUDA_VISIBLE_DEVICES`` for the job, so slot *i* means the *i*-th device in
  that assignment; only on bare metal, where the variable is unset, is a slot a
  physical index. Writing raw slot numbers into a worker's
  ``CUDA_VISIBLE_DEVICES`` stacks the whole fleet onto the wrong devices — the
  job collides with its neighbours while its own assignment idles, and its
  processes are untracked by the scheduler's ledger (observed live on a
  co-scheduled generation pass; finding recorded in the canonical-ops notes).
  :func:`resolve_physical_gpus` is the translation.

* **The worker environment.** A worker inherits the launcher's environment plus
  its device and a single-threaded BLAS pin. The pin is not a courtesy: feature
  extraction is CPU-bound, so many workers each opening a core-count thread pool
  oversubscribe the machine, and a multi-threaded reduction can also change the
  summation order behind a feature. One thread per worker gives clean N-way
  parallelism across workers and keeps the arithmetic a worker does independent
  of how many siblings it has.

* **The path-of-record guard.** A multi-cell roster driven by a shell loop over
  a single-cell launcher reloads the model once per cell. Prose saying "use the
  multicell path" does not prevent it, so :func:`enforce_single_cell_guard`
  refuses the pattern at the second invocation and points at the launcher that
  loads once. A resumed cell re-runs against the same output directory and is
  never refused; deliberate sequential single-cell work says so with
  ``--single-cell-ok``.
"""

from __future__ import annotations

import getpass
import json
import logging
import os
import tempfile
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

VISIBLE_DEVICES_ENV = "CUDA_VISIBLE_DEVICES"
"""How a scheduler names the devices a job may use, and how a launcher names the
one device a worker may use."""

THREAD_LIMIT_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
"""The thread-pool variables a worker is pinned through, one per numeric runtime
that would otherwise size its pool to the whole machine."""

ESCAPE_FLAG = "--single-cell-ok"
ESCAPE_ENV = "ANAMNESIS_SINGLE_CELL_OK"
CONTEXT_ENV = "ANAMNESIS_JOB_CONTEXT"
TTL_ENV = "ANAMNESIS_GUARD_TTL_HOURS"
GUARD_DIR_ENV = "ANAMNESIS_GUARD_DIR"
DEFAULT_TTL_HOURS = 12.0
STALE_FILE_FACTOR = 4.0
"""Guard files untouched for this multiple of the time-to-live are deleted
opportunistically, so a machine does not accumulate bookkeeping forever."""


def resolve_physical_gpus(requested: Iterable[str]) -> list[str]:
    """Map logical device slots onto the scheduler's assignment.

    With no ``CUDA_VISIBLE_DEVICES`` in the environment the slots are already
    physical indices and come back as themselves. With one, slot *i* resolves to
    its *i*-th entry. Either way the result is one device string per slot, in the
    order asked for.

    Raises
    ------
    ValueError
        When a slot is not a non-negative integer, or names a slot beyond the
        assignment — which is a request for devices the job was not given, and is
        refused rather than silently narrowed. A slot is validated whether or not
        the job carries an assignment, because the same typo is the same mistake on
        bare metal and would otherwise be caught on one box and not another.
    """
    wanted = [_slot_index(item) for item in requested]
    parent = os.environ.get(VISIBLE_DEVICES_ENV, "").strip()
    if not parent:
        return [str(index) for index in wanted]
    visible = [g.strip() for g in parent.split(",") if g.strip()]
    out: list[str] = []
    for index in wanted:
        if index >= len(visible):
            raise ValueError(
                f"device slot {index} exceeds the scheduler assignment "
                f"{VISIBLE_DEVICES_ENV}={parent!r} ({len(visible)} device(s)); "
                f"request more devices at submit time, or ask for fewer slots"
            )
        out.append(visible[index])
    logger.info(f"device slots {wanted} -> physical {out} ({VISIBLE_DEVICES_ENV}={parent!r})")
    return out


def _slot_index(slot: object) -> int:
    """One ``--gpus`` entry as the index it has to be.

    Negative indices are refused rather than allowed to mean what they mean in
    Python: ``-1`` would select the last device of the assignment, so a fleet asked
    for slots ``0,-1`` would put two workers on one device and report two.
    """
    text = str(slot).strip()
    try:
        index = int(text)
    except ValueError as exc:
        raise ValueError(f"device slot {text!r} is not an integer") from exc
    if index < 0:
        raise ValueError(
            f"device slot {index} is negative; a slot is a position in the job's device "
            f"assignment, counted from 0"
        )
    return index


def worker_environment(
    device: str | None = None,
    *,
    extra: Mapping[str, str] | None = None,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The environment one worker subprocess runs under.

    Inherits ``base`` (the launcher's own environment by default), pins every
    thread-pool variable in :data:`THREAD_LIMIT_ENV` to one, keeps ``PYTHONPATH``
    resolvable from the working directory, and — when ``device`` is given —
    confines the worker to that single device. ``extra`` is applied last, so a
    caller that means to raise a limit can.
    """
    env = dict(os.environ if base is None else base)
    env["PYTHONPATH"] = env.get("PYTHONPATH", ".")
    for name in THREAD_LIMIT_ENV:
        env[name] = "1"
    if device is not None:
        env[VISIBLE_DEVICES_ENV] = str(device)
    env.update(extra or {})
    return env


@dataclass(frozen=True)
class _Entry:
    """One recorded launcher invocation inside a job context."""

    ts: float
    script: str
    model_path: str
    out_dir: str


def _ttl_seconds() -> float:
    try:
        return float(os.environ.get(TTL_ENV, DEFAULT_TTL_HOURS)) * 3600.0
    except ValueError:
        return DEFAULT_TTL_HOURS * 3600.0


def _context_key() -> str:
    """What counts as one job.

    An explicit ``ANAMNESIS_JOB_CONTEXT`` wins. Otherwise the parent process id:
    a shell chain or loop shares one parent, while a new scheduler attempt or a
    new terminal is a new context, which is the distinction the guard needs.
    """
    explicit = os.environ.get(CONTEXT_ENV)
    if explicit:
        return explicit
    return f"ppid{os.getppid()}"


def _guard_dir() -> Path:
    override = os.environ.get(GUARD_DIR_ENV)
    if override:
        return Path(override)
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 — a nameless user still gets a private dir
        user = f"uid{os.getuid()}"
    return Path(tempfile.gettempdir()) / f"anamnesis_single_cell_guard_{user}"


def _record_path() -> Path:
    # An explicitly set context key can carry path-hostile characters.
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in _context_key())
    return _guard_dir() / f"{safe}.jsonl"


def _load_entries(path: Path, now: float, ttl: float) -> list[_Entry]:
    entries: list[_Entry] = []
    try:
        if not path.exists():
            return entries
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                entry = _Entry(
                    ts=float(row["ts"]),
                    script=str(row["script"]),
                    model_path=str(row["model_path"]),
                    out_dir=str(row["out_dir"]),
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue  # a corrupt line is skipped, never fatal to a production launcher
            if now - entry.ts <= ttl:
                entries.append(entry)
    except OSError as err:
        logger.warning(f"single-cell guard: could not read {path}: {err}")
    return entries


def _write_entries(path: Path, entries: list[_Entry]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(asdict(e)) + "\n" for e in entries))
        tmp.replace(path)
    except OSError as err:
        logger.warning(f"single-cell guard: could not write {path}: {err}")


def _prune_stale_files(guard_dir: Path, now: float, ttl: float) -> None:
    try:
        if not guard_dir.is_dir():
            return
        for f in guard_dir.glob("*.jsonl"):
            try:
                if now - f.stat().st_mtime > ttl * STALE_FILE_FACTOR:
                    f.unlink()
            except OSError:
                pass
    except OSError:
        pass


def enforce_single_cell_guard(
    script: str,
    model_path: str,
    out_dir: Path | str,
    *,
    allow_repeat: bool,
    multicell_pointer: str,
) -> None:
    """Refuse a per-cell loop over a single-cell launcher, and record this call.

    The refused pattern is: the same script, against the same model, inside one
    job context, writing to a *different* output directory than a previous call.
    That is a roster being walked one cell at a time, and it pays a model load
    per cell. A resumed cell writes to the same directory and passes.

    All bookkeeping is best-effort — an unusable temporary directory degrades to
    a warning, because losing a guard record must never kill a run. Only the
    refusal itself raises.

    Parameters
    ----------
    script
        Fully qualified module name of the calling launcher.
    model_path
        The model this invocation would load.
    out_dir
        This invocation's output root.
    allow_repeat
        The command-line escape hatch; ``ANAMNESIS_SINGLE_CELL_OK`` set to
        anything but the empty string or ``0`` has the same effect, for
        invocations buried inside a driver.
    multicell_pointer
        One line naming the launcher that loads once and loops cells.

    Raises
    ------
    SystemExit
        When the per-cell-loop pattern is detected and no escape hatch is set.
    """
    now = time.time()
    ttl = _ttl_seconds()
    out_str = str(Path(out_dir).expanduser().resolve())
    expanded = os.path.expanduser(str(model_path))
    model_str = str(Path(model_path).expanduser().resolve()) if os.path.exists(expanded) else str(model_path)
    allow = allow_repeat or os.environ.get(ESCAPE_ENV, "") not in ("", "0")

    path = _record_path()
    entries = _load_entries(path, now, ttl)
    prior = [
        e for e in entries
        if e.script == script and e.model_path == model_str and e.out_dir != out_str
    ]

    if prior and not allow:
        prior_dirs = sorted({e.out_dir for e in prior})
        shown = "\n".join(f"    {d}" for d in prior_dirs[:8])
        raise SystemExit(
            f"\nPATH-OF-RECORD GUARD ({script}):\n"
            f"  invocation #{len(prior_dirs) + 1} against the same model within one job "
            f"context ({_context_key()}), each with a DIFFERENT out dir:\n{shown}\n"
            f"    {out_str}  <- this invocation (REFUSED)\n"
            f"  model: {model_str}\n"
            f"  A per-cell loop reloads the model once per cell. Multi-cell rosters go "
            f"through the multicell path:\n"
            f"    {multicell_pointer}\n"
            f"  If sequential single-cell invocation is deliberate, re-run with "
            f"{ESCAPE_FLAG} (or {ESCAPE_ENV}=1).\n"
        )
    if prior and allow:
        logger.warning(
            f"single-cell guard: repeat single-cell invocation allowed by escape hatch "
            f"({len(prior)} prior in this context; multicell path: {multicell_pointer})"
        )

    keep = [
        e for e in entries
        if (e.script, e.model_path, e.out_dir) != (script, model_str, out_str)
    ]
    keep.append(_Entry(ts=now, script=script, model_path=model_str, out_dir=out_str))
    _write_entries(path, keep)
    _prune_stale_files(_guard_dir(), now, ttl)
