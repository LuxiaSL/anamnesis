"""What the vLLM lane is: its engine settings, environment, models and identity.

A vLLM lane computes the fast lane's features from a vLLM engine instead of a
hooked eager forward. It is only the lane it claims to be inside the envelope it
its fixtures were produced in, so everything that fixes its arithmetic is declared here, once,
and refused by name when a run asks for anything else:

* :data:`SETTINGS` — the engine settings that never vary: one GPU, eager
  execution, the instrumented ``TRITON_ATTN`` backend, no prefix caching, no
  LoRA, no speculative decoding, a 1024-token context.
* :data:`REQUIRED_ENV` — the environment the engine process must be started in.
  Batch-invariant mode and the cuBLAS workspace are read when the libraries
  initialise, so the engine process refuses a wrong value rather than correcting
  it after start; :func:`anamnesis.extraction.vllm.runtime.run_step` exports them
  when it launches that process.
* :data:`PINNED_PACKAGES` — the versions the backend and the capture were written
  against. Both reach into engine internals, so another version is refused before
  an engine is built rather than trusted to behave.
* :data:`LANE_MODELS` — the models a lane exists for, and the facts that differ
  between them: the dtype, whether the sampler's logprob input is promoted to
  float32, and, for a model too large for one GPU, its tensor-parallel size.
* :data:`TP_SETTINGS` and :data:`TP_REQUIRED_ENV` — what a tensor-parallel lane
  adds: one worker process per GPU, spawned, reached through the capture's worker
  extension, with the engine's custom all-reduce off. A tensor-parallel lane sums
  each layer's partial products across GPUs in its own order, so it is its own
  lane, with its own id and its own fixtures.
* :data:`CONDITIONS` — the two execution conditions a run may use: one request at
  a time, and batches of eight. The install check captures every fixture under
  both, so a host that passes it has shown the two agree.

:func:`lane_id` is the lane id a model's fixtures carry: a digest of the
facts above that are fixed per model. :func:`enforce_lane_envelope` is the gate
every engine construction passes first.

A lane key is also an **extension lane** (:mod:`anamnesis.extraction.vllm.extensions`),
a declared fine-tune of a shipped model: it inherits its base's facts and settings
and has its own lane id. A shipped key never reads the extension file.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

SETTINGS: dict[str, Any] = dict(
    tensor_parallel_size=1,
    dtype="bfloat16",
    enforce_eager=True,
    compilation_config=0,
    attention_backend="TRITON_ATTN",
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
    async_scheduling=False,
    max_num_seqs=1,
    max_model_len=1024,
    max_num_batched_tokens=1024,
    kv_cache_memory_bytes=2 * 1024**3,
    enable_lora=False,
    speculative_config=None,
    seed=20260922,
)
"""Engine settings fixed by the lane. ``dtype`` and the batching keys in
:data:`CONDITION_KEYS` are set per model and condition; every other key is fixed: the
startup guard refuses a departure, and an install check's receipt is keyed by the
digest of the full settings."""

CONDITION_KEYS = frozenset({
    "dtype", "max_num_seqs", "enable_chunked_prefill", "max_num_batched_tokens",
    "long_prefill_token_threshold", "max_num_partial_prefills",
    "max_long_partial_prefills",
})
"""The settings a model or an execution condition may vary."""

PINNED_PACKAGES: dict[str, str] = {
    "vllm": "0.16.0",
    "torch": "2.9.1",
    "triton": "3.5.1",
}
"""Exact versions the engine process requires. The backend subclasses the engine's
Triton attention implementation and the capture wraps private model-runner
methods, so a different release can change behaviour without failing loudly."""

REQUIRED_ENV: dict[str, str] = {
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_BATCH_INVARIANT": "1",
    "VLLM_PLUGINS": "",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "VLLM_NO_USAGE_STATS": "1",
    "DO_NOT_TRACK": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
"""The environment the engine process must start in. ``VLLM_PLUGINS`` must be set
and empty: unset, the engine loads every installed plugin. Single-process mode
keeps the model runner in this process, where the capture reads it."""

READOUT_WORKSPACE = ":4096:8"
"""The cuBLAS workspace the reduction process runs at. The reduction refuses the
engine's batch-invariant mode, so it always runs in a process of its own."""

LANE_MODELS: dict[str, dict[str, Any]] = {
    "3b": dict(dtype="float16", logprob_wrapper="explicit-fp32-input"),
    "8b": dict(dtype="bfloat16", logprob_wrapper="none"),
    "70b": dict(dtype="bfloat16", logprob_wrapper="none"),
}
"""Per model: the engine dtype, and whether the sampler's logprob input is
promoted to float32. Float16 needs the promotion, because the invariant
log-softmax has no half-precision path; bfloat16 runs without it. A model may
also declare ``tensor_parallel_size`` above 1; without it the lane runs on one GPU."""

TP_WORKER_EXTENSION = "anamnesis.extraction.vllm.tp_worker.TPCaptureExtension"
"""The worker extension a tensor-parallel lane's engine loads into every worker:
importing it installs the instrumented backend there, and its methods are how the
capture reaches each worker's model."""

TP_SETTINGS: dict[str, Any] = dict(
    distributed_executor_backend="mp",
    disable_custom_all_reduce=True,
    worker_extension_cls=TP_WORKER_EXTENSION,
)
"""Engine settings a tensor-parallel lane adds to :data:`SETTINGS`, fixed like them.
The custom all-reduce kernel is off because batch-invariant mode fixes the
reduction order only for the collective library's all-reduce."""

TP_REQUIRED_ENV: dict[str, str] = {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
"""The environment a tensor-parallel lane adds to :data:`REQUIRED_ENV`: forked
workers inherit the parent's CUDA state and can hang, so they are spawned."""

CONDITIONS: dict[str, dict[str, Any]] = {
    "full-b1-order0": dict(condition_id="full-b1-order0", max_num_seqs=1),
    "full-b8-order0": dict(condition_id="full-b8-order0", max_num_seqs=8),
}
"""Execution conditions a run may use: whole-prompt prefill, one request at a time
or eight. Condition ids are written into every capture record."""


def canonical_digest(value: Any) -> str:
    """sha256 of ``value`` as compact, key-sorted JSON; NaN is refused."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _extension(model: str):
    """The extension lane declared as ``model``; see :mod:`anamnesis.extraction.vllm.extensions`.

    Raises
    ------
    ValueError
        When ``model`` names no lane.
    """
    from anamnesis.extraction.vllm.extensions import declared_lane, lane_keys

    entry = declared_lane(model)
    if entry is None:
        raise ValueError(f"{model!r} has no vLLM lane (declared: {', '.join(lane_keys())})")
    return entry


def lane_model(model: str) -> dict[str, str]:
    """The declared facts for ``model``: its own, or for an extension its base's.

    Raises
    ------
    ValueError
        When ``model`` has no vLLM lane.
    """
    return LANE_MODELS[model if model in LANE_MODELS else _extension(model).extends]


def lane_tensor_parallel_size(model: str) -> int:
    """The number of GPUs ``model``'s lane runs on: its declared tensor-parallel size,
    or 1."""
    return int(lane_model(model).get("tensor_parallel_size", 1))


def fixed_settings(model: str) -> dict[str, Any]:
    """The settings ``model``'s lane fixes: :data:`SETTINGS`, and for a
    tensor-parallel lane its size and :data:`TP_SETTINGS`."""
    tp = lane_tensor_parallel_size(model)
    if tp == 1:
        return dict(SETTINGS)
    return dict(SETTINGS, tensor_parallel_size=tp, **TP_SETTINGS)


def required_environment(model: str) -> dict[str, str]:
    """The environment ``model``'s engine process must start in."""
    if lane_tensor_parallel_size(model) == 1:
        return dict(REQUIRED_ENV)
    return dict(REQUIRED_ENV, **TP_REQUIRED_ENV)


def lane_preset(model: str) -> str:
    """The registry preset ``model``'s lane reads its layer plan from."""
    return model if model in LANE_MODELS else _extension(model).preset


def extension_lane_id(extends: str, checkpoint_sha256: str) -> str:
    """An extension's lane id: the digest of its base's identity, the base's key and
    the checkpoint digest, a shape no shipped identity has."""
    return canonical_digest(dict(base=lane_identity(extends), extends=extends,
                                 checkpoint_sha256=checkpoint_sha256))


def lane_identity(model: str) -> dict[str, Any]:
    """The facts a shipped model's lane fixes, as its lane id digests them."""
    if model not in LANE_MODELS:
        raise ValueError(f"{model!r} has no vLLM lane of its own; an extension's identity "
                         "is extension_lane_id's")
    facts = LANE_MODELS[model]
    identity = dict(model=model, dtype=facts["dtype"], invariant_mode=True,
                    attention_backend="TRITON_ATTN", tensor_parallel_size=1,
                    logprob_wrapper=facts["logprob_wrapper"])
    tp = lane_tensor_parallel_size(model)
    if tp > 1:
        identity.update(tensor_parallel_size=tp, collectives=dict(
            executor=TP_SETTINGS["distributed_executor_backend"],
            custom_all_reduce=not TP_SETTINGS["disable_custom_all_reduce"],
            worker_start=TP_REQUIRED_ENV["VLLM_WORKER_MULTIPROC_METHOD"]))
    return identity


def lane_id(model: str) -> str:
    """The lane id ``model``'s fixtures carry: the digest of its identity, or for an
    extension :func:`extension_lane_id`."""
    if model not in LANE_MODELS:
        entry = _extension(model)
        return extension_lane_id(entry.extends, entry.checkpoint_sha256)
    return canonical_digest(lane_identity(model))


def engine_settings(model: str, condition_id: str) -> dict[str, Any]:
    """The full engine settings for ``model`` under a declared condition.

    The scheduler's token budget is the batch capacity times the context length,
    so a whole prompt always fits one prefill step.

    Raises
    ------
    ValueError
        When the model or the condition is not declared.
    """
    facts = lane_model(model)
    if condition_id not in CONDITIONS:
        raise ValueError(f"{condition_id!r} is not a declared condition "
                         f"(declared: {', '.join(CONDITIONS)})")
    capacity = CONDITIONS[condition_id]["max_num_seqs"]
    return dict(fixed_settings(model), dtype=facts["dtype"], max_num_seqs=capacity,
                enable_chunked_prefill=False,
                max_num_batched_tokens=capacity * SETTINGS["max_model_len"],
                long_prefill_token_threshold=0, max_num_partial_prefills=1,
                max_long_partial_prefills=1)


def request_groups(generation_ids: Sequence[int], capacity: int) -> list[list[dict]]:
    """Split rows into engine batches of exactly ``capacity`` requests.

    Every row is a target exactly once. A short final batch is filled with other
    rows, marked ``retain=False``: they keep the batch composition the condition
    declares and are never read back as features.

    Raises
    ------
    ValueError
        On duplicate or non-integer ids, a capacity outside 1, 2, 4 or 8, or fewer
        rows than one batch holds.
    """
    ids = sorted(generation_ids)
    if (not ids or len(set(ids)) != len(ids) or any(type(i) is not int for i in ids)
            or capacity not in (1, 2, 4, 8) or len(ids) < capacity):
        raise ValueError("unique integer ids, a capacity of 1, 2, 4 or 8, and at least "
                         "one full batch of rows are required")
    groups = []
    for start in range(0, len(ids), capacity):
        targets = ids[start:start + capacity]
        members = [(gid, True) for gid in targets]
        fillers = (gid for gid in ids if gid not in targets)
        while len(members) < capacity:
            members.append((next(fillers), False))
        groups.append([dict(generation_id=gid, retain=retain,
                            occurrence_id=f"group-{len(groups):04d}-slot-{slot}")
                       for slot, (gid, retain) in enumerate(members)])
    return groups


def require_environment(environment: Mapping[str, str] | None = None,
                        model: str | None = None) -> None:
    """Refuse an engine process not started in :data:`REQUIRED_ENV`, or, given
    ``model``, in :func:`required_environment`.

    Raises
    ------
    RuntimeError
        Naming every variable that differs.
    """
    environment = os.environ if environment is None else environment
    required = REQUIRED_ENV if model is None else required_environment(model)
    wrong = {key: environment.get(key) for key, value in required.items()
             if environment.get(key) != value}
    if wrong:
        raise RuntimeError("the vLLM lane's environment must be exported before the "
                           f"engine imports; differing: {wrong}")


def require_pinned_packages() -> dict[str, str]:
    """Refuse an installation whose engine packages differ from :data:`PINNED_PACKAGES`.

    Returns the installed versions checked.

    Raises
    ------
    RuntimeError
        Naming each missing or differing package; this package's ``vllm`` extra
        installs the pinned set.
    """
    found, wrong = {}, {}
    for name, version in PINNED_PACKAGES.items():
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = "missing"
        if found[name].split("+")[0] != version:
            wrong[name] = found[name]
    if wrong:
        raise RuntimeError(f"the vLLM lane needs {PINNED_PACKAGES}; this installation has "
                           f"{wrong} (install this package with its vllm extra: "
                           f"uv pip install -e '.[vllm]' from a clone)")
    return found


def _refuse(lane: str, reason: str) -> None:
    raise ValueError(f"lane {lane} refuses this configuration: {reason}")


def enforce_lane_envelope(settings: Mapping[str, Any], environment: Mapping[str, str], *,
                          lane: str, model: str) -> dict[str, Any]:
    """Refuse engine settings or an environment outside the declared envelope.

    Called between assembling the settings and constructing the engine, with the
    environment the engine will see. The engine refuses some of these
    configurations on its own; the lane never relies on that. Passing asserts only
    that the run asked for a declared configuration: whether the arithmetic agrees
    is the install check's question.

    Returns a record of what was checked, written into the capture record.

    Raises
    ------
    ValueError
        Naming ``lane`` and the first departure found.
    """
    if not isinstance(lane, str) or not lane:
        raise ValueError("a lane id is required before any engine construction")
    if model not in LANE_MODELS:
        from anamnesis.extraction.vllm.extensions import admit, declared_lane

        if declared_lane(model) is None:
            _refuse(lane, f"model {model!r} has no vLLM lane")
        try:
            admit(model)
        except ValueError as exc:
            _refuse(lane, str(exc))
    declared = lane_model(model)["dtype"]
    dtype = settings.get("dtype")
    if dtype != declared:
        _refuse(lane, f"dtype {dtype!r} for model {model} "
                      f"(declared: {declared})")
    if environment.get("VLLM_BATCH_INVARIANT") != "1":
        _refuse(lane, "non-invariant mode (VLLM_BATCH_INVARIANT must be 1)")
    if environment.get("VLLM_PLUGINS") != "":
        _refuse(lane, "engine plugins (VLLM_PLUGINS must be set and empty; unset loads "
                      "every installed plugin)")
    if settings.get("enable_prefix_caching") is not False:
        _refuse(lane, "prefix caching")
    if settings.get("enable_lora") is not False:
        _refuse(lane, "LoRA adapters")
    if settings.get("speculative_config") is not None:
        _refuse(lane, "speculative decoding")
    if settings.get("enable_chunked_prefill") is not False:
        _refuse(lane, "chunked prefill")
    partial = settings.get("max_num_partial_prefills", 1)
    if type(partial) is not int or partial != 1:
        _refuse(lane, f"concurrent partial prefill (max_num_partial_prefills={partial!r})")
    capacity = settings.get("max_num_seqs")
    if capacity not in {c["max_num_seqs"] for c in CONDITIONS.values()}:
        _refuse(lane, f"batch capacity max_num_seqs={capacity!r} (declared: "
                      f"{sorted(c['max_num_seqs'] for c in CONDITIONS.values())})")
    required = required_environment(model)
    for key, expected in required.items():
        if environment.get(key) != expected:
            _refuse(lane, f"environment {key}={environment.get(key)!r} "
                          f"(declared: {expected!r})")
    fixed = fixed_settings(model)
    undeclared = sorted(set(settings) - set(fixed) - CONDITION_KEYS)
    if undeclared:
        _refuse(lane, f"undeclared engine settings {undeclared}")
    for key in sorted(set(fixed) - CONDITION_KEYS):
        if key not in settings or settings[key] != fixed[key]:
            _refuse(lane, f"{key}={settings.get(key)!r} (the lane fixes it to "
                          f"{fixed[key]!r})")
    record = dict(lane_id=lane, model=model, dtype=dtype,
                  named_exclusions_checked=[
                      "prefix_caching", "plugins", "lora", "speculative_decoding",
                      "non_invariant_mode", "dtype", "chunked_prefill",
                      "concurrent_partial_prefill"],
                  fixed_settings_verified=sorted(set(fixed) - CONDITION_KEYS),
                  environment_verified=sorted(required))
    if fixed["tensor_parallel_size"] > 1:
        record["tensor_parallel_size"] = fixed["tensor_parallel_size"]
    return record
