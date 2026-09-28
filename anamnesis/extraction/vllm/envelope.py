"""What the vLLM lane is: its engine settings, environment, models and identity.

A vLLM lane computes the fast lane's features from a vLLM engine instead of a
hooked eager forward. It is only the lane it claims to be inside the envelope it
was qualified in, so everything that fixes its arithmetic is declared here, once,
and refused by name when a run asks for anything else:

* :data:`SETTINGS` — the engine settings that never vary: one GPU, eager
  execution, the instrumented ``TRITON_ATTN`` backend, no prefix caching, no
  LoRA, no speculative decoding, a 1024-token context.
* :data:`REQUIRED_ENV` — the environment the engine process must be started in.
  Batch-invariant mode and the cuBLAS workspace are read when the libraries
  initialise, so they are refused rather than set.
* :data:`PINNED_PACKAGES` — the versions the backend and the capture were written
  against. Both reach into engine internals, so another version is refused before
  an engine is built rather than trusted to behave.
* :data:`LANE_MODELS` — the models a lane exists for, and the two facts that
  differ between them: the dtype, and whether the sampler's logprob input is
  promoted to float32.
* :data:`CONDITIONS` — the two execution conditions a run may use: one request at
  a time, and batches of eight. The install check captures every fixture under
  both, so a host that passes it has shown the two agree.

:func:`lane_id` is the qualified lane's identity for a model: a digest of the
facts above that are fixed per model. :func:`enforce_lane_envelope` is the gate
every engine construction passes first.
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
:data:`CONDITION_KEYS` are set per model and condition; every other key is part of
the lane identity and is refused if it differs."""

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

LANE_MODELS: dict[str, dict[str, str]] = {
    "3b": dict(dtype="float16", logprob_wrapper="explicit-fp32-input"),
    "8b": dict(dtype="bfloat16", logprob_wrapper="none"),
    "70b": dict(dtype="bfloat16", logprob_wrapper="none"),
}
"""Per model: the engine dtype, and whether the sampler's logprob input is
promoted to float32. Float16 needs the promotion, because the invariant
log-softmax has no half-precision path; bfloat16 runs without it."""

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


def lane_model(model: str) -> dict[str, str]:
    """The declared facts for ``model``.

    Raises
    ------
    ValueError
        When ``model`` has no vLLM lane.
    """
    if model not in LANE_MODELS:
        raise ValueError(f"{model!r} has no vLLM lane (declared: "
                         f"{', '.join(sorted(LANE_MODELS))})")
    return LANE_MODELS[model]


def lane_identity(model: str) -> dict[str, Any]:
    """The facts a model's qualified lane fixes, as the lane id digests them."""
    facts = lane_model(model)
    return dict(model=model, dtype=facts["dtype"], invariant_mode=True,
                attention_backend="TRITON_ATTN", tensor_parallel_size=1,
                logprob_wrapper=facts["logprob_wrapper"])


def lane_id(model: str) -> str:
    """The qualified lane's id for ``model``: the digest of its identity."""
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
    return dict(SETTINGS, dtype=facts["dtype"], max_num_seqs=capacity,
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


def require_environment(environment: Mapping[str, str] | None = None) -> None:
    """Refuse an engine process not started in :data:`REQUIRED_ENV`.

    Raises
    ------
    RuntimeError
        Naming every variable that differs.
    """
    environment = os.environ if environment is None else environment
    wrong = {key: environment.get(key) for key, value in REQUIRED_ENV.items()
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
        Naming each missing or differing package; ``anamnesis[vllm]`` installs the
        pinned set.
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
                           f"{wrong} (install anamnesis[vllm])")
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
        _refuse(lane, f"model {model!r} has no vLLM lane")
    dtype = settings.get("dtype")
    if dtype != LANE_MODELS[model]["dtype"]:
        _refuse(lane, f"dtype {dtype!r} for model {model} "
                      f"(declared: {LANE_MODELS[model]['dtype']})")
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
    for key, expected in REQUIRED_ENV.items():
        if environment.get(key) != expected:
            _refuse(lane, f"environment {key}={environment.get(key)!r} "
                          f"(declared: {expected!r})")
    undeclared = sorted(set(settings) - set(SETTINGS) - CONDITION_KEYS)
    if undeclared:
        _refuse(lane, f"undeclared engine settings {undeclared}")
    for key in sorted(set(SETTINGS) - CONDITION_KEYS):
        if key not in settings or settings[key] != SETTINGS[key]:
            _refuse(lane, f"{key}={settings.get(key)!r} (the lane fixes it to "
                          f"{SETTINGS[key]!r})")
    return dict(lane_id=lane, model=model, dtype=dtype,
                named_exclusions_checked=[
                    "prefix_caching", "plugins", "lora", "speculative_decoding",
                    "non_invariant_mode", "dtype", "chunked_prefill",
                    "concurrent_partial_prefill"],
                fixed_settings_verified=sorted(set(SETTINGS) - CONDITION_KEYS),
                environment_verified=sorted(REQUIRED_ENV))
