"""The vLLM lane's declared envelope, and the startup guard that refuses everything else.

A vLLM lane is only the lane it claims to be inside the configuration it declares in
:mod:`anamnesis.extraction.vllm.envelope`. These cases pin that declaration: the lane
ids each model's fixtures carry, the exact engine settings each execution condition
produces, how rows are split into engine batches, and the guard every engine
construction passes first. Each guard case perturbs exactly one axis of an otherwise
declared configuration and requires a refusal that names the lane, so a failure
record says whose scope refused.

Nothing here builds an engine: the environment and the installed package versions
are monkeypatched. Whether a host's engine computes the same arithmetic inside this
envelope is the install check's question, and needs a device and the engine.
"""

from __future__ import annotations

import importlib.metadata

import pytest

from anamnesis.extraction.vllm import envelope
from anamnesis.extraction.vllm.envelope import (
    CONDITION_KEYS,
    CONDITIONS,
    LANE_MODELS,
    PINNED_PACKAGES,
    REQUIRED_ENV,
    SETTINGS,
    canonical_digest,
    engine_settings,
    enforce_lane_envelope,
    lane_id,
    lane_identity,
    lane_model,
    request_groups,
    require_environment,
    require_pinned_packages,
)

LANE = "lane-abc123"

FIXED = dict(
    async_scheduling=False,
    attention_backend="TRITON_ATTN",
    compilation_config=0,
    enable_chunked_prefill=False,
    enable_lora=False,
    enable_prefix_caching=False,
    enforce_eager=True,
    kv_cache_memory_bytes=2147483648,
    long_prefill_token_threshold=0,
    max_long_partial_prefills=1,
    max_model_len=1024,
    max_num_partial_prefills=1,
    seed=20260922,
    speculative_config=None,
    tensor_parallel_size=1,
)
"""The settings every model and condition shares."""


def expected_settings(dtype: str, capacity: int) -> dict:
    return dict(FIXED, dtype=dtype, max_num_seqs=capacity,
                max_num_batched_tokens=capacity * 1024)


def declared(model: str = "3b", condition_id: str = "full-b1-order0"):
    return engine_settings(model, condition_id), dict(REQUIRED_ENV)


# --- identity -------------------------------------------------------------------


def test_lane_ids_are_the_ids_the_shipped_fixtures_carry() -> None:
    """The lane id is a digest of the per-model facts. The shipped conformance
    fixtures (:class:`anamnesis.extraction.vllm.conformance.FixtureSet`) carry
    exactly these ids, so a change to any fact a lane id digests fails here
    rather than stranding every fixture set."""
    assert lane_id("3b") == "8fb19f3039e7e3ef657d4fdb60510917f7132c8f0b5cc8bd98a412f264a6ca79"
    assert lane_id("8b") == "0a06952777a72a871951155b96800dac32db974ac1f2b3b304764bd64fe5ad9e"
    assert lane_id("70b") == "c34f17d8d1a4e42731d613a6bc6378828a8b19b4258c476753be3376451067d2"


def test_lane_identity_digests_the_per_model_facts() -> None:
    for model in LANE_MODELS:
        identity = lane_identity(model)
        assert identity == dict(model=model, dtype=LANE_MODELS[model]["dtype"],
                                invariant_mode=True, attention_backend="TRITON_ATTN",
                                tensor_parallel_size=1,
                                logprob_wrapper=LANE_MODELS[model]["logprob_wrapper"])
        assert lane_id(model) == canonical_digest(identity)
    assert len({lane_id(model) for model in LANE_MODELS}) == len(LANE_MODELS)


def test_canonical_digest_is_key_order_free_and_refuses_nan() -> None:
    assert canonical_digest({"a": 1, "b": 2}) == canonical_digest({"b": 2, "a": 1})
    assert canonical_digest({"a": 1}) != canonical_digest({"a": 2})
    with pytest.raises(ValueError):
        canonical_digest({"a": float("nan")})


def test_unknown_model_has_no_lane() -> None:
    assert lane_model("8b") == LANE_MODELS["8b"]
    for call in (lane_model, lane_identity, lane_id):
        with pytest.raises(ValueError, match="has no vLLM lane"):
            call("405b")


# --- engine settings --------------------------------------------------------------


def test_exactly_two_conditions_are_declared() -> None:
    assert {cid: c["max_num_seqs"] for cid, c in CONDITIONS.items()} == {
        "full-b1-order0": 1, "full-b8-order0": 8}
    assert all(c["condition_id"] == cid for cid, c in CONDITIONS.items())


@pytest.mark.parametrize("model,dtype", [
    ("3b", "float16"), ("8b", "bfloat16"), ("70b", "bfloat16")])
@pytest.mark.parametrize("condition_id,capacity", [
    ("full-b1-order0", 1), ("full-b8-order0", 8)])
def test_engine_settings_are_exactly_the_declared_dicts(model, dtype, condition_id,
                                                        capacity) -> None:
    """Whole-prompt prefill: the token budget holds ``capacity`` full contexts, so
    every prompt fits one prefill step and no partial prefill is ever scheduled."""
    assert engine_settings(model, condition_id) == expected_settings(dtype, capacity)


def test_engine_settings_refuse_undeclared_models_and_conditions() -> None:
    with pytest.raises(ValueError, match="has no vLLM lane"):
        engine_settings("405b", "full-b1-order0")
    for condition_id in ("chunk128-b1-order0", "full-b8-order1", "full-b2-order0"):
        with pytest.raises(ValueError, match="not a declared condition"):
            engine_settings("3b", condition_id)


def test_condition_keys_stay_inside_the_settings_vocabulary() -> None:
    settings, _ = declared()
    assert CONDITION_KEYS <= set(settings)
    assert set(SETTINGS) <= set(settings)


# --- request groups ---------------------------------------------------------------


@pytest.mark.parametrize("capacity", [1, 2, 4, 8])
def test_every_target_once_in_full_batches_with_explicit_fillers(capacity) -> None:
    ids = list(range(160)) + list(range(10000, 10030))
    groups = request_groups(ids, capacity)
    targets = [r["generation_id"] for g in groups for r in g if r["retain"]]
    assert sorted(targets) == ids
    assert all(len(g) == capacity for g in groups)
    assert len({r["occurrence_id"] for g in groups for r in g}) == len(groups) * capacity
    for group in groups:
        members = [r["generation_id"] for r in group]
        assert len(set(members)) == capacity


def test_short_final_batch_is_filled_with_other_rows_never_retained() -> None:
    groups = request_groups([5, 1, 3], 2)
    assert [[(r["generation_id"], r["retain"]) for r in g] for g in groups] == [
        [(1, True), (3, True)], [(5, True), (1, False)]]
    assert [r["occurrence_id"] for r in groups[1]] == [
        "group-0001-slot-0", "group-0001-slot-1"]


@pytest.mark.parametrize("ids,capacity", [
    ([1, 1], 2), ([], 1), ([1, 2, 3], 3), ([1, 2, 3], 16), ([1], 2), ([1, 2.0], 1),
    ([True, 2], 1)])
def test_invalid_rosters_and_capacities_refused(ids, capacity) -> None:
    with pytest.raises(ValueError, match="unique integer ids"):
        request_groups(ids, capacity)


# --- environment and package pins -------------------------------------------------


def test_required_environment_passes_exactly_and_names_every_difference() -> None:
    require_environment(dict(REQUIRED_ENV))
    env = dict(REQUIRED_ENV, VLLM_BATCH_INVARIANT="0")
    del env["VLLM_PLUGINS"]
    with pytest.raises(RuntimeError, match="differing") as caught:
        require_environment(env)
    assert "VLLM_BATCH_INVARIANT" in str(caught.value)
    assert "VLLM_PLUGINS" in str(caught.value)
    assert "OMP_NUM_THREADS" not in str(caught.value)


def test_required_environment_reads_the_process_environment(monkeypatch) -> None:
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    require_environment()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        require_environment()


def _installed(versions):
    def version(name):
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]
    return version


def test_pinned_packages_pass_exactly_and_allow_a_local_build_tag(monkeypatch) -> None:
    assert set(PINNED_PACKAGES) == {"vllm", "torch", "triton"}
    versions = dict(PINNED_PACKAGES, torch=PINNED_PACKAGES["torch"] + "+cu128")
    monkeypatch.setattr(envelope.importlib.metadata, "version", _installed(versions))
    assert require_pinned_packages() == versions


def test_pinned_packages_refuse_a_differing_or_missing_package(monkeypatch) -> None:
    versions = dict(PINNED_PACKAGES, vllm="0.17.0")
    del versions["triton"]
    monkeypatch.setattr(envelope.importlib.metadata, "version", _installed(versions))
    with pytest.raises(RuntimeError, match="vllm extra") as caught:
        require_pinned_packages()
    message = str(caught.value)
    assert "'vllm': '0.17.0'" in message
    assert "'triton': 'missing'" in message
    assert "'torch'" not in message.split("this installation has")[1]


# --- the startup guard ------------------------------------------------------------


@pytest.mark.parametrize("model", sorted(LANE_MODELS))
@pytest.mark.parametrize("condition_id", sorted(CONDITIONS))
def test_declared_configurations_pass_and_the_record_names_the_lane(model,
                                                                    condition_id) -> None:
    settings, env = declared(model, condition_id)
    record = enforce_lane_envelope(settings, env, lane=LANE, model=model)
    assert record["lane_id"] == LANE
    assert record["model"] == model
    assert record["dtype"] == LANE_MODELS[model]["dtype"]
    assert "concurrent_partial_prefill" in record["named_exclusions_checked"]
    assert "chunked_prefill" in record["named_exclusions_checked"]
    assert record["fixed_settings_verified"] == sorted(set(SETTINGS) - CONDITION_KEYS)
    assert record["environment_verified"] == sorted(REQUIRED_ENV)


@pytest.mark.parametrize("key,value,fragment", [
    ("enable_prefix_caching", True, "prefix caching"),
    ("enable_lora", True, "LoRA"),
    ("speculative_config", {"method": "eagle"}, "speculative"),
    ("dtype", "bfloat16", "dtype"),
    ("max_num_seqs", 16, "capacity"),
    ("max_num_seqs", 2, "capacity"),
    ("enable_chunked_prefill", True, "chunked prefill"),
    ("max_num_partial_prefills", 2, "partial prefill"),
    ("tensor_parallel_size", 2, "tensor_parallel_size"),
    ("enforce_eager", False, "enforce_eager"),
    ("attention_backend", "FLASH_ATTN", "attention_backend"),
    ("max_model_len", 4096, "max_model_len"),
])
def test_each_named_setting_axis_refused(key, value, fragment) -> None:
    settings, env = declared()
    settings[key] = value
    with pytest.raises(ValueError, match=LANE) as caught:
        enforce_lane_envelope(settings, env, lane=LANE, model="3b")
    assert fragment in str(caught.value)


def test_chunked_prefill_refused_at_every_declared_capacity() -> None:
    for condition_id in CONDITIONS:
        settings, env = declared("8b", condition_id)
        settings.update(enable_chunked_prefill=True, max_num_batched_tokens=128,
                        long_prefill_token_threshold=128)
        with pytest.raises(ValueError, match=LANE) as caught:
            enforce_lane_envelope(settings, env, lane=LANE, model="8b")
        assert "chunked prefill" in str(caught.value)


@pytest.mark.parametrize("key,value,fragment", [
    ("VLLM_BATCH_INVARIANT", "0", "non-invariant"),
    ("VLLM_PLUGINS", "steering", "plugins"),
    ("CUBLAS_WORKSPACE_CONFIG", ":16:8", "CUBLAS_WORKSPACE_CONFIG"),
    ("VLLM_ENABLE_V1_MULTIPROCESSING", "1", "VLLM_ENABLE_V1_MULTIPROCESSING"),
])
def test_each_named_environment_axis_refused(key, value, fragment) -> None:
    settings, env = declared()
    env[key] = value
    with pytest.raises(ValueError, match=LANE) as caught:
        enforce_lane_envelope(settings, env, lane=LANE, model="3b")
    assert fragment in str(caught.value)


def test_unset_plugins_and_unset_invariance_both_refused() -> None:
    for key in ("VLLM_PLUGINS", "VLLM_BATCH_INVARIANT"):
        settings, env = declared()
        del env[key]
        with pytest.raises(ValueError, match=LANE):
            enforce_lane_envelope(settings, env, lane=LANE, model="3b")


def test_missing_fixed_setting_and_undeclared_setting_refused() -> None:
    settings, env = declared()
    del settings["enable_prefix_caching"]
    with pytest.raises(ValueError, match="prefix caching"):
        enforce_lane_envelope(settings, env, lane=LANE, model="3b")
    settings, env = declared()
    del settings["seed"]
    with pytest.raises(ValueError, match="seed"):
        enforce_lane_envelope(settings, env, lane=LANE, model="3b")
    settings, env = declared()
    settings["kv_transfer_config"] = object()
    with pytest.raises(ValueError, match="undeclared"):
        enforce_lane_envelope(settings, env, lane=LANE, model="3b")


def test_lane_identity_and_model_are_required() -> None:
    settings, env = declared()
    with pytest.raises(ValueError, match="lane id"):
        enforce_lane_envelope(settings, env, lane="", model="3b")
    with pytest.raises(ValueError, match="has no vLLM lane"):
        enforce_lane_envelope(settings, env, lane=LANE, model="405b")
