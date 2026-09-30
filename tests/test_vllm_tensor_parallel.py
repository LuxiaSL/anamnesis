"""Tensor-parallel lanes: head ownership, rank-order gathers, identity and envelope.

The engine is not needed: ranks are simulated by splitting full tensors the way
tensor parallelism partitions them, and gathering them back must reproduce the
single-GPU layouts exactly. The last tests pin every shipped lane's identity,
settings and fixture bytes, which a tensor-parallel lane must leave unchanged.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from anamnesis.extraction.vllm import envelope, tensor_parallel
from anamnesis.extraction.vllm.envelope import (
    CONDITIONS,
    LANE_MODELS,
    TP_REQUIRED_ENV,
    TP_SETTINGS,
    canonical_digest,
    engine_settings,
    enforce_lane_envelope,
    lane_id,
    lane_identity,
    required_environment,
)

WORLD = 4
HEADS, KV_HEADS = 32, 8


class FakeGroup:
    """A tensor-parallel group whose ranks' tensors are all in this process."""

    def __init__(self, parts):
        self.parts, self.world_size = parts, len(parts)

    def all_gather(self, tensor, dim):
        return torch.cat(self.parts, dim=dim)


def test_ranks_own_contiguous_head_slices_in_rank_order():
    entries = [tensor_parallel.head_ownership(r, WORLD, HEADS // WORLD, KV_HEADS // WORLD)
               for r in range(WORLD)]
    assert tensor_parallel.ownership_problems(entries) == []
    assert [e["query_heads"] for e in entries] == [[0, 8], [8, 16], [16, 24], [24, 32]]
    assert [e["kv_heads"] for e in entries] == [[0, 2], [2, 4], [4, 6], [6, 8]]
    assert tensor_parallel.ownership_problems(list(reversed(entries))) == []


def test_a_dropped_doubled_or_missing_rank_is_named():
    entries = [tensor_parallel.head_ownership(r, WORLD, 8, 2) for r in range(WORLD)]
    assert tensor_parallel.ownership_problems(entries[:3])
    doubled = entries[:2] + [dict(entries[1], rank=2)] + entries[3:]
    assert any("do not continue" in p for p in tensor_parallel.ownership_problems(doubled))
    with pytest.raises(ValueError):
        tensor_parallel.head_ownership(0, WORLD, 6, 4)


def test_per_head_statistics_gather_back_to_the_single_gpu_layout():
    full = torch.randn(5, HEADS, 7)
    parts = list(full.chunk(WORLD, dim=1))
    gathered = tensor_parallel.gather(parts[0], dim=1, group=FakeGroup(parts))
    assert torch.equal(gathered, full)


def test_packed_gate_slices_gather_back_to_the_full_gate():
    tokens, width = 6, 64
    gate, up = torch.randn(tokens, width), torch.randn(tokens, width)
    local = width // WORLD
    packed = [torch.cat([gate[:, r * local:(r + 1) * local], up[:, r * local:(r + 1) * local]],
                        dim=-1) for r in range(WORLD)]
    slices = [p[:, :local] for p in packed]
    assert torch.equal(tensor_parallel.gather(slices[0], dim=-1, group=FakeGroup(slices)), gate)


def test_qkv_shards_gather_back_to_full_width_queries_keys_and_values():
    tokens, d = 3, 4
    q = torch.randn(tokens, HEADS * d)
    k, v = torch.randn(tokens, KV_HEADS * d), torch.randn(tokens, KV_HEADS * d)
    per_rank = [(q.chunk(WORLD, -1)[r], k.chunk(WORLD, -1)[r], v.chunk(WORLD, -1)[r])
                for r in range(WORLD)]
    for i, full in enumerate((q, k, v)):
        parts = [p[i] for p in per_rank]
        assert torch.equal(tensor_parallel.gather(parts[0], dim=-1, group=FakeGroup(parts)),
                           full)


def test_replicated_kv_heads_are_refused():
    tensor_parallel.require_unreplicated(8, 16, 1, 128, 8)
    tensor_parallel.require_unreplicated(WORLD, HEADS // WORLD, KV_HEADS // WORLD, HEADS,
                                         KV_HEADS)
    # Sixteen ranks over eight KV heads: each rank holds a copy of one KV head, and
    # contiguous ownership records would still cover sixteen "heads".
    entries = [tensor_parallel.head_ownership(r, 16, 8, 1) for r in range(16)]
    assert tensor_parallel.ownership_problems(entries) == []
    with pytest.raises(ValueError, match="do not partition the model's 128 query and 8 KV"):
        tensor_parallel.require_unreplicated(16, 8, 1, 128, 8)
    with pytest.raises(ValueError, match="no more ranks than KV heads"):
        tensor_parallel.require_unreplicated(8, 16, 2, 128, 8)


def test_one_rank_is_returned_unchanged():
    tensor = torch.randn(2, 3)
    assert tensor_parallel.gather(tensor, dim=0, group=FakeGroup([tensor])) is tensor


def test_rank_agreement_names_the_tensors_that_differ():
    receipt = dict(sha256="a", tensors={"hidden": "h", "gates/0": "g"})
    other = dict(sha256="b", tensors={"hidden": "h", "gates/0": "x"})
    finished = [dict(rank=0, receipts={"r1": receipt}), dict(rank=1, receipts={"r1": other})]
    agreement = tensor_parallel.rank_agreement(finished, {"r1": 7})
    assert agreement == {"7": dict(joint_equal=False, differing_tensors=["gates/0"])}
    finished[1] = dict(rank=1, receipts={"r1": receipt})
    assert tensor_parallel.rank_agreement(finished, {"r1": 7})["7"]["joint_equal"]


@pytest.fixture()
def tp_model(monkeypatch):
    monkeypatch.setitem(LANE_MODELS, "tp-test", dict(dtype="bfloat16", logprob_wrapper="none",
                                                     tensor_parallel_size=2))
    return "tp-test"


def test_a_tensor_parallel_lane_is_its_own_lane(tp_model):
    identity = lane_identity(tp_model)
    assert identity["tensor_parallel_size"] == 2
    assert identity["collectives"] == dict(executor="mp", custom_all_reduce=False,
                                           worker_start="spawn")
    single = dict(identity, tensor_parallel_size=1)
    del single["collectives"]
    assert lane_id(tp_model) != canonical_digest(single)


def test_a_tensor_parallel_lane_declares_its_settings_and_environment(tp_model):
    settings = engine_settings(tp_model, "full-b1-order0")
    assert settings["tensor_parallel_size"] == 2
    assert {k: settings[k] for k in TP_SETTINGS} == TP_SETTINGS
    env = required_environment(tp_model)
    assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    record = enforce_lane_envelope(settings, env, lane=lane_id(tp_model), model=tp_model)
    assert record["tensor_parallel_size"] == 2


@pytest.mark.parametrize("change", [
    ("settings", "disable_custom_all_reduce", False),
    ("settings", "tensor_parallel_size", 4),
    ("env", "VLLM_WORKER_MULTIPROC_METHOD", "fork"),
])
def test_a_tensor_parallel_departure_is_refused(tp_model, change):
    settings = engine_settings(tp_model, "full-b1-order0")
    env = required_environment(tp_model)
    where, key, value = change
    (settings if where == "settings" else env)[key] = value
    with pytest.raises(ValueError, match="refuses"):
        enforce_lane_envelope(settings, env, lane=lane_id(tp_model), model=tp_model)


def test_the_declared_worker_extension_is_this_package_s():
    """The engine loads the extension by its path, so the path must name a module here;
    importing it registers the backend, which needs the engine."""
    import importlib.util

    module, _, name = TP_SETTINGS["worker_extension_cls"].rpartition(".")
    assert importlib.util.find_spec(module) is not None
    pytest.importorskip("vllm")
    from anamnesis.extraction.vllm import tp_worker

    assert tp_worker.__name__ == module and hasattr(tp_worker, name)


def test_a_single_gpu_lane_refuses_tensor_parallel_settings():
    settings = dict(engine_settings("8b", "full-b1-order0"), **TP_SETTINGS)
    env = dict(required_environment("8b"), **TP_REQUIRED_ENV)
    with pytest.raises(ValueError, match="undeclared engine settings"):
        enforce_lane_envelope(settings, env, lane=lane_id("8b"), model="8b")


# The shipped lanes, pinned: their identities, settings and fixture bytes.
SHIPPED_LANE_IDS = {
    "3b": "8fb19f3039e7e3ef657d4fdb60510917f7132c8f0b5cc8bd98a412f264a6ca79",
    "8b": "0a06952777a72a871951155b96800dac32db974ac1f2b3b304764bd64fe5ad9e",
    "70b": "c34f17d8d1a4e42731d613a6bc6378828a8b19b4258c476753be3376451067d2",
}
SHIPPED_SETTINGS = {
    ("3b", "full-b1-order0"): "3443362e851cbdabef4f582d734ea3f8de55eb8067903bfbdaf7b68c52db0875",
    ("3b", "full-b8-order0"): "c93b69e647b5915d2194abcff0b0c3e2f508782da5f46b2382bb530a15be6ef2",
    ("8b", "full-b1-order0"): "aa66c79f9067d9e886b2f9fd07f3154bd6065962033b23a8c4cdced951269262",
    ("8b", "full-b8-order0"): "fa9df61f51c87a0fd814be875cada4f0c2c13ed9368a632938b7188f14dc09b0",
    ("70b", "full-b1-order0"): "aa66c79f9067d9e886b2f9fd07f3154bd6065962033b23a8c4cdced951269262",
    ("70b", "full-b8-order0"): "fa9df61f51c87a0fd814be875cada4f0c2c13ed9368a632938b7188f14dc09b0",
}
SHIPPED_FIXTURE_FILES = {
    "3b/fixtures.json": "50fad8b95485d58dde36e7909fee9a963860f1f44434c5705326e4d6e3157d1f",
    "3b/tolerance.json": "7cc7722cf18bc63a2e3c4546929033bcdd2529c9609c65f992ec31dd8b133d98",
    "3b/transfer_rules.json": "856772497637c72348abfd6ab8e6c71290809b6043c7bf3066b42c408c9b576a",
    "3b/vectors.npz": "98afbd197b973b21a0b6451fa30346b4bcb60f7518d0ca6d9f308d4d3d9c13da",
    "8b/fixtures.json": "1287a28dd792584df04c920e3ac716138df7ffd2a366f9e5b256b59810e8c96a",
    "8b/tolerance.json": "02a5cdb4c3bc6e7b6b8d340cf8569d09bb9d5d2385cb56bb08d30e68c39afbf8",
    "8b/transfer_rules.json": "443fcd9df5cdb6a31454a93e86a401c9a954618b6f40905cc8d4154f75cc0ac9",
    "8b/vectors.npz": "5654cf2bdb882719bcbf6e6619da25b7b78d46f252e1c13309d75d5093b5a162",
    "70b/fixtures.json": "2854b5575a39294afd2d6452622a09db311e561431af6039760df8539b924e99",
    "70b/tolerance.json": "b0a2f1383621e4cbc60c2da3ee9ec2591cbe679f5805945fb946d68257fa9ecf",
    "70b/transfer_rules.json": "ba7d2e6f11d66e18de50dde6ae3b8681347bbf95d14aae56d44200b4e623247b",
    "70b/vectors.npz": "d9b9cbd0c21d33e0198e3c8f7012cb3582eaff2b057383d8eed302de8c9ab797",
}


def test_every_shipped_lane_keeps_its_id_and_settings():
    assert {m: lane_id(m) for m in LANE_MODELS} == SHIPPED_LANE_IDS
    assert {(m, c): canonical_digest(engine_settings(m, c)) for m in LANE_MODELS
            for c in CONDITIONS} == SHIPPED_SETTINGS
    assert all(required_environment(m) == envelope.REQUIRED_ENV for m in LANE_MODELS)


def test_every_shipped_fixture_file_keeps_its_bytes():
    """A subset pin: each pinned file exists with its bytes; new files may join."""
    root = Path(envelope.__file__).parent / "fixtures"
    found = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
             if (root / name).is_file() else None for name in SHIPPED_FIXTURE_FILES}
    assert found == SHIPPED_FIXTURE_FILES
