"""A lane binds its weights; a contrast across lanes is declared, never accidental.

Three things are asserted here.

1. **The digest.** :func:`weights_digest` is pinned to a constant on a small module
   holding a float32 linear layer, a bfloat16 parameter, a persistent scalar buffer, a
   non-persistent buffer (not read), and a tied pair (read under each key). The same
   constant is computed by hand from packed bytes, so a copy of the definition in
   another codebase can assert against it. The digest is of the loaded tensors: the
   same weights read back from safetensors, a ``.pt`` file, or another device give it.
2. **The binding.** A registry row may declare ``checkpoint_sha256`` (and an adapter's
   ``adapter_sha256``, a ``parent`` link, a ``lineage`` kind). The row's fields are
   checked against one another and the links against the merged registry. Loading
   or receiving a model for a bound row refuses other weights by name, in the loader
   and in the fast lane's preparation. A row that binds nothing computes no digest
   and leaves every configuration dump and lane id as it was.
3. **The guard.** One lane id over two weights digests is refused as a mixed lane.
   :func:`require_declared_contrast` admits rows across lanes under a declared
   :class:`LineageContrast` (different weights traced through ``parent`` links, with
   a calibration policy) or :class:`BridgeContrast` (one set of weights under two
   computors, with a measured floor), and refuses each mode's rows under the other.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from test_fast_lane_equivalence import tiny_loaded
from test_loaded_model_seam import TINY_ROW, calib_dir, lane_arithmetic, tiny_preset  # noqa: F401
from test_onboarding import save_tiny_tokenizer

from anamnesis.analysis.lane_guard import (
    BridgeContrast,
    ContrastRecord,
    LineageContrast,
    MixedLaneError,
    gate_banked_signatures,
    gate_declared_contrast,
    require_declared_contrast,
    require_single_lane,
)
from anamnesis.config import LINEAGE_KINDS, ModelConfig, ModelPreset, resolve_preset
from anamnesis.config.models import (
    MODELS_ENV,
    ModelRegistryError,
    load_registry,
)
from anamnesis.extraction.fast.runtime import prepare_fast_lane
from anamnesis.extraction.model_loader import load_model
from anamnesis.extraction.replay_config import native_replay_configs
from anamnesis.extraction.weights_binding import (
    WeightsMismatchError,
    require_weights,
    weights_record,
)
from anamnesis.extraction.weights_digest import WEIGHTS_DIGEST_VERSION, weights_digest

# --------------------------------------------------------------------------- digest

PINNED_DIGEST = "1176665bb75973e4742f40273ca02227d0587f84c951bee891e22b231d2e31e0"
"""weights-v1 of :class:`Pinned`. A copy of the definition elsewhere asserts this."""


class Pinned(torch.nn.Module):
    """Every case the definition names, with values exact in their dtypes."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.bf = torch.nn.Parameter(torch.tensor([1.0, -1.5], dtype=torch.bfloat16))
        self.register_buffer("steps", torch.tensor(7, dtype=torch.int64))
        self.register_buffer("scratch", torch.ones(3), persistent=False)
        self.emb = torch.nn.Embedding(2, 1)
        self.head = torch.nn.Linear(1, 2, bias=False)
        self.head.weight = self.emb.weight
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.0, -2.0]]))
            self.linear.bias.copy_(torch.tensor([0.5]))
            self.emb.weight.copy_(torch.tensor([[0.25], [2.0]]))


def hand_digest() -> str:
    """The definition applied by hand, from packed little-endian bytes."""

    def f32(*values: float) -> bytes:
        return b"".join(struct.pack("<f", v) for v in values)

    def bf16(*values: float) -> bytes:
        # bfloat16 is the upper half of float32; these values are exact in both.
        return b"".join(struct.pack("<f", v)[2:] for v in values)

    entries = sorted([
        ("linear.weight", "torch.float32", "1,2", f32(1.0, -2.0)),
        ("linear.bias", "torch.float32", "1", f32(0.5)),
        ("bf", "torch.bfloat16", "2", bf16(1.0, -1.5)),
        ("steps", "torch.int64", "", struct.pack("<q", 7)),
        ("emb.weight", "torch.float32", "2,1", f32(0.25, 2.0)),
        ("head.weight", "torch.float32", "2,1", f32(0.25, 2.0)),
    ])
    digest = hashlib.sha256()
    for key, dtype, shape, raw in entries:
        digest.update(key.encode() + b"\0" + dtype.encode() + b"\0" + shape.encode() + b"\0" + raw)
    return digest.hexdigest()


def test_the_digest_is_the_pinned_constant_and_the_hand_computation() -> None:
    module = Pinned()
    assert set(module.state_dict()) == {
        "linear.weight", "linear.bias", "bf", "steps", "emb.weight", "head.weight"}
    assert weights_digest(module) == hand_digest() == PINNED_DIGEST
    assert WEIGHTS_DIGEST_VERSION == "weights-v1"


def test_the_digest_reads_values_dtypes_shapes_and_names() -> None:
    base = weights_digest(Pinned())
    changed = Pinned()
    with torch.no_grad():
        changed.linear.bias.add_(1e-6)
    cast = Pinned().to(torch.float64)
    renamed = Pinned()
    renamed.register_buffer("steps_", renamed.steps)
    del renamed._buffers["steps"]
    reshaped = Pinned()
    reshaped.linear.weight = torch.nn.Parameter(reshaped.linear.weight.detach().reshape(2, 1))
    digests = {base, weights_digest(changed), weights_digest(cast),
               weights_digest(renamed), weights_digest(reshaped)}
    assert len(digests) == 5


def test_the_digest_is_of_the_loaded_tensors_not_the_file(tmp_path: Path) -> None:
    from safetensors.torch import load_file, save_file

    model = tiny_loaded().model
    expected = weights_digest(model)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, tmp_path / "weights.safetensors")
    torch.save(state, tmp_path / "weights.pt")
    for loaded_state in (load_file(tmp_path / "weights.safetensors"),
                         torch.load(tmp_path / "weights.pt")):
        fresh = type(model)(model.config).eval()
        fresh.load_state_dict(loaded_state)
        assert weights_digest(fresh) == expected
    # A strided tensor hashes its values in logical order, as its contiguous copy does.
    values = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    strided, packed = torch.nn.Module(), torch.nn.Module()
    strided.w = torch.nn.Parameter(values.t().contiguous().t())
    packed.w = torch.nn.Parameter(values.clone())
    assert not strided.w.is_contiguous()
    assert weights_digest(strided) == weights_digest(packed)


def test_unloaded_and_opaque_entries_are_refused() -> None:
    with torch.device("meta"):
        meta = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="meta device"):
        weights_digest(meta)

    class Opaque(torch.nn.Module):
        def get_extra_state(self) -> Any:
            return {"not": "a tensor"}

        def set_extra_state(self, state: Any) -> None:
            pass

    with pytest.raises(TypeError, match="not a tensor"):
        weights_digest(Opaque())


# --------------------------------------------------------------------------- registry

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def write_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                   presets: dict[str, dict[str, Any]], name: str = "models.json") -> None:
    path = tmp_path / name
    path.write_text(json.dumps({"presets": presets}))
    monkeypatch.setenv(MODELS_ENV, str(path))


def lineage_rows() -> dict[str, dict[str, Any]]:
    """A bound base, an SFT round on it, a DPO round on that, and a sibling LoRA."""
    return {
        "tiny-base": {**TINY_ROW, "name": "tiny-base", "checkpoint_sha256": SHA_A,
                      "calibration_dir": "calibration/tiny-base"},
        "tiny-sft": {"extends": "tiny-base", "checkpoint_sha256": SHA_B, "parent": "tiny-base",
                     "parent_checkpoint_sha256": SHA_A, "lineage": "sft",
                     "calibration_dir": "calibration/tiny-sft"},
        "tiny-dpo": {"extends": "tiny-sft", "checkpoint_sha256": SHA_C, "parent": "tiny-sft",
                     "parent_checkpoint_sha256": SHA_B, "lineage": "dpo",
                     "calibration_dir": "calibration/tiny-dpo"},
        "tiny-lora": {"extends": "tiny-base", "checkpoint_sha256": "d" * 64,
                      "adapter_sha256": "e" * 64, "parent": "tiny-base",
                      "parent_checkpoint_sha256": SHA_A, "lineage": "lora",
                      "calibration_dir": "calibration/tiny-lora"},
        "tiny-other": {**TINY_ROW, "name": "tiny-other", "checkpoint_sha256": "f" * 64,
                       "calibration_dir": "calibration/tiny-other"},
    }


def test_the_lineage_kinds_are_one_literal() -> None:
    assert LINEAGE_KINDS == ("sft", "dpo", "lora", "pretrain-step", "family")
    for kind in LINEAGE_KINDS:
        if kind != "lora":
            tiny_preset(checkpoint_sha256=SHA_B, parent="x", parent_checkpoint_sha256=SHA_A,
                        lineage=kind)
    with pytest.raises(ValueError):
        tiny_preset(checkpoint_sha256=SHA_B, parent="x", parent_checkpoint_sha256=SHA_A,
                    lineage="rlhf")


@pytest.mark.parametrize("fields, message", [
    (dict(checkpoint_sha256="A" * 64), "pattern"),
    (dict(adapter_sha256=SHA_A), "without checkpoint_sha256"),
    (dict(checkpoint_sha256=SHA_B, parent="x"), "given without"),
    (dict(parent="x", parent_checkpoint_sha256=SHA_A, lineage="sft"), "binds its own weights"),
    (dict(checkpoint_sha256=SHA_A, parent="x", parent_checkpoint_sha256=SHA_A, lineage="sft"),
     "equals its parent"),
    (dict(checkpoint_sha256=SHA_B, parent="x", parent_checkpoint_sha256=SHA_A, lineage="lora"),
     "adapter_sha256"),
    (dict(checkpoint_sha256=SHA_B, parent="tiny-llama", parent_checkpoint_sha256=SHA_A,
          lineage="sft"), "not its own parent"),
])
def test_a_row_s_weights_fields_agree_with_one_another(fields: dict[str, Any],
                                                        message: str) -> None:
    with pytest.raises(ValueError, match=message):
        tiny_preset(**fields)


def test_a_lineage_resolves_and_traces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_registry(tmp_path, monkeypatch, lineage_rows())
    registry = load_registry()
    assert registry.ancestry("tiny-dpo") == ("tiny-dpo", "tiny-sft", "tiny-base")
    assert registry.ancestry("tiny-other") == ("tiny-other",)
    dpo = resolve_preset("tiny-dpo")
    assert (dpo.checkpoint_sha256, dpo.lineage, dpo.parent) == (SHA_C, "dpo", "tiny-sft")


@pytest.mark.parametrize("change, message", [
    ({"tiny-sft": {"parent": "nowhere"}}, "which no registry file defines"),
    ({"tiny-sft": {"parent_checkpoint_sha256": "9" * 64}}, "the link names the weights"),
    ({"tiny-sft": {"calibration_dir": "calibration/tiny-base"}}, "own calibration"),
    ({"tiny-base": {"parent": "tiny-dpo", "parent_checkpoint_sha256": SHA_C, "lineage": "family"}},
     "cycle"),
])
def test_a_link_the_registry_cannot_trace_is_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, dict[str, Any]],
        message: str) -> None:
    rows = lineage_rows()
    for key, fields in change.items():
        rows[key] = {**rows[key], **fields}
    write_registry(tmp_path, monkeypatch, rows)
    with pytest.raises(ModelRegistryError, match=message):
        load_registry()


def test_an_unbound_row_is_what_it_was(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No shipped row binds weights, no dump that feeds an identity gains a key, and
    binding a row moves no lane id: the binding is a check at construction."""
    for name in load_registry().names():
        row = resolve_preset(name)
        assert row.checkpoint_sha256 is row.parent is row.lineage is None
        assert weights_record(row) is None
    plain = tiny_preset()
    bound = tiny_preset(checkpoint_sha256=SHA_A)
    for preset in (plain, bound):
        dumped = ModelConfig.from_preset(preset).model_dump()
        assert "checkpoint_sha256" not in dumped and "_checkpoint_sha256" not in dumped
    assert ModelConfig.from_preset(plain).model_dump() == ModelConfig.from_preset(bound).model_dump()
    assert ModelConfig.from_preset(bound).checkpoint_sha256 == SHA_A
    assert ModelConfig.from_preset(plain).checkpoint_sha256 is None
    for a, b in zip(native_replay_configs(plain), native_replay_configs(bound)):
        assert a.model_dump(mode="json") == b.model_dump(mode="json")


# --------------------------------------------------------------------------- binding

def test_require_weights_refuses_other_weights_by_name() -> None:
    model = tiny_loaded().model
    digest = weights_digest(model)
    bound = tiny_preset(checkpoint_sha256=digest)
    assert require_weights(model, bound) == dict(
        digest="weights-v1", preset="tiny-llama", checkpoint_sha256=digest)
    assert require_weights(tiny_loaded(), bound) is not None  # a LoadedModel is unwrapped
    with torch.no_grad():
        model.lm_head.weight[0, 0] += 1.0
    with pytest.raises(WeightsMismatchError, match="preset 'tiny-llama' binds weights") as error:
        require_weights(model, bound)
    assert digest in str(error.value) and weights_digest(model) in str(error.value)
    assert "own preset key and its own calibration" in str(error.value)
    with pytest.raises(WeightsMismatchError):
        require_weights(model, bound, digest="0" * 64)


def test_an_unbound_row_computes_no_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    from anamnesis.extraction import weights_binding

    def refuse(_model: Any) -> str:
        raise AssertionError("an unbound row computed a digest")

    monkeypatch.setattr(weights_binding, "weights_digest", refuse)
    assert require_weights(tiny_loaded().model, tiny_preset()) is None


def test_the_loader_refuses_other_weights(tmp_path: Path) -> None:
    model = tiny_loaded().model
    model.save_pretrained(tmp_path)
    save_tiny_tokenizer(tmp_path, model.config.vocab_size)
    digest = weights_digest(model)

    def load(preset: ModelPreset) -> Any:
        config = ModelConfig.from_preset(preset, model_id=str(tmp_path), device_map="cpu")
        return load_model(config, sampled_layers=[0, 1, 2])

    loaded = load(tiny_preset(checkpoint_sha256=digest))
    assert weights_digest(loaded.model) == digest
    with pytest.raises(WeightsMismatchError, match="tiny-llama"):
        load(tiny_preset(checkpoint_sha256=SHA_A))


def test_the_fast_lane_refuses_a_handed_in_model_with_other_weights(
        lane_arithmetic: None, calib_dir: Path) -> None:  # noqa: F811
    loaded = tiny_loaded()
    digest = weights_digest(loaded.model)
    prepared = prepare_fast_lane(tiny_preset(checkpoint_sha256=digest), calib_dir,
                                 device="cpu", loaded=loaded)
    assert prepared.weights == weights_record(tiny_preset(checkpoint_sha256=digest))
    assert prepared.provenance()["weights"]["checkpoint_sha256"] == digest
    with pytest.raises(WeightsMismatchError):
        prepare_fast_lane(tiny_preset(checkpoint_sha256=SHA_A), calib_dir, device="cpu",
                          loaded=loaded)
    plain = prepare_fast_lane(tiny_preset(), calib_dir, device="cpu", loaded=loaded)
    assert plain.weights is None and "weights" not in plain.provenance()
    bound_lane = prepared.lane(20)[0]
    assert bound_lane.lane_id == plain.lane(20)[0].lane_id


def test_a_bound_harvest_stamps_the_record_the_guard_reads(
        lane_arithmetic: None, calib_dir: Path) -> None:  # noqa: F811
    from anamnesis.extraction.fast.harvest import harvest_loaded

    loaded = tiny_loaded()
    digest = weights_digest(loaded.model)
    prepared = prepare_fast_lane(tiny_preset(checkpoint_sha256=digest), calib_dir,
                                 device="cpu", loaded=loaded)
    ids = np.random.default_rng(3).integers(1, 64, size=27).tolist()
    bound = harvest_loaded(prepared, ids, prompt_len=7)
    assert bound.receipt["weights"]["checkpoint_sha256"] == digest
    plain = harvest_loaded(prepare_fast_lane(tiny_preset(), calib_dir, device="cpu",
                                             loaded=loaded), ids, prompt_len=7)
    assert "weights" not in plain.receipt
    np.testing.assert_array_equal(bound.features, plain.features)
    row = {"lane_id": bound.receipt["lane_id"], "extraction_lane": bound.receipt}
    assert require_single_lane([row, row]) == bound.receipt["lane_id"]


# --------------------------------------------------------------------------- guard

def row(lane: str, preset: str, checkpoint: str, *, calibration: str = "1" * 64,
        schema: str = "5" * 64, **extra: Any) -> dict[str, Any]:
    weights = dict(digest="weights-v1", preset=preset, checkpoint_sha256=checkpoint)
    return {"lane_id": lane, "extraction_lane": dict(
        lane_id=lane, calibration_sha256=calibration, feature_schema_sha256=schema,
        weights=weights, **extra)}


@pytest.fixture
def lineage_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    write_registry(tmp_path, monkeypatch, lineage_rows())
    return load_registry()


OWN = LineageContrast(calibration="own")


def test_an_ordinary_contrast_refuses_one_lane_over_two_weights() -> None:
    same = [row("L", "tiny-base", SHA_A), row("L", "tiny-base", SHA_A)]
    assert require_single_lane(same) == "L"
    with pytest.raises(MixedLaneError, match="2 weights digests"):
        require_single_lane([row("L", "tiny-base", SHA_A), row("L", "tiny-sft", SHA_B)])
    # A row from before the record existed is unbound, not a second set of weights.
    unbound = row("L", "tiny-base", SHA_A)
    del unbound["extraction_lane"]["weights"]
    assert require_single_lane([row("L", "tiny-base", SHA_A), unbound]) == "L"
    with pytest.raises(MixedLaneError, match="mixed arithmetic lanes"):
        require_single_lane([row("L1", "tiny-base", SHA_A), row("L2", "tiny-sft", SHA_B)])


def test_a_lineage_contrast_under_own_calibrations_is_admitted_and_recorded(
        lineage_registry: Any) -> None:
    rows = [row("L1", "tiny-base", SHA_A, calibration="1" * 64),
            row("L1", "tiny-base", SHA_A, calibration="1" * 64),
            row("L2", "tiny-dpo", SHA_C, calibration="2" * 64),
            row("L3", "tiny-lora", "d" * 64, calibration="3" * 64)]
    record = require_declared_contrast(rows, OWN)
    assert record.kind == "lineage" and record.root == "tiny-base"
    assert record.calibration == "own"
    assert [(m.lane_id, m.preset, m.rows) for m in record.members] == [
        ("L1", "tiny-base", 2), ("L2", "tiny-dpo", 1), ("L3", "tiny-lora", 1)]
    assert record.ancestry["tiny-dpo"] == (
        ("tiny-dpo", "dpo"), ("tiny-sft", "sft"), ("tiny-base", "root"))
    json.dumps(record.model_dump(mode="json"))
    with pytest.raises(ValueError, match="no bridge floor"):
        record.relative_to_floor(1.0)


def test_a_lineage_contrast_under_one_reference_calibration_shares_a_lane_id(
        lineage_registry: Any) -> None:
    """One calibration over both weights gives both one lane id; the records part them."""
    reference = "7" * 64
    rows = [row("L", "tiny-base", SHA_A, calibration=reference),
            row("L", "tiny-sft", SHA_B, calibration=reference)]
    with pytest.raises(MixedLaneError):
        require_single_lane(rows)
    record = require_declared_contrast(rows, LineageContrast(calibration=f"reference:{reference}"))
    assert record.calibration == f"reference:{reference}" and len(record.members) == 2
    with pytest.raises(MixedLaneError, match="declares reference calibration"):
        require_declared_contrast(rows, LineageContrast(calibration="reference:" + "8" * 64))
    with pytest.raises(MixedLaneError, match="declare 'reference"):
        require_declared_contrast(rows, OWN)


@pytest.mark.parametrize("rows, message", [
    ([row("L1", "tiny-base", SHA_A), row("L2", "tiny-other", "f" * 64, calibration="2" * 64)],
     "do not trace to one another"),
    ([row("L1", "tiny-base", SHA_A), row("L2", "tiny-sft", "9" * 64, calibration="2" * 64)],
     "which binds"),
    ([row("L1", "tiny-base", SHA_A), row("L2", "nowhere", SHA_B, calibration="2" * 64)],
     "does not hold"),
    ([row("L1", "tiny-base", SHA_A), row("L2", "tiny-base", SHA_A, calibration="2" * 64)],
     "BridgeContrast"),
    ([row("L1", "tiny-base", SHA_A), row("L2", "tiny-sft", SHA_B, calibration="2" * 64),
      row("L3", "tiny-sft", SHA_B, calibration="3" * 64)], "BridgeContrast"),
    ([row("L1", "tiny-base", SHA_A), row("L1", "tiny-base", SHA_A)], "ordinary contrast"),
    ([row("L1", "tiny-base", SHA_A), {"lane_id": "L2"}], "no weights record"),
    ([row("L1", "tiny-base", SHA_A), {}], "untagged"),
    ([row("L1", "tiny-base", SHA_A), row("L2", "tiny-sft", SHA_B, calibration="2" * 64,
                                         schema="6" * 64)], "feature schemas"),
    ([row("L1", "tiny-base", SHA_A), row("L2", "tiny-sft", SHA_B, calibration="2" * 64,
                                         eff_dim="norms")], "effective dimension"),
    ([row("L1", "tiny-base", SHA_A, calibration="1" * 64),
      row("L1", "tiny-base", SHA_A, calibration="2" * 64),
      row("L2", "tiny-sft", SHA_B, calibration="3" * 64)], "inconsistent calibration_sha256"),
])
def test_a_lineage_contrast_refuses_what_it_cannot_trace(
        lineage_registry: Any, rows: list[dict[str, Any]], message: str) -> None:
    with pytest.raises(MixedLaneError, match=message):
        require_declared_contrast(rows, OWN)


def test_the_calibration_policy_is_declared_in_one_of_two_forms() -> None:
    for bad in ("", "shared", "reference:", "reference:" + "A" * 64):
        with pytest.raises(ValueError, match="calibration policy"):
            LineageContrast(calibration=bad)
    assert LineageContrast(calibration="reference:" + "0" * 64).reference_calibration == "0" * 64
    assert OWN.reference_calibration is None


def test_a_bridge_contrast_joins_two_computors_of_one_weights_against_its_floor() -> None:
    bridge = BridgeContrast(floor=2.46, floor_provenance="engine vs replay, identical tokens")
    rows = [row("replay", "tiny-base", SHA_A, calibration="1" * 64),
            row("engine", "tiny-base", SHA_A, calibration="2" * 64)]
    record = require_declared_contrast(rows, bridge, registry=None)
    assert isinstance(record, ContrastRecord) and record.kind == "bridge"
    assert (record.bridge_floor, record.bridge_floor_provenance) == (
        2.46, "engine vs replay, identical tokens")
    assert {m.lane_id for m in record.members} == {"replay", "engine"}
    assert record.relative_to_floor(4.92) == pytest.approx(2.0)
    with pytest.raises(MixedLaneError, match="LineageContrast"):
        require_declared_contrast(
            [row("replay", "tiny-base", SHA_A), row("engine", "tiny-sft", SHA_B)], bridge)
    with pytest.raises(MixedLaneError, match="ordinary contrast"):
        require_declared_contrast([row("replay", "tiny-base", SHA_A)] * 2, bridge)


@pytest.mark.parametrize("fields", [
    dict(floor=0.0, floor_provenance="x"),
    dict(floor=-1.0, floor_provenance="x"),
    dict(floor=float("nan"), floor_provenance="x"),
    dict(floor=float("inf"), floor_provenance="x"),
    dict(floor=1.0, floor_provenance=""),
    dict(floor=1.0),
])
def test_a_bridge_declares_a_measured_floor_and_its_provenance(fields: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        BridgeContrast(**fields)


def test_the_banked_gate_reads_sidecars_and_refuses_an_unreadable_one(
        tmp_path: Path, lineage_registry: Any) -> None:
    paths = []
    for i, metadata in enumerate([row("L1", "tiny-base", SHA_A, calibration="1" * 64),
                                  row("L2", "tiny-sft", SHA_B, calibration="2" * 64)]):
        path = tmp_path / f"gen_{i:03d}.npz"
        np.savez(path, features=np.zeros(2, np.float32))
        path.with_suffix(".json").write_text(json.dumps(metadata))
        paths.append(path)
    with pytest.raises(MixedLaneError):
        gate_banked_signatures(paths)
    assert gate_declared_contrast(paths, OWN).root == "tiny-base"
    missing = tmp_path / "gen_009.npz"
    with pytest.raises(MixedLaneError, match="no readable metadata"):
        gate_declared_contrast([*paths, missing], OWN)
