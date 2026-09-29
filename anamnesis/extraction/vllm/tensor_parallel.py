"""The lane's capture across tensor-parallel workers.

A model too large for one GPU runs with its weights split over ``tp`` ranks, one
worker process per GPU. Each rank owns a contiguous slice of the query heads
(rank r: heads ``[r*H/tp, (r+1)*H/tp)``) and of the KV heads, the matching slice
of the packed QKV projection, and a contiguous slice of the MLP's intermediate
dimension (the column-parallel gate/up projection packs ``[gate_r | up_r]`` per
rank). The residual stream, the norms' outputs and the logits are replicated.

The capture runs inside every worker and stores the same full-width layouts a
single-GPU capture stores:

* **Per-head attention statistics** are computed by the owning rank over its own
  heads and gathered along the head axis in rank order, which is the global head
  order (:class:`TPInstrumentedImpl`).
* **Cross-head products** (coverage, agreement, spectral, decay, span and entropy
  rows) are never reduced per rank: each rank's second-pass rows and row sums are
  gathered first, and the unchanged reduction runs once over all heads.
* **QKV and gate observations** are gathered along the feature axis before they
  are stored (:class:`TPLaneCapture`).

Every gather is a copy; none adds arithmetic. Every rank runs the same hooks in
the same order, so the collectives line up. The driver reaches the workers only
through the extension in :mod:`anamnesis.extraction.vllm.tp_worker`, and
:func:`capture_rows_tp` is :func:`anamnesis.extraction.vllm.runtime.capture_rows`
with the capture moved into the workers: rank 0 writes each retained row, the
driver writes the lane's receipts and schedule records, and a group refuses
unless every rank captured the same bytes.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from anamnesis.extraction.vllm import backend as lane_backend
from anamnesis.extraction.vllm.capture import LaneCapture

OWNERSHIP: dict[str, dict] = {}
"""Per attention layer, the heads this rank computed: filled on the layer's first
armed forward and checked on every later one."""

NCCL_ENV_PREFIXES = ("NCCL_", "TORCH_NCCL")
"""Environment variables recorded per rank: the collective library's settings."""


def head_ownership(rank: int, world: int, local_heads: int, local_kv_heads: int) -> dict:
    """The query and KV heads rank ``rank`` of ``world`` owns: contiguous slices in
    rank order."""
    if not 0 <= rank < world or local_heads <= 0 or local_kv_heads <= 0 \
            or local_heads % local_kv_heads:
        raise ValueError("a rank owns a positive, grouped slice of the heads")
    return dict(rank=rank, world=world, local_query_heads=local_heads,
                local_kv_heads=local_kv_heads, total_query_heads=local_heads * world,
                total_kv_heads=local_kv_heads * world,
                query_heads=[rank * local_heads, (rank + 1) * local_heads],
                kv_heads=[rank * local_kv_heads, (rank + 1) * local_kv_heads])


def ownership_problems(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    """What is wrong with one layer's per-rank ownership records: a rank missing or
    repeated, or a head dropped, doubled or out of rank order."""
    ranks = sorted(entries, key=lambda e: e["rank"])
    world = len(ranks)
    problems = []
    if [e["rank"] for e in ranks] != list(range(world)) \
            or any(e["world"] != world for e in ranks):
        problems.append("every rank must report its heads exactly once")
    for kind, total in (("query_heads", "total_query_heads"), ("kv_heads", "total_kv_heads")):
        cursor = 0
        for e in ranks:
            lo, hi = e[kind]
            if lo != cursor or hi <= lo:
                problems.append(f"rank {e['rank']}: {kind} {lo}-{hi} do not continue at {cursor}")
            cursor = hi
        if ranks and cursor != ranks[0][total]:
            problems.append(f"{kind} cover {cursor} of {ranks[0][total]}")
    return problems


def tp_group():
    from vllm.distributed import get_tp_group

    return get_tp_group()


def gather(tensor: Tensor, dim: int, group=None) -> Tensor:
    """``tensor`` from every rank, concatenated along ``dim`` in rank order."""
    group = tp_group() if group is None else group
    if group.world_size == 1:
        return tensor
    return group.all_gather(tensor.contiguous(), dim=dim).contiguous()


def rank_agreement(finished: Sequence[Mapping[str, Any]],
                   generation_ids: Mapping[str, int]) -> dict[str, dict]:
    """Per retained row, whether every rank captured the same bytes, and which
    tensors differ from rank 0's."""
    lead, rest = finished[0], finished[1:]
    out = {}
    for internal, gid in generation_ids.items():
        per_rank = [f["receipts"][internal]["sha256"] for f in finished]
        out[str(gid)] = dict(
            joint_equal=len(set(per_rank)) == 1,
            differing_tensors=sorted(
                name for name, entry in lead["receipts"][internal]["tensors"].items()
                if any(f["receipts"][internal]["tensors"][name] != entry for f in rest)))
    return out


if lane_backend.HAVE_VLLM:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    class TPInstrumentedImpl(lane_backend.InstrumentedTritonAttentionImpl):
        """The lane's instrumented impl, with per-head outputs gathered across ranks."""

        def _tp_world(self) -> int:
            return tp_group().world_size

        def _record_heads(self, layer_name: str) -> None:
            group = tp_group()
            entry = head_ownership(group.rank_in_group, group.world_size, self.num_heads,
                                   self.num_kv_heads)
            if OWNERSHIP.setdefault(layer_name, entry) != entry:
                raise RuntimeError(f"{layer_name}: head ownership changed between steps")

        def _gather_heads(self, tensor: Tensor) -> Tensor:
            return gather(tensor, dim=1)

    class TPInstrumentedBackend(lane_backend.InstrumentedTritonAttentionBackend):
        """TRITON_ATTN with the tensor-parallel instrumented impl; the name is unchanged."""

        @staticmethod
        def get_impl_cls() -> type[TPInstrumentedImpl]:
            return TPInstrumentedImpl

    def register_tp_backend() -> str:
        """Override TRITON_ATTN's resolved class in this process with the
        tensor-parallel backend."""
        path = f"{TPInstrumentedBackend.__module__}.TPInstrumentedBackend"
        register_backend(AttentionBackendEnum.TRITON_ATTN, path)
        if AttentionBackendEnum.TRITON_ATTN.get_class() is not TPInstrumentedBackend:
            raise RuntimeError("the registry did not resolve the tensor-parallel backend")
        return path

else:  # pragma: no cover - exercised only without the engine

    def register_tp_backend() -> str:
        raise RuntimeError("the engine is required to register the tensor-parallel backend")


class TPLaneCapture(LaneCapture):
    """The lane's capture inside one worker, storing full-width QKV and gate tensors."""

    def _qkv(self, layer):
        def observe(module, args, output):
            attn = self.blocks[layer].self_attn
            q, k, d = int(attn.q_size), int(attn.kv_size), int(attn.head_dim)
            if q * tp_group().world_size != self.model.config.hidden_size \
                    or q % d or k % d or q % k:
                raise ValueError("the packed QKV projection does not match a "
                                 "tensor-parallel Llama shard")
            if (not isinstance(output, tuple) or len(output) != 2
                    or output[0].shape != (self.step["total"], q + 2 * k)):
                raise ValueError("native packed QKV shape mismatch")
            for kind, value in zip(("queries", "keys", "values"),
                                   output[0].split([q, k, k], dim=-1), strict=True):
                full = gather(value, dim=-1)
                self._packed(f"{kind}/{layer}", full, full.shape[-1], d)

        return observe

    def _gate(self, layer):
        def observe(module, args, output):
            width = int(self.model.config.intermediate_size)
            world = tp_group().world_size
            if width % world:
                raise ValueError("the intermediate width does not split over the ranks")
            local = width // world
            if (not isinstance(output, tuple) or len(output) != 2
                    or output[0].shape != (self.step["total"], 2 * local)):
                raise ValueError("native packed gate shape mismatch")
            self._packed(f"gates/{layer}", gather(output[0][:, :local], dim=-1), width)

        return observe


def topology() -> str | None:
    """``nvidia-smi topo -m``: the interconnect the collectives ran over, when readable."""
    try:
        result = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True,
                                text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def capture_groups_tp(llm, rows, groups, condition, sampled_layers, sampling, tokens_prompt,
                      out, *, attention_rounding: bool) -> dict:
    """:func:`anamnesis.extraction.vllm.runner.capture_groups` with the capture in the
    workers: the same refusals and records, plus every rank's per-tensor receipts."""
    from anamnesis.extraction.vllm.runner import (
        _normalized,
        _validate_groups,
        validate_schedule,
        write_json,
    )
    from anamnesis.provenance import file_sha

    by_id = _validate_groups(rows, groups, condition)
    if getattr(sampling, "max_tokens", None) != 1 or getattr(sampling, "n", 1) != 1:
        raise ValueError("one output token and one completion required")
    if (getattr(sampling, "prompt_logprobs", None) != 0 or getattr(sampling, "logprobs", None)
            != 0 or getattr(sampling, "temperature", None) != 0):
        raise ValueError("deterministic prompt/sample logprob capture required")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob("row-*")) or list(out.glob("group-*")):
        raise FileExistsError("capture output files already exist")
    records, group_records = [], []
    for group_index, members in enumerate(groups):
        begun = time.perf_counter()
        stem = f"group-{group_index:04d}"
        mapping, hooked = {}, False
        try:
            if llm.llm_engine.has_unfinished_requests():
                raise RuntimeError("engine has requests before bounded group")
            prompts = [tokens_prompt(prompt_token_ids=by_id[m["generation_id"]]["input_ids"])
                       for m in members]
            control_outputs = llm.generate(prompts, sampling, use_tqdm=False)
            controls = _normalized(control_outputs, members, by_id)
            del control_outputs
            write_json(out / f"{stem}.control.json",
                       dict(condition=condition, group_index=group_index,
                            occurrences=controls))
            if llm.llm_engine.has_unfinished_requests():
                raise RuntimeError("control left pending requests")
            requests, expected_external = {}, []
            for member, prompt in zip(members, prompts, strict=True):
                internal = llm._add_request(prompt, sampling)
                if internal in requests:
                    raise ValueError("duplicate assigned internal request ID")
                state = llm.llm_engine.output_processor.request_states[internal]
                external = state.external_req_id
                if not isinstance(external, str) or not external.isdecimal():
                    raise ValueError("the engine assigned a non-numeric external request id")
                expected_external.append(external)
                row = by_id[member["generation_id"]]
                requests[internal] = dict(input_ids=row["input_ids"],
                                          start=row["prompt_length"], end=row["end"],
                                          retain=member["retain"])
                mapping[internal] = dict(**member, external_request_id=external)
            llm.collective_rpc("lane_tp_begin", args=(requests, list(sampled_layers),
                                                      attention_rounding))
            hooked = True
            observed_outputs = llm._run_engine(use_tqdm=False)
            llm.collective_rpc("lane_tp_close")
            observed = _normalized(observed_outputs, members, by_id, expected_external)
            del observed_outputs
            gids = {r: m["generation_id"] for r, m in mapping.items() if requests[r]["retain"]}
            finished = sorted(llm.collective_rpc("lane_tp_finish", args=(str(out), gids)),
                              key=lambda x: x["rank"])
            hooked = False
            lead = finished[0]
            if [f["rank"] for f in finished] != list(range(len(finished))):
                raise RuntimeError("every rank must report its capture once")
            agreement = rank_agreement(finished, gids)
            schedules_equal = all(json.dumps(f["schedule"], sort_keys=True)
                                  == json.dumps(lead["schedule"], sort_keys=True)
                                  for f in finished[1:])
            evidence = validate_schedule(lead["schedule"], requests, condition["max_num_seqs"])
            if llm.llm_engine.has_unfinished_requests():
                raise RuntimeError("capture left pending requests")
            unchanged = all(a["generation"] == b["generation"]
                            for a, b in zip(controls, observed, strict=True))
            layers = sorted(lead["ownership"])
            ownership = {layer: ownership_problems([f["ownership"][layer] for f in finished])
                         for layer in layers}
            schedule = dict(
                capture_contract="lane-capture/1", condition=condition,
                attention_rounding=attention_rounding, group_index=group_index,
                internal_requests=mapping, trace=lead["schedule"], evidence=evidence,
                peak_fragment_bytes=lead["peak_fragment_bytes"],
                hook_noninterference=unchanged, controls=controls, observed=observed,
                tensor_parallel_size=len(finished), rank_schedules_equal=schedules_equal,
                rank_agreement=agreement,
                rank_receipts={str(f["rank"]): {str(gids[r]): f["receipts"][r]["sha256"]
                                                 for r in gids} for f in finished},
                ownership={str(f["rank"]): f["ownership"] for f in finished},
                ownership_problems={k: v for k, v in ownership.items() if v})
            schedule_path = out / f"{stem}.schedule.json"
            write_json(schedule_path, schedule)
            schedule_digest = file_sha(schedule_path)
            if set(lead["written"]) != set(gids):
                raise ValueError("capture retained request set differs from protocol")
            for internal, gid in gids.items():
                raw = out / f"row-{gid:05d}.pt"
                record = dict(
                    capture_contract="lane-capture/1", generation_id=gid,
                    raw_sha256=file_sha(raw), substrate=lead["receipts"][internal],
                    hook_noninterference=unchanged, schedule_sha256=schedule_digest,
                    schedule_file=schedule_path.name,
                    occurrence_id=mapping[internal]["occurrence_id"],
                    internal_request_id=internal,
                    ranks_agree=agreement[str(gid)]["joint_equal"])
                write_json(raw.with_suffix(".json"), record)
                records.append(record)
            group_records.append(dict(group_index=group_index, schedule_file=schedule_path.name,
                                      schedule_sha256=schedule_digest, retained_rows=len(gids),
                                      seconds=time.perf_counter() - begun, **evidence))
            if not unchanged:
                raise RuntimeError("the capture hooks changed the generation or its logprobs")
            if not schedules_equal:
                raise RuntimeError("the ranks traced different schedules")
            if not all(a["joint_equal"] for a in agreement.values()):
                raise RuntimeError("the ranks captured different bytes")
            if schedule["ownership_problems"]:
                raise RuntimeError(f"head ownership: {schedule['ownership_problems']}")
        except BaseException as exc:
            if hooked:
                try:
                    llm.collective_rpc("lane_tp_abort")
                except BaseException:  # noqa: BLE001 - the original error is the one raised
                    pass
            write_json(out / f"{stem}.failed.json",
                       dict(condition=condition, group_index=group_index,
                            exception=type(exc).__name__, message=str(exc),
                            internal_requests=mapping))
            raise
    if sorted(r["generation_id"] for r in records) != sorted(by_id):
        raise RuntimeError("retained population differs from complete roster")
    return dict(rows=sorted(records, key=lambda r: r["generation_id"]), groups=group_records)


def capture_rows_tp(spec: Mapping[str, Any]) -> None:
    """:func:`anamnesis.extraction.vllm.runtime.capture_rows` for a tensor-parallel lane:
    the engine's workers capture, and the capture record adds every rank's device,
    collective settings and the interconnect topology."""
    from anamnesis.extraction.vllm.envelope import (
        CONDITIONS,
        engine_settings,
        enforce_lane_envelope,
        lane_id,
        lane_model,
        lane_preset,
        lane_tensor_parallel_size,
        request_groups,
        require_environment,
        require_pinned_packages,
    )
    from anamnesis.extraction.vllm.runtime import ATTENTION_ROUNDING

    model, condition_id = spec["model"], spec["condition_id"]
    require_environment(model=model)
    versions = require_pinned_packages()
    settings = engine_settings(model, condition_id)
    guard = enforce_lane_envelope(settings, os.environ, lane=lane_id(model), model=model)
    tp = lane_tensor_parallel_size(model)
    if lane_model(model)["logprob_wrapper"] != "none":
        raise ValueError("a tensor-parallel lane runs without logprob promotion (bfloat16)")
    resolved = register_tp_backend()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from anamnesis.config import resolve_preset

    rows = [dict(generation_id=int(r["generation_id"]), input_ids=list(r["input_ids"]),
                 prompt_length=int(r["prompt_length"]), end=int(r["end"]))
            for r in spec["rows"]]
    condition = CONDITIONS[condition_id]
    groups = request_groups([r["generation_id"] for r in rows], condition["max_num_seqs"])
    llm = LLM(model=str(spec["model_path"]), **settings)
    ranks = sorted(llm.collective_rpc("lane_tp_info"), key=lambda x: x["rank"])
    for rank in ranks:
        if rank["model_class"] != "LlamaForCausalLM":
            raise ValueError("the engine did not resolve the model to its native Llama")
        if rank["attention_impls"] != ["TPInstrumentedImpl"]:
            raise ValueError(f"rank {rank['rank']} runs {rank['attention_impls']}")
        if rank["world"] != tp or not rank["disable_custom_all_reduce"] \
                or rank["custom_allreduce"]:
            raise ValueError(f"rank {rank['rank']}: parallel settings differ from the lane's")
    sampling = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0, logprobs=0,
                              detokenize=False, seed=settings["seed"])
    preset = resolve_preset(lane_preset(model))
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=False)
    for index in range(int(spec["passes"])):
        capture_groups_tp(llm, rows, groups, condition, preset.sampled_layers, sampling,
                          TokensPrompt, out / f"pass-{index}",
                          attention_rounding=ATTENTION_ROUNDING)
    (out / "capture.json").write_text(json.dumps(dict(
        lane_id=lane_id(model), condition=condition, settings=settings,
        passes=int(spec["passes"]), attention_rounding=ATTENTION_ROUNDING,
        startup_guard=guard, resolved_backend=resolved, packages=versions,
        tensor_parallel_size=tp, ranks=ranks, topology=topology(),
        device=torch.cuda.get_device_name(0)), indent=2, default=str) + "\n")
