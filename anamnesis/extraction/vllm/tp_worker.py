"""The worker extension a tensor-parallel lane's engine loads into every worker.

The engine resolves the extension class in each worker before the worker builds
its model, so importing this module is what installs the tensor-parallel
instrumented backend there. The driver reaches each worker's model runner only
through ``LLM.collective_rpc`` calls on these methods, all prefixed ``lane_tp_``
so none collides with the worker's own attributes.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from anamnesis.extraction.vllm import tensor_parallel
from anamnesis.extraction.vllm.receipts import assert_substrate_fields, capture_receipt

REGISTERED_BACKEND = tensor_parallel.register_tp_backend()


class TPCaptureExtension:
    """Capture control for one worker; the driver calls each method on every rank."""

    def lane_tp_info(self) -> dict:
        """This rank's identity, device, collective settings and model wiring."""
        group = tensor_parallel.tp_group()
        runner = self.model_runner
        model = runner.model
        communicator = group.device_communicator
        parallel = self.vllm_config.parallel_config
        properties = torch.cuda.get_device_properties(runner.device)
        return dict(
            rank=group.rank_in_group, world=group.world_size, device=str(runner.device),
            gpu_name=properties.name, gpu_uuid=str(properties.uuid),
            nccl_version=".".join(str(x) for x in torch.cuda.nccl.version()),
            torch=torch.__version__, model_class=type(model).__name__,
            attention_impls=sorted({type(b.self_attn.attn.impl).__qualname__
                                    for b in model.model.layers}),
            local_heads=sorted({(b.self_attn.attn.num_heads, b.self_attn.attn.num_kv_heads)
                                for b in model.model.layers})[0],
            model_heads=(self.vllm_config.model_config.hf_config.num_attention_heads,
                         self.vllm_config.model_config.hf_config.num_key_value_heads),
            registered_backend=REGISTERED_BACKEND,
            tensor_parallel_size=parallel.tensor_parallel_size,
            disable_custom_all_reduce=parallel.disable_custom_all_reduce,
            device_communicator=None if communicator is None
            else type(communicator).__qualname__,
            custom_allreduce=None if communicator is None
            else getattr(communicator, "ca_comm", None) is not None,
            collective_environment={k: v for k, v in sorted(os.environ.items())
                                    if k.startswith(tensor_parallel.NCCL_ENV_PREFIXES)})

    def lane_tp_begin(self, requests: dict, sampled_layers: list,
                      attention_rounding: bool) -> int:
        """Hook this rank's runner for one group of requests."""
        if getattr(self, "_lane_tp_tap", None) is not None:
            raise RuntimeError("a capture is already open on this rank")
        tap = tensor_parallel.TPLaneCapture(self.model_runner, requests, sampled_layers,
                                            attention_rounding=attention_rounding)
        tap.__enter__()
        self._lane_tp_tap = tap
        return tensor_parallel.tp_group().rank_in_group

    def lane_tp_close(self) -> int:
        """Unhook once the group has run."""
        self._lane_tp_tap.__exit__(None, None, None)
        return tensor_parallel.tp_group().rank_in_group

    def lane_tp_abort(self) -> int:
        """Unhook after a failure, if a capture is open."""
        tap = getattr(self, "_lane_tp_tap", None)
        if tap is not None:
            tap.__exit__(RuntimeError, RuntimeError("aborted by the driver"), None)
        self._lane_tp_tap = None
        return tensor_parallel.tp_group().rank_in_group

    def lane_tp_finish(self, out_dir: str, generation_ids: dict) -> dict:
        """Finish the group. Rank 0 writes each retained row's capture; every rank
        returns its schedule and per-tensor receipts, so the driver can check that the
        ranks hold the same bytes."""
        tap, self._lane_tp_tap = self._lane_tp_tap, None
        rank = tensor_parallel.tp_group().rank_in_group
        captures = tap.finish()
        receipts, written = {}, {}
        for internal, capture in captures.items():
            gid = int(generation_ids[internal])
            assert_substrate_fields(capture, context=f"rank {rank} row {gid}")
            receipts[internal] = capture_receipt(capture)
            if rank == 0:
                raw = Path(out_dir) / f"row-{gid:05d}.pt"
                if raw.exists():
                    raise FileExistsError(f"duplicate retained row {gid}")
                torch.save(capture, raw)
                restored = torch.load(raw, map_location="cpu", weights_only=True)
                if capture_receipt(restored) != receipts[internal]:
                    raise RuntimeError("capture bytes differ after the serialization round trip")
                written[internal] = str(raw)
                del restored
        return dict(rank=rank, schedule=tap.schedule,
                    peak_fragment_bytes=tap.peak_fragment_bytes, receipts=receipts,
                    written=written, ownership=dict(tensor_parallel.OWNERSHIP))
