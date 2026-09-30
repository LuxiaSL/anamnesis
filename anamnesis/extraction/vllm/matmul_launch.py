"""Fit the pinned engine's invariant matrix multiply in device shared memory.

The float16 persistent kernel's three-stage launch needs 104 KiB per block for
transposed weights. Devices with a 99 KiB limit cannot start that launch, even
when the checkpoint and cache fit in global memory. Two stages need 56 KiB;
one stage needs 48 KiB. The selected cap depends only on the device, never on
request length or batch composition. Tiles, reduction order, dtype and the
kernel itself stay the engine's.

Only the capture process installs this launcher, before engine profiling. Its
policy is recorded in the capture and hashed into the host's receipt; the full
fixture check still decides repeatability, batch invariance and conformance.
The upstream launch this adapter wraps is pinned with the engine:
https://github.com/vllm-project/vllm/blob/v0.16.0/vllm/model_executor/layers/batch_invariant.py
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class MatmulLaunch:
    """Device capacity and the float16 staging cap used throughout one capture."""

    shared_memory_per_block: int
    float16_num_stages: int


def matmul_launch(shared_memory_per_block: int) -> MatmulLaunch:
    """Choose staging from capacity alone; compilation still checks the launch.

    The byte thresholds describe the pinned kernel's transposed-weight launch.
    A device whose compiler needs more memory still refuses at launch rather
    than switching arithmetic or choosing a policy per request.
    """
    if shared_memory_per_block <= 0:
        raise ValueError("a positive shared-memory capacity is required")
    stages = (3 if shared_memory_per_block >= 104 * 1024
              else 2 if shared_memory_per_block >= 56 * 1024 else 1)
    return MatmulLaunch(shared_memory_per_block, stages)


class _StageLimitedKernel:
    """Delegate the engine's kernel launch with a fixed float16 staging cap."""

    def __init__(self, kernel: Any, stages: int) -> None:
        self.kernel = kernel
        self.stages = stages

    def __getitem__(self, grid: Any) -> Any:
        launch = self.kernel[grid]

        def run(*args: Any, **kwargs: Any) -> Any:
            if args[0].dtype == torch.float16:
                if kwargs.get("num_stages") != 3:
                    raise RuntimeError("the pinned float16 matmul launch requires three stages")
                kwargs["num_stages"] = self.stages
            return launch(*args, **kwargs)

        return run


def configure_matmul(device: int = 0) -> dict[str, int]:
    """Install the device's staging cap in the capture process, idempotently.

    Devices fitting the stock launch retain the original kernel object. Other
    dtypes retain their engine launch settings on every device. Importing this
    module alone never imports the engine or changes torch operations.
    """
    from vllm.model_executor.layers import batch_invariant

    properties = torch.cuda.get_device_properties(device)
    policy = matmul_launch(properties.shared_memory_per_block_optin)
    kernel = batch_invariant.matmul_kernel_persistent
    if isinstance(kernel, _StageLimitedKernel):
        if kernel.stages != policy.float16_num_stages:
            raise RuntimeError("the capture process already has another matmul staging cap")
    elif policy.float16_num_stages < 3:
        batch_invariant.matmul_kernel_persistent = _StageLimitedKernel(
            kernel, policy.float16_num_stages)
    return asdict(policy)
