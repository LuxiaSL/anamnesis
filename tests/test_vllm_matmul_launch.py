"""The engine's launch cap preserves its arithmetic and the readout boundary."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from anamnesis.extraction.vllm.matmul_launch import configure_matmul, matmul_launch
from anamnesis.extraction.vllm.runtime import settings_digest


@pytest.mark.parametrize("capacity,stages", [
    (227 * 1024, 3), (163 * 1024, 3), (104 * 1024, 3),
    (99 * 1024, 2), (56 * 1024, 2), (48 * 1024, 1),
])
def test_the_policy_uses_capacity_instead_of_device_names(capacity, stages):
    policy = matmul_launch(capacity)
    assert policy.shared_memory_per_block == capacity
    assert policy.float16_num_stages == stages


def test_unknown_capacity_is_refused():
    with pytest.raises(ValueError, match="positive"):
        matmul_launch(0)


class RecordingKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
            return "launched"
        return launch


@pytest.fixture
def engine(monkeypatch):
    kernel = RecordingKernel()
    module = SimpleNamespace(matmul_kernel_persistent=kernel)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers",
                        SimpleNamespace(batch_invariant=module))
    properties = SimpleNamespace(shared_memory_per_block_optin=99 * 1024)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: properties)
    return module, kernel, properties


def test_low_capacity_changes_only_float16_staging_and_installs_once(engine):
    module, original, _ = engine
    record = configure_matmul()
    installed = module.matmul_kernel_persistent
    assert configure_matmul() == record
    assert module.matmul_kernel_persistent is installed
    for dtype, stages in [(torch.float16, 2), (torch.bfloat16, 3), (torch.float32, 3)]:
        tensor = torch.ones((1, 1), dtype=dtype)
        assert installed[(1,)](tensor, num_stages=3, BLOCK_SIZE_K=64) == "launched"
        grid, args, options = original.calls[-1]
        assert grid == (1,) and args[0] is tensor
        assert options == {"num_stages": stages, "BLOCK_SIZE_K": 64}


def test_high_capacity_keeps_the_engine_kernel_object(engine):
    module, original, properties = engine
    properties.shared_memory_per_block_optin = 227 * 1024
    assert configure_matmul()["float16_num_stages"] == 3
    assert module.matmul_kernel_persistent is original


def test_a_process_cannot_silently_switch_launch_policies(engine):
    _, _, properties = engine
    configure_matmul()
    properties.shared_memory_per_block_optin = 227 * 1024
    with pytest.raises(RuntimeError, match="another matmul staging cap"):
        configure_matmul()


def test_an_unexpected_upstream_stage_setting_is_refused(engine):
    module, original, _ = engine
    configure_matmul()
    with pytest.raises(RuntimeError, match="pinned float16"):
        module.matmul_kernel_persistent[(1,)](torch.ones(1, dtype=torch.float16), num_stages=4)
    assert not original.calls


def test_receipts_separate_launch_policies():
    small = asdict(matmul_launch(99 * 1024))
    large = asdict(matmul_launch(227 * 1024))
    assert settings_digest("3b", matmul_policy=small) != settings_digest("3b", matmul_policy=large)
    assert settings_digest("3b", matmul_policy=small) != settings_digest("3b")


def test_importing_the_policy_does_not_import_the_engine():
    result = subprocess.run([sys.executable, "-c", """
import sys
from anamnesis.extraction.vllm.matmul_launch import matmul_launch
assert matmul_launch(101376).float16_num_stages == 2
assert not any(n == 'vllm' or n.startswith('vllm.') for n in sys.modules)
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None or not torch.cuda.is_available(),
    reason="the real invariant matrix multiply needs the pinned engine and a CUDA device",
)
def test_transposed_weight_gemm_repeats_across_batch_splits_and_orders():
    # A fresh process keeps the engine's imports out of readout tests.
    result = subprocess.run([sys.executable, "-c", """
import torch
from anamnesis.extraction.vllm.matmul_launch import configure_matmul
from vllm.model_executor.layers.batch_invariant import matmul_persistent
torch.manual_seed(42)
configure_matmul()
a = torch.randn(137, 3072, device='cuda', dtype=torch.float16) / 3072**0.5
b = torch.randn(5120, 3072, device='cuda', dtype=torch.float16).T
whole = matmul_persistent(a, b)
assert torch.equal(whole, matmul_persistent(a, b))
split = torch.cat([matmul_persistent(a[:5], b), matmul_persistent(a[5:], b)])
assert torch.equal(whole, split)
order = torch.randperm(len(a), device='cuda')
assert torch.equal(whole[order], matmul_persistent(a[order], b))
reference = (a.double() @ b.double()).half()
torch.testing.assert_close(whole, reference, rtol=1e-3, atol=2e-3)
"""], capture_output=True, text=True, env={**os.environ, "VLLM_NO_USAGE_STATS": "1"},
                            timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
