"""NumPy's CPU dispatch is recorded beside every reduction, outside every lane id.

NumPy chooses its compiled loops per CPU at runtime, and its ``argsort`` orders tied
values differently under different instruction sets, so the gate's top-k overlap can
differ between two hosts that differ only in CPU. The dispatch is provenance: it is
in the vLLM host fingerprint, the readout's record and the fast lane's provenance,
and in neither lane's identity hash.
"""

from __future__ import annotations

import numpy as np

from anamnesis.extraction.vllm.conformance import HostFingerprint
from anamnesis.provenance import numpy_dispatch


def test_the_dispatch_names_numpy_its_baseline_and_the_enabled_targets():
    record = numpy_dispatch()
    assert record["numpy"] == np.__version__
    assert isinstance(record["baseline"], list) and isinstance(record["dispatch"], list)
    assert all(isinstance(name, str) for name in record["baseline"] + record["dispatch"])


def test_two_cpus_are_two_host_fingerprints():
    fields = dict(gpu_name="g", gpu_uuid="u", driver="d", cuda_runtime="c", torch="t",
                  vllm="v", anamnesis="a", checkpoint_sha256="0" * 64,
                  engine_settings_sha256="e" * 64, fixture_digest="f" * 64,
                  tolerance_digest="1" * 64)
    avx512 = HostFingerprint(**fields, numpy_dispatch=dict(numpy="2.2.6", baseline=["SSE"],
                                                           dispatch=["AVX2", "AVX512_SKX"]))
    avx2 = HostFingerprint(**fields, numpy_dispatch=dict(numpy="2.2.6", baseline=["SSE"],
                                                         dispatch=["AVX2"]))
    assert avx512.digest != avx2.digest


def test_the_readout_record_carries_the_dispatch():
    from anamnesis.extraction.vllm.session import _device_record

    assert _device_record()["numpy_dispatch"] == numpy_dispatch()
