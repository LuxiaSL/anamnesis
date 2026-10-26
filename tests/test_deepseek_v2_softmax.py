"""DeepSeek-V2's attention softmax scale: refused unless named, applied when asked, kept when asked.

transformers' native deepseek_v2 port omitted the YaRN mscale² DeepSeek's own code applies to the
softmax scale (4.57 and 5.3 do; later releases apply it). These run a tiny randomly initialised
native-port model through :func:`anamnesis.extraction.model_loader.resolve_deepseek_v2_softmax`,
as the omitting port (``unscaled``) wherever the installed release already applies the factor, so
every path is exercised whatever transformers is installed.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from anamnesis.extraction.model_loader import (
    DEEPSEEK_V2_SOFTMAX_ENV,
    DeepseekV2SoftmaxScaleError,
    deepseek_v2_mscale_factor,
    resolve_deepseek_v2_softmax,
)

transformers = pytest.importorskip("transformers")
DeepseekV2Config = getattr(transformers, "DeepseekV2Config", None)
DeepseekV2ForCausalLM = getattr(transformers, "DeepseekV2ForCausalLM", None)
pytestmark = pytest.mark.skipif(DeepseekV2Config is None, reason="transformers has no native deepseek_v2")

YARN = {"type": "yarn", "rope_type": "yarn", "factor": 40.0, "mscale": 0.707, "mscale_all_dim": 0.707,
        "beta_fast": 32, "beta_slow": 1, "original_max_position_embeddings": 64}


def tiny(rope_scaling=YARN):
    torch.manual_seed(0)
    config = DeepseekV2Config(
        vocab_size=97, hidden_size=32, intermediate_size=48, moe_intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, n_routed_experts=4, n_shared_experts=1,
        num_experts_per_tok=2, first_k_dense_replace=1, kv_lora_rank=16, q_lora_rank=None,
        qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8, max_position_embeddings=256,
        rope_scaling=rope_scaling)
    return DeepseekV2ForCausalLM(config).eval()


def unscaled(model):
    """The model as the omitting port builds it: every attention at qk_head_dim**-0.5."""
    for m in model.modules():
        if hasattr(m, "scaling") and hasattr(m, "qk_head_dim"):
            m.scaling = m.qk_head_dim ** (-0.5)
    return model


def scalings(model):
    return [m.scaling for m in model.modules() if hasattr(m, "scaling") and hasattr(m, "qk_head_dim")]


def test_the_factor_is_deepseeks_yarn_mscale_squared():
    m = 0.1 * 0.707 * math.log(40.0) + 1.0
    assert deepseek_v2_mscale_factor(SimpleNamespace(rope_scaling=YARN)) == pytest.approx(m * m)
    assert deepseek_v2_mscale_factor(SimpleNamespace(rope_scaling=None)) is None


def test_a_native_port_load_without_a_choice_is_refused():
    model = unscaled(tiny())
    with pytest.raises(DeepseekV2SoftmaxScaleError, match="official"):
        resolve_deepseek_v2_softmax(model, None)


def test_official_applies_the_factor_to_every_attention():
    model = unscaled(tiny())
    before = scalings(model)
    assert resolve_deepseek_v2_softmax(model, "official") == "official"
    factor = deepseek_v2_mscale_factor(model.config)
    after = scalings(model)
    assert len(after) == 2
    assert all(math.isclose(a, b * factor) for a, b in zip(after, before))


def test_official_on_a_port_that_already_applies_it_changes_nothing():
    model = unscaled(tiny())
    resolve_deepseek_v2_softmax(model, "official")
    applied = scalings(model)
    assert resolve_deepseek_v2_softmax(model, "official") == "official"
    assert scalings(model) == applied
    with pytest.raises(DeepseekV2SoftmaxScaleError, match="cannot be reproduced"):
        resolve_deepseek_v2_softmax(model, "native_unscaled")


def test_native_unscaled_keeps_the_port_and_changes_the_output():
    a, b = unscaled(tiny()), unscaled(tiny())
    assert resolve_deepseek_v2_softmax(a, "native_unscaled") == "native_unscaled"
    resolve_deepseek_v2_softmax(b, "official")
    ids = torch.randint(0, 97, (1, 12))
    with torch.no_grad():
        assert not torch.equal(a(ids).logits, b(ids).logits)


def test_an_unknown_choice_is_refused():
    with pytest.raises(DeepseekV2SoftmaxScaleError, match="choose one of"):
        resolve_deepseek_v2_softmax(tiny(), "huh")


def test_a_model_without_yarn_mscale_is_left_alone():
    model = tiny(rope_scaling=None)
    before = scalings(model)
    assert resolve_deepseek_v2_softmax(model, None) is None
    assert scalings(model) == before


def test_other_architectures_are_not_its_concern():
    other = torch.nn.Linear(2, 2)
    other.config = SimpleNamespace(model_type="llama")
    assert resolve_deepseek_v2_softmax(other, None) is None


def test_the_refusal_names_both_ways_to_choose():
    with pytest.raises(DeepseekV2SoftmaxScaleError) as info:
        resolve_deepseek_v2_softmax(unscaled(tiny()), None)
    assert "deepseek_v2_softmax" in str(info.value) and DEEPSEEK_V2_SOFTMAX_ENV in str(info.value)
