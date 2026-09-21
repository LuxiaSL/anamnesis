"""The attention reduction, against every reference consumer of the same numbers.

Attention is the strongest source in the signature, and it is also the largest tensor
a forward produces. The lane's answer is to reduce inside the hook: one layer's weights
at a time, never all layers at once. That makes the reduction the one place where a
formula is rewritten rather than reused, so every number it emits is checked here
against whichever reference function owns it — the two baseline attention blocks, the
attention-flow family, and the per-head family.

What the parametrization is for. Span length changes the *set of names* the families
emit, not only their values: below two steps attention-flow falls to its zero-filled
name contract, below four steps per-head does, and the entropy and head-agreement
series subsample above 60 and 30 steps respectively. The lengths 1, 2, 7, 31, 61 and
127 sit on both sides of each of those boundaries, because a reduction that agrees on
the common case and silently emits a different name set on a short span corrupts a
bank without failing anything.

`temporal` is checked separately against the numpy/scipy operator it reimplements,
including the STFT padding branches at 8, 64 and 65 samples.

Pure CPU torch; a real replay on real weights is the equivalence suite's business.
"""
import numpy as np
import pytest
import torch

from anamnesis.config import ExtractionConfig
from anamnesis.extraction.feature_families.attention_flow import extract_attention_flow
from anamnesis.extraction.feature_families.per_head import extract_per_head
from anamnesis.extraction.feature_families.operators import apply_operators
from anamnesis.extraction.fast.attention import AttentionReducer
from anamnesis.extraction.fast.ops import FeatureCollector, temporal
from anamnesis.extraction.state_extractor import RawGenerationData, extract_tier2, extract_tier2_5


@pytest.mark.parametrize('steps',[1,2,7,31,61,127])
def test_attention_matches_all_reference_consumers(steps):
    rng=np.random.default_rng(7351)
    layers,heads,prefix,width=2,3,11,8
    attention=[]
    hidden=[rng.normal(size=(layers+1,width)).astype(np.float32) for _ in range(steps)]
    keys={layer:[rng.normal(size=(2,4)).astype(np.float32) for _ in range(steps)] for layer in range(layers)}
    for t in range(steps):
        a=rng.uniform(.001,1,size=(layers,heads,prefix+t+1)).astype(np.float32)
        a/=a.sum(axis=-1,keepdims=True)
        attention.append(a)
    raw=RawGenerationData(hidden_states=hidden,attentions=attention,logits=[],
                          chosen_token_ids=np.zeros(steps,dtype=np.float32),pre_rope_keys=keys,
                          prompt_length=prefix)
    config=ExtractionConfig(sampled_layers=[0,1],pca_layers=[0,1],
                            early_layer_cutoff=8,late_layer_cutoff=24)
    expected={}
    for fn in (extract_tier2,extract_tier2_5):
        f,n=fn(raw,config)
        expected.update(zip(n,f,strict=True))
    for fn in (extract_attention_flow,extract_per_head):
        r=fn(raw,sampled_layers=[0,1])
        expected.update(zip(r.feature_names,r.features,strict=True))
    out=FeatureCollector('cpu')
    reducer=AttentionReducer(out,steps=steps,prefix_length=prefix,sampled_layers=[0,1])
    for layer in range(layers):
        full=torch.zeros((1,heads,steps+1,prefix+steps+1))
        for t in range(steps):
            full[0,:,t,:prefix+t+1]=torch.from_numpy(attention[t][layer])
        reducer.consume(layer,full)
        reducer.smoothness(layer,torch.from_numpy(np.stack([h[layer+1] for h in hidden])))
        reducer.head_features(layer,torch.from_numpy(np.stack(keys[layer])))
    for name,actual in out.values.items():
        assert name in expected,name
        np.testing.assert_allclose(actual.numpy(),expected[name],rtol=3e-5,atol=5e-6,err_msg=name)
    assert not reducer.spectral and not reducer.per_head


@pytest.mark.parametrize('n',[1,3,4,7,8,31,64,65,127])
def test_temporal_matches_scipy_and_numpy(n):
    x=np.random.default_rng(n).normal(size=n).astype(np.float32)
    expected,_=apply_operators(x,prefix='x')
    actual=temporal(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-7)
