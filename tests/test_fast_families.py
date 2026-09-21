"""The geometry, gate and residual reductions, against the families that define them.

Everything the lane computes outside attention: key/value/query geometry, the SwiGLU
gate surface, residual trajectories, and the cross-layer CKA and epoch aggregates that
only exist once every sampled layer has been seen. Each is checked against the family
function that owns the definition, and `set(actual) == set(expected)` is asserted
before any value is compared — a reduction that agrees on the features it emits while
emitting the wrong set of them is not agreement.

Two details the fixtures exist for. The gate activations are rounded to one decimal
place deliberately, because the top-k Jaccard feature reads an argsort over tied
magnitudes and numpy's tie order is part of the feature's definition; ties have to be
present for that to be exercised. And the step counts run from 1 to 127, since the
short-span branches emit zero-filled names under the full contract rather than
emitting nothing.

Pure CPU torch over synthetic arrays; no model.
"""
import numpy as np
import pytest
import torch

from anamnesis.config import ExtractionConfig,FeaturePipelineConfig
from anamnesis.extraction.feature_families.gate_features import extract_gate_features
from anamnesis.extraction.feature_families.value_geometry import extract_value_geometry
from anamnesis.extraction.feature_families.qk_geometry import extract_qk_geometry
from anamnesis.extraction.feature_families.key_cka import extract_key_cka
from anamnesis.extraction.feature_families.residual_stream import extract_residual_trajectory
from anamnesis.extraction.fast.families import FamilyReducer
from anamnesis.extraction.fast.ops import FeatureCollector
from anamnesis.extraction.state_extractor import RawGenerationData,extract_cache_and_keys


@pytest.mark.parametrize('steps',[1,3,16,80,127])
def test_geometry_gate_and_residual_match_reference(steps):
    rng=np.random.default_rng(6024)
    layers,heads,prefix,width=3,2,11,8
    hidden=[rng.normal(size=(layers+1,width)).astype(np.float32) for _ in range(steps)]
    def head_surface():
        return {layer:[rng.normal(size=(heads,4)).astype(np.float32) for _ in range(steps)] for layer in range(layers)}
    keys,values,queries=head_surface(),head_surface(),head_surface()
    # Rounded values deliberately produce top-k ties.
    gates={layer:[np.round(rng.normal(size=40),1).astype(np.float32) for _ in range(steps)] for layer in range(layers)}
    attention=[np.ones((layers,heads,prefix+t+1),dtype=np.float32)/(prefix+t+1) for t in range(steps)]
    raw=RawGenerationData(hidden_states=hidden,attentions=attention,logits=[],
                          chosen_token_ids=np.zeros(steps,dtype=np.float32),pre_rope_keys=keys,
                          prompt_length=prefix,v_proj_values=values,queries=queries,gate_activations=gates)
    config=ExtractionConfig(sampled_layers=list(range(layers)),pca_layers=list(range(layers)),
                            early_layer_cutoff=8,late_layer_cutoff=24)
    families=FeaturePipelineConfig(trajectory_layers=list(range(layers)),
                                   contrastive_layers=list(range(layers)))
    expected={}
    f,n=extract_cache_and_keys(raw,config)
    expected.update((name,value) for name,value in zip(n,f,strict=True) if not name.startswith('cache_'))
    for fn in (extract_value_geometry,extract_qk_geometry,extract_key_cka,extract_gate_features):
        r=fn(raw,sampled_layers=config.sampled_layers)
        expected.update(zip(r.feature_names,r.features,strict=True))
    r=extract_residual_trajectory(raw,layer_indices=config.sampled_layers)
    expected.update(zip(r.feature_names,r.features,strict=True))
    out=FeatureCollector('cpu')
    reducer=FamilyReducer(out,config,families)
    for layer in config.sampled_layers:
        reducer.key(layer,torch.from_numpy(np.stack(keys[layer])))
        reducer.value(layer,torch.from_numpy(np.stack(values[layer])))
        reducer.query(layer,torch.from_numpy(np.stack(queries[layer])))
        reducer.gate(layer,torch.from_numpy(np.stack(gates[layer])))
        h=torch.from_numpy(np.stack([v[layer+1] for v in hidden]))
        reducer.residual_trajectory(layer,h,h[0].norm())
    reducer.finish()
    assert set(out.values)==set(expected)
    for name,actual in out.values.items():
        np.testing.assert_allclose(actual.numpy(),expected[name],rtol=3e-5,atol=5e-6,err_msg=name)
