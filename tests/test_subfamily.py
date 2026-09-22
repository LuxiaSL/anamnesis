"""Sub-family decomposition: the name classifiers, and the mapping they rest on.

A decomposition is a map from feature names to column indices, so the failure mode is not
a wrong accuracy — it is a right accuracy attributed to the wrong part of a family. The
tests are therefore mostly about names and alignment:

  * each family's classifier reads its own naming convention, including the one signal
    that has no layer to belong to (the cross-layer gate features);
  * a name the convention does not cover reads as that family's unknown bucket rather than
    being silently filed under a real sub-family;
  * feature names come from the loader, and a **length mismatch between the names and the
    matrix is refused** — that is the exact condition under which a decomposition would map
    indices onto the wrong features and still print a plausible table;
  * a sub-family with no features returns zero with a reason rather than an accuracy;
  * the whole family is scored alongside its parts, because a part is only readable against
    the whole;
  * the output path differs for a mode-filtered pass, since that is a different measurement.

CPU only; the corpus is synthetic and the classifier is the real one at small parameters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anamnesis.analysis.gauntlet.signature_io import Run4Data, SampleMeta
from anamnesis.analysis.subfamily import (
    FULL_FAMILY,
    SUBFAMILY_CLASSIFIERS,
    accuracy_on_subset,
    classify_attention_flow,
    classify_gate_features,
    decompose_family,
    decompose_run,
    decomposition_document,
    default_output_path,
    feature_names_for,
)


def test_attention_flow_names_are_read_by_their_allocation_statistic() -> None:
    assert classify_attention_flow("attn_flow_L8_sysprompt_mass_mean") == "af_sysprompt_mass"
    assert classify_attention_flow("attn_flow_L8_recency_bias_std") == "af_recency_bias"
    assert classify_attention_flow("attn_flow_L8_region_early_gen_mean") == "af_region_early_gen"
    assert classify_attention_flow("af_head_diversity_recency_L8") == "af_head_diversity_recency"
    assert classify_attention_flow("attn_flow_L8_something_mean") == "af_unknown"


def test_a_name_the_convention_does_not_span_reads_as_unknown() -> None:
    """The bucket is the honest answer, and it is what makes a stale cut visible.

    ``attn_flow_L8_sysprompt_decay_rate`` is read to the signal ``sysprompt`` and no
    further, because the operator suffix ``decay`` ends the signal — so it lands in the
    unknown bucket rather than under a sub-family it resembles. A cut whose unknown bucket
    is large is telling the reader that the convention it reads has moved.
    """
    assert classify_attention_flow("attn_flow_L8_sysprompt_decay_rate") == "af_unknown"


def test_gate_names_keep_cross_layer_features_out_of_a_layer() -> None:
    assert classify_gate_features("gate_cross_layer_agreement") == "gf_cross_layer"
    assert classify_gate_features("gate_layer_sparsity_diversity") == "gf_cross_layer"
    assert classify_gate_features("gate_L16_sparsity_mean") == "gf_sparsity"
    assert classify_gate_features("gate_L16_eff_dim_std") == "gf_eff"
    assert classify_gate_features("some_sparsity_thing") == "gf_sparsity"
    assert classify_gate_features("gate_mystery") == "gf_unknown"


def test_the_classifier_table_names_the_families_that_have_a_convention() -> None:
    assert set(SUBFAMILY_CLASSIFIERS) == {
        "attention_flow",
        "gate_features",
    }


GF_NAMES = [
    "gate_L16_sparsity_mean",
    "gate_L16_sparsity_std",
    "gate_L16_sparsity_w0_mean",
    "gate_L16_drift_mean",
    "gate_L16_drift_std",
    "gate_L16_eff_dim_mean",
    "gate_cross_layer_agreement",
    "gate_layer_sparsity_diversity",
]


def synthetic_run(names: list[str], *, n_per_mode: int = 12, seed: int = 0) -> Run4Data:
    """Two modes separated on the first column only, so a cut's readout is known."""
    rng = np.random.default_rng(seed)
    rows = []
    modes = []
    samples = []
    for index in range(2 * n_per_mode):
        mode = "linear" if index < n_per_mode else "socratic"
        vector = 0.3 * rng.standard_normal(len(names))
        vector[0] += -3.0 if mode == "linear" else 3.0
        rows.append(vector.astype(np.float32))
        modes.append(mode)
        samples.append(
            SampleMeta(
                generation_id=index,
                topic=f"topic-{index % 4}",
                topic_idx=index % 4,
                mode=mode,
                mode_idx=0 if mode == "linear" else 1,
                num_generated_tokens=90,
                file_stem=f"gen_{index:03d}",
            )
        )
    X = np.stack(rows)
    return Run4Data(
        block_features={"gate_features": X},
        group_features={},
        all_features=X,
        block_feature_names={"gate_features": np.array(names)},
        samples=samples,
        modes=np.array(modes),
        topics=np.array([s.topic for s in samples]),
        mode_indices=np.array([s.mode_idx for s in samples], dtype=np.int64),
        topic_indices=np.array([s.topic_idx for s in samples], dtype=np.int64),
    )


def test_names_and_columns_must_agree_or_the_cut_is_refused() -> None:
    data = synthetic_run(GF_NAMES)
    assert feature_names_for(data, "gate_features") == GF_NAMES

    data.block_feature_names["gate_features"] = np.array(GF_NAMES[:-1])
    with pytest.raises(ValueError, match="would map names onto the wrong columns"):
        feature_names_for(data, "gate_features")

    data.block_feature_names["gate_features"] = np.array([])
    with pytest.raises(KeyError, match="no feature names"):
        feature_names_for(data, "gate_features")


def test_a_cut_scores_every_part_and_the_whole_family() -> None:
    data = synthetic_run(GF_NAMES)
    results = decompose_family(
        data, "gate_features", GF_NAMES, classify_gate_features
    )
    assert FULL_FAMILY in results
    assert results[FULL_FAMILY].n_features == len(GF_NAMES)
    assert results["gf_sparsity"].n_features == 3
    assert results["gf_sparsity"].accuracy > 0.8, (
        "the planted signal is in the sparsity columns, so its cut has to find it"
    )
    assert results["gf_drift"].accuracy < results[FULL_FAMILY].accuracy


def test_an_empty_subset_returns_a_reason_rather_than_an_accuracy() -> None:
    data = synthetic_run(GF_NAMES)
    X = data.get_block("gate_features")
    empty = accuracy_on_subset(X, data.modes, np.zeros(X.shape[1], dtype=bool))
    assert empty.n_features == 0 and empty.accuracy == 0.0
    assert empty.error == "no features"


def test_a_run_is_decomposed_for_every_family_it_carries_a_convention_for() -> None:
    data = synthetic_run(GF_NAMES)
    out = decompose_run(data)
    assert set(out) == {"gate_features_by_signal"}
    document = decomposition_document({"8b_v2": out})
    assert document["8b_v2"]["gate_features_by_signal"]["gf_sparsity"]["n_features"] == 3


def test_a_family_whose_names_the_bank_cannot_assign_is_skipped_with_a_reason() -> None:
    data = synthetic_run(GF_NAMES)
    data.block_feature_names.clear()
    assert decompose_run(data) == {}, "a family with no usable names is not decomposed"


def test_a_filtered_pass_writes_beside_the_full_one(tmp_path: Path) -> None:
    assert default_output_path(tmp_path, mode_filter=None).name == "subfamily_decomp.json"
    assert (
        default_output_path(tmp_path, mode_filter=["a", "b", "c", "d", "e"]).name
        == "subfamily_decomp_5way.json"
    )
