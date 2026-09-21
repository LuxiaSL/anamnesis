"""Sub-family decomposition: the name classifiers, and the mapping they rest on.

A decomposition is a map from feature names to column indices, so the failure mode is not
a wrong accuracy — it is a right accuracy attributed to the wrong part of a family. The
tests are therefore mostly about names and alignment:

  * each family's classifier reads its own naming convention, including the two cases that
    need a fourth field (key drift against key novelty) and the one that has no layer to
    belong to (the cross-layer gate features);
  * a name the convention does not cover reads as that family's unknown bucket rather than
    being silently filed under a real sub-family;
  * feature names come from the loader, and a **length mismatch between the names and the
    matrix is refused** — that is the exact condition under which a decomposition would map
    indices onto the wrong features and still print a plausible table;
  * a sub-family with no features returns zero with a reason rather than an accuracy;
  * the whole family is scored alongside its parts, because a part is only readable against
    the whole;
  * the temporal-dynamics operator groups are nested, so each is the previous plus a window,
    and the feature counts have to grow with them;
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
    TD_COARSE_GROUPS,
    TD_OPERATOR_GROUPS,
    accuracy_on_subset,
    classify_attention_flow,
    classify_contrastive_projection,
    classify_gate_features,
    classify_temporal_dynamics,
    classify_temporal_operator,
    decompose_by_groups,
    decompose_family,
    decompose_run,
    decomposition_document,
    default_output_path,
    feature_names_for,
)


def test_temporal_dynamics_names_are_read_down_to_the_key_signal() -> None:
    assert classify_temporal_dynamics("td_L16_attn_entropy_w0_mean") == "td_T2_attn_entropy"
    assert classify_temporal_dynamics("td_L16_head_agreement_w1_std") == "td_T2_head_agreement"
    assert classify_temporal_dynamics("td_L16_key_drift_w0_mean") == "td_T2.5_key_drift"
    assert classify_temporal_dynamics("td_L16_key_novelty_w2_mean") == "td_T2.5_key_novelty"
    assert classify_temporal_dynamics("td_L16_key_other_w0") == "td_T2.5_key"
    assert classify_temporal_dynamics("td_L16_lookback_ratio_w0") == "td_T2.5_lookback_ratio"
    assert classify_temporal_dynamics("td_L16") == "unknown"
    assert classify_temporal_dynamics("td_L16_mystery_w0") == "unknown"


def test_the_operator_cut_names_windows_and_pools_the_spectrum() -> None:
    assert classify_temporal_operator("td_L16_attn_entropy_w0_mean") == "w0"
    assert classify_temporal_operator("td_L16_attn_entropy_w3_std") == "w3"
    for spectral in ("dominant_freq", "spectral_centroid", "bandwidth", "band_energy"):
        assert classify_temporal_operator(f"td_L16_attn_entropy_{spectral}") == "stft"
    assert classify_temporal_operator("td_L16_attn_entropy_mean") == "other"


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


def test_the_learned_projection_is_cut_by_temporal_position_only() -> None:
    assert classify_contrastive_projection("cp_L16_t3_d07") == "cp_t3"
    assert classify_contrastive_projection("cp_L16_d07") == "cp_unknown"


def test_the_classifier_table_names_the_families_that_have_a_convention() -> None:
    assert set(SUBFAMILY_CLASSIFIERS) == {
        "temporal_dynamics",
        "attention_flow",
        "gate_features",
        "contrastive_projection",
    }


TD_NAMES = [
    "td_L16_attn_entropy_w0_mean",
    "td_L16_attn_entropy_w1_mean",
    "td_L16_head_agreement_w0_mean",
    "td_L16_key_drift_w0_mean",
    "td_L16_key_novelty_w2_mean",
    "td_L16_lookback_ratio_w3_mean",
    "td_L16_attn_entropy_dominant_freq",
    "td_L16_key_drift_spectral_centroid",
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
        tier_features={"temporal_dynamics": X},
        group_features={},
        all_features=X,
        tier_feature_names={"temporal_dynamics": np.array(names)},
        samples=samples,
        modes=np.array(modes),
        topics=np.array([s.topic for s in samples]),
        mode_indices=np.array([s.mode_idx for s in samples], dtype=np.int64),
        topic_indices=np.array([s.topic_idx for s in samples], dtype=np.int64),
    )


def test_names_and_columns_must_agree_or_the_cut_is_refused() -> None:
    data = synthetic_run(TD_NAMES)
    assert feature_names_for(data, "temporal_dynamics") == TD_NAMES

    data.tier_feature_names["temporal_dynamics"] = np.array(TD_NAMES[:-1])
    with pytest.raises(ValueError, match="would map names onto the wrong columns"):
        feature_names_for(data, "temporal_dynamics")

    data.tier_feature_names["temporal_dynamics"] = np.array([])
    with pytest.raises(KeyError, match="no feature names"):
        feature_names_for(data, "temporal_dynamics")


def test_a_cut_scores_every_part_and_the_whole_family() -> None:
    data = synthetic_run(TD_NAMES)
    results = decompose_family(
        data, "temporal_dynamics", TD_NAMES, classify_temporal_dynamics
    )
    assert FULL_FAMILY in results
    assert results[FULL_FAMILY].n_features == len(TD_NAMES)
    assert results["td_T2_attn_entropy"].n_features == 3
    assert results["td_T2_attn_entropy"].accuracy > 0.8, (
        "the planted signal is in the attn-entropy columns, so its cut has to find it"
    )
    assert results["td_T2.5_key_novelty"].accuracy < results[FULL_FAMILY].accuracy


def test_an_empty_subset_returns_a_reason_rather_than_an_accuracy() -> None:
    data = synthetic_run(TD_NAMES)
    X = data.get_tier("temporal_dynamics")
    empty = accuracy_on_subset(X, data.modes, np.zeros(X.shape[1], dtype=bool))
    assert empty.n_features == 0 and empty.accuracy == 0.0
    assert empty.error == "no features"


def test_the_operator_groups_are_nested_so_their_widths_grow() -> None:
    data = synthetic_run(TD_NAMES)
    results = decompose_by_groups(
        data, "temporal_dynamics", TD_NAMES, classify_temporal_operator, TD_OPERATOR_GROUPS
    )
    widths = [results[group].n_features for group in ("td_w0_only", "td_w0_w1", "td_windowed")]
    assert widths == sorted(widths), "each group is the previous one plus a window"
    assert results["td_stft_only"].n_features == 2


def test_the_coarse_cut_groups_the_signals_by_substrate() -> None:
    data = synthetic_run(TD_NAMES)
    results = decompose_by_groups(
        data, "temporal_dynamics", TD_NAMES, classify_temporal_dynamics, TD_COARSE_GROUPS
    )
    assert set(results) == set(TD_COARSE_GROUPS)
    assert results["td_T2"].n_features == 4
    assert results["td_T2.5"].n_features == 4


def test_a_run_is_decomposed_for_every_family_it_carries_a_convention_for() -> None:
    data = synthetic_run(TD_NAMES)
    out = decompose_run(data)
    assert set(out) == {"temporal_dynamics_by_signal", "td_coarse", "td_by_operator"}
    document = decomposition_document({"8b_v2": out})
    assert document["8b_v2"]["td_coarse"]["td_T2"]["n_features"] == 4


def test_a_family_whose_names_the_bank_cannot_assign_is_skipped_with_a_reason() -> None:
    data = synthetic_run(TD_NAMES)
    data.tier_feature_names.clear()
    assert decompose_run(data) == {}, "a family with no usable names is not decomposed"


def test_a_filtered_pass_writes_beside_the_full_one(tmp_path: Path) -> None:
    assert default_output_path(tmp_path, mode_filter=None).name == "subfamily_decomp.json"
    assert (
        default_output_path(tmp_path, mode_filter=["a", "b", "c", "d", "e"]).name
        == "subfamily_decomp_5way.json"
    )
