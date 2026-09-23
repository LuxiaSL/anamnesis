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
    classify_signal,
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


def test_one_region_under_two_spellings_is_one_sub_family() -> None:
    """The prompt region is written ``sysprompt`` in one bank and ``prompt`` in another.

    Both are the mass a layer puts on the text it was given rather than on what it
    generated, so both read to one sub-family. A rule that knew only the older spelling
    would leave the newer corpus's whole prompt-allocation signal in the unknown bucket,
    which reads as a signal the family does not carry.
    """
    for spelling in ("attn_flow_L8_sysprompt_mass_mean", "attn_flow_L8_prompt_mass_w0_std"):
        assert classify_attention_flow(spelling) == "af_sysprompt_mass"
    for spelling in ("attn_flow_L8_region_sysprompt_mean", "attn_flow_L8_region_prompt_mean"):
        assert classify_attention_flow(spelling) == "af_region_sysprompt"
    for spelling in ("attn_flow_L8_sysprompt_decay_rate", "attn_flow_L8_prompt_decay_rate"):
        assert classify_attention_flow(spelling) == "af_sysprompt_decay"


def test_gate_names_keep_cross_layer_features_out_of_a_layer() -> None:
    assert classify_gate_features("gate_cross_layer_agreement") == "gf_cross_layer"
    assert classify_gate_features("gate_layer_sparsity_diversity") == "gf_cross_layer"
    assert classify_gate_features("gate_L16_sparsity_mean") == "gf_sparsity"
    assert classify_gate_features("some_sparsity_thing") == "gf_sparsity"
    assert classify_gate_features("gate_mystery") == "gf_unknown"


def test_a_gate_signal_is_labelled_by_its_whole_name() -> None:
    """The label is the signal, not the part of it before the first underscore.

    ``eff_dim`` and ``topk_overlap`` are two words each. Cutting them at the underscore
    labels the effective dimension ``gf_eff``, which is not a signal anything is called
    and not a name a reader can look up.
    """
    assert classify_gate_features("gate_L16_eff_dim_std") == "gf_eff_dim"
    assert classify_gate_features("gate_L16_topk_overlap_mean") == "gf_topk_overlap"


def test_the_classifier_table_names_the_families_that_have_a_convention() -> None:
    assert set(SUBFAMILY_CLASSIFIERS) == {
        "attention_flow",
        "gate_features",
    }


BANKED_SIGNALS: dict[str, tuple[str, ...]] = {
    # The four core blocks, whose names carry no family prefix at all.
    "activation_norm": ("activation_norm_mean_L0", "activation_norm_traj3_L16"),
    "attn_entropy": ("attn_entropy_mean_L0", "attn_entropy_std_L16"),
    "head_agreement": ("head_agreement_mean_L0", "head_agreement_std_L16"),
    "delta_cosine": ("delta_cosine_mean_L0",),
    "delta_norm": ("delta_norm_mean_L0", "delta_norm_std_L16"),
    "cache_anchor_strength": ("cache_anchor_strength_L0",),
    "cache_attn_decay_rate": ("cache_attn_decay_rate_L0",),
    "cache_coverage": ("cache_cache_coverage_L0",),
    "cache_lookback_ratio": ("cache_lookback_ratio_L0",),
    "cache_recency": ("cache_recency_bias_L0", "cache_recency_traj2_L16"),
    "cache_sink_mass": ("cache_sink_mass_L0",),
    "kv_key_drift": ("kv_key_drift_L0",),
    "kv_key_eff_dim": ("kv_key_eff_dim_L0",),
    "kv_key_novelty": ("kv_key_novelty_mean_L0", "kv_key_novelty_traj4_L16"),
    "kv_key_spread": ("kv_key_spread_L0",),
    "epoch_max_transition": ("epoch_max_transition_mean",),
    "epoch_n_transitions": ("epoch_n_transitions_mean",),
    "epoch_regularity": ("epoch_regularity_mean",),
    "spectral_fiedler": ("spectral_fiedler_L0",),
    "spectral_hfer": ("spectral_hfer_L0",),
    "spectral_smoothness": ("spectral_smoothness_L0",),
    "spectral_entropy": ("spectral_spectral_entropy_L0",),
    "cross_layer_keys": ("cross_layer_early_late_agreement", "cross_layer_overall_coherence"),
    "logit_entropy": ("logit_entropy_mean", "logit_entropy_traj0"),
    "top1_prob": ("top1_prob_mean",),
    "top5_mass": ("top5_mass_mean",),
    "chosen_rank": ("mean_chosen_rank", "std_chosen_rank"),
    "surprise": ("mean_surprise", "surprise_traj0"),
    "surprise_boundary": ("surprise_boundary_count",),
    # The engineered families, in the spellings the banks carry.
    "af_sysprompt_mass": ("attn_flow_L0_prompt_mass_mean", "attn_flow_L20_prompt_mass_w2_std"),
    "af_sysprompt_decay": ("attn_flow_L0_prompt_decay_rate",),
    "af_region_sysprompt": ("attn_flow_L0_region_prompt_mean",),
    "af_region_early_gen": ("attn_flow_L0_region_early_gen_mean",),
    "af_region_mid_gen": ("attn_flow_L0_region_mid_gen_std",),
    "af_region_recent": ("attn_flow_L0_region_recent_mean",),
    "af_recency_bias": ("attn_flow_L0_recency_bias_w0_mean", "af_L8_recency_bias"),
    "af_head_diversity_recency": ("attn_flow_L0_head_diversity_recency",),
    "af_head_diversity_sysprompt": ("attn_flow_L0_head_diversity_prompt",),
    "gf_sparsity": ("gate_L0_sparsity_mean", "gf_L16_sparsity_mean"),
    "gf_drift": ("gate_L20_drift_w2_std",),
    "gf_eff_dim": ("gate_L0_eff_dim_mean",),
    "gf_topk_overlap": ("gate_L0_topk_overlap_mean",),
    "gf_cross_layer": ("gate_cross_layer_agreement", "gate_cross_layer_sparsity_diversity"),
    "rt_velocity": ("res_traj_L8_velocity_norm_mean", "rt_L24_velocity_norm"),
    "rt_acceleration": ("res_traj_L8_acceleration_norm_mean",),
    "rt_direction_change": ("res_traj_L20_direction_change_w2_std",),
    "rt_directness": ("res_traj_L8_directness",),
    "ph_head_entropy": ("ph_L0_head_entropy_mean", "ph_L20_head_entropy_min"),
    "ph_head_role": ("ph_L0_head_role_stability",),
    "ph_key_spread": ("ph_L0_kv_key_spread_head_mean",),
    "ph_sink_head": ("ph_L0_sink_head_std",),
    "td_attn_entropy": ("td_L0_attn_entropy_w0_mean", "td_L20_attn_entropy_bandwidth"),
    "td_head_agreement": ("td_L0_head_agreement_w0_mean",),
    "td_key_drift": ("td_L0_key_drift_w0_mean",),
    "td_key_novelty": ("td_L0_key_novelty_spectral_centroid",),
    "td_lookback_ratio": ("td_L0_lookback_ratio_w3_slope",),
    "cp_t0": ("cp_L8_t0_d0",),
    "cp_t3": ("cp_L20_t3_d16",),
    "pca_t0": ("pca_L8_t0_c0",),
    "pca_t4": ("pca_L8_t4_c49",),
}
"""Every sub-family a banked signature file holds, with names spelled as the banks spell
them. Held here as data, because a test that read a bank would only run where that bank
is, and the whole failure this pins is a rule keyed on a spelling nothing carries."""


@pytest.mark.parametrize(
    ("name", "subfamily"),
    [(name, subfamily) for subfamily, names in BANKED_SIGNALS.items() for name in names],
)
def test_a_banked_name_is_grouped_under_the_signal_it_reads(name: str, subfamily: str) -> None:
    assert classify_signal(name) == subfamily


def test_every_banked_spelling_places_and_the_labels_do_not_collide() -> None:
    """No name falls to a fallback, and no two signals share a label.

    Placement alone is not enough: a rule whose mark is too wide swallows a second
    signal, and the table it feeds then attributes two signals' importance to one.
    """
    placed = {
        name: classify_signal(name)
        for names in BANKED_SIGNALS.values()
        for name in names
    }
    unplaced = [
        name
        for name, signal in placed.items()
        if signal.startswith("other(") or signal.endswith("_unknown")
    ]
    assert unplaced == []
    assert set(placed.values()) == set(BANKED_SIGNALS)


# The windowed family's naming grid: every layer, every signal, every operator over the
# window. This is the shape the banked file has, which is what makes the count below a
# measurement of the grouping rather than of the sample.
GRID_LAYERS = (0, 8, 16, 20, 24, 28, 31)
GRID_SIGNALS = ("attn_entropy", "head_agreement", "key_drift", "key_novelty", "lookback_ratio")
GRID_OPERATORS = (
    *(f"w{window}_{statistic}" for window in range(4) for statistic in ("mean", "std", "slope")),
    "dominant_freq",
    "spectral_centroid",
    "bandwidth",
    "low_band_energy",
    "mid_band_energy",
    "high_band_energy",
)


def test_a_familys_whole_naming_grid_collapses_to_its_signals() -> None:
    """The failure this pins: a classifier that places nothing degenerates into the name.

    Returning a fallback built from the name gives one bucket per feature, which is the
    opposite of a sub-family grouping and reads as a table of hundreds of findings. Over
    a family's whole grid the right answer is one bucket per signal, whatever the layer
    and whatever the operator — so the count is asserted and not only the labels.
    """
    names = [
        f"td_L{layer}_{signal}_{operator}"
        for layer in GRID_LAYERS
        for signal in GRID_SIGNALS
        for operator in GRID_OPERATORS
    ]
    assert len(names) == 630
    assert {classify_signal(name) for name in names} == {
        f"td_{signal}" for signal in GRID_SIGNALS
    }


def test_a_name_outside_every_convention_is_returned_as_itself() -> None:
    """Unplaced, and findable: the bucket names the column so a reader can go look."""
    assert classify_signal("a_signal_no_rule_names") == "other(a_signal_no_rule_names)"


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
