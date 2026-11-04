"""A preset may set how many span positions the residual PCA is sampled at."""

from __future__ import annotations

from anamnesis.config import resolve_preset
from anamnesis.config.experiment import ExtractionConfig
from anamnesis.extraction.replay_config import native_replay_configs


def test_unset_keeps_the_extraction_default_and_the_extraction_identity():
    preset = resolve_preset("8b")
    assert preset.pca_temporal_samples is None
    extraction, _ = native_replay_configs(preset)
    assert extraction.pca_temporal_samples == ExtractionConfig.model_fields["pca_temporal_samples"].default
    # an unset field leaves the dumped extraction (part of every lane id) as it was
    assert "pca_temporal_samples" in extraction.model_dump(mode="json")


def test_a_preset_value_reaches_the_extraction():
    preset = resolve_preset("8b").model_copy(update={"pca_temporal_samples": 17})
    extraction, _ = native_replay_configs(preset)
    assert extraction.pca_temporal_samples == 17
