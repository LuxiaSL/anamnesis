"""The result schemas, split per section, still describe one wire format.

Splitting one 1409-line schema module into twelve introduces exactly one class of
risk: a name that fails to reach its caller, or a section whose models drift out of
the composite. These tests close that off:

  * every model defined in a section module is re-exported by the package, and
    ``__all__`` and the modules agree in both directions;
  * the composite declares one field per section and its types are the sections'
    own result models, so a section renamed in one place fails here;
  * the two reshaping models — classification's tier-keys-at-top-level and
    contrastive's on-disk key vocabulary — round-trip, because those are the
    places where the model's shape and the file's shape are not the same shape;
  * ``extra="forbid"`` is live on every model, which is the whole reason a
    checkpoint can be trusted after validation.

CPU only; no data, no model.
"""

from __future__ import annotations

import importlib

import pytest
from pydantic import BaseModel, ValidationError

from anamnesis.analysis.gauntlet import SECTION_MODELS, SECTIONS
from anamnesis.analysis.gauntlet import schemas
from anamnesis.analysis.gauntlet.schemas.base import _FORBID

SECTION_MODULES = (
    "ccgp",
    "classification",
    "clustering",
    "contrastive",
    "integrity",
    "intrinsic_dimension",
    "manifold_geometry",
    "results",
    "scorecard",
    "semantic",
    "tier_ablation",
    "topology",
)


def models_in(module_name: str) -> dict[str, type[BaseModel]]:
    module = importlib.import_module(f"anamnesis.analysis.gauntlet.schemas.{module_name}")
    return {
        name: obj
        for name, obj in vars(module).items()
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj.__module__ == module.__name__
    }


def test_every_section_model_is_exported_and_nothing_is_exported_twice() -> None:
    defined: dict[str, str] = {}
    for module_name in SECTION_MODULES:
        for name in models_in(module_name):
            assert name not in defined, f"{name} defined in both {defined.get(name)} and {module_name}"
            defined[name] = module_name
    assert set(defined) == set(schemas.__all__), (
        f"missing from __all__: {sorted(set(defined) - set(schemas.__all__))}; "
        f"in __all__ but undefined: {sorted(set(schemas.__all__) - set(defined))}"
    )
    for name in schemas.__all__:
        assert getattr(schemas, name).__name__ == name


def test_the_composite_covers_every_section_the_orchestrator_runs() -> None:
    fields = schemas.AnalysisResults.model_fields
    section_keys = {spec.key for spec in SECTIONS}
    assert section_keys <= set(fields), sorted(section_keys - set(fields))
    assert set(SECTION_MODELS) == section_keys
    # The rehydration registry and the composite must name the same model for a
    # key, or a resumed run validates against a different schema than a fresh one.
    for key, model in SECTION_MODELS.items():
        annotation = str(fields[key].annotation)
        assert model.__name__ in annotation, f"{key}: {model.__name__} not in {annotation}"


def test_every_model_forbids_unknown_keys() -> None:
    for name in schemas.__all__:
        model = getattr(schemas, name)
        assert model.model_config.get("extra") == "forbid", name
    assert _FORBID["extra"] == "forbid"


def test_a_stray_key_in_a_checkpointed_section_is_an_error_not_a_shrug() -> None:
    good = {"nan": 0, "inf": 0}
    assert schemas.NanInfCount.model_validate(good).nan == 0
    with pytest.raises(ValidationError):
        schemas.NanInfCount.model_validate({**good, "nans": 3})


def test_error_stubs_round_trip_through_exclude_none() -> None:
    stub = schemas.ClassifierWithConfusionResult.model_validate({"error": "analogical absent"})
    assert stub.model_dump(exclude_none=True) == {"error": "analogical absent"}
    length_only = schemas.LengthOnlyResult.model_validate(
        {"accuracy": None, "error": "no length metadata"}
    )
    assert length_only.model_dump(mode="json") == {
        "accuracy": None,
        "error": "no length metadata",
    }


def tier_classification_payload() -> dict[str, object]:
    return {
        "rf_5way": {"accuracy": 0.8, "fold_accuracies": [0.8], "confusion_matrix": [[1]],
                    "labels": ["linear"]},
        "topic_heldout": {"accuracy": 0.7, "fold_accuracies": [0.7], "n_groups": 2},
        "linear_probe": {"accuracy": 0.6, "fold_accuracies": [0.6]},
        "pairwise_binary": {"linear_vs_socratic": {"accuracy": 0.9, "fold_accuracies": [0.9]}},
        "rf_4way_no_analogical": {"error": "analogical absent"},
    }


def test_classification_reshapes_tier_keys_both_ways() -> None:
    wire = {
        "T2+T2.5": tier_classification_payload(),
        "combined": tier_classification_payload(),
        "length_only": {"accuracy": None, "error": "no length metadata"},
    }
    parsed = schemas.ClassificationResult.model_validate(wire)
    assert set(parsed.by_tier) == {"T2+T2.5", "combined"}
    assert parsed.by_tier["combined"].rf_5way.accuracy == 0.8
    assert parsed.length_only is not None
    out = parsed.model_dump(mode="json")
    assert set(out) == set(wire)
    assert out["T2+T2.5"]["topic_heldout"]["n_groups"] == 2
    # Round-tripping twice is what a resumed run does, so it must be a fixed point.
    assert schemas.ClassificationResult.model_validate(out).model_dump(mode="json") == out


def test_a_classification_result_with_no_length_baseline_omits_the_key() -> None:
    parsed = schemas.ClassificationResult.model_validate({"T1": tier_classification_payload()})
    assert parsed.length_only is None
    assert "length_only" not in parsed.model_dump(mode="json")


def test_contrastive_reads_the_frozen_on_disk_key_vocabulary() -> None:
    """``T2.5_alone`` is not a Python identifier, so the key on disk and the
    field in the model differ. The translation is the contract: a banked file
    validates, and serializing gives that file's spelling back unchanged."""
    on_disk = {
        "T2_alone": 0.5,
        "T2.5_alone": 0.6,
        "T2+T2.5_pair": 0.75,
        "best_individual": 0.6,
        "gain": 0.15,
        "combined_knn": 0.7,
        "T2+T2.5_beats_combined": True,
    }
    parsed = schemas.ContrastiveSuperAdditivity.model_validate(on_disk)
    assert parsed.T2_5_alone == 0.6
    assert parsed.T2_T2_5_pair == 0.75
    assert parsed.T2_T2_5_beats_combined is True
    assert parsed.model_dump(mode="json") == on_disk
    # The Python spelling also validates, so a freshly constructed result and a
    # reloaded one are the same object.
    python_spelling = {
        "T2_alone": 0.5, "T2_5_alone": 0.6, "T2_T2_5_pair": 0.75,
        "best_individual": 0.6, "gain": 0.15, "combined_knn": 0.7,
        "T2_T2_5_beats_combined": True,
    }
    assert schemas.ContrastiveSuperAdditivity.model_validate(python_spelling) == parsed
