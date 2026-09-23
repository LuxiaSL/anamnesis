"""The mode registry: pinned prompts, pinned indices, extension, and the refusals.

The mode prompts are the protocol. A label in banked data means the exact text the
registry composes, so :data:`SET_PROMPTS_SHA256` pins that text and
:data:`MODE_INDICES` pins the label order: a mode's index reaches
:func:`anamnesis.extraction.generation_runner.make_seed`, so every banked
generation's seed depends on it and a changed index silently renames what a
coordinate produced. Both fail here rather than surfacing as an unexplained shift in
a later corpus.

The second half is the extension path, tested the way a stranger meets it: a
registry file outside the package, named in the environment, holding a set this
package never shipped — and then the refusals that keep an added set from
redefining a shipped one or reusing an index a parent already spent.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, get_args

import pytest

import anamnesis.modes as modes
from anamnesis.config import paths
from anamnesis.config.experiment import ProcessingMode
from anamnesis.modes import (
    CORE_MODE_SET,
    EXTENDED_MODE_SET,
    MODE_SETS_ENV,
    MODE_SETS_FILE,
    PROMPT_SWAP_PAIRS,
    ModeRegistryError,
    PromptSwapPair,
    UnknownModeSetError,
    easy_modes,
    format_constraint,
    hard_modes,
    label_vocabulary,
    load_mode_registry,
    mapping_wildcards,
    mode_indices,
    mode_mapping,
    mode_prompts,
    mode_set,
    mode_set_names,
)

CORE_MODES = ("linear", "analogical", "socratic", "contrastive", "dialectical")
ADDED_MODES = ("structured", "compressed", "associative")

MODE_INDICES: dict[str, int] = {
    "linear": 0,
    "analogical": 1,
    "socratic": 2,
    "contrastive": 3,
    "dialectical": 4,
    "structured": 5,
    "compressed": 6,
    "associative": 7,
}
"""Every shipped mode's label index. A seed coordinate depends on each of these."""

SET_PROMPTS_SHA256 = {
    "run4": "be2b0b7ab4fca49143901cf715e85a05bde53e5f56d4ba634097dde5c42a7432",
    "mixed": "d6a3f05a096ba3f799e507c5dc0c75167e29cef4f33146a0700c849a4b8693ad",
}
"""Digest of each set's composed prompts: instruction plus format constraint."""

MODE_SETS_FILE_SHA256 = "734a45ebd7b3bb0a86ef08796258cf080501ca92db12dbb85a2a1ca7f6adb887"
"""The registry file's own bytes, so a change to a gloss or a pair also fails here."""

SWAP_PAIRS_SHA256 = "0fea66a52815d3ee7530afb9b358218cf53cc977b4003d223d97a86eed9bd0a3"

PROCESS_MAPPING = "process_to_format"
PROCESS_VOCABULARY = "process5"

RECORD_ONLY_MODULES = (
    "anamnesis.modes.run3_original_modes",
    "anamnesis.modes.run4_modes",
    "anamnesis.modes.extended_modes",
)
"""Modules no longer backing a mode set: the sets are data, and run 3's is not shipped."""

EXPECTED_EXPORTS = {
    "CORE_MODE_SET",
    "DEFAULT_MODE_MAPPING",
    "DEFAULT_USER_TEMPLATE",
    "EXTENDED_MODE_SET",
    "MODE_SETS_ENV",
    "MODE_SETS_FILE",
    "Label",
    "LabelVocabulary",
    "MappingReference",
    "Mode",
    "ModeMapping",
    "ModeRegistry",
    "ModeRegistryError",
    "ModeRegistryFile",
    "ModeSet",
    "ModeSetSpec",
    "PROMPT_SWAP_PAIRS",
    "PromptSwapPair",
    "UnknownModeSetError",
    "easy_modes",
    "format_constraint",
    "hard_modes",
    "label_vocabulary",
    "load_mode_registry",
    "mapping_wildcards",
    "mode_indices",
    "mode_mapping",
    "mode_prompts",
    "mode_set",
    "mode_set_names",
}

ADDED_SET: dict[str, Any] = {
    "name": "my-modes",
    "description": "A researcher's own two modes.",
    "format_constraint": " Answer in one paragraph.",
    "modes": [
        {"name": "enumerative", "index": 0, "instruction": "List the parts, then count them."},
        {"name": "narrative", "index": 1, "instruction": "Tell it as a story with a middle."},
    ],
}


def digest(payload: object) -> str:
    """A stable digest of mode data: sorted keys, unescaped text, UTF-8 bytes."""
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_registry(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def added(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A user's own mode-set file, named in the environment."""
    path = write_registry(tmp_path / "my_modes.json", {"mode_sets": {"my-modes": ADDED_SET}})
    monkeypatch.setenv(MODE_SETS_ENV, str(path))
    return path


# ── What the package offers ───────────────────────────────────────────────────
def test_the_package_exports_exactly_the_registry_surface() -> None:
    assert set(modes.__all__) == EXPECTED_EXPORTS
    for name in EXPECTED_EXPORTS:
        assert hasattr(modes, name)


def test_no_export_names_a_record_only_mode_set() -> None:
    for name in dir(modes):
        assert "run3" not in name.lower()
    assert "run3" not in modes.__doc__.lower() if modes.__doc__ else True


@pytest.mark.parametrize("module", RECORD_ONLY_MODULES)
def test_a_record_only_module_is_not_importable(module: str) -> None:
    assert importlib.util.find_spec(module) is None
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


@pytest.mark.parametrize("module", RECORD_ONLY_MODULES)
def test_no_file_backs_a_record_only_module(module: str) -> None:
    leaf = module.rsplit(".", 1)[-1]
    assert not (paths.package_root() / "modes" / f"{leaf}.py").exists()


def test_importing_the_modes_does_not_pull_the_configuration() -> None:
    """The mode sets are data; loading them starts nothing else."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import anamnesis.modes, sys; "
            "assert 'anamnesis.config' not in sys.modules, sorted(sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


# ── The protocol, pinned ──────────────────────────────────────────────────────
def test_the_registry_file_matches_its_pinned_digest() -> None:
    assert hashlib.sha256(MODE_SETS_FILE.read_bytes()).hexdigest() == MODE_SETS_FILE_SHA256


def test_the_shipped_sets_are_the_two_of_record() -> None:
    assert mode_set_names() == (CORE_MODE_SET, EXTENDED_MODE_SET)


def test_the_five_core_modes_are_the_protocol() -> None:
    assert mode_set(CORE_MODE_SET).names() == CORE_MODES
    assert digest(mode_prompts(CORE_MODE_SET)) == SET_PROMPTS_SHA256[CORE_MODE_SET]


def test_the_eight_mode_set_extends_the_five_without_disturbing_them() -> None:
    core, extended = mode_set(CORE_MODE_SET), mode_set(EXTENDED_MODE_SET)
    assert extended.names() == CORE_MODES + ADDED_MODES
    assert extended.extends == CORE_MODE_SET
    for name in CORE_MODES:
        assert extended.prompts()[name] == core.prompts()[name]
        assert extended.indices()[name] == core.indices()[name]
    assert digest(mode_prompts(EXTENDED_MODE_SET)) == SET_PROMPTS_SHA256[EXTENDED_MODE_SET]


@pytest.mark.parametrize(("mode", "index"), sorted(MODE_INDICES.items()))
def test_every_shipped_modes_index_is_the_one_its_seeds_were_drawn_under(
    mode: str, index: int
) -> None:
    """A changed index renames what a banked generation coordinate produced."""
    assert mode_indices(EXTENDED_MODE_SET)[mode] == index


def test_the_label_order_is_declared_rather_than_derived() -> None:
    """Sorting or dict order would be an accident; the index field is the order."""
    for name in mode_set_names():
        row = mode_set(name)
        assert [mode.index for mode in row.modes] == list(range(len(row.modes)))
        assert row.names() != tuple(sorted(row.names()))


def test_every_mode_prompt_carries_the_format_constraint() -> None:
    clause = format_constraint(CORE_MODE_SET)
    assert clause.strip().startswith("Write in flowing paragraphs.")
    assert format_constraint(EXTENDED_MODE_SET) == clause, "the extending set inherits it"
    for name, prompt in mode_prompts(EXTENDED_MODE_SET).items():
        assert prompt.endswith(clause), name


def test_the_hard_subset_is_the_five_and_the_easy_one_is_what_the_eight_adds() -> None:
    assert hard_modes() == frozenset(CORE_MODES)
    assert easy_modes() == frozenset(ADDED_MODES)
    assert mode_set(CORE_MODE_SET).hard
    assert not mode_set(EXTENDED_MODE_SET).hard


def test_the_narrow_mode_literal_is_the_core_sets_labels() -> None:
    """A type that named other modes would type-check code the protocol cannot run."""
    assert get_args(ProcessingMode) == mode_set(CORE_MODE_SET).names()


# ── The process-mode vocabulary, and the mapping that predicts it ─────────────
def test_every_label_the_mapping_names_resolves_to_a_gloss() -> None:
    """The reader's complaint this row exists to answer: what is a deliberative mode?"""
    mapping = mode_mapping(PROCESS_MAPPING)
    source = label_vocabulary(mapping.source)
    target = label_vocabulary(mapping.target)
    assert mapping.source == PROCESS_VOCABULARY
    assert mapping.target == CORE_MODE_SET
    for label in (*source.names(), *mapping.pairs):
        assert source.gloss(label).strip()
    for label in (*target.names(), *mapping.pairs.values()):
        assert target.gloss(label).strip()
    for undefined in ("deliberative", "pedagogical"):
        assert undefined in source.names()
        assert source.gloss(undefined).strip()


def test_the_process_vocabulary_says_why_it_cannot_be_run() -> None:
    vocabulary = label_vocabulary(PROCESS_VOCABULARY)
    assert vocabulary.runnable_note.strip()
    assert PROCESS_VOCABULARY not in mode_set_names(), "it ships no prompts, so it is not a set"


def test_the_colliding_label_names_belong_to_two_vocabularies() -> None:
    """Three process labels spell modes of the eight-mode set and are not them."""
    process = set(label_vocabulary(PROCESS_VOCABULARY).names())
    extended = set(mode_set(EXTENDED_MODE_SET).names())
    assert process & extended == {"associative", "structured", "compressed"}
    assert process - extended == {"deliberative", "pedagogical"}


def test_the_reverse_direction_is_the_mapping_inverted() -> None:
    mapping = mode_mapping(PROCESS_MAPPING)
    assert mapping.reverse_pairs() == {v: k for k, v in mapping.pairs.items()}
    forward, reverse = mapping_wildcards(PROCESS_MAPPING)
    assert forward == "compressed"
    assert reverse == "contrastive"


def test_the_mappings_comparison_travels_with_its_pairs() -> None:
    reference = mode_mapping(PROCESS_MAPPING).reference
    assert reference is not None
    assert reference.model
    assert reference.forward_pair in mode_mapping(PROCESS_MAPPING).pairs
    assert reference.reverse_pair in mode_mapping(PROCESS_MAPPING).reverse_pairs()


def test_a_mode_set_answers_as_a_vocabulary_of_its_own_labels() -> None:
    assert label_vocabulary(CORE_MODE_SET).names() == mode_set(CORE_MODE_SET).names()


# ── Extension ─────────────────────────────────────────────────────────────────
def test_a_user_adds_a_mode_set_by_adding_data(added: Path) -> None:
    assert "my-modes" in mode_set_names()
    row = mode_set("my-modes")
    assert row.names() == ("enumerative", "narrative")
    assert row.indices() == {"enumerative": 0, "narrative": 1}
    assert row.prompts()["enumerative"].endswith(" Answer in one paragraph.")
    assert mode_set(CORE_MODE_SET).names() == CORE_MODES, "the shipped sets are untouched"


def test_an_added_set_can_extend_a_shipped_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_registry(
        tmp_path / "extra.json",
        {
            "mode_sets": {
                "six": {
                    "name": "six",
                    "extends": CORE_MODE_SET,
                    "modes": [
                        {"name": "enumerative", "index": 5, "instruction": "List the parts."}
                    ],
                }
            }
        },
    )
    monkeypatch.setenv(MODE_SETS_ENV, str(path))
    row = mode_set("six")
    assert row.names() == CORE_MODES + ("enumerative",)
    assert row.indices()["enumerative"] == 5
    assert row.prompts()["linear"] == mode_prompts(CORE_MODE_SET)["linear"]
    assert row.format_constraint == format_constraint(CORE_MODE_SET)


def test_a_user_adds_their_own_mapping_between_two_vocabularies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_registry(
        tmp_path / "mine.json",
        {
            "mode_sets": {"my-modes": ADDED_SET},
            "mode_mappings": {
                "mine": {
                    "name": "mine",
                    "source": "my-modes",
                    "target": CORE_MODE_SET,
                    "pairs": {"enumerative": "linear"},
                }
            },
        },
    )
    monkeypatch.setenv(MODE_SETS_ENV, str(path))
    mapping = mode_mapping("mine")
    assert mapping.reverse_pairs() == {"linear": "enumerative"}
    assert mapping.reference is None
    forward, reverse = mapping_wildcards("mine")
    assert forward == "narrative"
    assert reverse is None, "four target labels are unpaired, so there is no single wildcard"


def test_several_override_files_are_read_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = write_registry(tmp_path / "a.json", {"mode_sets": {"my-modes": ADDED_SET}})
    second = write_registry(
        tmp_path / "b.json",
        {
            "label_vocabularies": {
                "banked": {
                    "name": "banked",
                    "labels": [{"name": "terse", "gloss": "Says little."}],
                }
            }
        },
    )
    monkeypatch.setenv(MODE_SETS_ENV, os.pathsep.join([str(first), str(second)]))
    assert load_mode_registry().sources == (MODE_SETS_FILE, first, second)
    assert label_vocabulary("banked").names() == ("terse",)


# ── Refusals ──────────────────────────────────────────────────────────────────
def test_an_added_set_cannot_redefine_a_shipped_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clash = write_registry(
        tmp_path / "clash.json", {"mode_sets": {CORE_MODE_SET: {**ADDED_SET, "name": CORE_MODE_SET}}}
    )
    monkeypatch.setenv(MODE_SETS_ENV, str(clash))
    with pytest.raises(ModeRegistryError) as caught:
        load_mode_registry()
    message = str(caught.value)
    assert str(MODE_SETS_FILE) in message
    assert str(clash) in message


def test_an_extending_set_cannot_reuse_a_parents_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An index is a seed coordinate, so the parent's keep their meaning."""
    path = write_registry(
        tmp_path / "reuse.json",
        {
            "mode_sets": {
                "six": {
                    "name": "six",
                    "extends": CORE_MODE_SET,
                    "modes": [{"name": "enumerative", "index": 2, "instruction": "List."}],
                }
            }
        },
    )
    monkeypatch.setenv(MODE_SETS_ENV, str(path))
    with pytest.raises(ModeRegistryError, match="reuses indices"):
        load_mode_registry()


def test_an_extending_set_cannot_restate_a_parents_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_registry(
        tmp_path / "restate.json",
        {
            "mode_sets": {
                "six": {
                    "name": "six",
                    "extends": CORE_MODE_SET,
                    "modes": [{"name": "linear", "index": 5, "instruction": "Differently."}],
                }
            }
        },
    )
    monkeypatch.setenv(MODE_SETS_ENV, str(path))
    with pytest.raises(ModeRegistryError, match="restates"):
        load_mode_registry()


def test_an_extending_set_cannot_declare_its_own_format_constraint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_registry(
        tmp_path / "constraint.json",
        {
            "mode_sets": {
                "six": {
                    "name": "six",
                    "extends": CORE_MODE_SET,
                    "format_constraint": " Use bullet points.",
                    "modes": [{"name": "enumerative", "index": 5, "instruction": "List."}],
                }
            }
        },
    )
    monkeypatch.setenv(MODE_SETS_ENV, str(path))
    with pytest.raises(ModeRegistryError, match="different protocol"):
        load_mode_registry()


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"modes": []}, "declares no modes"),
        (
            {
                "modes": [
                    {"name": "a", "index": 0, "instruction": "x"},
                    {"name": "a", "index": 1, "instruction": "y"},
                ]
            },
            "names a mode twice",
        ),
        (
            {
                "modes": [
                    {"name": "a", "index": 1, "instruction": "x"},
                    {"name": "b", "index": 2, "instruction": "y"},
                ]
            },
            "no gaps",
        ),
        ({"modes": [{"name": "a", "index": 0, "instruction": "  "}]}, "non-empty"),
        ({"format_constraint": None}, "declares no format_constraint"),
        ({"prompts": {}}, "prompts"),
    ],
)
def test_a_malformed_set_refuses_and_names_what_is_wrong(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: dict[str, Any], fragment: str
) -> None:
    payload = {**ADDED_SET, **overrides}
    payload = {key: value for key, value in payload.items() if value is not None}
    path = write_registry(tmp_path / "bad.json", {"mode_sets": {"my-modes": payload}})
    monkeypatch.setenv(MODE_SETS_ENV, str(path))
    with pytest.raises(ModeRegistryError) as caught:
        load_mode_registry()
    assert fragment in str(caught.value)


def test_a_mapping_pairing_a_label_neither_side_holds_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_registry(
        tmp_path / "mapping.json",
        {
            "mode_mappings": {
                "mine": {
                    "name": "mine",
                    "source": PROCESS_VOCABULARY,
                    "target": CORE_MODE_SET,
                    "pairs": {"deliberative": "no-such-mode"},
                }
            }
        },
    )
    monkeypatch.setenv(MODE_SETS_ENV, str(path))
    with pytest.raises(ModeRegistryError, match="does not hold"):
        load_mode_registry()


def test_a_mapping_that_is_not_one_to_one_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reverse direction is the inverse, so two sources on one target has no inverse."""
    path = write_registry(
        tmp_path / "many.json",
        {
            "mode_mappings": {
                "mine": {
                    "name": "mine",
                    "source": PROCESS_VOCABULARY,
                    "target": CORE_MODE_SET,
                    "pairs": {"deliberative": "linear", "pedagogical": "linear"},
                }
            }
        },
    )
    monkeypatch.setenv(MODE_SETS_ENV, str(path))
    with pytest.raises(ModeRegistryError, match="one to one"):
        load_mode_registry()


def test_a_vocabulary_cannot_shadow_a_mode_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_registry(
        tmp_path / "shadow.json",
        {
            "label_vocabularies": {
                CORE_MODE_SET: {
                    "name": CORE_MODE_SET,
                    "labels": [{"name": "linear", "gloss": "Something else."}],
                }
            }
        },
    )
    monkeypatch.setenv(MODE_SETS_ENV, str(path))
    with pytest.raises(ModeRegistryError, match="both a mode set and a label vocabulary"):
        load_mode_registry()


def test_a_file_named_in_the_environment_that_is_absent_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not a fall-through: a silent skip would generate under the shipped prompts."""
    monkeypatch.setenv(MODE_SETS_ENV, str(tmp_path / "nope.json"))
    with pytest.raises(ModeRegistryError, match="unreadable"):
        load_mode_registry()


def test_invalid_json_says_where(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"mode_sets": ', encoding="utf-8")
    monkeypatch.setenv(MODE_SETS_ENV, str(bad))
    with pytest.raises(ModeRegistryError, match="invalid JSON at line"):
        load_mode_registry()


def test_a_row_whose_name_disagrees_with_its_key_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_registry(tmp_path / "mismatch.json", {"mode_sets": {"other": ADDED_SET}})
    monkeypatch.setenv(MODE_SETS_ENV, str(path))
    with pytest.raises(ModeRegistryError, match="are one thing"):
        load_mode_registry()


def test_an_unknown_set_names_the_files_it_looked_in() -> None:
    with pytest.raises(UnknownModeSetError) as caught:
        mode_set("no-such-set")
    message = str(caught.value)
    assert str(MODE_SETS_FILE) in message
    assert MODE_SETS_ENV in message


def test_an_edit_to_a_registry_file_is_seen_without_restarting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_registry(tmp_path / "live.json", {"mode_sets": {"my-modes": ADDED_SET}})
    monkeypatch.setenv(MODE_SETS_ENV, str(path))
    assert mode_set("my-modes").names() == ("enumerative", "narrative")
    revised = {**ADDED_SET, "modes": ADDED_SET["modes"][:1]}
    write_registry(tmp_path / "live.json", {"mode_sets": {"my-modes": revised}})
    assert mode_set("my-modes").names() == ("enumerative",)


# ── The swap pairs ────────────────────────────────────────────────────────────
def test_the_swap_pairs_are_the_confound_test() -> None:
    payload = [
        (pair.system_mode, pair.execution_mode, pair.user_directive, pair.label)
        for pair in PROMPT_SWAP_PAIRS
    ]
    assert digest(payload) == SWAP_PAIRS_SHA256
    labels = [pair.label for pair in PROMPT_SWAP_PAIRS]
    assert labels == ["socratic→linear", "dialectical→contrastive", "analogical→linear"]
    assert len(set(labels)) == len(labels)


def test_every_swap_pair_names_modes_from_the_five() -> None:
    prompts = mode_prompts(CORE_MODE_SET)
    for pair in PROMPT_SWAP_PAIRS:
        assert pair.system_mode in prompts
        assert pair.execution_mode in prompts
        assert pair.system_mode != pair.execution_mode
        assert pair.get_system_prompt() == prompts[pair.system_mode]


def test_a_swap_prompt_puts_the_override_ahead_of_the_topic() -> None:
    pair = PROMPT_SWAP_PAIRS[0]
    prompt = pair.format_user_prompt("the tides")
    assert prompt == f"{pair.user_directive}\n\nWrite about: the tides"
    assert prompt.startswith(pair.user_directive)
    assert prompt.endswith("the tides")


def test_a_swap_prompt_accepts_another_template() -> None:
    pair = PROMPT_SWAP_PAIRS[0]
    assert pair.format_user_prompt("x", template="Discuss {topic}").endswith("Discuss x")


def test_a_swap_pair_is_immutable() -> None:
    pair = PROMPT_SWAP_PAIRS[0]
    with pytest.raises(Exception):
        pair.label = "other"


def test_a_pair_naming_a_mode_outside_the_core_set_refuses_by_name() -> None:
    pair = PromptSwapPair(
        system_mode="structured",
        execution_mode="linear",
        user_directive="Write plainly.",
        label="structured→linear",
    )
    with pytest.raises(UnknownModeSetError, match="structured"):
        pair.get_system_prompt()
