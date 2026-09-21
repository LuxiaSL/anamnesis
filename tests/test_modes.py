"""The mode sets: what they contain, what they export, and what stays in the record.

The mode prompts are the protocol. A label in banked data means the exact text in
this package, so the digests below pin that text: a change to a prompt is a change
to what every stored label refers to, and it fails here rather than surfacing as
an unexplained shift in a later corpus.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import subprocess
import sys

import pytest

import anamnesis.modes as modes
from anamnesis.config import paths
from anamnesis.modes import (
    EXTENDED_MODE_INDEX,
    EXTENDED_MODES,
    FORMAT_CONSTRAINT,
    PROMPT_SWAP_PAIRS,
    PromptSwapPair,
    RUN4_MODE_INDEX,
    RUN4_MODES,
)

CORE_MODES = ("linear", "analogical", "socratic", "contrastive", "dialectical")
ADDED_MODES = ("structured", "compressed", "associative")

RUN4_PROMPTS_SHA256 = "be2b0b7ab4fca49143901cf715e85a05bde53e5f56d4ba634097dde5c42a7432"
EXTENDED_PROMPTS_SHA256 = "d6a3f05a096ba3f799e507c5dc0c75167e29cef4f33146a0700c849a4b8693ad"
SWAP_PAIRS_SHA256 = "0fea66a52815d3ee7530afb9b358218cf53cc977b4003d223d97a86eed9bd0a3"

RECORD_ONLY_MODULES = ("anamnesis.modes.run3_original_modes",)

EXPECTED_EXPORTS = {
    "DEFAULT_USER_TEMPLATE",
    "EXTENDED_MODES",
    "EXTENDED_MODE_INDEX",
    "FORMAT_CONSTRAINT",
    "PROMPT_SWAP_PAIRS",
    "PromptSwapPair",
    "RUN4_MODES",
    "RUN4_MODE_INDEX",
}


def digest(payload: object) -> str:
    """A stable digest of mode data: sorted keys, unescaped text, UTF-8 bytes."""
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_the_package_exports_exactly_the_ported_set() -> None:
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


def test_the_five_core_modes_are_the_protocol() -> None:
    assert tuple(RUN4_MODES) == CORE_MODES
    assert RUN4_MODE_INDEX == {name: index for index, name in enumerate(CORE_MODES)}
    assert digest(RUN4_MODES) == RUN4_PROMPTS_SHA256


def test_the_eight_mode_set_extends_the_five_without_disturbing_them() -> None:
    assert tuple(EXTENDED_MODES) == CORE_MODES + ADDED_MODES
    for name in CORE_MODES:
        assert EXTENDED_MODES[name] == RUN4_MODES[name]
        assert EXTENDED_MODE_INDEX[name] == RUN4_MODE_INDEX[name]
    assert EXTENDED_MODE_INDEX == {
        name: index for index, name in enumerate(CORE_MODES + ADDED_MODES)
    }
    assert digest(EXTENDED_MODES) == EXTENDED_PROMPTS_SHA256


def test_every_mode_prompt_carries_the_format_constraint() -> None:
    assert FORMAT_CONSTRAINT.strip().startswith("Write in flowing paragraphs.")
    for name, prompt in EXTENDED_MODES.items():
        assert prompt.endswith(FORMAT_CONSTRAINT), name


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
    for pair in PROMPT_SWAP_PAIRS:
        assert pair.system_mode in RUN4_MODES
        assert pair.execution_mode in RUN4_MODES
        assert pair.system_mode != pair.execution_mode
        assert pair.get_system_prompt() == RUN4_MODES[pair.system_mode]


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


def test_a_pair_naming_an_unknown_mode_fails_when_asked_for_its_prompt() -> None:
    pair = PromptSwapPair(
        system_mode="structured",
        execution_mode="linear",
        user_directive="Write plainly.",
        label="structured→linear",
    )
    with pytest.raises(KeyError):
        pair.get_system_prompt()
