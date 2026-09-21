"""The prompt-set data files: pinned bytes, and the shape readers rely on.

The prompt sets are the topics every generation in the corpus was produced from,
so their bytes are part of the record rather than an implementation detail. The
digests are the frozen values carried over from the extraction repository: a run
that cites a topic index means these files.

Two kinds of file live here and they are pinned the same way for the same reason,
but their shapes differ. The **topic sets** carry a template and four named sets a
generation draws from. The **calibration ruler** carries one flat list: the prompts a
model's positional means and residual basis are fitted over. A mean is subtracted
from every state before every corrected feature, so editing the ruler moves every
signature downstream of it while no argument anywhere changes — which is exactly why
its bytes are pinned rather than treated as a list somebody may extend.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from anamnesis.config import paths
from anamnesis.extraction import calibration_fit

TOPIC_SET_SHA256: dict[str, str] = {
    "prompt_sets.json": "2feeb2b63a535ee5684beee160600b0f1367741a9935e0435ff85f3307853b73",
    "prompt_sets_narrative.json": "3b9bbbff4a45411a02f31324cc17a8c9aba7b9269ad56a211b8ed351aeecf415",
}

CALIBRATION_SET_SHA256: dict[str, str] = {
    "calibration_prompts.json": (
        "817bcff40d91cd849b85308fd3240f04024cbc8f406b5c0baee4410fedd217d1"
    ),
}

PROMPT_SET_SHA256: dict[str, str] = {**TOPIC_SET_SHA256, **CALIBRATION_SET_SHA256}
"""Every prompt-set file the package ships, by name."""

TOPIC_SET_SIZES = {"set_a": 10, "set_b": 10, "set_c": 20, "set_d": 20}

CALIBRATION_RULER_SIZE = 50
"""Prompts in the calibration ruler; the count the banked artifacts were fitted at."""


def load(name: str) -> dict[str, Any]:
    payload = json.loads(paths.prompts_path(name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_the_directory_holds_exactly_the_pinned_files() -> None:
    present = sorted(path.name for path in paths.prompts_dir().glob("*.json"))
    assert present == sorted(PROMPT_SET_SHA256)


@pytest.mark.parametrize("name", sorted(PROMPT_SET_SHA256))
def test_bytes_match_the_pinned_digest(name: str) -> None:
    path: Path = paths.prompts_path(name)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == PROMPT_SET_SHA256[name]


@pytest.mark.parametrize("name", sorted(TOPIC_SET_SHA256))
def test_each_topic_file_parses_and_carries_the_expected_keys(name: str) -> None:
    payload = load(name)
    for key in ("description", "user_prompt_template", "num_repetitions", "topics"):
        assert key in payload, key
    assert "{topic}" in payload["user_prompt_template"]
    assert payload["num_repetitions"] >= 1


@pytest.mark.parametrize("name", sorted(TOPIC_SET_SHA256))
def test_topic_sets_are_the_sizes_the_protocol_uses(name: str) -> None:
    topics = load(name)["topics"]
    assert {key: len(value) for key, value in topics.items()} == TOPIC_SET_SIZES
    for key, value in topics.items():
        assert all(isinstance(topic, str) and topic for topic in value), key
        assert len(set(value)) == len(value), key


def test_the_calibration_ruler_is_a_flat_list_of_distinct_prompts() -> None:
    """The ruler's own shape, and the reader that the fitting pass goes through."""
    name = next(iter(CALIBRATION_SET_SHA256))
    payload = load(name)
    assert "description" in payload
    prompts = payload[calibration_fit.CALIBRATION_PROMPTS_KEY]
    assert len(prompts) == CALIBRATION_RULER_SIZE
    assert len(set(prompts)) == CALIBRATION_RULER_SIZE
    assert all(isinstance(text, str) and text for text in prompts)
    assert calibration_fit.calibration_prompts() == tuple(prompts)


def test_the_twenty_mode_topics_are_the_first_two_sets() -> None:
    topics = load("prompt_sets.json")["topics"]
    corpus = [*topics["set_a"], *topics["set_b"]]
    assert len(corpus) == 20
    assert len(set(corpus)) == 20
