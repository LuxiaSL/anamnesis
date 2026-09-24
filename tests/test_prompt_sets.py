"""The prompt-set data files: pinned bytes, and the shape readers rely on.

The prompt sets are the topics every generation in the corpus was produced from,
so what they say is part of the record rather than an implementation detail: a run
that cites a topic index means these files.

Two kinds of file live here, pinned differently. A **topic set** is pinned by its
*meaning* — the parsed file with every ``description`` removed, serialized in its own
key order — because the topics, the template, the strata and their order are what a
banked run depends on, while a description is prose about them that must be free to
state its reasons correctly. Key order stays inside the digest because order is
meaningful here: stratum order lays out a banked floor corpus's generation ids. The
**calibration ruler** is pinned by its bytes, because nothing in it is prose about
something else — every character is a prompt the positional means were fitted over.

The two kinds also differ in shape. The **topic sets** carry a template and four named sets a
generation draws from. The **calibration ruler** carries one flat list: the prompts a
model's positional means and residual basis are fitted over. A mean is subtracted
from every state before every corrected feature, so editing the ruler moves every
signature downstream of it while no argument anywhere changes — which is exactly why
its bytes are pinned. It grows only by appending, and the head the banked artifacts
were fitted over is pinned on its own, so a fit over that head is still provably a fit
over the banked ruler.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from anamnesis.config import paths
from anamnesis.extraction import calibration_fit

TOPIC_SET_MEANING_SHA256: dict[str, str] = {
    "prompt_sets.json": "25fa8e826546b9a3d5135d1d6f851a1d5793af1eb592fa258f19eee069ffb13c",
    "prompt_sets_narrative.json": "95133f704bdeb9628dc40b5547a1743783c768fff46babbdcacc8700742e5211",
}
"""Each topic set's meaning digest; see :func:`meaning_digest`."""

TOPIC_SET_RECORD_BYTES_SHA256: dict[str, str] = {
    "prompt_sets.json": "2feeb2b63a535ee5684beee160600b0f1367741a9935e0435ff85f3307853b73",
    "prompt_sets_narrative.json": "3b9bbbff4a45411a02f31324cc17a8c9aba7b9269ad56a211b8ed351aeecf415",
}
"""The byte digests of these files as the banked runs were produced from them.

Not asserted: only the descriptions have changed since, which the meaning digest
proves. Kept so a copy of a file found elsewhere can be matched to the record.
"""

CALIBRATION_SET_SHA256: dict[str, str] = {
    "calibration_prompts.json": (
        "4bce267a8ea7ef5ba58a1e02df5c07274b76c8670b9422ac6b9345a839ebc4c8"
    ),
}

BANKED_RULER_SIZE = 50
"""The leading prompts the banked positional means and residual bases were fitted over."""

BANKED_RULER_SHA256 = "e5de382609ce936f0e4ad9d7bc3f025d5757c70e50f0b84fab9ba4cbc695265e"
"""Digest of those prompts as a compact JSON list, so the banked ruler stays provable
inside the longer one: a fit over the first :data:`BANKED_RULER_SIZE` prompts is a fit
over exactly the ruler the banked artifacts used."""

CALIBRATION_RECORD_BYTES_SHA256 = "817bcff40d91cd849b85308fd3240f04024cbc8f406b5c0baee4410fedd217d1"
"""The byte digest of the fifty-prompt file the banked artifacts were fitted from.

Not asserted: the file has since grown by appending. Kept so a copy found elsewhere can
be matched to the record."""

PROMPT_SET_NAMES: frozenset[str] = frozenset(TOPIC_SET_MEANING_SHA256) | frozenset(CALIBRATION_SET_SHA256)
"""Every prompt-set file the package ships, by name."""

PROVENANCE_IN_PROSE = re.compile(r"prere[g]|ratifie[d]|addendu[m]|codici[l]|\b20\d\d-\d\d-\d\d\b|\bsession-\d")
"""Citations a description must not carry: it states the constraint, not the plan behind it."""

TOPIC_SET_SIZES = {"set_a": 10, "set_b": 10, "set_c": 20, "set_d": 20}

CALIBRATION_RULER_SIZE = 200
"""Prompts in the calibration ruler."""


def _without_descriptions(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _without_descriptions(v) for k, v in node.items() if k != "description"}
    if isinstance(node, list):
        return [_without_descriptions(v) for v in node]
    return node


def meaning_digest(payload: dict[str, Any]) -> str:
    """SHA-256 of the payload with every ``description`` removed, in its own key order."""
    canonical = json.dumps(_without_descriptions(payload), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _descriptions(node: Any) -> list[str]:
    if isinstance(node, dict):
        found = [node["description"]] if isinstance(node.get("description"), str) else []
        return found + [d for k, v in node.items() if k != "description" for d in _descriptions(v)]
    if isinstance(node, list):
        return [d for v in node for d in _descriptions(v)]
    return []


def load(name: str) -> dict[str, Any]:
    payload = json.loads(paths.prompts_path(name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_the_directory_holds_exactly_the_pinned_files() -> None:
    present = sorted(path.name for path in paths.prompts_dir().glob("*.json"))
    assert present == sorted(PROMPT_SET_NAMES)


@pytest.mark.parametrize("name", sorted(TOPIC_SET_MEANING_SHA256))
def test_each_topic_set_means_what_the_banked_runs_were_drawn_from(name: str) -> None:
    assert meaning_digest(load(name)) == TOPIC_SET_MEANING_SHA256[name]


@pytest.mark.parametrize("name", sorted(CALIBRATION_SET_SHA256))
def test_the_calibration_ruler_bytes_match_the_pinned_digest(name: str) -> None:
    path: Path = paths.prompts_path(name)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == CALIBRATION_SET_SHA256[name]


def test_the_meaning_digest_ignores_descriptions_and_nothing_else() -> None:
    """A description edit is invisible to the pin; a topic edit or a reorder is not."""
    payload = load("prompt_sets.json")
    pinned = meaning_digest(payload)
    assert meaning_digest({**payload, "description": "anything at all"}) == pinned
    topics = payload["topics"]
    edited = {**topics, "set_a": [topics["set_a"][0] + " ", *topics["set_a"][1:]]}
    assert meaning_digest({**payload, "topics": edited}) != pinned
    reordered = dict(reversed(list(topics.items())))
    assert meaning_digest({**payload, "topics": reordered}) != pinned


@pytest.mark.parametrize("name", sorted(PROMPT_SET_NAMES))
def test_descriptions_state_constraints_rather_than_cite_their_provenance(name: str) -> None:
    for text in _descriptions(load(name)):
        assert not PROVENANCE_IN_PROSE.search(text), text


@pytest.mark.parametrize("name", sorted(TOPIC_SET_MEANING_SHA256))
def test_each_topic_file_parses_and_carries_the_expected_keys(name: str) -> None:
    payload = load(name)
    for key in ("description", "user_prompt_template", "num_repetitions", "topics"):
        assert key in payload, key
    assert "{topic}" in payload["user_prompt_template"]
    assert payload["num_repetitions"] >= 1


@pytest.mark.parametrize("name", sorted(TOPIC_SET_MEANING_SHA256))
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


def test_the_banked_ruler_is_the_unchanged_head_of_the_ruler() -> None:
    """Growing the ruler appended to it: the banked fifty lead it, unchanged and in order."""
    head = list(calibration_fit.calibration_prompts()[:BANKED_RULER_SIZE])
    serial = json.dumps(head, ensure_ascii=False, separators=(",", ":")).encode()
    assert hashlib.sha256(serial).hexdigest() == BANKED_RULER_SHA256


def test_the_twenty_mode_topics_are_the_first_two_sets() -> None:
    topics = load("prompt_sets.json")["topics"]
    corpus = [*topics["set_a"], *topics["set_b"]]
    assert len(corpus) == 20
    assert len(set(corpus)) == 20
