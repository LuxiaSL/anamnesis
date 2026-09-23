"""The retired bin vocabulary appears in this repository only where it must.

The four numbered bins a feature was once assigned to are retired as a taxonomy: each
one spanned several substrates at once, one of them mixing attention-weight reads with
key-vector geometry, so a bin's accuracy localizes nothing. Nothing in this instrument
reasons in them, and nothing prints them.

Two places still hold the strings, and both are wire format rather than description:

  * the names the four core blocks are *stored* under, in
    `anamnesis/extraction/state_extractor.py` — every signature ever banked keys its
    arrays and its slice table with exactly them;
  * the labels older banked results were *reported* under, in
    `anamnesis/analysis/gauntlet/schemas/compat.py` — the table that reads such a file
    forward onto the labels now in use.

Everything else is a regression. This test assembles its patterns from those constants
at runtime, so it is not its own hit, and so a string that stops being retired stops
being searched for.

CPU only; reads the repository's own source.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from anamnesis.analysis.gauntlet.schemas.compat import BLOCK_LABEL_RENAMES
from anamnesis.feature_map import (
    STORED_FAMILY_ATTENTION_OTHER,
    STORED_FAMILY_ATTENTION_SPECTRAL,
    STORED_FAMILY_CACHE_AND_KEYS,
    STORED_FAMILY_NORMS_AND_OUTPUT_STATS,
    STORED_FAMILY_RESIDUAL_PCA,
)
from anamnesis.extraction.state_extractor import (
    STORED_ATTENTION_AND_DELTAS,
    STORED_BLOCK_SLICES_KEY,
    STORED_CACHE_AND_KEYS,
    STORED_NORMS_AND_OUTPUT_STATS,
    STORED_RESIDUAL_PCA,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

SCANNED_ROOTS = ("anamnesis", "tests", "tools")

OWNS_THE_VOCABULARY: dict[str, str] = {
    "anamnesis/extraction/state_extractor.py":
        "the stored block names, held as constants so every reader takes them from here",
    "anamnesis/feature_map.py":
        "the stored family labels, which banked floor results are keyed by",
    "anamnesis/analysis/gauntlet/schemas/compat.py":
        "the table that reads an older results file forward",
    "tests/test_schemas_compat.py":
        "its fixtures are banked documents, spelled the way those files are",
    "tests/test_generation_runner.py":
        "pins the written sidecar's slice table literally, which is what a wire pin is",
    "tests/test_lane_guard.py":
        "builds synthetic npz files keyed the way a banked one is",
    "tests/test_fail_closed.py":
        "builds synthetic npz files keyed the way a banked one is",
}
"""Files allowed a hit, and why. A file here with no hit left is a stale exemption."""


def retired_strings() -> tuple[str, ...]:
    """Every retired spelling, from the constants that hold it."""
    return (
        *BLOCK_LABEL_RENAMES,
        STORED_NORMS_AND_OUTPUT_STATS,
        STORED_ATTENTION_AND_DELTAS,
        STORED_CACHE_AND_KEYS,
        STORED_RESIDUAL_PCA,
        STORED_BLOCK_SLICES_KEY,
        STORED_FAMILY_CACHE_AND_KEYS,
        STORED_FAMILY_ATTENTION_SPECTRAL,
        STORED_FAMILY_ATTENTION_OTHER,
        STORED_FAMILY_RESIDUAL_PCA,
        STORED_FAMILY_NORMS_AND_OUTPUT_STATS,
    )


def retired_pattern() -> re.Pattern[str]:
    """One pattern over every retired spelling, longest first so the longer bin label wins.

    A bin label is matched on word boundaries — a two-character label inside a longer
    identifier is somebody else's name — while a stored name is matched anywhere,
    because it reaches disk inside ``features_<name>``.
    """
    bin_labels = sorted(BLOCK_LABEL_RENAMES, key=len, reverse=True)
    stored = sorted(set(retired_strings()) - set(BLOCK_LABEL_RENAMES), key=len, reverse=True)
    parts = [
        rf"(?<![A-Za-z0-9_]){re.escape(label)}(?![A-Za-z0-9_])" for label in bin_labels
    ]
    parts += [re.escape(name) for name in stored]
    return re.compile("|".join(parts))


def python_files() -> list[Path]:
    found: list[Path] = []
    for root in SCANNED_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            found.append(path)
    return found


def hits_by_file() -> dict[str, list[str]]:
    pattern = retired_pattern()
    out: dict[str, list[str]] = {}
    for path in python_files():
        relative = path.relative_to(REPO_ROOT).as_posix()
        lines = [
            f"{relative}:{number}: {line.strip()}"
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if pattern.search(line)
        ]
        if lines:
            out[relative] = lines
    return out


def test_no_module_outside_the_wire_format_carries_the_retired_vocabulary() -> None:
    hits = hits_by_file()
    unexpected = {
        name: lines for name, lines in hits.items() if name not in OWNS_THE_VOCABULARY
    }
    assert unexpected == {}, (
        "the retired bin vocabulary reappeared outside the files that hold the wire "
        f"format: {unexpected}"
    )


def test_every_exemption_is_still_earning_it() -> None:
    """An exemption is only for a file that does hold the vocabulary."""
    hits = hits_by_file()
    stale = sorted(set(OWNS_THE_VOCABULARY) - set(hits))
    assert stale == [], f"exemptions with nothing left to exempt: {stale}"


@pytest.mark.parametrize(
    "document", ["README.md", "CONTRIBUTING.md", "docs/ARCHITECTURE.md"]
)
def test_the_facing_documents_do_not_teach_the_retired_vocabulary(document: str) -> None:
    """Every document a newcomer reads first, held to the rule the code is held to.

    These three are outside the G3 checkers' scope, which is python prose, so the one
    mechanical guard on what they teach is here.
    """
    pattern = retired_pattern()
    text = (REPO_ROOT / document).read_text(encoding="utf-8")
    offenders = [line for line in text.splitlines() if pattern.search(line)]
    assert offenders == [], f"{document}: {offenders}"
