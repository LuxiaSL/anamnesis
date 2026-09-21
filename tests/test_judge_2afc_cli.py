"""The 2AFC entry point: what it asks for, what it refuses, and what it banks.

The script is a shim, so these cases test the shim's own obligations rather than
the harness's arithmetic: that a circular study is refused before a call is made,
that a dry run draws and banks a real contrast while spending nothing, and that
the packet it writes carries no answer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anamnesis.judging.harness import BlindPacket, CircularityError
from anamnesis.scripts import judge_2afc


def bank(path: Path, tag: str, *, groups: int = 10, per_group: int = 3) -> Path:
    """Write a metadata.json with generations long enough to survive the word filter."""
    path.mkdir(parents=True, exist_ok=True)
    generations = [
        {
            "generation_id": g * per_group + i,
            "topic_idx": g,
            "topic": f"topic {g}",
            "generated_text": " ".join(f"{tag}-{g}-{i}-w{w}" for w in range(30)),
        }
        for g in range(groups)
        for i in range(per_group)
    ]
    meta = path / "metadata.json"
    meta.write_text(json.dumps({"generations": generations}))
    return meta


def dry_run_argv(tmp_path: Path, **overrides: str) -> list[str]:
    target = bank(tmp_path / "cell_V3_a0.1", "steered")
    rider = bank(tmp_path / "cell_rider_a0.0", "rider")
    argv = [
        "--target", str(target),
        "--distractor", f"rider={rider}",
        "--prompt-set", "formality",
        "--scoring-instrument", "attention-allocation signature classifier",
        "--seed", "11",
        "--n-pairs", "20",
        "--out-dir", str(tmp_path / "out"),
        "--dry-run",
    ]
    for flag, value in overrides.items():
        argv += [f"--{flag.replace('_', '-')}", value]
    return argv


def test_a_dry_run_draws_and_banks_a_contrast_without_calling_anything(tmp_path: Path) -> None:
    assert judge_2afc.main(dry_run_argv(tmp_path)) == 0
    out = tmp_path / "out"
    packet = BlindPacket.read(packet_path=out / "packet.json", key_path=out / "key.json")
    assert packet.contrast == "cell_V3_a0.1"
    assert len(packet.pairs) == 20
    assert not (out / "results.json").exists()


def test_the_banked_packet_carries_no_answer(tmp_path: Path) -> None:
    judge_2afc.main(dry_run_argv(tmp_path))
    assert "target_side" not in (tmp_path / "out" / "packet.json").read_text()
    assert "target_side" in (tmp_path / "out" / "key.json").read_text()


def test_the_same_seed_banks_the_same_contrast(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    judge_2afc.main(dry_run_argv(first))
    judge_2afc.main(dry_run_argv(second))
    assert (first / "out" / "packet.json").read_text() == (second / "out" / "packet.json").read_text()


def test_a_circular_study_is_refused_before_anything_is_drawn(tmp_path: Path) -> None:
    argv = dry_run_argv(tmp_path)
    argv[argv.index("--scoring-instrument") + 1] = (
        "hand-written register criterion in the donor, not a marker battery"
    )
    with pytest.raises(CircularityError, match="two hats"):
        judge_2afc.main(argv)
    assert not (tmp_path / "out").exists()


def test_a_distractor_without_a_path_is_refused(tmp_path: Path) -> None:
    argv = dry_run_argv(tmp_path)
    argv[argv.index("--distractor") + 1] = "rider"
    with pytest.raises(judge_2afc.JudgingError, match="NAME=PATH"):
        judge_2afc.main(argv)


def test_the_seed_and_the_scoring_instrument_are_not_optional(tmp_path: Path) -> None:
    """A contrast nobody can redraw, or whose circularity nobody can check, is not
    a measurement this entry point offers to take."""
    for flag in ("--seed", "--scoring-instrument"):
        argv = dry_run_argv(tmp_path)
        index = argv.index(flag)
        del argv[index:index + 2]
        with pytest.raises(SystemExit):
            judge_2afc.main(argv)


def test_the_prompt_set_must_come_from_the_versioned_table(tmp_path: Path) -> None:
    argv = dry_run_argv(tmp_path)
    argv[argv.index("--prompt-set") + 1] = "improvised"
    with pytest.raises(SystemExit):
        judge_2afc.main(argv)


def test_the_parser_documents_itself_from_the_module(tmp_path: Path) -> None:
    assert judge_2afc.parser().description == judge_2afc.__doc__.splitlines()[0]


def test_the_script_defines_nothing_another_script_would_import() -> None:
    """The shim rule: capability lives in the package, entry points stay thin."""
    public = [
        name for name in vars(judge_2afc)
        if not name.startswith("_") and callable(getattr(judge_2afc, name))
        and getattr(getattr(judge_2afc, name), "__module__", "") == judge_2afc.__name__
    ]
    assert sorted(public) == ["main", "parser"]
