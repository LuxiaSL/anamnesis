"""The qualification entry point: what it asks for, and what its verdict says.

`qualify_box.py` is the golden path for "can I use this machine?". Running it needs a
checkpoint and a device, so what is testable without either is the part a user reads:
the arguments it requires, the refusals that stop a meaningless measurement before a
model is loaded, and the wording of the verdict.

The verdict wording is not cosmetic. Three states are distinguishable and must stay
distinguishable: the box repeats and its two paths agree; the box repeats and the
comparison is undecided because a declared lower bound was insufficient, which is more
work owed rather than a failure; and the box does not reproduce its own vectors, in
which case nothing else printed is a measurement. The exit status follows the same
three-way split, because a caller in a shell reads the status rather than the prose.

The one sentence that must survive any edit is the one about mixing: different hardware
gives different numbers, and signatures carrying different lane ids must not be combined
inside one contrast. That is the reason the script exists rather than a caveat attached
to it.
"""

from __future__ import annotations

import argparse

import pytest

from anamnesis.scripts.qualify_box import main, parser, qualify, render

BASE = [
    "--model",
    "3b",
    "--model-path",
    "/checkpoint",
    "--calib-dir",
    "/calibration",
    "--manifest",
    "/manifest.json",
]


def test_the_device_is_part_of_the_question():
    args = parser().parse_args(BASE)
    assert args.device == "cuda:0"
    assert parser().parse_args(BASE + ["--device", "cpu"]).device == "cpu"
    assert args.gen_ids is None


@pytest.mark.parametrize("missing", ["--model", "--model-path", "--calib-dir", "--manifest"])
def test_every_input_is_required(missing):
    trimmed = list(BASE)
    index = trimmed.index(missing)
    del trimmed[index : index + 2]
    with pytest.raises(SystemExit):
        parser().parse_args(trimmed)


def test_unknown_preset_is_refused_before_anything_loads():
    with pytest.raises(SystemExit):
        parser().parse_args(
            ["--model", "not-a-preset", "--model-path", "/c", "--calib-dir", "/d",
             "--manifest", "/m.json"]
        )


def test_the_arithmetic_settings_are_a_precondition(tmp_path, monkeypatch):
    """The verdict is about a lane, and the lane identity includes these settings."""
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    args = parser().parse_args(
        BASE[:4] + ["--calib-dir", str(tmp_path), "--manifest", str(tmp_path / "m.json")]
    )
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        qualify(args)


def test_a_single_row_cannot_be_qualified(tmp_path, monkeypatch):
    """A paired contrast needs two rows; one row would have nothing to be scaled against."""
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"entries": {"0": {"prompt_length": 4, "input_ids": [1,2,3,4,5,6]}}}')
    args = parser().parse_args(
        BASE[:4] + ["--calib-dir", str(tmp_path), "--manifest", str(manifest),
                    "--gen-ids", "0"]
    )
    with pytest.raises(ValueError, match="two or more distinct generations"):
        qualify(args)


def test_a_missing_input_exits_two_without_a_verdict(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    status = main(BASE[:4] + ["--calib-dir", str(tmp_path),
                              "--manifest", str(tmp_path / "absent.json")])
    assert status == 2
    captured = capsys.readouterr()
    assert "qualification did not run" in captured.err
    assert not captured.out


def _receipt(**overrides):
    receipt = dict(
        repeatable=True, agreement=True, lane_id="torch-eager-reduce-v1-abc",
        device="cuda:0", model="8b", rows=2, certified=False,
    )
    receipt.update(overrides)
    return receipt


def test_the_three_verdicts_are_distinguishable():
    agrees = render(_receipt())
    assert "AGREES" in agrees and "does not reproduce" not in agrees

    undecided = render(_receipt(agreement=None))
    assert "UNDECIDED" in undecided
    assert "more work rather than a failure" in undecided

    broken = render(_receipt(repeatable=False, agreement=False))
    assert "repeatable: NO" in broken
    assert "nothing else here is measured" in broken


def test_every_verdict_carries_the_do_not_mix_rule_and_the_lane_id():
    for receipt in (_receipt(), _receipt(agreement=None), _receipt(repeatable=False)):
        text = render(receipt)
        assert "must not be combined inside one contrast" in text
        assert "torch-eager-reduce-v1-abc" in text
        assert "expected" in text


def test_exit_status_splits_the_same_three_ways(monkeypatch):
    for receipt, expected in (
        (_receipt(), 0),
        (_receipt(agreement=None), 1),
        (_receipt(agreement=False), 1),
        (_receipt(repeatable=False, agreement=True), 1),
    ):
        monkeypatch.setattr(
            "anamnesis.scripts.qualify_box.qualify", lambda _args, r=receipt: r
        )
        assert main(BASE) == expected


def test_the_receipt_is_written_when_asked(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "anamnesis.scripts.qualify_box.qualify", lambda _args: _receipt()
    )
    destination = tmp_path / "receipt.json"
    assert main(BASE + ["--json", str(destination)]) == 0
    assert '"lane_id": "torch-eager-reduce-v1-abc"' in destination.read_text()


def test_qualify_takes_a_namespace_so_it_is_callable_without_a_shell():
    assert isinstance(parser().parse_args(BASE), argparse.Namespace)
