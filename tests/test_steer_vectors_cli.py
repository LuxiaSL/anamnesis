"""The steering CLI: the path from a contrast to a lever readout, and the shim rule it obeys.

A script may parse arguments and call into the package. It may not hold capability,
because two front ends over one capability must not be able to disagree about the
science. The structural check here is the one ``tests/test_scripts_are_shims.py``
generalizes: this module defines no function another module would want to import,
and every steering module the path names is reachable from it.

``sweep`` runs end to end on synthetic banks, which is what makes this a caller
rather than a declaration. The legs that need a checkpoint are exercised through
their argument parsing only.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from anamnesis.scripts import steer_vectors


def test_every_subcommand_is_reachable_and_dispatches() -> None:
    parsed = steer_vectors.parser().parse_args(
        ["sweep", "--positive-means", "a.npz", "--negative-means", "b.npz", "--out-json", "o.json"]
    )
    assert parsed.command == "sweep"
    for command in ("build", "screen", "gate", "null", "lever"):
        with pytest.raises(SystemExit):
            steer_vectors.parser().parse_args([command])


def test_the_cli_reaches_every_steering_module() -> None:
    """G4's reachability, asserted rather than inferred from the import graph.

    The steering path runs contrast → vectors → screens → on-policy gate → lever
    readout → judge, and the first five legs are subcommands here, so all four
    steering modules have a caller on the path rather than only in a test.
    """
    from anamnesis.steering import gates, readouts, screens, vectors

    assert steer_vectors.vectors is vectors
    assert steer_vectors.screens is screens
    assert steer_vectors.gates is gates
    assert steer_vectors.readouts is readouts


def test_the_script_defines_no_capability_another_module_would_import() -> None:
    """The shim rule: argument parsing, dispatch and IO glue, and nothing else."""
    public = {
        name for name in vars(steer_vectors)
        if not name.startswith("_") and callable(getattr(steer_vectors, name))
        and getattr(getattr(steer_vectors, name), "__module__", "") == steer_vectors.__name__
    }
    assert public == {
        "parser", "main", "capture_model",
        "run_sweep", "run_build", "run_screen", "run_gate", "run_null", "run_lever",
    }


def test_the_json_writer_is_the_packages_and_not_a_second_copy() -> None:
    """One writer, so a readout and a screen land in the same format.

    Two copies of "write this dict as JSON" is how two artifacts of one pass come to
    differ in indentation, key order or whether the directory is created — cosmetic
    until something diffs them.
    """
    from anamnesis.steering import readouts

    assert "write_json" not in vars(steer_vectors)
    assert callable(readouts.write_json)


def test_sweep_runs_end_to_end_and_names_the_planted_site(tmp_path) -> None:
    rng = np.random.default_rng(2)
    n_prompts, n_samples, n_layers, dim = 20, 3, 6, 8
    prompt_ids = [f"p{i}" for i in range(n_prompts) for _ in range(n_samples)]
    positive = rng.standard_normal((n_prompts * n_samples, n_layers, dim))
    negative = rng.standard_normal((n_prompts * n_samples, n_layers, dim))
    positive[:, 4, 0] += 8.0
    np.savez(tmp_path / "pos.npz", means=positive, prompt_ids=np.array(prompt_ids))
    np.savez(tmp_path / "neg.npz", means=negative, prompt_ids=np.array(prompt_ids))

    out = tmp_path / "sweep.json"
    steer_vectors.main([
        "sweep", "--positive-means", str(tmp_path / "pos.npz"),
        "--negative-means", str(tmp_path / "neg.npz"), "--k-splits", "10",
        "--out-json", str(out),
    ])
    report = json.loads(out.read_text())
    assert report["peak_layer"] == 4
    assert report["n_prompts"] == n_prompts
    assert len(report["d_mean"]) == n_layers
    assert "per-prompt averaged" in report["law"]


def test_null_subcommand_refuses_an_undeclared_construction() -> None:
    with pytest.raises(SystemExit):
        steer_vectors.parser().parse_args([
            "null", "--sigma", "s.npz", "--vectors", "v", "--key", "V3_L14",
            "--construction", "vibes",
        ])


def test_null_runs_over_a_banked_eigenbasis_and_bank(tmp_path, capsys) -> None:
    np.savez(tmp_path / "sigma.npz", evals=np.array([4.0, 4.0] + [1.0] * 6), evecs=np.eye(8))
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    rng = np.random.default_rng(5)
    vector = rng.standard_normal(8)
    np.savez(bank_dir / "a5_vectors.npz", R1=(vector / np.linalg.norm(vector)).astype(np.float32))
    (bank_dir / "a5_vectors_stamps.json").write_text(json.dumps({"model": "3b"}))

    out = tmp_path / "null.json"
    steer_vectors.main([
        "null", "--sigma", str(tmp_path / "sigma.npz"), "--vectors", str(bank_dir),
        "--key", "R1", "--construction", "isotropic_random", "--k", "2",
        "--n-draws", "100", "--out-json", str(out),
    ])
    report = json.loads(out.read_text())
    assert report["key"] == "R1"
    assert report["construction"] == "isotropic_random"
    assert "null" in report["verdict"]
    assert "isotropic_random" in capsys.readouterr().out
