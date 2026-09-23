"""Fail closed: a command that produced fewer units than it was asked for refuses.

The hazard this pins is one-directional. Four loops here keep going when one unit
fails — a replay's generation, a generation pass's spec, a gauntlet's section, a
recompute's tensor file — and that is right: one unsamplable prompt is not a reason
to abandon three hundred. What is wrong is the command above reporting success over
the short corpus, because a mean over four of five generations is a number and
nothing about the number says five were asked for.

So each site states its pass as expected-versus-produced in an
`anamnesis.shortfall.Shortfall` and refuses on it, and what the tests below check is
that the accounting says the same thing at all four:

* a clean pass is complete and exits zero;
* a pass with a failed unit exits non-zero and names the unit and the reason;
* `--allow-partial` exits with its own non-zero status and leaves a receipt;
* **a resumed pass that skips completed work exits zero** — the one way a correct
  refusal could still break a working corpus-building pass, so it is tested at
  every site where resume exists rather than argued about.

How far each site runs here is set by what the site needs:

* the replay loop runs for real against the tiny transformer in
  `tests/synthetic_runtime.py`, writing real signatures, so resume and the
  `--gen-ids` subset are read off the disk the loop wrote;
* the gauntlet runs end to end through `anamnesis.scripts.run_gauntlet`'s `main`,
  exit status and all, over a synthetic corpus;
* the recompute runs end to end over raw tensors written by the real saver;
* the generation loop samples on a device, so its accounting is checked on the
  typed object the loop returns and the records it would have banked. The object is
  the real one the command receives.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anamnesis.config import ExtractionConfig, FeaturePipelineConfig
from anamnesis.extraction.feature_pipeline import (
    RecomputeCount,
    recompute_all_features,
    recompute_shortfall,
)
from anamnesis.extraction.replay.cell import (
    ReplaySurface,
    cell_shortfall,
    replay_cell,
    signature_on_disk,
)
from anamnesis.extraction.state_extractor import RawGenerationData
from anamnesis.extraction.token_generation import GenerationCount, generation_shortfall
from anamnesis.scripts import run_gauntlet
from anamnesis.shortfall import (
    EXIT_SHORT,
    EXIT_SHORT_SANCTIONED,
    RECEIPT_STEM,
    Shortfall,
    ids_present,
    refuse_unless_complete,
    worker_shortfall_code,
)
from synthetic_runtime import HookPlan, loaded_tiny_model

# ── the tiny geometry the replay and recompute loops run under ────────────────

TINY_LAYERS = [0, 1, 2]
TINY_PLAN = HookPlan(
    key_layers=TINY_LAYERS, value_layers=TINY_LAYERS, query_layers=TINY_LAYERS,
    attn_output_layers=TINY_LAYERS, gate_layers=TINY_LAYERS,
)
#: A twelve-token sequence with a five-token prompt: seven generated, six banked.
TINY_IDS = list(range(1, 13))
TINY_PROMPT_LENGTH = 5


def tiny_extraction() -> ExtractionConfig:
    """A layer plan and window set the tiny model's three layers can carry.

    The residual-PCA and kNN-LM blocks want a fitted basis, which a random-weight
    model has no business having, so they are off; every other core block runs.
    """
    return ExtractionConfig(
        sampled_layers=TINY_LAYERS,
        pca_layers=[1],
        enable_residual_pca=False,
        enable_knnlm_baseline=False,
        early_layer_cutoff=0,
        late_layer_cutoff=2,
        spectral_subsample_step=1,
        epoch_window_size=2,
        epoch_stride=1,
        surprise_window=2,
        trajectory_points=3,
        pca_temporal_samples=3,
    )


def tiny_families() -> FeaturePipelineConfig:
    """The families that read only the sampled layers, over the tiny geometry."""
    return FeaturePipelineConfig(
        include_core_blocks=True,
        enable_residual_trajectory=True,
        enable_attention_flow=True,
        enable_gate_features=True,
        enable_temporal_dynamics=False,
        enable_per_head=False,
        enable_stft=False,
        enable_contrastive_projection=False,
        trajectory_layers=TINY_LAYERS,
        contrastive_layers=TINY_LAYERS,
    )


def write_manifest(
    run_dir: Path,
    gen_ids: list[int],
    *,
    unreplayable: list[int] | None = None,
    flagged: list[int] | None = None,
) -> Path:
    """A replay manifest over ``gen_ids``, in the shape the assembler writes.

    ``unreplayable`` entries are present and replayable-looking but carry a prompt
    length equal to the whole sequence, so the alignment arithmetic raises when the
    loop reaches them — a real failure through the real loop, rather than a
    simulated one. ``flagged`` ids go in the manifest's flagged list, which is how a
    manifest accounts for a generation it knows cannot be replayed at all.
    """
    broken = set(unreplayable or ())
    entries = {
        str(gen_id): {
            "input_ids": TINY_IDS,
            "prompt_length": len(TINY_IDS) if gen_id in broken else TINY_PROMPT_LENGTH,
        }
        for gen_id in gen_ids
    }
    rows = [
        {"gen_id": gen_id, "reason": "no raw_tensors / chosen_ids"}
        for gen_id in (flagged or ())
    ]
    path = run_dir / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "entries": entries, "n_ok": len(entries), "n_flagged": len(rows), "flagged": rows,
    }))
    return path


def replay(
    run_dir: Path, manifest: Path, *, gen_ids: list[int] | None = None, resume: bool = True
) -> Shortfall:
    """Replay a cell with the tiny model and state it as a shortfall.

    These two calls are what the command does with the cell it replayed: the loop
    reports, and `cell_shortfall` turns the report into the thing a refusal reads.
    """
    loaded, _ = loaded_tiny_model(TINY_PLAN)
    surface = ReplaySurface(
        loaded=loaded, extraction=tiny_extraction(), families=tiny_families()
    )
    result = replay_cell(
        surface, (None, None, None), run_dir, manifest,
        gen_ids=gen_ids,
        signatures_subdir="signatures",
        save_raw=False,
        resume=resume,
        label="t",
    )
    return cell_shortfall(result, manifest, command="test", label="t")


# ── the vocabulary ────────────────────────────────────────────────────────────


def a_shortfall(target: Path, **overrides: object) -> Shortfall:
    """A three-unit shortfall over ``target``, with fields substituted in."""
    fields: dict[str, object] = {
        "command": "test", "unit": "generation", "target": target,
        "requested": ("0", "1", "2"), "produced": ("0", "1", "2"),
    }
    fields.update(overrides)
    return Shortfall(**fields)  # type: ignore[arg-type]


def test_a_complete_pass_is_ok_and_names_nothing_missing(tmp_path: Path) -> None:
    shortfall = a_shortfall(tmp_path)
    assert shortfall.ok
    assert shortfall.missing == ()


def test_a_produced_id_that_was_never_requested_is_a_broken_accounting(tmp_path: Path) -> None:
    """The guard against a call site that listed a directory instead of its slice."""
    with pytest.raises(ValueError, match="did not request"):
        a_shortfall(tmp_path, requested=("0", "1"), produced=("0", "1", "2"))


def test_an_id_cannot_be_requested_and_excluded_at_once(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exclusion is not a request"):
        a_shortfall(tmp_path, excluded={"1": "flagged"})


def test_a_stale_artifact_does_not_cover_for_a_unit_that_raised(tmp_path: Path) -> None:
    """A pass with resume off leaves an earlier pass's output in place when it fails.

    Nothing is missing from the directory and the number would still be wrong, so
    the verdict reads the failures as well as the listing.
    """
    shortfall = a_shortfall(tmp_path, failures={"2": "ValueError: bad tensor"})
    assert shortfall.missing == ()
    assert not shortfall.ok


def test_ids_present_reads_the_predicate_the_site_owns(tmp_path: Path) -> None:
    (tmp_path / "gen_001.npz").write_bytes(b"x")
    present = ids_present(
        ("0", "1", "2"), lambda name: (tmp_path / f"gen_{int(name):03d}.npz").exists()
    )
    assert present == ("1",)


def test_a_complete_pass_exits_zero_and_clears_its_own_stale_receipt(tmp_path: Path) -> None:
    """A directory holding a receipt is short as of its last pass, not an earlier one."""
    shortfall = a_shortfall(tmp_path, label="w0")
    shortfall.receipt_path.write_text("{}")
    refuse_unless_complete([shortfall], allow_partial=False)
    assert not shortfall.receipt_path.exists()


def test_a_short_pass_refuses_with_the_unsanctioned_status(tmp_path: Path) -> None:
    shortfall = a_shortfall(
        tmp_path, produced=("0", "1"), failures={"2": "ValueError: bad tensor"}
    )
    with pytest.raises(SystemExit) as exit_info:
        refuse_unless_complete([shortfall], allow_partial=False)
    assert exit_info.value.code == EXIT_SHORT
    receipt = json.loads((tmp_path / f"{RECEIPT_STEM}.json").read_text())
    assert receipt["sanctioned"] is False
    assert receipt["missing"] == ["2"]
    assert receipt["failures"] == {"2": "ValueError: bad tensor"}
    assert receipt["n_requested"] == 3 and receipt["n_produced"] == 2


def test_a_sanctioned_short_pass_refuses_with_its_own_status(tmp_path: Path) -> None:
    """Permitted, and still non-zero: the failure is recorded, never hidden."""
    shortfall = a_shortfall(tmp_path, produced=("0", "1"))
    with pytest.raises(SystemExit) as exit_info:
        refuse_unless_complete([shortfall], allow_partial=True)
    assert exit_info.value.code == EXIT_SHORT_SANCTIONED
    assert exit_info.value.code != EXIT_SHORT
    receipt = json.loads((tmp_path / f"{RECEIPT_STEM}.json").read_text())
    assert receipt["sanctioned"] is True
    assert receipt["missing"] == ["2"]


def test_the_summary_names_the_unit_and_the_reason(tmp_path: Path) -> None:
    shortfall = a_shortfall(
        tmp_path, produced=("0", "1"), failures={"2": "FileNotFoundError: no tensor"}
    )
    summary = shortfall.summary()
    assert "2 of 3 generations" in summary
    assert "missing 2" in summary
    assert "FileNotFoundError: no tensor" in summary


def test_a_roster_reports_every_short_cell_before_it_refuses(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    for folder in (first, second):
        folder.mkdir()
    with pytest.raises(SystemExit):
        refuse_unless_complete(
            [a_shortfall(first, produced=("0",)), a_shortfall(second, produced=("1",))],
            allow_partial=True,
        )
    assert (first / f"{RECEIPT_STEM}.json").exists()
    assert (second / f"{RECEIPT_STEM}.json").exists()


def test_parallel_workers_over_one_directory_get_a_receipt_each(tmp_path: Path) -> None:
    """Two workers share a signature directory, so a shared receipt name would lose one."""
    with pytest.raises(SystemExit):
        refuse_unless_complete(
            [a_shortfall(tmp_path, produced=("0",), label="w0g0"),
             a_shortfall(tmp_path, produced=("1",), label="w1g1")],
            allow_partial=True,
        )
    assert (tmp_path / f"{RECEIPT_STEM}-w0g0.json").exists()
    assert (tmp_path / f"{RECEIPT_STEM}-w1g1.json").exists()


def test_a_fan_out_inherits_its_workers_shortfall_status() -> None:
    assert worker_shortfall_code({0: 0, 1: 0}) is None
    assert worker_shortfall_code({0: 0, 1: EXIT_SHORT}) == EXIT_SHORT
    assert worker_shortfall_code({0: EXIT_SHORT_SANCTIONED}) == EXIT_SHORT_SANCTIONED
    assert worker_shortfall_code(
        {0: EXIT_SHORT_SANCTIONED, 1: EXIT_SHORT}
    ) == EXIT_SHORT, "an unsanctioned worker outranks a sanctioned one"
    assert worker_shortfall_code({0: EXIT_SHORT, 1: 1}) is None, (
        "a crashed worker is the launcher's report to make, not a shortfall"
    )


# ── site 1: the replay loop ───────────────────────────────────────────────────


def test_a_clean_replay_produces_its_manifest_and_exits_zero(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, [0, 1, 2])
    shortfall = replay(tmp_path, manifest)
    assert shortfall.requested == ("0", "1", "2")
    assert shortfall.produced == ("0", "1", "2")
    assert shortfall.ok
    refuse_unless_complete([shortfall], allow_partial=False)


def test_a_replay_that_lost_one_generation_refuses_and_names_it(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, [0, 1, 2], unreplayable=[1])
    shortfall = replay(tmp_path, manifest)
    assert shortfall.produced == ("0", "2")
    assert shortfall.missing == ("1",)
    assert "1" in shortfall.failures and shortfall.failures["1"]
    with pytest.raises(SystemExit) as exit_info:
        refuse_unless_complete([shortfall], allow_partial=False)
    assert exit_info.value.code == EXIT_SHORT
    receipt = json.loads((tmp_path / "signatures" / f"{RECEIPT_STEM}-t.json").read_text())
    assert receipt["missing"] == ["1"]
    assert list(receipt["failures"]) == ["1"]


def test_a_resumed_replay_that_computes_nothing_is_complete(tmp_path: Path) -> None:
    """The hazard: a resume legitimately skipping finished work must not read as short.

    The second pass computes zero signatures because the first pass wrote all three,
    and that pass is complete — the corpus is whole. A shortfall counted from the
    work done rather than from the disk would refuse here, and refusing here would
    break every resumed corpus build.
    """
    manifest = write_manifest(tmp_path, [0, 1, 2])
    first = replay(tmp_path, manifest)
    assert first.ok

    second = replay(tmp_path, manifest, resume=True)
    assert second.requested == ("0", "1", "2"), "a resume narrows the work, not the request"
    assert second.produced == ("0", "1", "2")
    assert second.failures == {}
    assert second.ok
    refuse_unless_complete([second], allow_partial=False)


def test_a_resumed_replay_finishes_what_an_interrupted_one_started(tmp_path: Path) -> None:
    """The mixed case: some work skipped, some done, and the cell complete afterwards."""
    manifest = write_manifest(tmp_path, [0, 1, 2])
    replay(tmp_path, manifest, gen_ids=[0])
    resumed = replay(tmp_path, manifest)
    assert resumed.produced == ("0", "1", "2")
    assert resumed.ok
    refuse_unless_complete([resumed], allow_partial=False)


def test_half_a_signature_does_not_count_as_one(tmp_path: Path) -> None:
    """The predicate the resume filter, the fan-out and the accounting all read.

    A crash between the two writes leaves metadata with no vector. If resume
    skipped on the metadata alone, the accounting would report the generation
    missing on every pass while no re-run ever redid it — a refusal nobody could
    clear, which is worse than the bug this change closes.
    """
    (tmp_path / "gen_000.json").write_text("{}")
    assert not signature_on_disk(tmp_path, 0)
    (tmp_path / "gen_000.npz").write_bytes(b"")
    assert signature_on_disk(tmp_path, 0)


def test_a_resume_redoes_a_generation_whose_vector_never_landed(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, [0, 1])
    (tmp_path / "signatures").mkdir()
    (tmp_path / "signatures" / "gen_000.json").write_text("{}")

    shortfall = replay(tmp_path, manifest, resume=True)
    assert shortfall.produced == ("0", "1")
    assert shortfall.ok
    assert (tmp_path / "signatures" / "gen_000.npz").is_file(), "the half-written one was redone"


def test_a_worker_that_produced_its_own_share_is_complete(tmp_path: Path) -> None:
    """A cell narrowed by --gen-ids requests its slice, so the slice is the whole ask."""
    manifest = write_manifest(tmp_path, [0, 1, 2])
    shortfall = replay(tmp_path, manifest, gen_ids=[1])
    assert shortfall.requested == ("1",)
    assert shortfall.ok, "the other two generations are another worker's share"
    refuse_unless_complete([shortfall], allow_partial=False)


def test_a_generation_the_manifest_flags_is_an_exclusion_not_a_failure(tmp_path: Path) -> None:
    """A flagged record is accounted for, so a pass over a flagged bank still passes."""
    manifest = write_manifest(tmp_path, [0, 1], flagged=[7])
    shortfall = replay(tmp_path, manifest)
    assert shortfall.requested == ("0", "1")
    assert shortfall.excluded == {"7": "no raw_tensors / chosen_ids"}
    assert shortfall.failures == {}
    assert shortfall.ok
    refuse_unless_complete([shortfall], allow_partial=False)


def test_a_sanctioned_short_replay_leaves_the_receipt_beside_the_signatures(
    tmp_path: Path,
) -> None:
    manifest = write_manifest(tmp_path, [0, 1, 2], unreplayable=[2])
    shortfall = replay(tmp_path, manifest)
    with pytest.raises(SystemExit) as exit_info:
        refuse_unless_complete([shortfall], allow_partial=True)
    assert exit_info.value.code == EXIT_SHORT_SANCTIONED
    receipt = json.loads((tmp_path / "signatures" / f"{RECEIPT_STEM}-t.json").read_text())
    assert receipt["sanctioned"] is True
    assert receipt["requested"] == ["0", "1", "2"]
    assert receipt["missing"] == ["2"]


def test_a_fan_out_carries_the_escape_hatch_down_and_the_verdict_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sanctioned fan-out has to sanction its workers, and inherit what they say.

    The flag is what each worker reads; without it a worker under a sanctioned
    parent refuses unsanctioned, and the parent reports the wrong one of the two
    non-zero statuses. No subprocess is spawned: the launcher is replaced by one
    that records the argv it was going to run and answers with the real
    `anamnesis.orchestration.launch.LaunchResult`.
    """
    from anamnesis.orchestration.launch import LaunchPlan, LaunchResult
    from anamnesis.scripts import run_replay

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    manifest = write_manifest(tmp_path, [0, 1, 2])
    spawned: list[list[str]] = []

    def record_and_report(
        plan: LaunchPlan, command_for: object, *, stem: str, workers: object, **_: object
    ) -> LaunchResult:
        indices = list(workers)  # type: ignore[call-overload]
        for worker in indices:
            spawned.append(list(command_for(worker)))  # type: ignore[operator]
        return LaunchResult(
            returncodes={worker: EXIT_SHORT_SANCTIONED for worker in indices},
            log_paths={worker: plan.log_path(worker, stem) for worker in indices},
            seconds=0.0,
        )

    monkeypatch.setattr("anamnesis.orchestration.launch.launch", record_and_report)
    with pytest.raises(SystemExit) as exit_info:
        run_replay.main([
            "--model", "8b", "--model-path", "/models/x", "--calib-dir", str(tmp_path),
            "--run-dir", str(tmp_path), "--manifest", str(manifest),
            "--gpus", "0", "--workers-per-gpu", "1",
            "--allow-partial", "--single-cell-ok",
        ])
    assert exit_info.value.code == EXIT_SHORT_SANCTIONED
    assert spawned and all("--allow-partial" in command for command in spawned)


# ── site 2: the generation loop ───────────────────────────────────────────────


def bank_record(out_dir: Path, gen_id: int) -> None:
    """One banked record, which is what a generated spec leaves behind."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"gen_{gen_id:03d}.json").write_text(json.dumps({"generation_id": gen_id}))


def test_a_clean_generation_pass_is_complete(tmp_path: Path) -> None:
    for gen_id in (0, 1, 2):
        bank_record(tmp_path, gen_id)
    count = GenerationCount(
        n_done=3, failed={}, requested=(0, 1, 2), out_dir=tmp_path, seconds=1.0
    )
    shortfall = generation_shortfall(count, command="test")
    assert shortfall.produced == ("0", "1", "2")
    assert shortfall.ok
    refuse_unless_complete([shortfall], allow_partial=False)


def test_a_generation_pass_that_lost_a_spec_refuses_and_names_it(tmp_path: Path) -> None:
    for gen_id in (0, 2):
        bank_record(tmp_path, gen_id)
    count = GenerationCount(
        n_done=2, failed={1: "RuntimeError: device out of memory"},
        requested=(0, 1, 2), out_dir=tmp_path, seconds=1.0,
    )
    shortfall = generation_shortfall(count, command="test")
    assert shortfall.missing == ("1",)
    assert count.n_failed == 1 and not count.ok
    with pytest.raises(SystemExit) as exit_info:
        refuse_unless_complete([shortfall], allow_partial=False)
    assert exit_info.value.code == EXIT_SHORT
    receipt = json.loads((tmp_path / f"{RECEIPT_STEM}.json").read_text())
    assert receipt["failures"] == {"1": "RuntimeError: device out of memory"}


def test_a_resumed_generation_pass_that_banks_nothing_is_complete(tmp_path: Path) -> None:
    """Every record was already there, so the loop had nothing to do and nothing is short."""
    for gen_id in (0, 1, 2):
        bank_record(tmp_path, gen_id)
    count = GenerationCount(
        n_done=0, failed={}, requested=(0, 1, 2), out_dir=tmp_path, seconds=0.1
    )
    shortfall = generation_shortfall(count, command="test")
    assert shortfall.ok
    refuse_unless_complete([shortfall], allow_partial=False)


def test_a_sanctioned_short_generation_pass_keeps_its_own_status(tmp_path: Path) -> None:
    bank_record(tmp_path, 0)
    count = GenerationCount(
        n_done=1, failed={1: "ValueError: no chat template"},
        requested=(0, 1), out_dir=tmp_path, seconds=1.0,
    )
    with pytest.raises(SystemExit) as exit_info:
        refuse_unless_complete(
            [generation_shortfall(count, command="test", label="w0")], allow_partial=True
        )
    assert exit_info.value.code == EXIT_SHORT_SANCTIONED
    assert (tmp_path / f"{RECEIPT_STEM}-w0.json").exists()


# ── site 3: the gauntlet ──────────────────────────────────────────────────────

GAUNTLET_MODES = ["linear", "socratic", "contrastive", "dialectical", "analogical"]
GAUNTLET_TOPICS = 8
#: Sections wanting an optional dependency, a larger corpus, or minutes of CPU.
GAUNTLET_SKIP = ("2", "4", "5", "8", "9")
#: The same set with section 9 asked for, so the corpus without text fails it.
GAUNTLET_SKIP_WITH_SEMANTIC = ("2", "4", "5", "8")


@pytest.fixture(scope="module")
def gauntlet_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Five modes by eight topics of separable signatures, with no generated text.

    The absent text is deliberate and is this fixture's second job: section 9 reads
    it, and a corpus without it makes that section write the error stub whose
    refusal is under test — a real stub from the real section, needing no optional
    dependency to provoke.
    """
    folder = tmp_path_factory.mktemp("gauntlet_signatures")
    rng = np.random.default_rng(20260921)
    widths = {"tier1": 6, "tier2": 8, "tier2_5": 8, "tier3": 4}
    index = 0
    for mode_idx, mode in enumerate(GAUNTLET_MODES):
        for topic_idx in range(GAUNTLET_TOPICS):
            arrays: dict[str, np.ndarray] = {}
            names: list[str] = []
            slices: dict[str, list[int]] = {}
            cursor = 0
            for block, width in widths.items():
                signal = float(mode_idx) + 0.15 * rng.standard_normal(width)
                arrays[f"features_{block}"] = signal.astype(np.float32)
                slices[block] = [cursor, cursor + width]
                names.extend(f"{block}_feat_{i}" for i in range(width))
                cursor += width
            np.savez(
                folder / f"gen_{index:03d}.npz",
                feature_names=np.array(names),
                **arrays,
            )
            (folder / f"gen_{index:03d}.json").write_text(json.dumps({
                "generation_id": index,
                "topic": f"topic_{topic_idx}",
                "topic_idx": topic_idx,
                "mode": mode,
                "mode_idx": mode_idx,
                "num_generated_tokens": 200 + 5 * mode_idx + topic_idx,
                "prompt_length": 40,
                "system_prompt": f"system prompt for {mode}",
                "user_prompt": f"user prompt for topic {topic_idx}",
                "tier_slices": slices,
            }))
            index += 1
    return folder


def gauntlet(
    corpus: Path, out: Path, *extra: str, skip: tuple[str, ...] = GAUNTLET_SKIP
) -> int:
    """One end-to-end gauntlet invocation, through the command's own entry point."""
    return run_gauntlet.main(
        ["--run", "synthetic", "--sig-dir", str(corpus), "--output-dir", str(out),
         "--skip", *skip, *extra]
    )


def test_a_gauntlet_pass_over_the_sections_it_asked_for_exits_zero(
    gauntlet_corpus: Path, tmp_path: Path
) -> None:
    out = tmp_path / "analysis"
    assert gauntlet(gauntlet_corpus, out) == 0
    assert (out / "results.json").is_file()
    assert not (out / f"{RECEIPT_STEM}.json").exists()


def test_a_gauntlet_pass_holding_an_error_stub_refuses_and_names_the_section(
    gauntlet_corpus: Path, tmp_path: Path
) -> None:
    """Section 9 is asked for over a corpus with no text, so it writes a stub.

    The results file is still written and still worth reading, which is exactly why
    the status has to carry the verdict: nothing about the file says a section in it
    is a stub.
    """
    out = tmp_path / "analysis"
    with pytest.raises(SystemExit) as exit_info:
        gauntlet(gauntlet_corpus, out, skip=GAUNTLET_SKIP_WITH_SEMANTIC)
    assert exit_info.value.code == EXIT_SHORT
    assert (out / "results.json").is_file(), "the pass's own output survives the refusal"
    receipt = json.loads((out / f"{RECEIPT_STEM}.json").read_text())
    assert receipt["missing"] == ["semantic"], "a stub is not a result, so the section is absent"
    assert list(receipt["failures"]) == ["semantic"]
    assert "No generated text available" in receipt["failures"]["semantic"]
    assert "semantic" not in receipt["excluded"], "the section was asked for, not skipped"


def test_a_sanctioned_gauntlet_pass_keeps_the_stub_and_its_own_status(
    gauntlet_corpus: Path, tmp_path: Path
) -> None:
    out = tmp_path / "analysis"
    with pytest.raises(SystemExit) as exit_info:
        gauntlet(
            gauntlet_corpus, out, "--allow-partial", skip=GAUNTLET_SKIP_WITH_SEMANTIC
        )
    assert exit_info.value.code == EXIT_SHORT_SANCTIONED
    assert json.loads((out / f"{RECEIPT_STEM}.json").read_text())["sanctioned"] is True


def test_a_skipped_section_is_an_exclusion_so_the_narrowed_pass_is_complete(
    gauntlet_corpus: Path, tmp_path: Path
) -> None:
    """--skip is how a pass is narrowed on purpose, and a narrowed pass can be whole."""
    out = tmp_path / "analysis"
    assert gauntlet(gauntlet_corpus, out) == 0
    document = json.loads((out / "results.json").read_text())
    assert "semantic" not in document, "a skipped section stays unpopulated"


def test_a_resumed_gauntlet_pass_whose_sections_came_from_the_checkpoint_exits_zero(
    gauntlet_corpus: Path, tmp_path: Path
) -> None:
    """The hazard again, for sections: resume skips completed work and is complete."""
    out = tmp_path / "analysis"
    assert gauntlet(gauntlet_corpus, out) == 0
    assert gauntlet(gauntlet_corpus, out, "--resume") == 0
    assert not (out / f"{RECEIPT_STEM}.json").exists()


# ── site 4: the recompute ─────────────────────────────────────────────────────

RECOMPUTE_STEPS = 6
RECOMPUTE_HIDDEN = 16
RECOMPUTE_HEADS = 4
RECOMPUTE_KV_HEADS = 2
RECOMPUTE_HEAD_DIM = 4
RECOMPUTE_PROMPT = 4


def write_raw_tensors(raw_dir: Path, gen_id: int) -> None:
    """One generation's raw capture, written through the real saver.

    Going through `anamnesis.extraction.raw_saver.save_raw_tensors_all_layer` rather than
    hand-writing an npz is what makes the recompute under test the real one: the
    loop reads these files back with the loader the instrument uses.
    """
    from anamnesis.extraction.raw_saver import save_raw_tensors_all_layer

    rng = np.random.default_rng(gen_id)
    hidden = [
        rng.standard_normal((len(TINY_LAYERS) + 1, RECOMPUTE_HIDDEN)).astype(np.float32)
        for _ in range(RECOMPUTE_STEPS)
    ]
    attentions = []
    for step in range(RECOMPUTE_STEPS):
        weights = rng.random(
            (len(TINY_LAYERS), RECOMPUTE_HEADS, RECOMPUTE_PROMPT + step + 1)
        ).astype(np.float32)
        attentions.append(weights / weights.sum(axis=-1, keepdims=True))
    per_layer = {
        layer: [
            rng.standard_normal((RECOMPUTE_KV_HEADS, RECOMPUTE_HEAD_DIM)).astype(np.float32)
            for _ in range(RECOMPUTE_STEPS)
        ]
        for layer in TINY_LAYERS
    }
    raw = RawGenerationData(
        hidden_states=hidden,
        attentions=attentions,
        logits=[rng.standard_normal(32).astype(np.float32) for _ in range(RECOMPUTE_STEPS)],
        chosen_token_ids=np.arange(RECOMPUTE_STEPS, dtype=np.float32),
        pre_rope_keys=per_layer,
        prompt_length=RECOMPUTE_PROMPT,
        gate_activations={
            layer: [rng.standard_normal(24).astype(np.float32) for _ in range(RECOMPUTE_STEPS)]
            for layer in TINY_LAYERS
        },
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    save_raw_tensors_all_layer(
        raw, gen_id, raw_dir,
        prompt_length=RECOMPUTE_PROMPT,
        input_ids=list(range(RECOMPUTE_PROMPT + RECOMPUTE_STEPS)),
        top_k_logits=4,
    )


def recompute(raw_dir: Path, out_dir: Path, *, n_workers: int = 1) -> RecomputeCount:
    """Recompute every vector the raw directory offers, over the tiny geometry."""
    return recompute_all_features(
        raw_dir=raw_dir,
        output_dir=out_dir,
        config=tiny_extraction(),
        family_config=tiny_families(),
        metadata_dir=raw_dir.parent / "signatures",
        n_workers=n_workers,
    )


def test_a_clean_recompute_writes_a_vector_per_tensor_file(tmp_path: Path) -> None:
    raw_dir, out_dir = tmp_path / "raw_tensors", tmp_path / "signatures_v3"
    for gen_id in range(3):
        write_raw_tensors(raw_dir, gen_id)
    result = recompute(raw_dir, out_dir)
    assert result.requested == (0, 1, 2) and result.n_done == 3 and result.ok
    shortfall = recompute_shortfall(result, command="test")
    assert shortfall.produced == ("0", "1", "2")
    refuse_unless_complete([shortfall], allow_partial=False)


def test_a_recompute_that_wrote_four_of_five_refuses_and_says_which(tmp_path: Path) -> None:
    """The demonstrated failure: a bank one tensor file short of readable.

    The pass wrote four vectors and reported success, which is how a five-condition
    contrast quietly became a four-condition one.
    """
    raw_dir, out_dir = tmp_path / "raw_tensors", tmp_path / "signatures_v3"
    for gen_id in range(5):
        write_raw_tensors(raw_dir, gen_id)
    (raw_dir / "gen_003.npz").write_bytes(b"not an npz at all")

    result = recompute(raw_dir, out_dir)
    assert result.requested == (0, 1, 2, 3, 4)
    assert result.n_done == 4
    assert 3 in result.failed and result.failed[3]
    shortfall = recompute_shortfall(result, command="test")
    assert shortfall.missing == ("3",)
    with pytest.raises(SystemExit) as exit_info:
        refuse_unless_complete([shortfall], allow_partial=False)
    assert exit_info.value.code == EXIT_SHORT
    receipt = json.loads((out_dir / f"{RECEIPT_STEM}.json").read_text())
    assert receipt["n_requested"] == 5 and receipt["n_produced"] == 4
    assert receipt["missing"] == ["3"]


def test_a_recompute_accounts_for_a_failure_in_the_pool_as_well(tmp_path: Path) -> None:
    """The parallel path has its own accounting, so it gets its own case."""
    raw_dir, out_dir = tmp_path / "raw_tensors", tmp_path / "signatures_v3"
    for gen_id in range(3):
        write_raw_tensors(raw_dir, gen_id)
    (raw_dir / "gen_001.npz").write_bytes(b"not an npz at all")

    result = recompute(raw_dir, out_dir, n_workers=2)
    assert sorted(result.failed) == [1]
    assert recompute_shortfall(result, command="test").missing == ("1",)


def test_a_recompute_over_an_empty_directory_asks_for_nothing(tmp_path: Path) -> None:
    """Nothing requested is not the same as something lost, so it is not a refusal."""
    raw_dir, out_dir = tmp_path / "raw_tensors", tmp_path / "signatures_v3"
    raw_dir.mkdir()
    result = recompute(raw_dir, out_dir)
    assert result.requested == () and result.ok
    refuse_unless_complete([recompute_shortfall(result, command="test")], allow_partial=False)


def test_a_sanctioned_short_recompute_keeps_its_own_status(tmp_path: Path) -> None:
    raw_dir, out_dir = tmp_path / "raw_tensors", tmp_path / "signatures_v3"
    for gen_id in range(2):
        write_raw_tensors(raw_dir, gen_id)
    (raw_dir / "gen_000.npz").write_bytes(b"not an npz at all")
    result = recompute(raw_dir, out_dir)
    with pytest.raises(SystemExit) as exit_info:
        refuse_unless_complete(
            [recompute_shortfall(result, command="test")], allow_partial=True
        )
    assert exit_info.value.code == EXIT_SHORT_SANCTIONED
    assert json.loads((out_dir / f"{RECEIPT_STEM}.json").read_text())["missing"] == ["0"]
