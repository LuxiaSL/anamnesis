"""Running the vLLM lane: the engine step, the readout step, and the host they ran on.

A pass through the lane is two processes that must never be one. The engine
step (:func:`capture_rows`) builds the vLLM engine inside the declared envelope
and captures every requested row's substrate to disk. The readout step
(:func:`reduce_rows`) reduces those captures to feature vectors in a process
that has never imported the engine: the engine's batch-invariant mode replaces
torch's matrix products process-wide, and the readout's arithmetic is only the
fast lane's arithmetic without them. :func:`run_step` launches each in its own
interpreter with the environment that step requires, so a caller sets nothing
by hand.

Both entry points build on one resolution. :func:`check_install` is the install
conformance check: it captures the shipped fixtures twice one at a time and
once in batches of eight, reduces them, and decides the host's tier with
:func:`anamnesis.extraction.vllm.conformance.decide`, caching the receipt against
the host's fingerprint. :func:`replay_bank` banks a manifest's generations and
refuses to start without a cached receipt for this host that is not
``refused``. The fingerprint, the fixtures and the engine settings they are
decided against are computed here, once, so the receipt a bank cites describes
the configuration the bank ran under.

Captured substrate is large (about a gigabyte per row for the largest model),
so a replay works through its rows in chunks: capture a chunk, reduce it,
delete its captures, and move on.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from anamnesis.extraction.vllm.conformance import (
    CapturedFixture,
    ConformanceReceipt,
    FixtureSet,
    HostFingerprint,
    ReceiptCache,
    Tolerance,
    decide,
)
from anamnesis.extraction.vllm.envelope import (
    CONDITIONS,
    LANE_MODELS,
    READOUT_WORKSPACE,
    REQUIRED_ENV,
    SETTINGS,
    canonical_digest,
    engine_settings,
    enforce_lane_envelope,
    lane_id,
    lane_model,
    lane_preset,
    request_groups,
    require_environment,
    require_pinned_packages,
)
from anamnesis.provenance import digest_of_shas, file_sha

CALIBRATION_FILES = ("positional_means.npz", "pca_model.pkl")
"""The calibration artifacts the lane reads, and whose digest the fixtures pin."""

ATTENTION_ROUNDING = True
"""The second pass rounds each head's probabilities to the model dtype, the
rounding the fast lane's materialized attention carries."""

FIXTURES_ROOT = Path(__file__).parent / "fixtures"
"""Shipped fixture sets and tolerances, one directory per model."""

READOUT_DEVICE = "cuda:0"
"""The device the readout step runs on: the first device the process can see."""

STEP_MODULE = "anamnesis.extraction.vllm.steps"
"""The module each child process runs, with the step name and a spec file."""


def default_cache_dir() -> Path:
    """Where install-check receipts are cached unless a command names another place:
    under the output root, never inside the installation."""
    from anamnesis.config import outputs_root

    return outputs_root() / "vllm_conformance"


def fixtures_dir(model: str) -> Path:
    """The shipped fixture directory for ``model``.

    Raises
    ------
    ValueError
        When the model has no vLLM lane or no shipped fixtures.
    """
    if model not in LANE_MODELS:
        raise ValueError(f"{model!r} has no vLLM lane (declared: {', '.join(LANE_MODELS)})")
    path = FIXTURES_ROOT / model
    if not (path / "fixtures.json").is_file():
        raise ValueError(f"no conformance fixtures ship for {model!r}")
    return path


def load_fixtures(model: str) -> tuple[FixtureSet, Tolerance]:
    """The fixture set and tolerance for ``model``, digest-checked: shipped, or an
    extension lane's through its guard, :func:`anamnesis.extraction.vllm.extensions.admit`."""
    if model not in LANE_MODELS:
        from anamnesis.extraction.vllm.extensions import admit, declared_lane

        if declared_lane(model) is not None:
            admitted = admit(model)
            return admitted.fixtures, admitted.tolerance
    path = fixtures_dir(model)
    fixtures = FixtureSet.load(path)
    tolerance = Tolerance.model_validate_json((path / "tolerance.json").read_text())
    if fixtures.model != model or tolerance.model != model:
        raise ValueError(f"the shipped fixtures under {path} are not {model!r}'s")
    if fixtures.lane_id != lane_id(model):
        raise ValueError(f"the shipped fixtures name lane {fixtures.lane_id}, not "
                         f"{model!r}'s lane {lane_id(model)}")
    return fixtures, tolerance


def calibration_digest(calib_dir: Path) -> str:
    """One digest over the calibration artifacts in :data:`CALIBRATION_FILES`."""
    return digest_of_shas({name: file_sha(Path(calib_dir) / name)
                           for name in CALIBRATION_FILES})


def require_fixture_calibration(fixtures: FixtureSet, calib_dir: Path) -> None:
    """Refuse a calibration other than the one the fixtures were reduced with.

    Raises
    ------
    ValueError
        When the digests differ. A different calibration gives different vectors
        on every row, so the check could only refuse after a full capture.
    """
    if calibration_digest(calib_dir) != fixtures.calibration_sha256:
        raise ValueError(f"the calibration in {calib_dir} is not the one the {fixtures.model} "
                         "fixtures were reduced with")


def lane_source_digest() -> str:
    """Digest of the source a pass runs: every module of this package, and the fast
    lane's reducers, configuration and schema it reduces with."""
    fast = Path(__file__).parents[1] / "fast"
    files = sorted(Path(__file__).parent.glob("*.py")) + [
        fast / name for name in ("features.py", "ops.py", "attention.py", "families.py",
                                 "batch_layout.py", "schema.py")] + [
        Path(__file__).parents[1] / "replay_config.py"]
    return digest_of_shas({f"{p.parent.name}/{p.name}": file_sha(p) for p in files})


def settings_digest(model: str) -> str:
    """Digest of every engine setting and capture switch a pass for ``model`` runs
    under, and of the source it runs. A receipt keyed by it is never served to a
    run with other settings or other code."""
    return canonical_digest(dict(
        settings={c: engine_settings(model, c) for c in CONDITIONS},
        attention_rounding=ATTENTION_ROUNDING, source=lane_source_digest()))


def child_environment(step: str, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment a step's process starts in.

    ``capture`` adds :data:`anamnesis.extraction.vllm.envelope.REQUIRED_ENV`;
    ``reduce`` removes every ``VLLM_`` variable, fixes the readout workspace and
    pins the BLAS thread pools to one. Everything else, device visibility
    included, passes through unchanged.
    """
    env = dict(os.environ if base is None else base)
    if step == "capture":
        env.update(REQUIRED_ENV)
    elif step == "reduce":
        for key in list(env):
            if key.startswith("VLLM_"):
                del env[key]
        env.update(CUBLAS_WORKSPACE_CONFIG=READOUT_WORKSPACE, OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    else:
        raise ValueError(f"unknown step {step!r}")
    return env


def run_step(step: str, spec: Mapping[str, Any], spec_path: Path) -> None:
    """Write ``spec`` and run one step in a fresh interpreter.

    Raises
    ------
    RuntimeError
        When the step exits non-zero; its own error is on its stderr.
    """
    spec_path.write_text(json.dumps(spec, indent=2, default=str) + "\n")
    result = subprocess.run([sys.executable, "-m", STEP_MODULE, step, str(spec_path)],
                            env=child_environment(step), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"the {step} step exited {result.returncode} ({spec_path})")


def promote_logprob_inputs(sampler) -> None:
    """Give the sampler's logprob computation a float32 input.

    Model outputs are untouched: only the sampler's own copy is promoted, because
    the invariant log-softmax has no float16 path.
    """
    original = sampler.compute_logprobs

    def promoted(logits):
        return original(logits.float())

    sampler.compute_logprobs = promoted


def capture_rows(spec: Mapping[str, Any]) -> None:
    """The engine step: capture every row of ``spec`` ``passes`` times.

    ``spec`` holds ``model``, ``model_path``, ``condition_id``, ``passes``,
    ``out`` and ``rows`` (each with ``generation_id``, ``input_ids``,
    ``prompt_length`` and ``end``). Pass ``i`` is written to its own directory under
    ``out``, beside a capture record of the settings, the guard's record and the
    device.
    """
    require_environment()
    versions = require_pinned_packages()
    model, condition_id = spec["model"], spec["condition_id"]
    settings = engine_settings(model, condition_id)
    guard = enforce_lane_envelope(settings, os.environ, lane=lane_id(model), model=model)

    from anamnesis.extraction.vllm.backend import register_instrumented_backend
    resolved = register_instrumented_backend()
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from anamnesis.config import resolve_preset
    from anamnesis.extraction.vllm.runner import capture_groups

    rows = [dict(generation_id=int(r["generation_id"]), input_ids=list(r["input_ids"]),
                 prompt_length=int(r["prompt_length"]), end=int(r["end"]))
            for r in spec["rows"]]
    condition = CONDITIONS[condition_id]
    groups = request_groups([r["generation_id"] for r in rows], condition["max_num_seqs"])
    llm = LLM(model=str(spec["model_path"]), **settings)
    runner = (llm.llm_engine.engine_core.engine_core.model_executor
              .driver_worker.worker.model_runner)
    if type(runner.model).__name__ != "LlamaForCausalLM":
        raise ValueError("the engine did not resolve the model to its native Llama")
    if lane_model(model)["logprob_wrapper"] == "explicit-fp32-input":
        promote_logprob_inputs(runner.sampler)
    sampling = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0, logprobs=0,
                              detokenize=False, seed=settings["seed"])
    preset = resolve_preset(lane_preset(model))
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=False)
    for index in range(int(spec["passes"])):
        capture_groups(llm, runner, rows, groups, condition, preset.sampled_layers, sampling,
                       TokensPrompt, out / f"pass-{index}",
                       attention_rounding=ATTENTION_ROUNDING)
    (out / "capture.json").write_text(json.dumps(dict(
        lane_id=lane_id(model), condition=condition, settings=settings,
        passes=int(spec["passes"]), attention_rounding=ATTENTION_ROUNDING,
        startup_guard=guard, resolved_backend=resolved, packages=versions,
        device=torch.cuda.get_device_name(0)), indent=2, default=str) + "\n")


def verify_capture_receipt(path: Path, receipt: Mapping[str, Any], generation_id: int) -> None:
    """Refuse a capture whose receipt does not describe these bytes and this row.

    Raises
    ------
    ValueError
        On a different generation id, a failed hook comparison, a missing schedule
        digest, or file bytes other than the receipt's.
    """
    if type(receipt.get("generation_id")) is not int or receipt["generation_id"] != generation_id:
        raise ValueError("the capture receipt names another generation")
    if receipt.get("hook_noninterference") is not True:
        raise ValueError("the capture's hooks changed its generation")
    schedule = receipt.get("schedule_sha256")
    if not isinstance(schedule, str) or len(schedule) != 64:
        raise ValueError("the capture receipt has no schedule digest")
    if receipt.get("raw_sha256") != file_sha(path):
        raise ValueError("the capture file's bytes differ from its receipt")


def readout_lane(model: str, calib_dir: Path, feature_names: Sequence[str],
                 device: str = READOUT_DEVICE):
    """The fast lane's reduction configuration for ``model``, at ``feature_names``."""
    from anamnesis.config import resolve_preset
    from anamnesis.extraction.calibration import load_calibration
    from anamnesis.extraction.fast.features import GpuFeatureLane
    from anamnesis.extraction.replay_config import native_replay_configs

    extraction, families = native_replay_configs(resolve_preset(lane_preset(model)))
    positional_means, components, mean = load_calibration(Path(calib_dir), True)
    return GpuFeatureLane(extraction, families, list(feature_names), positional_means,
                          components, mean, device=device,
                          calibration_sha256=calibration_digest(calib_dir),
                          replay_path="full")


def reduce_rows(spec: Mapping[str, Any]) -> None:
    """The readout step: reduce every captured pass of ``spec`` and delete its raws.

    ``spec`` holds ``model``, ``calib_dir``, ``feature_names``, ``captures`` (the
    engine step's ``out``) and ``rows``. Each pass directory gains one npz of the
    arrays ``generation_ids``, ``vectors`` and, when the model's configuration
    enables it, ``knnlm`` (the final hidden state at each span's last position). A
    raw capture is deleted once its vector is written; with ``keep_hidden``, the first
    pass keeps each row's block outputs beside its vectors, one file per row.
    """
    import torch

    from anamnesis.extraction.vllm.readout import assert_clean_readout_process, reduce_capture
    from anamnesis.extraction.vllm.receipts import capture_receipt

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert_clean_readout_process(READOUT_DEVICE)
    lane = readout_lane(spec["model"], Path(spec["calib_dir"]), spec["feature_names"])
    rows = {int(r["generation_id"]): r for r in spec["rows"]}
    passes = sorted(p for p in Path(spec["captures"]).glob("pass-*") if p.is_dir())
    if not passes:
        raise ValueError(f"no capture passes under {spec['captures']}")

    def on_device(value):
        return ({k: on_device(v) for k, v in value.items()}
                if isinstance(value, dict) else value.to(READOUT_DEVICE))

    for directory in passes:
        gids, vectors, knnlm = [], [], []
        for gid in sorted(rows):
            path = directory / f"row-{gid:05d}.pt"
            receipt = json.loads((directory / f"row-{gid:05d}.json").read_text())
            verify_capture_receipt(path, receipt, gid)
            raw = torch.load(path, map_location="cpu", weights_only=True)
            if capture_receipt(raw) != receipt["substrate"]:
                raise ValueError(f"row {gid}: capture tensors differ from their receipt")
            with torch.inference_mode():
                readout = reduce_capture(lane, on_device(raw),
                                         start=int(rows[gid]["prompt_length"]),
                                         end=int(rows[gid]["end"]), model=spec["model"])
            gids.append(gid)
            vectors.append(np.asarray(readout.features, dtype=np.float32))
            if spec.get("keep_hidden") and directory.name == "pass-0":
                torch.save(raw["hidden"].clone(), directory / f"row-{gid:05d}.hidden.pt")
            if lane.config.enable_knnlm_baseline:
                knnlm.append(raw["hidden"][-1][-1].float().numpy().copy())
        arrays = dict(generation_ids=np.asarray(gids, dtype=np.int64),
                      vectors=np.stack(vectors))
        if knnlm:
            arrays["knnlm"] = np.stack(knnlm)
        np.savez(directory / "vectors.npz", **arrays)
        for gid in gids:
            (directory / f"row-{gid:05d}.pt").unlink()


def pass_vectors(directory: Path) -> dict[int, np.ndarray]:
    """A reduced pass's vectors by generation id."""
    with np.load(Path(directory) / "vectors.npz", allow_pickle=False) as z:
        return {int(g): z["vectors"][i].copy() for i, g in enumerate(z["generation_ids"])}


def host_fingerprint(fixtures: FixtureSet, tolerance: Tolerance,
                     model_path: Path) -> HostFingerprint:
    """What makes this host this host for ``fixtures.model``'s receipt.

    Reads the first visible device's name and uuid, the driver, the CUDA runtime,
    the installed torch, vLLM and anamnesis versions, and the checkpoint's bytes.
    """
    import importlib.metadata

    import torch

    from anamnesis.extraction.fast.runtime import weight_file_digests

    properties = torch.cuda.get_device_properties(0)
    try:
        import pynvml
        pynvml.nvmlInit()
        driver = str(pynvml.nvmlSystemGetDriverVersion())
    except Exception:  # noqa: BLE001 - the driver is recorded when it can be read
        driver = "unreadable"
    try:
        anamnesis_version = importlib.metadata.version("anamnesis")
    except importlib.metadata.PackageNotFoundError:
        anamnesis_version = "source-tree"
    return HostFingerprint(
        gpu_name=properties.name, gpu_uuid=str(properties.uuid), driver=driver,
        cuda_runtime=str(torch.version.cuda), torch=torch.__version__,
        vllm=importlib.metadata.version("vllm"), anamnesis=anamnesis_version,
        checkpoint_sha256=digest_of_shas(weight_file_digests(model_path)),
        engine_settings_sha256=settings_digest(fixtures.model),
        fixture_digest=fixtures.digest, tolerance_digest=tolerance.digest)


def fixture_rows(fixtures: FixtureSet) -> list[dict[str, Any]]:
    """The fixture rows as the engine step takes them."""
    return [dict(generation_id=r.generation_id, input_ids=list(r.input_ids),
                 prompt_length=r.prompt_length, end=r.end) for r in fixtures.rows]


def check_install(model: str, model_path: Path, calib_dir: Path, work_dir: Path,
                  cache_dir: Path, *, refresh: bool = False) -> tuple[ConformanceReceipt, bool]:
    """Decide this host's tier for ``model``; return the receipt and whether it was cached.

    The fixtures, the calibration and the installed engine packages are checked
    first, then the checkpoint's digest. A cached receipt for an equal fingerprint
    is returned without capturing, unless ``refresh``. Otherwise the fixtures are
    captured twice under
    ``full-b1-order0`` and once under ``full-b8-order0`` into ``work_dir``, which
    must not exist, reduced, and decided.
    """
    fixtures, tolerance = load_fixtures(model)
    require_fixture_calibration(fixtures, calib_dir)
    require_pinned_packages()
    fingerprint = host_fingerprint(fixtures, tolerance, model_path)
    cache = ReceiptCache(cache_dir)
    cached = cache.load(fingerprint)
    if cached is not None and not refresh:
        return cached, True
    if fingerprint.checkpoint_sha256 != fixtures.checkpoint_sha256:
        raise ValueError(f"the checkpoint in {model_path} is not the one the {model} fixtures "
                         f"were produced from: its config and weights digest to "
                         f"{fingerprint.checkpoint_sha256}, the fixtures to "
                         f"{fixtures.checkpoint_sha256}")
    captured = capture_repeats(model, model_path, calib_dir, fixture_rows(fixtures),
                               fixtures.feature_names, work_dir)
    receipt = decide(fixtures, tolerance, fingerprint, captured)
    cache.store(receipt)
    return receipt, False


def capture_repeats(model: str, model_path: Path, calib_dir: Path,
                    rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str],
                    work_dir: Path, *, keep_hidden: bool = False) -> list[CapturedFixture]:
    """Each row's vectors captured twice under ``full-b1-order0`` and once under
    ``full-b8-order0`` into ``work_dir`` (which must not exist); a row missing from
    any pass is left out. With ``keep_hidden``, the first single pass keeps each row's
    block outputs for a caller that compares them with the anchor's and deletes them."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=False)
    rows = list(rows)
    for label, condition_id, passes in (("single", "full-b1-order0", 2),
                                        ("batched", "full-b8-order0", 1)):
        run_step("capture", dict(model=model, model_path=str(model_path),
                                 condition_id=condition_id, passes=passes,
                                 out=str(work_dir / label), rows=rows),
                 work_dir / f"{label}.capture.json")
        run_step("reduce", dict(model=model, calib_dir=str(calib_dir),
                                feature_names=list(feature_names),
                                captures=str(work_dir / label), rows=rows,
                                keep_hidden=keep_hidden and label == "single"),
                 work_dir / f"{label}.reduce.json")
    first = pass_vectors(work_dir / "single" / "pass-0")
    repeat = pass_vectors(work_dir / "single" / "pass-1")
    batched = pass_vectors(work_dir / "batched" / "pass-0")
    common = sorted(set(first) & set(repeat) & set(batched))
    return [CapturedFixture(generation_id=g, first=first[g], repeat=repeat[g],
                            batched=batched[g]) for g in common]


def usable_receipt(model: str, model_path: Path, cache_dir: Path) -> ConformanceReceipt:
    """This host's cached receipt for ``model``, which must not be ``refused``.

    Raises
    ------
    ValueError
        When no receipt is cached for this host's fingerprint, naming the command
        that produces one, or when the cached one refused the host.
    """
    fixtures, tolerance = load_fixtures(model)
    require_pinned_packages()
    fingerprint = host_fingerprint(fixtures, tolerance, model_path)
    receipt = ReceiptCache(cache_dir).load(fingerprint)
    if receipt is None:
        raise ValueError(f"no install check is cached for {model} on this host; run "
                         "python -m anamnesis.scripts.qualify_vllm first")
    if receipt.tier == "refused":
        raise ValueError(f"the install check refused this host for {model}: "
                         f"{'; '.join(receipt.reasons[:3])}")
    return receipt


def replay_rows(entries: Mapping[str, Mapping[str, Any]],
                ids: Sequence[int]) -> list[dict[str, Any]]:
    """The selected manifest entries as the engine step takes them.

    Raises
    ------
    ValueError
        When a span is shorter than two predicted positions or longer than the
        lane's context, :data:`anamnesis.extraction.vllm.envelope.SETTINGS`
        ``max_model_len``.
    """
    rows = []
    for gid in ids:
        entry = entries[str(gid)]
        input_ids = [int(x) for x in entry["input_ids"]]
        prompt_length = int(entry["prompt_length"])
        end = len(input_ids)
        if not 0 < prompt_length < end - 1:
            raise ValueError(f"generation {gid}: its span needs a prompt and at least "
                             "two generated tokens")
        if end > SETTINGS["max_model_len"]:
            raise ValueError(f"generation {gid}: {end} tokens exceed the lane's context "
                             f"of {SETTINGS['max_model_len']}")
        rows.append(dict(generation_id=gid, input_ids=input_ids,
                         prompt_length=prompt_length, end=end))
    return rows


def source_metadata(path: Path | None) -> dict[int, dict]:
    """Per-generation source metadata by id, from a ``metadata.json``-shaped file.

    Raises
    ------
    ValueError
        When the file holds no generation list or names a generation twice.
    """
    if path is None:
        return {}
    document = json.loads(Path(path).read_text())
    generations = document["generations"] if isinstance(document, dict) else document
    if not isinstance(generations, list):
        raise ValueError("source metadata must contain a generation list")
    result: dict[int, dict] = {}
    for record in generations:
        key = int(record["generation_id"])
        if key in result:
            raise ValueError(f"source metadata names generation {key} twice")
        result[key] = dict(record)
    return result


def feature_schema_sha256(feature_names: Sequence[str]) -> str:
    """The schema digest the fast lane stamps, over the ordered feature names."""
    return hashlib.sha256(json.dumps(list(feature_names)).encode()).hexdigest()


def span_schemas(model: str, calib_dir: Path, rows: Sequence[Mapping[str, Any]],
                 feature_names: Sequence[str]) -> dict[int, Any]:
    """Each row's fast-lane schema, which must name exactly ``feature_names``.

    Raises
    ------
    ValueError
        For a span so short that its schema differs from the fixtures'; the lane's
        agreement was measured on the fixtures' schema only.
    """
    from anamnesis.config import resolve_preset
    from anamnesis.extraction.calibration import load_calibration
    from anamnesis.extraction.fast.schema import resolve_gpu_schema
    from anamnesis.extraction.replay_config import native_replay_configs

    preset = resolve_preset(lane_preset(model))
    extraction, families = native_replay_configs(preset)
    _, components, _ = load_calibration(Path(calib_dir), True)
    schemas, by_steps = {}, {}
    for row in rows:
        steps = int(row["end"]) - int(row["prompt_length"]) - 1
        if steps not in by_steps:
            by_steps[steps] = resolve_gpu_schema(preset.num_layers, steps, extraction,
                                                 families, components)
            if tuple(by_steps[steps].feature_names) != tuple(feature_names):
                raise ValueError(f"a {steps}-step span resolves to a schema other than the "
                                 "fixtures'; the vLLM lane does not cover it")
        schemas[int(row["generation_id"])] = by_steps[steps]
    return schemas


def replay_bank(model: str, model_path: Path, calib_dir: Path, rows: Sequence[Mapping[str, Any]],
                output: Path, work_dir: Path, cache_dir: Path, *, chunk_rows: int,
                metadata: Mapping[int, Mapping[str, Any]], provenance: Mapping[str, Any]) -> int:
    """Bank ``rows`` through the lane into ``output``; return the rows written.

    Rows are captured and reduced ``chunk_rows`` at a time under
    ``full-b8-order0`` (``full-b1-order0`` for a final chunk of fewer than eight),
    each chunk's raw captures deleted once reduced. Every row is written with
    :func:`anamnesis.extraction.feature_pipeline.save_features`, stamped with the
    lane id of this host's install check and an ``extraction_lane`` receipt naming
    its tier and receipt digest. ``output`` and ``work_dir`` must not exist.
    """
    import shutil

    from anamnesis.extraction.feature_pipeline import save_features
    from anamnesis.extraction.state_extractor import ExtractionResult

    if chunk_rows < 8:
        raise ValueError("a chunk holds at least eight rows, one full batch")
    fixtures, _ = load_fixtures(model)
    require_fixture_calibration(fixtures, calib_dir)
    receipt = usable_receipt(model, model_path, cache_dir)
    names = list(fixtures.feature_names)
    schemas = span_schemas(model, calib_dir, rows, names)
    schema_sha = feature_schema_sha256(names)
    output, work_dir = Path(output), Path(work_dir)
    output.mkdir(parents=True, exist_ok=False)
    work_dir.mkdir(parents=True, exist_ok=False)
    (output / "deployment.json").write_text(json.dumps(dict(
        provenance, lane_id=receipt.lane_id, qualified_lane_id=receipt.qualified_lane_id,
        conformance_tier=receipt.tier, conformance_receipt_sha256=receipt.digest,
        conformance_fingerprint=receipt.fingerprint.model_dump(mode="json"),
        fixture_digest=fixtures.digest, feature_schema_sha256=schema_sha,
        calibration_sha256=fixtures.calibration_sha256, raw_tensors_saved=False,
        certified=False), indent=2, default=str) + "\n")
    written = 0
    for index, start in enumerate(range(0, len(rows), chunk_rows)):
        chunk = list(rows[start:start + chunk_rows])
        condition_id = "full-b8-order0" if len(chunk) >= 8 else "full-b1-order0"
        captures = work_dir / f"chunk-{index:04d}"
        run_step("capture", dict(model=model, model_path=str(model_path),
                                 condition_id=condition_id, passes=1, out=str(captures),
                                 rows=chunk), work_dir / f"chunk-{index:04d}.capture.json")
        run_step("reduce", dict(model=model, calib_dir=str(calib_dir), feature_names=names,
                                captures=str(captures), rows=chunk),
                 work_dir / f"chunk-{index:04d}.reduce.json")
        with np.load(captures / "pass-0" / "vectors.npz", allow_pickle=False) as z:
            gids = [int(g) for g in z["generation_ids"]]
            vectors = z["vectors"].copy()
            knnlm = z["knnlm"].copy() if "knnlm" in z.files else None
        by_id = {int(r["generation_id"]): r for r in chunk}
        for i, gid in enumerate(gids):
            row = by_id[gid]
            result = ExtractionResult(vectors[i], names, schemas[gid].family_slices,
                                      None if knnlm is None else knnlm[i])
            record = dict(metadata.get(gid, {}))
            record.update(
                generation_id=gid,
                lane_id=receipt.lane_id,
                extraction_lane=dict(
                    lane_id=receipt.lane_id, engine="vllm",
                    qualified_lane_id=receipt.qualified_lane_id,
                    conformance_tier=receipt.tier,
                    conformance_receipt_sha256=receipt.digest,
                    condition_id=condition_id, span_start=int(row["prompt_length"]),
                    span_end=int(row["end"]),
                    input_tokens_sha256=hashlib.sha256(
                        np.asarray(row["input_ids"], dtype="<i8").tobytes()).hexdigest(),
                    feature_schema_sha256=schema_sha,
                    calibration_sha256=fixtures.calibration_sha256, certified=False),
                raw_tensors_saved=False,
                certified=False,
            )
            save_features(gid, result, record, output)
            written += 1
        shutil.rmtree(captures / "pass-0", ignore_errors=False)
    return written
