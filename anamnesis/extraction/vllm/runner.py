"""Capture groups of requests through an engine the caller built, one group at a time.

:func:`capture_groups` takes rows already split into engine batches (see
:func:`anamnesis.extraction.vllm.envelope.request_groups`). For each group it
first generates without hooks, then again under :class:`LaneCapture`, and
refuses unless the hooked run produced the same tokens and logprobs: capturing
must not change what the model computes. It proves the scheduler realized the
declared batch capacity and prefilled every prompt whole, and writes each
retained row's substrate as a torch file named for its generation id, beside a
JSON receipt of its tensor contents, and each group's schedule record.

Nothing here builds an engine or deletes a file; the command that calls it owns
both.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from anamnesis.extraction.vllm.capture import LaneCapture
from anamnesis.extraction.vllm.receipts import assert_substrate_fields, capture_receipt
from anamnesis.provenance import file_sha


def write_json(path, value) -> None:
    """Write ``value`` as key-sorted JSON; NaN is refused."""
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)
                          + '\n')


def generation_record(outputs, expected_ids):
    """One request's output as comparable JSON: the prompt ids, the one sampled
    token, the prompt and sample logprobs (with ranks and decoded tokens) and
    the finish reason, without request ids or timings.

    Raises
    ------
    ValueError
        When the output is not exactly one finished request over
        ``expected_ids`` with one sampled token and every logprob present.
    """
    if len(outputs) != 1:
        raise ValueError('expected exactly one generation request')
    request = outputs[0]
    if (list(request.prompt_token_ids) != expected_ids or len(request.outputs) != 1
            or not request.finished):
        raise ValueError('generation request identity/completion mismatch')

    def logs(values):
        if values is None:
            raise ValueError('requested logprobs absent')
        return [
            None if row is None else {
                str(token): dict(logprob=float(lp.logprob), rank=lp.rank,
                                 decoded_token=lp.decoded_token)
                for token, lp in sorted(row.items())}
            for row in values]

    completion = request.outputs[0]
    if request.prompt_logprobs is None:
        raise ValueError('requested logprobs absent')
    if len(completion.token_ids) != 1 or len(request.prompt_logprobs) != len(expected_ids):
        raise ValueError('generation/logprob lengths differ from the request')
    result = dict(
        prompt_token_ids=list(request.prompt_token_ids),
        token_ids=list(completion.token_ids),
        prompt_logprobs=logs(request.prompt_logprobs),
        logprobs=logs(completion.logprobs),
        finish_reason=completion.finish_reason,
        stop_reason=completion.stop_reason,
    )
    if len(result['logprobs']) != 1:
        raise ValueError('sampling logprob length mismatch')
    return json.loads(json.dumps(result, sort_keys=True, allow_nan=False))


def _normalized(outputs, members, by_id, expected_external=None):
    if len(outputs) != len(members):
        raise ValueError("output count differs from submitted occurrence count")
    external = [x.request_id for x in outputs]
    if any(not isinstance(x, str) or not x.isdecimal() for x in external):
        raise ValueError("engine outputs require numeric external request ids")
    numeric = [int(x) for x in external]
    if (
        len(set(external)) != len(external)
        or numeric != sorted(numeric)
        or len(set(numeric)) != len(numeric)
    ):
        raise ValueError("engine outputs are not in unique numeric submission order")
    if numeric != list(range(numeric[0], numeric[0] + len(numeric))):
        raise ValueError("unexpected gap in group request IDs")
    if expected_external is not None and external != expected_external:
        raise ValueError(
            "hooked output IDs differ from exact queued occurrence mapping"
        )
    return [
        dict(
            occurrence_id=member["occurrence_id"],
            generation_id=member["generation_id"],
            retain=member["retain"],
            external_request_id=ext,
            generation=generation_record(
                [output], by_id[member["generation_id"]]["input_ids"]
            ),
        )
        for output, member, ext in zip(outputs, members, external, strict=True)
    ]


def validate_schedule(trace, requests, capacity):
    """Prove from the recorded trace, not from the settings, how the group ran:
    the first step held ``capacity`` requests and every prompt was prefilled
    whole in one step."""
    if not trace or len(trace[0]["requests"]) != capacity:
        raise ValueError("requested first-step concurrency was not realized")
    cursor = {r: 0 for r in requests}
    for index, step in enumerate(trace):
        if (
            step["step"] != index
            or not step["requests"]
            or len(step["requests"]) > capacity
        ):
            raise ValueError("invalid scheduling step/concurrency")
        seen = set()
        offset = 0
        for item in step["requests"]:
            r, s, n = item["request_id"], item["start"], item["count"]
            if r not in requests or r in seen or type(n) is not int or n <= 0:
                raise ValueError("unknown/duplicate/empty scheduled request")
            seen.add(r)
            if s != cursor[r] or item["offset"] != offset or s + n > requests[r]["end"]:
                raise ValueError("scheduler span missing, duplicated or mispacked")
            if s != 0 or n != requests[r]["end"]:
                raise ValueError("a prompt was prefilled in chunks")
            cursor[r] = s + n
            offset += n
        if offset != step["total_tokens"]:
            raise ValueError("scheduler packed total mismatch")
    if any(cursor[r] != spec["end"] for r, spec in requests.items()):
        raise ValueError("scheduler omitted request prompt positions")
    return dict(
        actual_first_step_concurrency=len(trace[0]["requests"]),
        max_concurrency=max(len(s["requests"]) for s in trace),
        max_request_chunk=max(i["count"] for s in trace for i in s["requests"]),
        one_token_chunks=sum(i["count"] == 1 for s in trace for i in s["requests"]),
        full_prompt_coverage=True,
    )


def _validate_groups(rows, groups, condition):
    capacity = condition["max_num_seqs"]
    if capacity not in (1, 2, 4, 8):
        raise ValueError("invalid condition capacity")
    by_id = {r["generation_id"]: r for r in rows}
    if not rows or len(by_id) != len(rows):
        raise ValueError("nonempty unique row roster required")
    for row in rows:
        if (
            not isinstance(row["input_ids"], list)
            or row["end"] != len(row["input_ids"])
            or not 0 <= row["prompt_length"] < row["end"] - 1
        ):
            raise ValueError("invalid banked row span")
    targets = []
    occurrences = set()
    for group in groups:
        if not any(member.get("retain") is True for member in group):
            raise ValueError("each bounded group requires a retained target")
        if len(group) != capacity:
            raise ValueError("every group must realize declared capacity")
        for member in group:
            if (
                member["generation_id"] not in by_id
                or type(member["retain"]) is not bool
            ):
                raise ValueError("unknown row or ambiguous retain flag")
            occurrence = member["occurrence_id"]
            if (
                not isinstance(occurrence, str)
                or not occurrence
                or occurrence in occurrences
            ):
                raise ValueError("occurrence IDs must be nonempty and globally unique")
            occurrences.add(occurrence)
            if member["retain"]:
                targets.append(member["generation_id"])
    if len(targets) != len(by_id) or set(targets) != set(by_id):
        raise ValueError("groups must retain each roster row exactly once")
    return by_id


def capture_groups(
    llm,
    runner,
    rows,
    groups,
    condition,
    sampled_layers,
    sampling,
    tokens_prompt,
    out,
    *,
    attention_rounding: bool,
):
    """Capture every group; return the row receipts and per-group evidence.

    ``rows`` carry ``generation_id``, ``input_ids``, ``prompt_length`` and
    ``end``; ``groups`` are the engine batches over them; ``sampling`` must ask
    for one greedy token with prompt and sample logprobs. The caller owns the
    environment and the engine. The engine queue must be empty on entry and is
    left empty.

    Raises
    ------
    ValueError, RuntimeError
        On any departure from the declared groups, the scheduling or the
        substrate, and when the hooks changed the generation. A failing group
        leaves a failure record beside its captures naming why.
    """
    by_id = _validate_groups(rows, groups, condition)
    if getattr(sampling, "max_tokens", None) != 1 or getattr(sampling, "n", 1) != 1:
        raise ValueError("one output token and one completion required")
    if (
        getattr(sampling, "prompt_logprobs", None) != 0
        or getattr(sampling, "logprobs", None) != 0
        or getattr(sampling, "temperature", None) != 0
    ):
        raise ValueError("deterministic prompt/sample logprob capture required")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob("row-*")) or list(out.glob("group-*")):
        raise FileExistsError("capture output files already exist")
    records = []
    group_records = []
    for group_index, members in enumerate(groups):
        begun = time.perf_counter()
        stem = f"group-{group_index:04d}"
        tap = None
        mapping = {}
        try:
            if llm.llm_engine.has_unfinished_requests():
                raise RuntimeError("engine has requests before bounded group")
            prompts = [
                tokens_prompt(prompt_token_ids=by_id[m["generation_id"]]["input_ids"])
                for m in members
            ]
            control_outputs = llm.generate(prompts, sampling, use_tqdm=False)
            controls = _normalized(control_outputs, members, by_id)
            del control_outputs
            write_json(
                out / f"{stem}.control.json",
                dict(
                    condition=condition, group_index=group_index, occurrences=controls
                ),
            )
            if llm.llm_engine.has_unfinished_requests():
                raise RuntimeError("control left pending requests")
            requests = {}
            expected_external = []
            for member, prompt in zip(members, prompts, strict=True):
                internal = llm._add_request(prompt, sampling)
                if internal in requests:
                    raise ValueError("duplicate assigned internal request ID")
                state = llm.llm_engine.output_processor.request_states[internal]
                external = state.external_req_id
                if not isinstance(external, str) or not external.isdecimal():
                    raise ValueError("the engine assigned a non-numeric external request id")
                expected_external.append(external)
                row = by_id[member["generation_id"]]
                requests[internal] = dict(
                    input_ids=row["input_ids"],
                    start=row["prompt_length"],
                    end=row["end"],
                    retain=member["retain"],
                )
                mapping[internal] = dict(**member, external_request_id=external)
            with LaneCapture(
                runner,
                requests,
                sampled_layers,
                attention_rounding=attention_rounding,
            ) as tap:
                observed_outputs = llm._run_engine(use_tqdm=False)
            observed = _normalized(observed_outputs, members, by_id, expected_external)
            del observed_outputs
            captures = tap.finish()
            evidence = validate_schedule(
                tap.schedule, requests, condition["max_num_seqs"])
            if llm.llm_engine.has_unfinished_requests():
                raise RuntimeError("capture left pending requests")
            unchanged = all(
                a["generation"] == b["generation"]
                for a, b in zip(controls, observed, strict=True)
            )
            schedule = dict(
                capture_contract="lane-capture/1",
                condition=condition,
                attention_rounding=attention_rounding,
                group_index=group_index,
                internal_requests=mapping,
                trace=tap.schedule,
                evidence=evidence,
                peak_fragment_bytes=tap.peak_fragment_bytes,
                hook_noninterference=unchanged,
                controls=controls,
                observed=observed,
            )
            schedule_path = out / f"{stem}.schedule.json"
            write_json(schedule_path, schedule)
            schedule_digest = file_sha(schedule_path)
            expected_retained = {r for r, s in requests.items() if s["retain"]}
            if set(captures) != expected_retained:
                raise ValueError("capture retained request set differs from protocol")
            # Rows are written before the comparison refuses, so a failure keeps its evidence.
            for internal, capture in captures.items():
                gid = mapping[internal]["generation_id"]
                raw = out / f"row-{gid:05d}.pt"
                if raw.exists():
                    raise FileExistsError(f"duplicate retained row {gid}")
                assert_substrate_fields(capture, context=f"retained row {gid}")
                substrate = capture_receipt(capture)
                torch.save(capture, raw)
                restored = torch.load(raw, map_location="cpu", weights_only=True)
                if capture_receipt(restored) != substrate:
                    raise RuntimeError(
                        "capture bytes differ after the serialization round trip"
                    )
                del restored
                record = dict(
                    capture_contract="lane-capture/1",
                    generation_id=gid,
                    raw_sha256=file_sha(raw),
                    substrate=substrate,
                    hook_noninterference=unchanged,
                    schedule_sha256=schedule_digest,
                    schedule_file=schedule_path.name,
                    occurrence_id=mapping[internal]["occurrence_id"],
                    internal_request_id=internal,
                )
                write_json(raw.with_suffix(".json"), record)
                records.append(record)
            group_records.append(
                dict(
                    group_index=group_index,
                    schedule_file=schedule_path.name,
                    schedule_sha256=schedule_digest,
                    retained_rows=len(captures),
                    seconds=time.perf_counter() - begun,
                    **evidence,
                )
            )
            del capture, captures, tap, prompts, observed, controls
            tap = None
            if not unchanged:
                raise RuntimeError(
                    "the capture hooks changed the generation or its logprobs"
                )
        except BaseException as exc:
            write_json(
                out / f"{stem}.failed.json",
                dict(
                    condition=condition,
                    group_index=group_index,
                    exception=type(exc).__name__,
                    message=str(exc),
                    internal_requests=mapping,
                    trace=[] if tap is None else tap.schedule,
                ),
            )
            raise
    if sorted(r["generation_id"] for r in records) != sorted(by_id):
        raise RuntimeError("retained population differs from complete roster")
    return dict(
        rows=sorted(records, key=lambda r: r["generation_id"]), groups=group_records
    )
