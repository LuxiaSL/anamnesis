"""Generation that banks token sequences and nothing else.

This is the first half of the replay gateway. A pass here samples text and writes
down the *realized* token ids — prompt and generation together — and no hooks, no
calibration and no feature arithmetic are involved. The signatures come later,
from a replay over exactly those ids. Splitting the two is what makes the capture
surface free to change: adding a family or hooking another layer costs a replay,
not a re-generation, and the text under both is provably the same text because it
is the same token ids.

Banking the full sequence rather than the decoded string is the point. A decoded
string has to be re-tokenised to be replayed, and the first generated token is not
recoverable from the decode at all — the reconstruction path in
:mod:`anamnesis.extraction.replay.manifest` exists for banks that predate this
and has to validate its way back to the ids. A pass here has them, so its manifest
is exact.

Every generation is seeded from its own coordinates, which is what makes a pass
partitionable: a worker's share, and the order it runs, cannot affect what any
generation produces. The same property lets one loaded model walk a whole roster
of cells, re-arming the intervention per cell, and still produce what a
cell-per-process fan-out would have.

The sampler is called with its defaults unless a cell asks otherwise: a
repetition penalty of one is not passed at all, so the default path's sampling
stream is the one the record was banked under.

A spec that raises is named in the result and the loop continues, because one
unsamplable prompt is not a reason to abandon the other three hundred. What the
result must not permit is a caller reporting success over the short corpus, so
:func:`generation_shortfall` states the pass as expected-versus-produced and the
command layer refuses on it.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from anamnesis.extraction.interventions import InjectionSpec, check_injection_gating
from anamnesis.shortfall import Shortfall, ids_present

logger = logging.getLogger(__name__)

NEUTRAL_REPETITION_PENALTY = 1.0
"""The value at which the penalty is not a penalty. At exactly this value the
argument is withheld from the sampler rather than passed, so the default path's
logits processors are the ones the banked corpus was produced under."""


class DecodePolicy(BaseModel):
    """How a pass samples, shared across every cell it runs.

    ``date_string`` pins the chat template's rendered date. A template that
    renders today's date makes the prompt tokens a function of the wall clock, so
    a pass crossing midnight would silently change its own prompts and break
    matched-history pairing between cells.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    temperature: float = Field(gt=0.0, description="Sampling temperature")
    top_p: float = Field(gt=0.0, le=1.0, description="Nucleus mass kept")
    max_new_tokens: int = Field(gt=0, description="Token budget per generation")
    eos_token_ids: tuple[int, ...] = Field(
        min_length=1, description="Stop tokens; model-specific, never assumed"
    )
    repetition_penalty: float = Field(
        default=NEUTRAL_REPETITION_PENALTY,
        gt=0.0,
        description="Above one discourages repeating context tokens, below one encourages it",
    )
    date_string: str | None = Field(
        default=None, description="Pinned chat-template date; unset leaves the template's own"
    )

    def sampler_extras(self) -> dict[str, Any]:
        """Sampler arguments beyond the defaults, which is usually nothing."""
        if self.repetition_penalty == NEUTRAL_REPETITION_PENALTY:
            return {}
        return {"repetition_penalty": float(self.repetition_penalty)}

    def with_repetition_penalty(self, penalty: float | None) -> DecodePolicy:
        """This policy under a cell's own penalty, or unchanged when it names none."""
        if penalty is None:
            return self
        return self.model_copy(update={"repetition_penalty": float(penalty)})

    @model_validator(mode="after")
    def _template_date_is_not_empty(self) -> Self:
        if self.date_string is not None and not self.date_string.strip():
            raise ValueError("date_string is present but blank; omit it to use the template's own")
        return self


@dataclass
class GenerationCount:
    """How a cell's generation went: the specs asked for, and which of them raised.

    ``requested`` is every spec the call was handed, including the ones whose
    record already existed — the asked-for corpus, not the work this pass did, so
    a resumed pass is complete rather than short. ``failed`` maps a generation id
    to the message its exception carried, which is what lets a refusal name the
    spec rather than only count it.
    """

    n_done: int
    failed: dict[int, str]
    requested: tuple[int, ...]
    out_dir: Path
    seconds: float

    @property
    def n_failed(self) -> int:
        return len(self.failed)

    @property
    def ok(self) -> bool:
        return self.n_failed == 0


def generation_shortfall(
    result: GenerationCount, *, command: str, label: str = ""
) -> Shortfall:
    """State a generated cell as expected-versus-produced, for a command to refuse on.

    Produced means a banked record on disk for a requested generation id, which is
    the same predicate the loop resumes on: a pass that generated three records
    because seventeen were already there has produced twenty and is complete.
    """
    requested = tuple(str(gen_id) for gen_id in result.requested)
    return Shortfall(
        command=command,
        unit="spec",
        target=result.out_dir,
        requested=requested,
        produced=ids_present(
            requested,
            lambda name: (result.out_dir / f"gen_{int(name):03d}.json").exists(),
        ),
        failures={str(gen_id): reason for gen_id, reason in result.failed.items()},
        label=label,
    )


def _prompt_ids(tokenizer: Any, spec: dict[str, Any], policy: DecodePolicy) -> Any:
    """The prompt as token ids, through the chat template when there is one.

    A base checkpoint has no chat template and takes the bare user prompt. A
    system prompt against such a checkpoint is refused rather than dropped: a
    mode is its system prompt, so a mode label over a prompt that never carried
    it would be a mislabelled corpus.
    """
    system_prompt = spec.get("system_prompt")
    if tokenizer.chat_template is None:
        if system_prompt:
            raise ValueError(
                f"generation {spec['generation_id']}: a system prompt was given but this "
                "checkpoint has no chat template, and a base model takes bare user prompts"
            )
        return tokenizer(spec["user_prompt"], return_tensors="pt")["input_ids"]
    messages = [{"role": "user", "content": spec["user_prompt"]}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    template_kwargs = {"date_string": policy.date_string} if policy.date_string else {}
    result = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", **template_kwargs
    )
    import torch

    return result if isinstance(result, torch.Tensor) else result["input_ids"]


def generate_specs(
    model: Any,
    tokenizer: Any,
    specs: Sequence[dict[str, Any]],
    out_dir: Path,
    policy: DecodePolicy,
    *,
    pad_token_id: int,
    write_handle: Any | None = None,
    injection: InjectionSpec | None = None,
    perturbation: dict[str, Any] | None = None,
    label: str = "w",
) -> GenerationCount:
    """Generate one cell's specs, banking one record per generation.

    Resume is per record: a spec whose record already exists is skipped, so a
    killed pass re-run over the same directory continues. A generation that
    raises is counted, logged with its traceback, and does not stop the cell.

    ``write_handle`` is an armed residual write and ``injection`` its spec; the
    write's start position is set per generation from that generation's prompt
    length, and its gating is checked afterwards against the generated span
    unless the magnitude is zero. ``perturbation`` is recorded on each record as
    provenance; arming it is the caller's, since it is model-wide.

    Returns
    -------
    GenerationCount
        Every spec handed in, the count that landed, and each id that raised with
        the message its exception carried. A caller turns that into a refusal
        through :func:`generation_shortfall`; nothing here decides the pass's fate.
    """
    import numpy as np
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [
        s for s in specs
        if not (out_dir / f"gen_{s['generation_id']:03d}.json").exists()
    ]
    logger.info(f"[{label}] {len(todo)}/{len(specs)} specs to generate -> {out_dir}")

    extras = policy.sampler_extras()
    n_done = 0
    failed: dict[int, str] = {}
    started = time.time()
    for index, spec in enumerate(todo):
        try:
            gen_id = spec["generation_id"]
            input_ids = _prompt_ids(tokenizer, spec, policy).to("cuda")
            attention_mask = torch.ones_like(input_ids)
            prompt_length = int(input_ids.shape[1])

            torch.manual_seed(spec["seed"])
            torch.cuda.manual_seed_all(spec["seed"])
            np.random.seed(spec["seed"] % (2**32))

            if write_handle is not None:
                write_handle.spec.start_pos = prompt_length
                write_handle.reset_stats()

            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=policy.max_new_tokens,
                    do_sample=True,
                    temperature=policy.temperature,
                    top_p=policy.top_p,
                    eos_token_id=list(policy.eos_token_ids),
                    pad_token_id=pad_token_id,
                    **extras,
                )
            full_sequence = out[0].tolist()
            generated_ids = full_sequence[prompt_length:]

            record: dict[str, Any] = {
                "generation_id": gen_id,
                "prompt_set": spec["prompt_set"],
                "topic": spec["topic"],
                "topic_idx": spec["topic_idx"],
                "mode": spec["mode"],
                "mode_idx": spec["mode_idx"],
                "system_prompt": spec["system_prompt"],
                "user_prompt": spec["user_prompt"],
                "seed": spec["seed"],
                "repetition": spec["repetition"],
                "condition": "standard",
                "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
                "num_generated_tokens": len(generated_ids),
                "prompt_length": prompt_length,
                "input_ids": full_sequence,
            }
            if extras:
                # Provenance for a sampler-actuator cell; absent from every
                # default-path record, so the record schema stays stable.
                record["sampler_repetition_penalty"] = float(policy.repetition_penalty)
            if injection is not None and write_handle is not None:
                if injection.alpha != 0.0:
                    check_injection_gating(
                        write_handle, max(0, len(generated_ids) - 1), f"generation {gen_id}"
                    )
                record["injection"] = {
                    **injection.metadata(),
                    "positions_injected": int(dict(write_handle.stats).get("positions", 0)),
                }
            if perturbation is not None:
                record["perturbation"] = perturbation
            (out_dir / f"gen_{gen_id:03d}.json").write_text(json.dumps(record))
            n_done += 1
            if (index + 1) % 20 == 0 or index == 0:
                elapsed = time.time() - started
                rate = (index + 1) / elapsed if elapsed else 0.0
                eta = (len(todo) - index - 1) / rate if rate else 0.0
                logger.info(
                    f"[{label}] {index + 1}/{len(todo)} gen_{gen_id:03d}: "
                    f"{len(generated_ids)} tokens, {elapsed:.0f}s ({rate:.2f}/s, ETA {eta:.0f}s)"
                )
        except Exception as exc:  # noqa: BLE001 — one generation's failure is not the cell's
            failed[int(spec["generation_id"])] = f"{type(exc).__name__}: {exc}"
            logger.error(
                f"[{label}] spec {spec.get('generation_id')} FAILED: {exc}", exc_info=True
            )
    seconds = time.time() - started
    logger.info(f"[{label}] cell done: {n_done} ok, {len(failed)} failed in {seconds:.0f}s")
    return GenerationCount(
        n_done=n_done,
        failed=failed,
        requested=tuple(int(s["generation_id"]) for s in specs),
        out_dir=out_dir,
        seconds=seconds,
    )
