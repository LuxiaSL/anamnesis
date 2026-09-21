"""The replay manifest: the token sequences a run can be replayed over.

A replay manifest is the run artifact that makes a banked generation
reproducible. It holds, per generation, the full realized token sequence
``[prompt + generated]`` together with the prompt length that splits it — which
is exactly what :func:`anamnesis.extraction.replay.extract.replay_extract`
teacher-forces. Its on-disk name is ``replay_manifest.json`` beside the run's
``metadata.json``.

This module is the manifest's one home, in two directions:

* **Writing.** A run that generates its own text knows the realized ids as it
  produces them, so it records them: :func:`entry_from_ids` builds one row and
  :func:`write_replay_manifest` writes the file.
  :func:`anamnesis.extraction.generation_runner.run_experiment` does this for
  every pass it runs, which is why a run made here needs no reconstruction.
* **Reconstruction.** A run banked before manifests existed has only its decoded
  text and its banked ``chosen_ids``, and the first generated token is missing
  from both. :func:`build_replay_manifest` recovers it and validates the
  recovery, so a text-only bank becomes replayable without trusting
  re-tokenization.

Reconstruction, per generation — only ``g_0`` is recovered, the rest are the
exact banked tokens:

* ``prompt_ids = chat_template(system, user)``, which must MATCH the banked
  ``prompt_length``, else the tokenizer or chat template drifted against the
  original extraction and the generation is flagged;
* ``retok = encode(generated_text, add_special_tokens=False)``;
* ``g_0 = retok[0]``, with the strong validation
  ``retok == [g_0] + chosen_ids`` (minus a trailing end-of-sequence tail), which
  confirms both ``g_0`` and that ``chosen_ids`` is the canonical tokenization of
  the text;
* ``full_gen_ids = [g_0] + chosen_ids``, keeping the banked tokens exactly,
  including a trailing end-of-sequence token;
* ``input_ids = prompt_ids + full_gen_ids``.

Generations that fail any check are flagged with a reason and are NOT silently
dropped: a manifest says how many of a run's generations it covers, so a replay
over it cannot quietly be a replay over a subset.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

# Known Llama 3.x end-of-sequence ids (3.2-3B / 3.1-8B Instruct both ship this set).
DEFAULT_EOS_IDS = (128001, 128008, 128009)

MANIFEST_NAME = "replay_manifest.json"
"""The manifest's filename inside a run directory."""

INPUT_IDS_KEY = "input_ids"
"""Key the realized token ids travel under, in a manifest row and in the
in-memory metadata a generation pass carries them on. The key is named here
because the generation pass strips it before writing metadata to disk: the ids
belong to the manifest, and duplicating them into ``metadata.json`` would make
two files answer one question."""


class ReplayEntry(BaseModel):
    """One generation's replay row: the sequence, and where the prompt ends."""

    model_config = ConfigDict(extra="forbid")

    input_ids: list[int] = Field(
        min_length=2, description="The realized [prompt + generated] token ids"
    )
    prompt_length: int = Field(gt=0, description="Tokens of prompt; states before this are not banked")
    n_gen: int = Field(gt=0, description="Generated tokens, which is len(input_ids) - prompt_length")

    @model_validator(mode="after")
    def _split_adds_up(self) -> Self:
        if self.prompt_length + self.n_gen != len(self.input_ids):
            raise ValueError(
                f"prompt_length {self.prompt_length} + n_gen {self.n_gen} != "
                f"{len(self.input_ids)} token ids; the prompt/generated split must partition "
                "the sequence, because replay slices its outputs at that boundary"
            )
        return self


class FlaggedGeneration(BaseModel):
    """A generation the manifest does not cover, and why."""

    model_config = ConfigDict(extra="forbid")

    gen_id: int = Field(ge=0, description="The generation's id in the run")
    reason: str = Field(min_length=1, description="Which check failed, in terms a reader can act on")


class ReplayManifest(BaseModel):
    """Every replayable generation of a run, and every one that is not.

    Entry keys are generation ids as strings, which is the on-disk form: JSON
    object keys are strings, and the banked manifests are read by scripts that
    index them directly.
    """

    model_config = ConfigDict(extra="forbid")

    entries: dict[str, ReplayEntry] = Field(description="Replayable generations, keyed by id")
    n_ok: int = Field(ge=0, description="Count of replayable generations")
    n_flagged: int = Field(ge=0, description="Count of generations that failed a check")
    flagged: list[FlaggedGeneration] = Field(description="The failures, with reasons")

    @model_validator(mode="after")
    def _counts_match(self) -> Self:
        if self.n_ok != len(self.entries):
            raise ValueError(f"n_ok {self.n_ok} != {len(self.entries)} entries")
        if self.n_flagged != len(self.flagged):
            raise ValueError(f"n_flagged {self.n_flagged} != {len(self.flagged)} flagged rows")
        return self

    def gen_ids(self) -> tuple[int, ...]:
        """The replayable generation ids, ascending."""
        return tuple(sorted(int(key) for key in self.entries))

    def entry(self, gen_id: int) -> ReplayEntry:
        """One generation's row.

        Raises
        ------
        KeyError
            When the manifest does not cover that generation, naming whether it
            was flagged and why.
        """
        key = str(int(gen_id))
        if key in self.entries:
            return self.entries[key]
        for row in self.flagged:
            if row.gen_id == int(gen_id):
                raise KeyError(f"generation {gen_id} is flagged, not replayable: {row.reason}")
        raise KeyError(f"generation {gen_id} is not in the manifest")


def entry_from_ids(input_ids: list[int], prompt_length: int) -> ReplayEntry:
    """A manifest row from a sequence a generation pass just realized."""
    ids = [int(x) for x in input_ids]
    return ReplayEntry(
        input_ids=ids, prompt_length=int(prompt_length), n_gen=len(ids) - int(prompt_length)
    )


def manifest_from_entries(
    entries: dict[str, ReplayEntry] | dict[int, ReplayEntry],
    flagged: list[FlaggedGeneration] | None = None,
) -> ReplayManifest:
    """A manifest around rows a pass produced, with the counts derived."""
    keyed = {str(int(key)): value for key, value in entries.items()}
    failures = list(flagged or [])
    return ReplayManifest(
        entries=keyed, n_ok=len(keyed), n_flagged=len(failures), flagged=failures
    )


def manifest_path(run_dir: Path) -> Path:
    """Where a run's manifest lives."""
    return Path(run_dir) / MANIFEST_NAME


def write_replay_manifest(path: Path, manifest: ReplayManifest) -> Path:
    """Write a manifest, creating its parent directory. Returns the path written.

    ``path`` may be a run directory, in which case the manifest lands at its
    conventional name inside it.
    """
    target = manifest_path(path) if path.is_dir() else Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        json.dump(manifest.model_dump(), f)
    return target


def load_replay_manifest(path: Path) -> ReplayManifest:
    """Read a manifest, validating its schema.

    ``path`` may be a run directory or the manifest file itself.

    Raises
    ------
    FileNotFoundError
        When neither the path nor the conventional name inside it exists.
    pydantic.ValidationError
        When the file's shape is not a manifest, which is the case a replay must
        not proceed past.
    """
    candidate = Path(path)
    target = manifest_path(candidate) if candidate.is_dir() else candidate
    if not target.is_file():
        raise FileNotFoundError(f"no replay manifest at {target}")
    with open(target) as f:
        payload = json.load(f)
    return ReplayManifest.model_validate(payload)


def _load_chosen_ids(
    raw_dir: Path, gen_id: int, chosen_map: dict[int, Any] | None = None,
) -> np.ndarray | None:
    """Load banked chosen_ids (= g_1..g_{N-1}) for one generation.

    Prefers an explicit chosen_map (a pre-extracted bundle — used when raw_tensors
    aren't co-located, e.g. building the manifest on the node before extraction),
    falling back to the run's raw_tensors npz.
    """
    if chosen_map is not None and gen_id in chosen_map:
        return np.asarray(chosen_map[gen_id]).astype(np.int64)
    p = raw_dir / f"gen_{gen_id:03d}.npz"
    if not p.exists():
        return None
    with np.load(p, allow_pickle=True) as z:
        if "chosen_ids" not in z.files:
            return None
        return z["chosen_ids"].astype(np.int64)


def build_prompt_ids(tokenizer: Any, system_prompt: str, user_prompt: str) -> list[int]:
    """Replicate generation_runner.format_prompt's chat template exactly (ids only)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    result = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors=None,
    )
    # tf returns either a flat list[int] or a (Batch)Encoding mapping with "input_ids".
    # BatchEncoding subclasses UserDict → it is NOT a `dict` instance; check for the key.
    if isinstance(result, (list, tuple)):
        ids = list(result)
    else:
        ids = result["input_ids"]
    if len(ids) > 0 and isinstance(ids[0], (list, tuple)):
        ids = ids[0]  # un-batch
    return [int(x) for x in ids]


def reconstruct_one(
    tokenizer: Any,
    gen_meta: dict[str, Any],
    chosen_ids: np.ndarray,
    eos_ids: set[int],
) -> tuple[list[int] | None, int, bool, str]:
    """Reconstruct + validate the full token sequence for one generation.

    Returns (input_ids | None, prompt_length, ok, reason).
    """
    system = gen_meta.get("system_prompt", "")
    user = gen_meta.get("user_prompt", "")
    gen_text = gen_meta.get("generated_text", "")
    meta_plen = int(gen_meta["prompt_length"])
    meta_ngen = int(gen_meta.get("num_generated_tokens", -1))

    prompt_ids = build_prompt_ids(tokenizer, system, user)
    if len(prompt_ids) != meta_plen:
        return (None, len(prompt_ids), False,
                f"prompt_len {len(prompt_ids)} != banked {meta_plen} (chat-template/tokenizer drift)")

    chosen = [int(x) for x in chosen_ids.tolist()]
    if not gen_text:
        return None, meta_plen, False, "empty generated_text"

    retok = [int(x) for x in tokenizer.encode(gen_text, add_special_tokens=False)]
    if len(retok) < 1:
        return None, meta_plen, False, "empty re-tokenization"

    # chosen may carry trailing EOS that skip-special decode dropped from gen_text.
    chosen_core = list(chosen)
    while chosen_core and chosen_core[-1] in eos_ids:
        chosen_core = chosen_core[:-1]

    g0 = retok[0]
    full_gen_ids = [g0] + chosen  # g_0 recovered + g_1..g_{N-1} exact banked (incl EOS)

    # PRIMARY: strict round-trip (retok == [g_0] + chosen_core).
    strict_ok = (len(retok) - 1 == len(chosen_core) and retok[1:] == chosen_core)
    if strict_ok:
        reason = "ok"
    else:
        # FALLBACK: a downstream tokenization ambiguity can make retok != [g_0]+chosen_core while
        # g_0 itself is still correct. Accept iff [g_0]+banked-chosen decodes back to the exact
        # generated_text — g_1..g_{N-1} remain the exact banked tokens; only g_0 is "recovered".
        if tokenizer.decode(full_gen_ids, skip_special_tokens=True) != gen_text:
            return (None, meta_plen, False,
                    f"round-trip mismatch (retok-1={len(retok) - 1}, chosen_core={len(chosen_core)})")
        reason = "ok-decode-fallback"

    if meta_ngen >= 0 and len(full_gen_ids) != meta_ngen:
        return (None, meta_plen, False, f"n_gen {len(full_gen_ids)} != banked {meta_ngen}")

    input_ids = prompt_ids + full_gen_ids
    return input_ids, meta_plen, True, reason


def build_replay_manifest(
    run_dir: Path,
    tokenizer: Any,
    eos_ids: set[int] | None = None,
    chosen_ids_map: dict[int, Any] | None = None,
) -> ReplayManifest:
    """Build the replay manifest for all banked generations in a run.

    Parameters
    ----------
    run_dir : Path
        Run directory containing metadata.json + raw_tensors/.
    tokenizer : transformers tokenizer
        Must be the model's tokenizer (same chat template as the original extraction).
    eos_ids : set[int], optional
        EOS token ids to strip from the chosen-ids tail (default: Llama 3.x set).

    Returns
    -------
    ReplayManifest — the covered generations, and every one that failed a check.
    """
    eos = set(eos_ids) if eos_ids is not None else set(DEFAULT_EOS_IDS)
    raw_dir = run_dir / "raw_tensors"
    with open(run_dir / "metadata.json") as f:
        meta = json.load(f)
    gens = meta["generations"] if isinstance(meta, dict) and "generations" in meta else meta

    entries: dict[str, ReplayEntry] = {}
    flagged: list[FlaggedGeneration] = []
    for g in gens:
        gid = int(g["generation_id"])
        chosen = _load_chosen_ids(raw_dir, gid, chosen_ids_map)
        if chosen is None:
            flagged.append(FlaggedGeneration(gen_id=gid, reason="no raw_tensors / chosen_ids"))
            continue
        input_ids, plen, ok, reason = reconstruct_one(tokenizer, g, chosen, eos)
        if ok and input_ids is not None:
            entries[str(gid)] = entry_from_ids(input_ids, plen)
        else:
            flagged.append(FlaggedGeneration(gen_id=gid, reason=reason))

    return manifest_from_entries(entries, flagged)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build + validate a replay manifest for a run")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run dir (has metadata.json + raw_tensors/)")
    parser.add_argument("--model-path", type=str, required=True, help="Model/tokenizer path or HF id")
    parser.add_argument("--out", type=Path, required=True, help="Output manifest JSON path")
    parser.add_argument("--eos-ids", type=int, nargs="+", default=None, help="Override EOS ids")
    parser.add_argument("--chosen-ids", type=Path, default=None,
                        help="JSON {gen_id: [ids]} bundle of banked chosen_ids "
                             "(fallback when raw_tensors are not co-located on this host)")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    chosen_ids_map: dict[int, Any] | None = None
    if args.chosen_ids is not None:
        with open(args.chosen_ids) as f:
            raw_map = json.load(f)
        chosen_ids_map = {int(k): v for k, v in raw_map.items()}
        logger.info(f"Loaded chosen_ids bundle: {len(chosen_ids_map)} gens")

    manifest = build_replay_manifest(
        args.run_dir, tokenizer,
        eos_ids=set(args.eos_ids) if args.eos_ids else None,
        chosen_ids_map=chosen_ids_map,
    )
    write_replay_manifest(args.out, manifest)

    logger.info(
        f"Manifest: {manifest.n_ok} ok, {manifest.n_flagged} flagged → {args.out}"
    )
    if manifest.flagged:
        logger.warning("Flagged gens (reasons):")
        # group reasons
        from collections import Counter
        reasons = Counter(x.reason.split("(")[0].strip() for x in manifest.flagged)
        for r, c in reasons.most_common():
            logger.warning(f"  {c:4d}× {r}")
        sample = manifest.flagged[:10]
        logger.warning(f"  e.g.: {[(x.gen_id, x.reason) for x in sample]}")


if __name__ == "__main__":
    main()
