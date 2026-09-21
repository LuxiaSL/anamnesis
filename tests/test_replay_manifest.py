"""The replay manifest: its schema, its round trip, and what reconstruction refuses.

A manifest is what makes a banked run replayable, so the interesting cases are the
ones where it declines to cover a generation. Reconstruction recovers exactly one
token — the first generated one, which the bank never stored — and validates the
recovery against the banked tokens. Every way that validation can fail is a flag
with a reason, never a silent drop, because a replay over a manifest that quietly
lost a third of its generations reads as a clean result.

No model and no GPU: a mock tokenizer with a fixed chat template and a
character-level encoding is enough to exercise every branch, which is the point of
the reconstruction living apart from the forward pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from anamnesis.extraction.replay.manifest import (
    DEFAULT_EOS_IDS,
    MANIFEST_NAME,
    FlaggedGeneration,
    ReplayEntry,
    ReplayManifest,
    build_prompt_ids,
    build_replay_manifest,
    entry_from_ids,
    load_replay_manifest,
    manifest_from_entries,
    manifest_path,
    reconstruct_one,
    write_replay_manifest,
)

EOS = 128009
PROMPT_HEAD = [200, 201]
PROMPT_TAIL = [202]


class MockTokenizer:
    """A tokenizer with a fixed chat template and a one-character-per-token encoding.

    The template is `PROMPT_HEAD + system + user + PROMPT_TAIL`, each message
    contributing one token per character, so a prompt length is predictable and a
    template change is simulated by lengthening `PROMPT_HEAD`.
    """

    def __init__(self, prompt_head: list[int] | None = None) -> None:
        self.prompt_head = list(prompt_head if prompt_head is not None else PROMPT_HEAD)

    def apply_chat_template(
        self, messages: list[dict[str, str]], add_generation_prompt: bool = True,
        return_tensors: Any = None,
    ) -> list[int]:
        ids = list(self.prompt_head)
        for message in messages:
            ids += [ord(c) for c in message["content"]]
        if add_generation_prompt:
            ids += list(PROMPT_TAIL)
        return ids

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        keep = [i for i in ids if not (skip_special_tokens and i in set(DEFAULT_EOS_IDS))]
        return "".join(chr(i) for i in keep)


def prompt_length_for(system: str, user: str, tokenizer: MockTokenizer) -> int:
    return len(build_prompt_ids(tokenizer, system, user))


def gen_meta(
    gen_id: int, text: str, tokenizer: MockTokenizer, *,
    system: str = "S", user: str = "U", n_gen: int | None = None,
    prompt_length: int | None = None,
) -> dict[str, Any]:
    """A banked generation's metadata row, as `metadata.json` carries it."""
    plen = prompt_length if prompt_length is not None else prompt_length_for(system, user, tokenizer)
    return {
        "generation_id": gen_id,
        "system_prompt": system,
        "user_prompt": user,
        "generated_text": text,
        "prompt_length": plen,
        "num_generated_tokens": len(text) if n_gen is None else n_gen,
    }


def bank_run(
    tmp_path: Path, rows: list[tuple[dict[str, Any], list[int] | None]],
) -> Path:
    """Write a run directory: `metadata.json` plus `raw_tensors/` chosen ids.

    A None chosen-ids list means the generation has no banked tensors, which is the
    first thing reconstruction can fail on.
    """
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw_tensors"
    raw_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps({"generations": [row for row, _ in rows]})
    )
    for row, chosen in rows:
        if chosen is None:
            continue
        np.savez(
            raw_dir / f"gen_{row['generation_id']:03d}.npz",
            chosen_ids=np.asarray(chosen, dtype=np.int64),
        )
    return run_dir


# ── the schema ────────────────────────────────────────────────────────────────


def test_an_entry_partitions_its_sequence() -> None:
    entry = entry_from_ids([1, 2, 3, 4, 5], prompt_length=2)
    assert entry.n_gen == 3
    assert entry.prompt_length + entry.n_gen == len(entry.input_ids)


def test_an_entry_whose_split_does_not_add_up_is_refused() -> None:
    """Replay slices its outputs at the prompt boundary, so a wrong boundary is not
    a wrong number — it is a different set of positions."""
    with pytest.raises(ValidationError, match="must partition"):
        ReplayEntry(input_ids=[1, 2, 3, 4], prompt_length=2, n_gen=5)


def test_manifest_counts_are_derived_not_asserted() -> None:
    entries = {"0": entry_from_ids([1, 2, 3], 1), "1": entry_from_ids([4, 5, 6, 7], 2)}
    flagged = [FlaggedGeneration(gen_id=2, reason="no raw_tensors / chosen_ids")]
    manifest = manifest_from_entries(entries, flagged)
    assert manifest.n_ok == 2 and manifest.n_flagged == 1
    assert manifest.gen_ids() == (0, 1)


def test_a_manifest_whose_counts_disagree_with_its_rows_is_refused() -> None:
    with pytest.raises(ValidationError, match="n_ok"):
        ReplayManifest(entries={}, n_ok=3, n_flagged=0, flagged=[])


def test_entry_lookup_says_why_a_generation_is_missing() -> None:
    manifest = manifest_from_entries(
        {"0": entry_from_ids([1, 2, 3], 1)},
        [FlaggedGeneration(gen_id=7, reason="round-trip mismatch (retok-1=4, chosen_core=3)")],
    )
    assert manifest.entry(0).n_gen == 2
    with pytest.raises(KeyError, match="round-trip mismatch"):
        manifest.entry(7)
    with pytest.raises(KeyError, match="not in the manifest"):
        manifest.entry(99)


def test_integer_and_string_keys_reach_the_same_row() -> None:
    manifest = manifest_from_entries({3: entry_from_ids([1, 2, 3], 1)})
    assert "3" in manifest.entries
    assert manifest.entry(3) == manifest.entries["3"]


# ── the round trip ────────────────────────────────────────────────────────────


def test_a_written_manifest_reads_back_identical(tmp_path: Path) -> None:
    manifest = manifest_from_entries({"0": entry_from_ids([9, 8, 7, 6], 2)})
    written = write_replay_manifest(tmp_path, manifest)
    assert written == tmp_path / MANIFEST_NAME
    assert load_replay_manifest(tmp_path) == manifest
    assert load_replay_manifest(written) == manifest


def test_the_written_json_is_the_banked_shape(tmp_path: Path) -> None:
    """The banked manifests are read by name, so the keys are the schema.

    Entry keys are generation ids as strings and the four top-level keys are in the
    order every banked file has them.
    """
    path = write_replay_manifest(
        tmp_path / "out.json", manifest_from_entries({12: entry_from_ids([1, 2, 3], 1)})
    )
    payload = json.loads(path.read_text())
    assert list(payload) == ["entries", "n_ok", "n_flagged", "flagged"]
    assert list(payload["entries"]) == ["12"]
    assert list(payload["entries"]["12"]) == ["input_ids", "prompt_length", "n_gen"]


def test_a_missing_manifest_is_an_error_not_an_empty_one(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_replay_manifest(tmp_path)


def test_a_file_that_is_not_a_manifest_is_refused(tmp_path: Path) -> None:
    """A replay must not proceed past a manifest it could not validate."""
    path = tmp_path / MANIFEST_NAME
    path.write_text(json.dumps({"entries": {"0": {"input_ids": [1, 2, 3]}}, "n_ok": 1,
                                "n_flagged": 0, "flagged": []}))
    with pytest.raises(ValidationError):
        load_replay_manifest(tmp_path)


def test_manifest_path_is_the_conventional_name(tmp_path: Path) -> None:
    assert manifest_path(tmp_path).name == "replay_manifest.json"


# ── reconstruction, one generation at a time ──────────────────────────────────


def test_the_first_generated_token_is_recovered_and_the_rest_are_the_banked_ones() -> None:
    tokenizer = MockTokenizer()
    text = "hello"
    chosen = [ord(c) for c in text[1:]]          # g_1..g_{N-1}, as the bank stores them
    meta = gen_meta(0, text, tokenizer)
    ids, plen, ok, reason = reconstruct_one(tokenizer, meta, np.asarray(chosen), {EOS})
    assert ok and reason == "ok"
    assert ids is not None
    assert ids[plen:] == [ord(c) for c in text]
    assert ids[:plen] == build_prompt_ids(tokenizer, "S", "U")


def test_a_trailing_end_of_sequence_token_is_kept_in_the_sequence_and_stripped_for_the_check() -> None:
    """Decoding with skip_special_tokens drops the end-of-turn token from the text,
    so the round-trip check compares against the banked ids minus that tail while the
    sequence keeps it: replay has to see the token the model actually emitted."""
    tokenizer = MockTokenizer()
    text = "abc"
    chosen = [ord("b"), ord("c"), EOS]
    meta = gen_meta(0, text, tokenizer, n_gen=4)
    ids, plen, ok, reason = reconstruct_one(tokenizer, meta, np.asarray(chosen), {EOS})
    assert ok and reason == "ok"
    assert ids is not None and ids[-1] == EOS
    assert ids[plen:] == [ord("a"), ord("b"), ord("c"), EOS]


def test_every_end_of_sequence_id_of_the_default_set_strips() -> None:
    tokenizer = MockTokenizer()
    for eos in DEFAULT_EOS_IDS:
        chosen = [ord("b"), eos]
        meta = gen_meta(0, "ab", tokenizer, n_gen=3)
        ids, _, ok, _ = reconstruct_one(tokenizer, meta, np.asarray(chosen), set(DEFAULT_EOS_IDS))
        assert ok, f"eos {eos} did not strip"
        assert ids is not None and ids[-1] == eos


def test_a_changed_chat_template_is_flagged_not_absorbed() -> None:
    """The prompt is re-rendered, so a template or tokenizer that drifted since the
    bank was made produces a different prompt length — and a replay over it would be
    a replay over a different sequence."""
    banked = MockTokenizer()
    drifted = MockTokenizer(prompt_head=[*PROMPT_HEAD, 203])
    text = "abc"
    meta = gen_meta(0, text, banked)
    ids, plen, ok, reason = reconstruct_one(
        drifted, meta, np.asarray([ord("b"), ord("c")]), {EOS}
    )
    assert not ok and ids is None
    assert "chat-template/tokenizer drift" in reason
    assert plen == meta["prompt_length"] + 1


def test_an_empty_generated_text_is_flagged() -> None:
    tokenizer = MockTokenizer()
    meta = gen_meta(0, "", tokenizer, n_gen=0)
    _, _, ok, reason = reconstruct_one(tokenizer, meta, np.asarray([1, 2]), {EOS})
    assert not ok and reason == "empty generated_text"


def test_a_round_trip_mismatch_is_flagged_when_the_text_does_not_decode_back() -> None:
    """The fallback accepts a re-tokenization that differs from the banked ids only
    when the banked ids still decode to the exact banked text. Here they do not."""
    tokenizer = MockTokenizer()
    meta = gen_meta(0, "abc", tokenizer)
    _, _, ok, reason = reconstruct_one(tokenizer, meta, np.asarray([ord("z")]), {EOS})
    assert not ok and "round-trip mismatch" in reason


def test_the_decode_fallback_accepts_a_tokenization_ambiguity() -> None:
    """A tokenizer whose re-encoding splits differently from the bank still yields the
    right first token, so the row is kept — marked as resting on the decode check
    rather than on the strict round trip."""

    class SplitDifferently(MockTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            # One extra token at the tail: the strict comparison fails, the decode holds.
            return [*(ord(c) for c in text), 0]

        def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
            return "".join(chr(i) for i in ids if i != 0)

    tokenizer = SplitDifferently()
    text = "abc"
    meta = gen_meta(0, text, tokenizer, prompt_length=prompt_length_for("S", "U", tokenizer))
    ids, plen, ok, reason = reconstruct_one(
        tokenizer, meta, np.asarray([ord("b"), ord("c")]), {EOS}
    )
    assert ok and reason == "ok-decode-fallback"
    assert ids is not None and ids[plen] == ord("a")


def test_a_generated_token_count_that_disagrees_with_the_bank_is_flagged() -> None:
    tokenizer = MockTokenizer()
    meta = gen_meta(0, "abc", tokenizer, n_gen=9)
    _, _, ok, reason = reconstruct_one(
        tokenizer, meta, np.asarray([ord("b"), ord("c")]), {EOS}
    )
    assert not ok and "n_gen 3 != banked 9" in reason


# ── reconstruction over a whole run ───────────────────────────────────────────


def test_a_whole_run_reconstructs_and_reports_what_it_could_not_cover(tmp_path: Path) -> None:
    tokenizer = MockTokenizer()
    rows = [
        (gen_meta(0, "hello", tokenizer), [ord(c) for c in "ello"]),
        (gen_meta(1, "world", tokenizer), [ord(c) for c in "orld"]),
        (gen_meta(2, "abc", tokenizer), None),                       # no banked tensors
        (gen_meta(3, "abc", tokenizer), [ord("z")]),                 # fails the round trip
    ]
    run_dir = bank_run(tmp_path, rows)
    manifest = build_replay_manifest(run_dir, tokenizer)

    assert manifest.n_ok == 2 and manifest.gen_ids() == (0, 1)
    assert manifest.n_flagged == 2
    reasons = {row.gen_id: row.reason for row in manifest.flagged}
    assert reasons[2] == "no raw_tensors / chosen_ids"
    assert "round-trip mismatch" in reasons[3]
    # A flagged generation is named, not dropped: the manifest's coverage is legible.
    assert manifest.n_ok + manifest.n_flagged == len(rows)


def test_reconstruction_reads_a_flat_generation_list_too(tmp_path: Path) -> None:
    """`metadata.json` wraps generations under the generations key; an older bank is a
    bare list, and both are read."""
    tokenizer = MockTokenizer()
    run_dir = tmp_path / "flat"
    (run_dir / "raw_tensors").mkdir(parents=True)
    row = gen_meta(0, "hello", tokenizer)
    (run_dir / "metadata.json").write_text(json.dumps([row]))
    np.savez(
        run_dir / "raw_tensors" / "gen_000.npz",
        chosen_ids=np.asarray([ord(c) for c in "ello"], dtype=np.int64),
    )
    assert build_replay_manifest(run_dir, tokenizer).n_ok == 1


def test_a_chosen_ids_bundle_stands_in_for_absent_raw_tensors(tmp_path: Path) -> None:
    """The manifest is sometimes built where the tensors are not, so the banked ids can
    arrive as a bundle instead."""
    tokenizer = MockTokenizer()
    rows = [(gen_meta(0, "hello", tokenizer), None)]
    run_dir = bank_run(tmp_path, rows)
    manifest = build_replay_manifest(
        run_dir, tokenizer, chosen_ids_map={0: [ord(c) for c in "ello"]}
    )
    assert manifest.n_ok == 1 and manifest.n_flagged == 0


def test_an_end_of_sequence_override_reaches_the_tail_strip(tmp_path: Path) -> None:
    tokenizer = MockTokenizer()
    custom_eos = 7
    rows = [(gen_meta(0, "ab", tokenizer, n_gen=3), [ord("b"), custom_eos])]
    run_dir = bank_run(tmp_path, rows)
    assert build_replay_manifest(run_dir, tokenizer, eos_ids={custom_eos}).n_ok == 1
    # Without the override the token is not an end-of-sequence marker, so the strict
    # round trip fails on a length that does not match.
    assert build_replay_manifest(run_dir, tokenizer).n_flagged == 1


def test_a_reconstructed_manifest_is_writable_and_readable(tmp_path: Path) -> None:
    """Reconstruction and generation produce the same object, so a reconstructed bank
    is replayed through exactly the path a fresh run is."""
    tokenizer = MockTokenizer()
    run_dir = bank_run(tmp_path, [(gen_meta(0, "hello", tokenizer), [ord(c) for c in "ello"])])
    manifest = build_replay_manifest(run_dir, tokenizer)
    write_replay_manifest(run_dir, manifest)
    assert load_replay_manifest(run_dir) == manifest
