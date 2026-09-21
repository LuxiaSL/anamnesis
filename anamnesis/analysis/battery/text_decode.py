"""Shared byte-level-BPE text decode for banked generations.

Some models (e.g. DeepSeek-V2-Lite / M6) bank `generated_text` still byte-BPE-encoded
(Ġ=space, Ċ=newline — the GPT-2 bytes_to_unicode alphabet). Any reader of banked text
(judge, marker, coherence) must decode it first. One home for the map so we never roll
a third copy (2026-07-18 sweep).
"""
from __future__ import annotations


def byte_decoder() -> dict[str, int]:
    """GPT-2 byte-level BPE unicode→byte map (inverse of bytes_to_unicode)."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTE_DEC = byte_decoder()


def maybe_decode(t: str) -> str:
    """Decode byte-BPE text iff the Ġ/Ċ markers are present (idempotent no-op otherwise)."""
    if "Ġ" in t or "Ċ" in t:
        return bytearray(BYTE_DEC.get(c, 32) for c in t).decode("utf-8", errors="replace")
    return t
