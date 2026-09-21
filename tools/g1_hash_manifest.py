#!/usr/bin/env python3
"""G1 — array-content manifests for a signatures directory, and their diff.

G1 asks whether refactored code reads banked artifacts and reproduces their
features byte-identically. The comparison is pre-refactor code against
post-refactor code, on the same box in the same session: floating-point results
differ between machines by design, so a manifest built here is only ever
compared with a manifest built here.

The hash is over array *contents*, never over the `.npz` file bytes. An npz is a
zip container whose byte layout depends on the numpy version and the compression
settings, so identical arrays can produce different files. Each entry therefore
carries the sha256 of `array.tobytes()` together with the dtype and shape, and a
drift in any of the three fails the diff — a shape or dtype change is a
compatibility break even when the bytes happen to line up.

JSON sidecars beside the arrays are hashed over their canonicalised content
(parsed, re-serialised with sorted keys), so key order and whitespace do not
register as drift while values do.

Usage
-----
    python tools/g1_hash_manifest.py --signatures <dir> --out reference.txt
    python tools/g1_hash_manifest.py --signatures <dir> --out candidate.txt
    python tools/g1_hash_manifest.py --compare reference.txt candidate.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TextIO

import numpy as np

FORMAT_LINE = "#format anamnesis-g1-manifest/1"
FIELD_SEPARATOR = "\t"
JSON_KEY = "json:content"
JSON_DTYPE = "json"
ABSENT = "-"


class ManifestError(RuntimeError):
    """A condition that makes the manifest unbuildable rather than mismatched."""


@dataclass(frozen=True)
class Entry:
    """One hashed array or sidecar, keyed by (file name, array key)."""

    name: str
    key: str
    dtype: str
    shape: str
    digest: str

    def render(self) -> str:
        return FIELD_SEPARATOR.join((self.name, self.key, self.dtype, self.shape, self.digest))

    @property
    def identity(self) -> tuple[str, str]:
        return (self.name, self.key)


@dataclass(frozen=True)
class Difference:
    """One (file, key) pair that does not agree between two manifests."""

    name: str
    key: str
    field: str
    left: str
    right: str


def array_digest(array: np.ndarray) -> str:
    """sha256 over an array's contents in C order."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def json_digest(payload: object) -> str:
    """sha256 over a JSON document's canonical serialisation."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def npz_entries(path: Path) -> list[Entry]:
    """Every array in one `.npz`, sorted by key."""
    try:
        with np.load(path, allow_pickle=False) as handle:
            keys = sorted(handle.files)
            entries: list[Entry] = []
            for key in keys:
                array = handle[key]
                entries.append(
                    Entry(
                        name=path.name,
                        key=key,
                        dtype=str(array.dtype),
                        shape=str(array.shape),
                        digest=array_digest(array),
                    )
                )
            return entries
    except (OSError, ValueError, EOFError) as exc:
        raise ManifestError(f"{path}: unreadable as npz ({exc})") from exc


def json_entry(path: Path) -> Entry:
    """The canonicalised hash of one JSON sidecar."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ManifestError(f"{path}: unreadable ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path}: invalid JSON at line {exc.lineno} ({exc.msg})") from exc
    return Entry(
        name=path.name,
        key=JSON_KEY,
        dtype=JSON_DTYPE,
        shape=ABSENT,
        digest=json_digest(payload),
    )


def build_manifest(
    signatures: Path,
    pattern: str = "gen_*.npz",
    json_pattern: str = "*.json",
    include_json: bool = True,
) -> list[Entry]:
    """Hash every array and sidecar in a signatures directory, deterministically."""
    signatures = signatures.resolve()
    if not signatures.is_dir():
        raise ManifestError(f"--signatures is not a directory: {signatures}")
    npz_files = sorted(signatures.glob(pattern), key=lambda p: p.name)
    if not npz_files:
        raise ManifestError(f"no files matching {pattern!r} under {signatures}")
    entries: list[Entry] = []
    for path in npz_files:
        entries.extend(npz_entries(path))
    if include_json:
        for path in sorted(signatures.glob(json_pattern), key=lambda p: p.name):
            entries.append(json_entry(path))
    return entries


def render_manifest(entries: Sequence[Entry]) -> str:
    """The manifest text: a format line, then one line per entry."""
    lines = [FORMAT_LINE]
    lines.extend(entry.render() for entry in entries)
    return "\n".join(lines) + "\n"


def parse_manifest(path: Path) -> list[Entry]:
    """Read a manifest back, failing loudly on a line that does not parse."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ManifestError(f"{path}: unreadable ({exc})") from exc
    entries: list[Entry] = []
    saw_format = False
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if line.startswith("#"):
            if number == 1:
                saw_format = line.strip() == FORMAT_LINE
            continue
        fields = line.split(FIELD_SEPARATOR)
        if len(fields) != 5:
            raise ManifestError(f"{path}:{number}: expected 5 tab-separated fields, found {len(fields)}")
        entries.append(Entry(*fields))
    if not saw_format:
        raise ManifestError(f"{path}: missing the {FORMAT_LINE!r} header line")
    return entries


def compare_manifests(left: Sequence[Entry], right: Sequence[Entry]) -> tuple[list[Difference], list[tuple[str, str]], list[tuple[str, str]]]:
    """Differences, entries only on the left, and entries only on the right."""
    left_map = {entry.identity: entry for entry in left}
    right_map = {entry.identity: entry for entry in right}
    differences: list[Difference] = []
    for identity in sorted(left_map.keys() & right_map.keys()):
        a = left_map[identity]
        b = right_map[identity]
        for field in ("dtype", "shape", "digest"):
            first = getattr(a, field)
            second = getattr(b, field)
            if first != second:
                differences.append(
                    Difference(name=a.name, key=a.key, field=field, left=first, right=second)
                )
    only_left = sorted(left_map.keys() - right_map.keys())
    only_right = sorted(right_map.keys() - left_map.keys())
    return differences, only_left, only_right


def print_comparison(
    left_path: Path,
    right_path: Path,
    differences: Sequence[Difference],
    only_left: Sequence[tuple[str, str]],
    only_right: Sequence[tuple[str, str]],
    stream: TextIO | None = None,
) -> None:
    """Human-readable diff receipt."""
    out = sys.stdout if stream is None else stream

    def line(text: str = "") -> None:
        out.write(text + "\n")

    line("G1 manifest comparison")
    line(f"  reference: {left_path}")
    line(f"  candidate: {right_path}")
    line(f"  entries only in reference: {len(only_left)}")
    for name, key in only_left:
        line(f"    {name} {key}")
    line(f"  entries only in candidate: {len(only_right)}")
    for name, key in only_right:
        line(f"    {name} {key}")
    line(f"  differing entries: {len(differences)}")
    for difference in differences:
        line(
            f"    {difference.name} {difference.key} {difference.field}: "
            f"{difference.left} != {difference.right}"
        )
    identical = not differences and not only_left and not only_right
    line(f"  verdict: {'PASS' if identical else 'FAIL'}")


def run_build(args: argparse.Namespace) -> int:
    entries = build_manifest(
        signatures=args.signatures,
        pattern=args.pattern,
        json_pattern=args.json_pattern,
        include_json=not args.skip_json,
    )
    text = render_manifest(entries)
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        names = {entry.name for entry in entries}
        print("G1 manifest built")
        print(f"  signatures: {args.signatures.resolve()}")
        print(f"  files: {len(names)}   entries: {len(entries)}")
        print(f"  manifest: {args.out}")
        print(f"  manifest sha256: {hashlib.sha256(text.encode('utf-8')).hexdigest()}")
    return 0


def run_compare(args: argparse.Namespace) -> int:
    left_path, right_path = args.compare
    left = parse_manifest(left_path)
    right = parse_manifest(right_path)
    differences, only_left, only_right = compare_manifests(left, right)
    print_comparison(left_path, right_path, differences, only_left, only_right)
    return 0 if not (differences or only_left or only_right) else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="g1_hash_manifest.py",
        description="G1: array-content manifests over a signatures directory, and their diff.",
    )
    parser.add_argument("--signatures", type=Path, default=None, help="signatures directory to hash")
    parser.add_argument("--out", type=Path, default=None, help="write the manifest here")
    parser.add_argument("--pattern", default="gen_*.npz", help="array-file glob within the directory")
    parser.add_argument("--json-pattern", default="*.json", help="sidecar glob within the directory")
    parser.add_argument("--skip-json", action="store_true", help="hash arrays only, no sidecars")
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        default=None,
        metavar=("REFERENCE", "CANDIDATE"),
        help="diff two manifests instead of building one",
    )
    args = parser.parse_args(argv)

    if args.compare is not None and args.signatures is not None:
        parser.error("--compare and --signatures are separate modes; pass one")
    if args.compare is None and args.signatures is None:
        parser.error("pass --signatures to build a manifest, or --compare to diff two")

    try:
        return run_compare(args) if args.compare is not None else run_build(args)
    except ManifestError as exc:
        print(f"g1_hash_manifest: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
