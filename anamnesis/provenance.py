"""Digests that say which inputs a result was produced from.

A number this instrument writes down is only as good as the record of what went
into it, and a filename is not that record: a calibration directory, a checkpoint
and a manifest can all be replaced in place. So the paths that bank results stamp
content digests beside them — of the calibration artifacts, of the weight files, of
the manifest, and of the source modules whose arithmetic defines the vector.

The digest is SHA-256 over file bytes, read in blocks so a multi-gigabyte weight
file does not have to fit in memory. That is deliberately the plainest possible
definition: anyone can reproduce it with a command-line tool, which is what makes a
stamped receipt checkable by someone who does not trust this code.

This lives at the package root rather than under either layer because both read it:
extraction stamps what it produced, and an analysis deciding whether two banks may
be joined inside one contrast reads the stamps back.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

BLOCK_BYTES = 8 * 1024**2
"""How much of a file is read at a time. Large enough that hashing a checkpoint is
bound by the disk rather than by Python, small enough to hash anything."""


def file_sha(path: Path) -> str:
    """SHA-256 of a file's bytes, as hexadecimal."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def numpy_dispatch() -> dict[str, Any]:
    """NumPy's version and the CPU instruction sets its compiled loops dispatch to here.

    NumPy picks a loop per CPU at runtime, and its ``argsort`` orders tied values
    differently under AVX-512, AVX2 and scalar loops, which the gate's top-k overlap
    reads. So a reduction's record names the dispatch, beside the device: the
    baseline the build assumes, and each dispatch target this CPU enables.
    """
    import numpy

    try:
        from numpy._core import _multiarray_umath as umath
    except ImportError:  # NumPy 1.x
        from numpy.core import _multiarray_umath as umath  # type: ignore[no-redef]
    features = dict(getattr(umath, "__cpu_features__", {}))
    return dict(numpy=numpy.__version__,
                baseline=list(getattr(umath, "__cpu_baseline__", [])),
                dispatch=[name for name in getattr(umath, "__cpu_dispatch__", [])
                          if features.get(name)])


def digest_of_shas(shas: Mapping[str, str]) -> str:
    """One digest over a set of named file digests.

    Sorted by name before hashing, so the result names the *set* of inputs and not
    the order they happened to be listed in.
    """
    return hashlib.sha256(json.dumps(dict(shas), sort_keys=True).encode()).hexdigest()
