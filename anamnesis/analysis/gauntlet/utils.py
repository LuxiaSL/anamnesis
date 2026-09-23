"""Shared utilities for the unified analysis runner.

Three of them are about what a section is allowed to assume. A section reads
either the signature data or another section's result, and neither is guaranteed
to hold what it wants: a corpus narrower than the suite that defined the unions
has fewer blocks, and a section that could not run returns a stub carrying its
reason instead of its numbers. :func:`absence_reason` answers the first,
:func:`is_error_stub` and :func:`section_reading` the second, and every consumer
goes through them — so an absence is stated rather than raised, wherever it
comes from.

:func:`topic_fold_partition` is here for a different reason: sections 5, 8 and 9
all hold out whole topics, and two of them compare their numbers against each
other. One partitioner means they cannot disagree about what a fold is.
"""

from __future__ import annotations

import textwrap
import time
from contextlib import contextmanager
from typing import Any, Generator, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from .signature_io import (
    ALL_CORE,
    ALL_FAMILIES,
    ATTENTION_AND_CACHE,
    ATTENTION_AND_CACHE_WITH_FAMILIES,
    ATTENTION_AND_DELTAS,
    CACHE_AND_KEYS,
    EVERYTHING,
    NORMS_AND_OUTPUT_STATS,
    RESIDUAL_PCA,
)


def standardize(X: NDArray[np.floating]) -> NDArray[np.float64]:
    """Z-score standardize features. Constant features get std=1."""
    X = X.astype(np.float64)
    std = X.std(axis=0)
    std[std < 1e-12] = 1.0
    return (X - X.mean(axis=0)) / std


def remove_constant(X: NDArray[np.float64], threshold: float = 1e-12) -> NDArray[np.float64]:
    """Remove features with near-zero variance."""
    variance = X.var(axis=0)
    mask = variance > threshold
    return X[:, mask]


def clean_for_json(obj: object) -> object:
    """Recursively make objects JSON-serializable.

    Handles: NaN/Inf floats (→ None), numpy scalars/arrays, dicts, lists,
    and pydantic BaseModel instances (dumped with ``exclude_none=True`` so
    Optional error fields vanish when unset).
    """
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, BaseModel):
        return clean_for_json(obj.model_dump(mode="json", exclude_none=True))
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    return obj


@contextmanager
def timer(label: str = "") -> Generator[dict[str, float], None, None]:
    """Context manager that tracks elapsed time."""
    result: dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - t0
        if label:
            print(f"  [{label}] {result['elapsed']:.1f}s")


# Default block lists — used when analysis modules don't get dynamic lists.
# For v2 data, use get_available_blocks() to discover what's actually present.
ALL_BLOCKS = [NORMS_AND_OUTPUT_STATS, ATTENTION_AND_DELTAS, CACHE_AND_KEYS, RESIDUAL_PCA, ATTENTION_AND_CACHE, ALL_CORE]
KEY_BLOCKS = [ATTENTION_AND_CACHE, ALL_CORE]

BLOCK_READOUT_LIMIT = (
    "Most of these blocks span more than one substrate, so a block's accuracy — and "
    "any ordering of blocks by accuracy — localizes nothing: it makes no substrate "
    "load-bearing. The decomposition of record cuts by family and sub-family: "
    "anamnesis/scripts/run_subfamily_decomp.py, over anamnesis/analysis/subfamily.py, "
    "with the taxonomy in anamnesis/feature_map.py."
)
"""The limit on every reading taken per block, in one place.

Three surfaces state it — the per-block readout's own output, the scorecard row that
orders three blocks, and the summary printer — and a reader who meets any of them
without it reads a block ordering as a claim about substrates. One string, because a
limit stated in three wordings is a limit a reader can believe was three different
limits.
"""

CAVEAT_WIDTH = 88
"""Line width a printed caveat wraps to, wide enough that a limit stays a paragraph."""


def print_caveat(text: str, *, indent: str = "  ") -> None:
    """Print a limit beside the numbers it constrains, wrapped and indented.

    A caveat is carried as one string so that the result file and the terminal say
    the same thing; printing it needs the wrapping this applies, because an
    unwrapped paragraph in a column of numbers is a paragraph a reader skips.
    """
    for line in textwrap.wrap(text, width=CAVEAT_WIDTH - len(indent)):
        print(f"{indent}{line}")


def absence_reason(data: object, *blocks: str) -> str | None:
    """Why a reading over ``blocks`` cannot be taken here, or None when it can.

    A union is built only when every member it names is present, so a corpus
    narrower than the suite that defined it holds fewer blocks than a section
    asks for. A section calls this before reading, and puts what comes back in
    its own ``error`` field: an absence is a fact about the corpus, and a pass
    that states it is worth more than one that dies on a missing key.
    """
    run4 = getattr(data, "run4", data)
    missing = [b for b in blocks if not run4.has_block(b)]
    if not missing:
        return None
    present = sorted(set(run4.block_features) | set(run4.group_features))
    return (
        f"{', '.join(missing)} not in this corpus "
        f"(blocks and unions present: {', '.join(present) or 'none'})"
    )


def is_error_stub(value: object) -> bool:
    """Whether a section's result is a stub carrying a reason rather than numbers.

    A section that cannot run — an optional dependency absent, a block this corpus
    does not hold — returns its own result model with only ``error`` set. That is a
    legitimate value of the section, not a failure to produce one, so every reader
    of a section's output has to test for it: the orchestrator to decide what a
    resume may skip, :func:`section_reading` on behalf of a consuming section.
    """
    if isinstance(value, BaseModel):
        return bool(getattr(value, "error", None))
    if isinstance(value, dict):
        return bool(value.get("error"))
    return False


def error_stub_reason(value: object) -> str:
    """The reason a stub carries, however the stub is spelled."""
    if isinstance(value, BaseModel):
        return str(getattr(value, "error", "") or "no reason recorded")
    if isinstance(value, dict):
        return str(value.get("error") or "no reason recorded")
    return "no reason recorded"


_Section = TypeVar("_Section", bound=BaseModel)


def section_reading(
    all_results: dict[str, Any], key: str, model: type[_Section],
) -> tuple[_Section | None, str | None]:
    """One section's result for a section that consumes it, or the reason there is none.

    A consuming section must treat the error stub as a possible value of its
    upstream, exactly as a reading section must treat an absent block as a
    possible state of its corpus. Three states come back as one shape: the typed
    result and no reason; nothing and a reason naming the section that did not
    run; nothing and a reason saying the section is absent or mis-shaped.

    Returns
    -------
    (result, reason)
        Exactly one of the two is None. A caller that can score without this
        upstream carries the reason beside the reading it could not take; one that
        cannot carries it as its own ``error``.
    """
    value = all_results.get(key)
    if isinstance(value, model):
        if is_error_stub(value):
            return None, f"section '{key}' did not run: {error_stub_reason(value)}"
        return value, None
    if value is None:
        return None, f"section '{key}' is not in these results"
    if is_error_stub(value):
        return None, f"section '{key}' did not run: {error_stub_reason(value)}"
    return None, f"section '{key}' did not validate into {model.__name__}"


class InsufficientTopicsError(ValueError):
    """A fold count larger than the number of topics there are to hold out."""


def topic_fold_partition(
    n_topics: int, n_folds: int, rng: np.random.Generator,
) -> list[NDArray[np.int64]]:
    """A shuffled partition of ``n_topics`` indices into ``n_folds`` held-out groups.

    Every index lands in exactly one group, and the group sizes differ by at most
    one. That is the property the callers need and the reason this is a partition
    rather than a stride: with a topic count that is not a multiple of the fold
    count, taking ``n // n_folds`` indices per fold leaves the remainder in no
    held-out group at all, and the variant is then computed over a subset of the
    topics while reporting the whole.

    ``rng`` is consumed for exactly one permutation of ``n_topics``, so a caller
    that seeded it gets the same partition it always did; for a topic count that
    divides the fold count evenly this returns the same groups, in the same order,
    that consecutive equal slices of that permutation give.

    Raises
    ------
    InsufficientTopicsError
        When ``n_folds`` exceeds ``n_topics``, which cannot be a partition into
        non-empty groups. The alternative is empty held-out groups, which reach
        the classifier as a zero-row matrix and fail there instead of here.
    """
    if n_folds < 1:
        raise InsufficientTopicsError(f"n_folds={n_folds}: a fold count is at least 1")
    if n_folds > n_topics:
        raise InsufficientTopicsError(
            f"{n_folds} folds over {n_topics} topics: a held-out fold would be empty. "
            f"Use at most {n_topics} folds on this corpus."
        )
    return [
        group.astype(np.int64)
        for group in np.array_split(rng.permutation(n_topics), n_folds)
    ]


def get_available_blocks(data: object) -> tuple[list[str], list[str]]:
    """Discover which blocks and groups are available in loaded data.

    Parameters
    ----------
    data : AnalysisData or Run4Data
        Loaded data object with block_features and group_features.

    Returns
    -------
    all_blocks : list[str]
        All individual blocks + groups that are present.
    key_blocks : list[str]
        Key composite groups for expensive analyses.
    """
    run4 = getattr(data, "run4", data)
    individual = list(run4.block_features.keys())
    groups = list(run4.group_features.keys())

    all_blocks = individual + groups
    # Key blocks: composites that include multiple families
    key_blocks = [g for g in groups if g in {
        ATTENTION_AND_CACHE, ALL_CORE, ALL_FAMILIES, EVERYTHING,
        ATTENTION_AND_CACHE_WITH_FAMILIES,
    }]

    return all_blocks, key_blocks
