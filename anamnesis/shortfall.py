"""Fail closed: a command that produced less than it was asked for refuses.

The library layer reports partial failure in a typed object — a replay's
:class:`anamnesis.extraction.replay.cell.CellResult`, a generation pass's
:class:`anamnesis.extraction.token_generation.GenerationCount`, a recompute's
:class:`anamnesis.extraction.feature_pipeline.RecomputeCount`. A command that
discards one of those reports success over an incomplete corpus, and that is the
one failure a downstream contrast cannot see: a mean over four of five
generations is a number, and nothing about the number says five were asked for.

So a command that produces a set of units ends by stating the set in a
:class:`Shortfall` and handing it to :func:`refuse_unless_complete`. The shape is
the same at every site:

* ``requested`` — the ids this invocation was asked for, taken from the artifact
  that names them: a replay manifest's entries, a spec file's generation ids, a
  raw-tensor directory's listing, the gauntlet's section registry.
* ``produced`` — the requested ids whose output artifact is on disk when the pass
  ends. On disk, rather than computed by this pass: a resumed pass that skips
  seventeen existing signatures and computes three has produced twenty and is
  complete. Reading the disk rather than the work done is what keeps a resume
  from registering as a short pass.
* ``excluded`` — ids this invocation was never going to produce, and says so: a
  generation the replay manifest flags as unreplayable, a section named in
  ``--skip``. An expected exclusion is not a failure.
* ``failures`` — requested ids that were attempted and raised, with the reason.

An invocation narrowed on purpose — one worker's ``--gen-ids`` share, a spec file
holding four of a run's four hundred specs — requests exactly its own slice, so a
deliberate subset is complete when it produced that subset.

Exit statuses, so a wrapper can tell the three outcomes apart:

* ``0`` — complete.
* :data:`EXIT_SHORT` — short, and nobody asked for a short pass.
* :data:`EXIT_SHORT_SANCTIONED` — short, and the invocation passed
  ``--allow-partial``. Still non-zero, because partial work is permitted and
  hiding it is not.

Either short outcome leaves a receipt beside the output it describes, named for
the worker that wrote it so parallel workers over one directory do not overwrite
each other's. A complete pass deletes its own receipt, so a directory carrying
one is short as of its last pass rather than as of some earlier one.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Self, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

EXIT_SHORT = 3
"""Exit status for a short pass nobody sanctioned. Above the status a bare
``SystemExit`` message produces and above argparse's, so a wrapper reading the
status can tell a refusal from a crash and from a usage error."""

EXIT_SHORT_SANCTIONED = 4
"""Exit status for a short pass the invocation asked for with ``--allow-partial``.
Distinct from :data:`EXIT_SHORT` so a wrapper can accept the sanctioned case
without accepting the other, and non-zero so neither is mistaken for complete."""

RECEIPT_STEM = "shortfall"
"""Stem of the receipt file a short pass leaves in its output directory."""

#: How many ids a one-line summary names before it stops and gives the count.
SUMMARY_IDS = 10


class Shortfall(BaseModel):
    """What one pass over one output directory was asked for, and what it produced.

    Ids are strings whatever they number, because the sites do not agree on a
    type: a generation id is an integer, a gauntlet section is a name. A caller
    formats its own ids once, here, and everything downstream — the refusal
    message, the receipt, a test's assertion — reads the same spelling.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    command: str = Field(min_length=1, description="Module a reader would re-run")
    unit: str = Field(min_length=1, description="What one id names, singular")
    target: Path = Field(description="Directory the produced units land in")
    requested: tuple[str, ...] = Field(description="Ids this invocation was asked for")
    produced: tuple[str, ...] = Field(description="Requested ids whose artifact is on disk")
    excluded: dict[str, str] = Field(
        default_factory=dict, description="Id to the reason it was never attempted"
    )
    failures: dict[str, str] = Field(
        default_factory=dict, description="Id to the reason the attempt raised"
    )
    label: str = Field(
        default="",
        description="Worker label, so parallel workers over one directory get one receipt each",
    )

    @model_validator(mode="after")
    def _the_accounting_closes(self) -> Self:
        """Produced and failed ids are requested ones, and an exclusion is not a request.

        A call site that listed a whole directory instead of its own slice, or
        that counted a skipped unit as failed, would otherwise report a shortfall
        against a set it never asked for.
        """
        requested = set(self.requested)
        for name, ids in (("produced", set(self.produced)), ("failures", set(self.failures))):
            stray = sorted(ids - requested)
            if stray:
                raise ValueError(
                    f"{name} names {stray}, which this pass did not request"
                )
        both = sorted(requested & set(self.excluded))
        if both:
            raise ValueError(
                f"{both} is requested and excluded at once; an exclusion is not a request"
            )
        return self

    @property
    def missing(self) -> tuple[str, ...]:
        """Requested ids with no artifact on disk, in requested order.

        A unit that raised is usually missing too, but not always: a pass run
        with resume off over a directory holding an earlier pass's output leaves
        the stale artifact in place, which is why :attr:`ok` reads both.
        """
        produced = set(self.produced)
        return tuple(i for i in self.requested if i not in produced)

    @property
    def ok(self) -> bool:
        """True when every requested unit is on disk and none of them raised."""
        return not self.missing and not self.failures

    @property
    def receipt_path(self) -> Path:
        """Where this pass's receipt goes, inside the directory it describes."""
        stem = f"{RECEIPT_STEM}-{self.label}" if self.label else RECEIPT_STEM
        return self.target / f"{stem}.json"

    def summary(self) -> str:
        """One line naming the count, then what is missing and what raised.

        Bounded at :data:`SUMMARY_IDS` ids per list, because a pass that lost
        four hundred generations says so in the count and the receipt carries
        the rest.
        """
        head = (
            f"{self.command}: {len(self.produced)} of {len(self.requested)} "
            f"{self.unit}s in {self.target}"
        )
        parts = [head]
        if self.missing:
            parts.append(f"missing {_capped(self.missing)}")
        if self.failures:
            parts.append(
                "failed "
                + "; ".join(
                    f"{key} ({self.failures[key]})"
                    for key in sorted(self.failures)[:SUMMARY_IDS]
                )
                + (f" and {len(self.failures) - SUMMARY_IDS} more"
                   if len(self.failures) > SUMMARY_IDS else "")
            )
        if self.excluded:
            parts.append(f"{len(self.excluded)} excluded by the manifest or by a flag")
        return " — ".join(parts)

    def receipt(self, *, sanctioned: bool) -> dict[str, Any]:
        """The receipt's content: every missing id and every failure, with reasons."""
        return {
            "command": self.command,
            "unit": self.unit,
            "target": str(self.target),
            "label": self.label,
            "sanctioned": sanctioned,
            "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_requested": len(self.requested),
            "n_produced": len(self.produced),
            "requested": list(self.requested),
            "missing": list(self.missing),
            "failures": dict(self.failures),
            "excluded": dict(self.excluded),
        }

    def write_receipt(self, *, sanctioned: bool) -> Path | None:
        """Write the receipt beside the output, returning where it landed.

        A receipt that cannot be written is logged and returns ``None``: the
        refusal is the part that must happen, and an unwritable output directory
        is one of the ways a pass comes up short in the first place.
        """
        path = self.receipt_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.receipt(sanctioned=sanctioned), indent=2))
        except OSError as exc:
            logger.error(f"could not write the shortfall receipt to {path}: {exc}")
            return None
        return path

    def clear_receipt(self) -> None:
        """Remove this pass's own receipt, which a complete pass has earned."""
        try:
            self.receipt_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.error(f"could not clear the shortfall receipt at {self.receipt_path}: {exc}")


def _capped(ids: Sequence[str]) -> str:
    """Up to :data:`SUMMARY_IDS` ids, then a count of the rest."""
    shown = ", ".join(ids[:SUMMARY_IDS])
    if len(ids) <= SUMMARY_IDS:
        return shown
    return f"{shown} and {len(ids) - SUMMARY_IDS} more"


def ids_present(ids: Iterable[str], has_artifact: Callable[[str], bool]) -> tuple[str, ...]:
    """The subset of ``ids`` whose output artifact is on disk, in the order given.

    ``has_artifact`` decides what produced means for one site, because the naming
    convention belongs to the site: a signature is a vector npz beside its
    metadata json, a banked record is one json, a section is a key in a results
    document.
    """
    return tuple(name for name in ids if has_artifact(name))


def refuse_unless_complete(
    shortfalls: Sequence[Shortfall], *, allow_partial: bool
) -> None:
    """Exit non-zero when any pass came up short, after leaving its receipt.

    Every shortfall is reported before the first raise, so an invocation that
    walked a roster of cells names all of the short ones rather than the first.
    A complete pass clears its own receipt, so a directory holding one is short
    as of its last pass.

    Raises
    ------
    SystemExit
        :data:`EXIT_SHORT_SANCTIONED` when ``allow_partial`` is set, otherwise
        :data:`EXIT_SHORT`. The status carries the verdict; the message is
        logged, because an integer status is what a wrapper reads.
    """
    for shortfall in shortfalls:
        if shortfall.ok:
            shortfall.clear_receipt()
    short = [shortfall for shortfall in shortfalls if not shortfall.ok]
    if not short:
        return
    for shortfall in short:
        logger.error(shortfall.summary())
        path = shortfall.write_receipt(sanctioned=allow_partial)
        if path is not None:
            logger.error(f"  receipt: {path}")
    if allow_partial:
        logger.error(
            f"{len(short)} of {len(shortfalls)} passes came up short; "
            f"sanctioned by --allow-partial, exiting {EXIT_SHORT_SANCTIONED}"
        )
        raise SystemExit(EXIT_SHORT_SANCTIONED)
    logger.error(
        f"{len(short)} of {len(shortfalls)} passes came up short; "
        f"re-run to fill the gap, or pass --allow-partial to accept it on the record"
    )
    raise SystemExit(EXIT_SHORT)


def worker_shortfall_code(returncodes: Mapping[int, int]) -> int | None:
    """The status a fan-out inherits from workers that refused, or ``None``.

    A worker that came up short exits :data:`EXIT_SHORT` or
    :data:`EXIT_SHORT_SANCTIONED`, and a launcher that collapsed both to a
    generic failure would lose the distinction its own wrapper needs. ``None``
    means the failures are not all shortfalls — a worker crashed, or died on a
    device — and the launcher's own failure report is the one that should speak.
    """
    failed = [code for code in returncodes.values() if code != 0]
    if not failed or any(code not in (EXIT_SHORT, EXIT_SHORT_SANCTIONED) for code in failed):
        return None
    return EXIT_SHORT if EXIT_SHORT in failed else EXIT_SHORT_SANCTIONED
