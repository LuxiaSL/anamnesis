"""Rate a bank's texts on every mode at once, and correlate purity with geometry.

The Likert paradigm, which is the other judging channel: one text, five ratings, a
primary classification. What it uniquely yields is **purity** — the intended mode's
rating minus the mean of the other four — a graded per-text readout that a rate over
pairs cannot produce, and the quantity that makes the cross-channel question sayable:
do the texts a judge reads as purely in-mode sit nearest their mode's centroid in
signature space?

It is not the paradigm of record for hardening a claim. A forced choice keeps the
answer out of the judge's context entirely; here the judge sees the text and the topic
but never the mode instruction, which is weaker, and the numbers are reported as
ratings rather than quoted as hardened rates. The forced-choice command beside this one
is ``judge_2afc.py``.

The retry stays on **one** model rather than falling to another, which is the opposite
of the forced-choice ladder's rule and right for the opposite reason: a Likert number is
a reading on one judge's internal scale, and a second model's rating is a different
scale wearing the same number.

    python -m anamnesis.scripts.judge_likert --sig-dir outputs/runs/8b_fat_01/signatures \\
        --out outputs/analysis/8b_v2/judge_scores.json
    python -m anamnesis.scripts.judge_likert --sig-dir <dir> --out <file> --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from anamnesis.judging.harness import AnthropicBackend, JudgingError, OpenRouterBackend

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
"""The rating judge. A mid-grade model is the default because a Likert pass scores every
generation of a bank, and the paradigm's ceiling is set by its blinding rather than by
the reader's grade."""

PROGRESS_EVERY = 10
"""Generations between partial writes. The receipt is written as it fills so a killed
pass resumes from what it scored rather than from nothing."""


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="judge_likert.py", description=__doc__.splitlines()[0])
    p.add_argument("--sig-dir", type=Path, required=True, help="Signature directory holding the texts")
    p.add_argument("--out", type=Path, required=True, help="Where the receipt is written")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Judge model (default: {DEFAULT_MODEL})")
    p.add_argument("--family", choices=("anthropic", "openrouter"), default="anthropic")
    p.add_argument("--all-reps", action="store_true", help="Score every repetition, not one per pair")
    p.add_argument("--resume", action="store_true", help="Skip generations already in the receipt")
    p.add_argument("--retries", type=int, default=3, help="Attempts per generation, on one model")
    p.add_argument(
        "--no-correlation", action="store_true",
        help="Skip the purity-against-centroid-distance readout",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)

    from anamnesis.judging.likert import (
        likert_receipt,
        load_generations,
        purity_signature_correlation,
        read_scores,
        score_bank,
        summarize,
    )

    backend = AnthropicBackend() if args.family == "anthropic" else OpenRouterBackend()
    generations = load_generations(args.sig_dir, core_only=not args.all_reps)
    already = read_scores(args.out) if args.resume and args.out.exists() else []
    if already:
        logger.info(f"resuming: {len(already)} generations already scored")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    def bank_partial(scores: list, failed: list) -> None:
        if len(scores) % PROGRESS_EVERY:
            return
        args.out.write_text(
            json.dumps(likert_receipt(model=args.model, scores=scores, failed=failed), indent=2),
            encoding="utf-8",
        )

    scores, failed, usage = score_bank(
        backend,
        generations,
        model=args.model,
        already_scored=already,
        retries=args.retries,
        on_progress=bank_partial,
    )
    if not scores:
        logger.error("no generation was scored — this is a non-run, not a null result")
        return 2

    summary = summarize(scores)
    correlation = (
        None if args.no_correlation else purity_signature_correlation(scores, args.sig_dir)
    )
    args.out.write_text(
        json.dumps(
            likert_receipt(
                model=args.model,
                scores=scores,
                failed=failed,
                summary=summary,
                correlation=correlation,
                usage=usage,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        f"{summary.n} scored: accuracy {summary.overall_accuracy:.1%}, "
        f"mean purity {summary.mean_purity:.2f} (sd {summary.std_purity:.2f}), "
        f"{len(failed)} failed, {usage.calls} calls -> {args.out}"
    )
    if correlation is not None:
        logger.info(
            f"purity against centroid distance: r={correlation.purity_distance_correlation:+.3f} "
            f"over {correlation.n_samples} generations"
        )
    elif not args.no_correlation:
        logger.info("purity-against-geometry not reported: too few generations have signatures")
    if failed:
        logger.warning(f"{len(failed)} generations failed; rerun with --resume to retry them")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except JudgingError as exc:
        print(f"judge_likert: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
