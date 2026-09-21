"""Run one blind 2AFC study: draw the pairs, bank the key apart, judge, report.

A command line names a target bank, one or more distractor banks, a prompt set
from the versioned table, and the instrument whose numbers the result will sit
beside. It draws the contrast, writes the packet and the key to two files, runs
the judge ladder, optionally runs a ceiling contrast and the coherence gate, and
writes one receipt carrying every rate beside the law it was taken under.

Two things it refuses rather than warns about. A study whose judge criterion
comes from the instrument scoring it is stopped before a call is made, because
that table's two columns would be one column. And ``--dry-run`` exists so a
contrast can be drawn, inspected and banked without spending anything, which is
also how a packet is prepared for a judge that is not an API.

    python -m anamnesis.scripts.judge_2afc \\
        --target outputs/battery/cell_V3_L14_a0.1/metadata.json \\
        --distractor rider=outputs/battery/cell_rider_a0.0/metadata.json \\
        --prompt-set socratic --variant socratic \\
        --scoring-instrument "attention-allocation signature classifier" \\
        --judge claude-fable-5 --judge claude-opus-4-8 \\
        --out-dir outputs/judge/V3_L14_a0.1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from anamnesis.judging.harness import (
    AnthropicBackend,
    BlindPacket,
    Corpus,
    JudgeBackend,
    JudgingError,
    OpenRouterBackend,
    assert_not_circular,
    draw_pairs,
    interpret_with_ceiling,
    receipt,
    run_coherence_gate,
    run_contrast,
    run_reader_ladder,
    sample_texts,
    texts_by_topic,
)
from anamnesis.judging.prompts import PROMPT_SETS, prompt_set

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="judge_2afc.py", description=__doc__.splitlines()[0]
    )
    p.add_argument("--target", type=Path, required=True, help="metadata.json of the target bank")
    p.add_argument(
        "--distractor", action="append", required=True, metavar="NAME=PATH",
        help="a named distractor bank; repeat for target-versus-any-other",
    )
    p.add_argument("--prompt-set", choices=sorted(PROMPT_SETS), required=True)
    p.add_argument("--variant", default=None, help="which criterion of the prompt set to ask")
    p.add_argument(
        "--scoring-instrument", required=True,
        help="what scores the number this judgement sits beside (the anti-circularity check)",
    )
    p.add_argument("--contrast", default=None, help="name for the contrast (default: the target dir)")
    p.add_argument("--judge", action="append", default=None, help="judge model; repeat for a fallback ladder")
    p.add_argument(
        "--reader-ladder", default=None,
        help="comma list of judge models, strongest first, each read separately",
    )
    p.add_argument("--family", choices=("anthropic", "openrouter"), default="anthropic")
    p.add_argument("--n-pairs", type=int, default=40)
    p.add_argument("--per-group", type=int, default=None)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--min-chars", type=int, default=0, help="drop texts shorter than this")
    p.add_argument("--max-chars", type=int, default=2200, help="truncate each side of a pair")
    p.add_argument("--min-words", type=int, default=20, help="bank filter on load")
    p.add_argument("--threshold", type=float, default=0.70)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--ceiling-target", type=Path, default=None, help="metadata.json of the ceiling arm")
    p.add_argument("--ceiling-distractor", type=Path, default=None)
    p.add_argument("--coherence-n", type=int, default=0, help="0 skips the gate")
    p.add_argument("--coherence-window", choices=("head", "tail"), default="tail")
    p.add_argument("--coherence-chars", type=int, default=1200)
    p.add_argument("--coherence-floor", type=float, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true", help="draw and bank the packet; make no calls")
    return p


def _named_banks(specs: list[str], *, min_words: int) -> dict[str, Corpus]:
    banks: dict[str, Corpus] = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        if not path:
            raise JudgingError(f"--distractor takes NAME=PATH; got {spec!r}")
        banks[name] = texts_by_topic(Path(path), min_words=min_words)
    return banks


def _backend(family: str) -> JudgeBackend:
    return AnthropicBackend() if family == "anthropic" else OpenRouterBackend()


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    prompt = prompt_set(args.prompt_set)
    assert_not_circular(prompt, scoring_instrument=args.scoring_instrument)

    contrast = args.contrast or args.target.parent.name
    target = texts_by_topic(args.target, min_words=args.min_words)
    packet = draw_pairs(
        contrast=contrast,
        target=target,
        distractors=_named_banks(args.distractor, min_words=args.min_words),
        n_pairs=args.n_pairs,
        seed=args.seed,
        per_group=args.per_group,
        min_chars=args.min_chars,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    packet.write(packet_path=args.out_dir / "packet.json", key_path=args.out_dir / "key.json")
    logger.info("%s: %d pairs drawn, key banked apart", contrast, len(packet.pairs))
    if args.dry_run:
        logger.info("dry run: no calls made")
        return 0

    backend = _backend(args.family)
    ladder = args.judge or ["claude-fable-5", "claude-opus-4-8"]
    shared = {
        "variant": args.variant,
        "workers": args.workers,
        "max_chars": args.max_chars,
        "threshold": args.threshold,
    }

    if args.reader_ladder:
        grades = [g.strip() for g in args.reader_ladder.split(",") if g.strip()]
        ladder_result = run_reader_ladder(backend, packet, prompt, grades=grades, **shared)
        results = [ladder_result.results[g] for g in grades if g in ladder_result.results]
        logger.info("reader ladder passes at: %s", ladder_result.weakest_pass or "no grade")
        primary = results[0]
    else:
        primary = run_contrast(backend, packet, prompt, models=ladder, **shared)
        results = [primary]

    ceiling = None
    if args.ceiling_target and args.ceiling_distractor:
        ceiling_packet = draw_pairs(
            contrast=f"{contrast}-CEIL",
            target=texts_by_topic(args.ceiling_target, min_words=args.min_words),
            distractors={"ceiling": texts_by_topic(args.ceiling_distractor, min_words=args.min_words)},
            n_pairs=args.n_pairs,
            seed=args.seed,
            per_group=args.per_group,
            min_chars=args.min_chars,
        )
        ceiling_packet.write(
            packet_path=args.out_dir / "ceiling_packet.json",
            key_path=args.out_dir / "ceiling_key.json",
        )
        ceiling = run_contrast(backend, ceiling_packet, prompt, models=ladder, **shared)
        results.append(ceiling)

    interpretation = interpret_with_ceiling(primary, ceiling)
    coherence = {}
    if args.coherence_n > 0:
        coherence[contrast] = run_coherence_gate(
            backend,
            sample_texts(target, args.coherence_n, seed=args.seed),
            models=ladder,
            window=args.coherence_window,
            chars=args.coherence_chars,
            workers=args.workers,
            floor=args.coherence_floor,
        )

    out = receipt(
        study=contrast, results=results, interpretations=[interpretation], coherence=coherence
    )
    (args.out_dir / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    logger.info(
        "%s: rate=%s wilson=%s verdict=%s (calls=%d)",
        contrast, primary.win_rate, primary.wilson95, interpretation.verdict,
        out["usage"]["calls"],
    )
    if primary.is_non_run:
        logger.error("ZERO scored pairs — this is a non-run, not a null; check the environment key")
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except JudgingError as exc:
        print(f"judge_2afc: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
