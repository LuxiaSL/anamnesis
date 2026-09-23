"""First contact with a new model: validate its preset before spending on it.

Loads the checkpoint through the instrument's own loader, resolves the layer plan and
the hook targets, generates a few tokens with the capture surface armed, and extracts one
feature vector. A routed checkpoint additionally has its router and compressed-key
surface checked, and the routing family's own arity and finiteness established.

Every step refuses with a reason rather than asserting, because the whole output of this
command is its refusals: the failures it catches — a fused attention kernel returning no
weights, a layer plan off by one, a hook target that does not exist on this architecture
— are the ones that otherwise produce a full-width vector of nothing.

Run it on one device before any calibration or floor pass.

    CUDA_VISIBLE_DEVICES=0 python -m anamnesis.scripts.onboard_model --model gemma3-27b
    CUDA_VISIBLE_DEVICES=0 python -m anamnesis.scripts.onboard_model --model 8b \\
        --model-path /models/llama-3.1-8b-instruct
"""

from __future__ import annotations

import argparse
import logging

from anamnesis.config import preset_names, resolve_preset
from anamnesis.extraction.onboarding import OnboardingError, onboard_model

logger = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="onboard_model.py", description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=list(preset_names()), required=True)
    p.add_argument(
        "--model-path", default=None,
        help="Local snapshot to load instead of the preset's identifier",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parser().parse_args(argv)
    try:
        report = onboard_model(resolve_preset(args.model), args.model_path)
    except OnboardingError as exc:
        logger.error(f"SMOKE FAIL: {exc}")
        return 1
    for line in report.lines():
        logger.info(line)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
