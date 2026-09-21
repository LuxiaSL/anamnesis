"""The prompt-swap test: is the signal what the model was told, or what it did?

A mode label names two things at once — the system prompt a generation was given,
and the processing it actually ran. A classifier that separates modes could be
reading either. The swap corpus is what pulls them apart: a generation is given
mode A's system prompt and a user instruction that asks for mode B's execution, so
the two axes disagree by construction and a prediction has to land on one of them.

The test is binary per swap pair, which is what makes the readout a count rather
than a rate over a confusion matrix. For ``swap_A→B``: train on the two pure
modes A and B from the core corpus, predict the swap samples, and count how many
land on B (the execution) against A (the system prompt). Signal that is
**execution-based** lands on B. Signal that is **prompt-based** lands on A.

The aggregate calls it at 1.5:1 — a majority is not enough for a direction, since
one confident pair can carry a small count. Below that the reading is
``ambiguous``, which is a verdict and not a missing one.

The swap prompts themselves are :mod:`anamnesis.modes.prompt_swap`; this is the
analysis that reads what they produced. Swap generations are excluded from the
core corpus by the signature loader, so they are loaded here directly from the
bank — the one place in the analysis layer that reads signature npz files without
going through :func:`~anamnesis.analysis.gauntlet.signature_io.load_run4`, because
what it needs is exactly the samples that loader is built to leave out.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from anamnesis.analysis.gauntlet.signature_io import (
    CORE_BLOCKS,
    FAMILY_BLOCKS,
    BLOCK_UNIONS,
    BLOCK_NPZ_KEYS,
    Run4Data,
    load_run4,
)

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]

SWAP_MODE_RE = re.compile(r"swap_(\w+)→(\w+)")
"""How a swap generation's mode label spells the pair: system prompt, then
execution. The arrow is in banked metadata, so it is matched rather than chosen."""

N_ESTIMATORS = 200
SEED = 42

DIRECTION_RATIO = 1.5
"""How much one side has to outweigh the other before the aggregate names a
direction. A bare majority over a few dozen samples is not a direction."""

EXECUTION_BASED = "execution_based"
PROMPT_BASED = "system_prompt_based"
AMBIGUOUS = "ambiguous"


def parse_swap_mode(mode: str) -> tuple[str, str] | None:
    """``swap_A→B`` as ``(system_prompt_mode, execution_mode)``, or None."""
    match = SWAP_MODE_RE.match(mode)
    return (match.group(1), match.group(2)) if match else None


class SwapSample(BaseModel):
    """One swap generation's identity: which two modes it holds apart."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    file_stem: str
    system_prompt_mode: str
    execution_mode: str
    swap_name: str
    topic: str = ""


def load_swap_samples(
    signature_dir: Path,
    addon_dirs: Sequence[Path] | None = None,
) -> tuple[list[SwapSample], dict[str, list[F32]]]:
    """Swap generations and their features, per block, from the primary bank and addons.

    A block is kept only where **every** swap sample has it: a block present for some
    samples and absent for others would train and predict over different feature
    sets under one name.

    A union is then built only where every member it names survived that filter, which
    is the rule the core loader joins under. A union built from the members that happen
    to be present is a narrower feature set wearing a label that names blocks it does not
    hold — and it is compared, under that label, against a training matrix built from all
    of them.
    """
    signature_dir = Path(signature_dir)
    npz_paths = sorted(signature_dir.glob("gen_*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"no gen_*.npz under {signature_dir}")

    samples: list[SwapSample] = []
    swap_paths: list[Path] = []
    for npz_path in npz_paths:
        json_path = npz_path.with_suffix(".json")
        if not json_path.exists():
            continue
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        parsed = parse_swap_mode(str(meta.get("mode", "")))
        if parsed is None:
            continue
        samples.append(
            SwapSample(
                file_stem=npz_path.stem,
                system_prompt_mode=parsed[0],
                execution_mode=parsed[1],
                swap_name=str(meta["mode"]),
                topic=str(meta.get("topic", "")),
            )
        )
        swap_paths.append(npz_path)

    if not samples:
        raise ValueError(f"no prompt-swap generations under {signature_dir}")

    per_block: dict[str, list[F32 | None]] = {}

    def absorb(index: int, npz_path: Path, *, only_missing: bool) -> None:
        data = np.load(npz_path, allow_pickle=True)
        for block, key in BLOCK_NPZ_KEYS.items():
            if key not in data.files:
                continue
            column = per_block.setdefault(block, [None] * len(swap_paths))
            if only_missing and column[index] is not None:
                continue
            column[index] = np.asarray(data[key], dtype=np.float32)

    for index, npz_path in enumerate(swap_paths):
        absorb(index, npz_path, only_missing=False)
    for addon_dir in [Path(d) for d in (addon_dirs or [])]:
        if not addon_dir.exists():
            logger.warning(f"addon dir not found: {addon_dir}")
            continue
        for index, sample in enumerate(samples):
            addon_npz = addon_dir / f"{sample.file_stem}.npz"
            if addon_npz.exists():
                absorb(index, addon_npz, only_missing=True)

    complete: dict[str, list[F32]] = {
        block: [array for array in column if array is not None]
        for block, column in per_block.items()
        if all(array is not None for array in column)
    }
    for group, members in BLOCK_UNIONS.items():
        missing = [m for m in members if m not in complete]
        if missing:
            logger.info(
                f"  union '{group}' not built over the swap samples: {sorted(missing)} "
                "absent, and a union built short would name blocks it does not hold"
            )
            continue
        complete[group] = [
            np.concatenate([complete[m][i] for m in members])
            for i in range(len(swap_paths))
        ]
    logger.info(f"{len(samples)} swap generations, {len(complete)} blocks with full coverage")
    return samples, complete


class BlockSwapResult(BaseModel):
    """One block's verdict on one swap pair: where the predictions landed."""

    model_config = ConfigDict(extra="forbid")

    n_execution: int = Field(ge=0)
    n_system: int = Field(ge=0)
    n_total: int = Field(gt=0)
    pct_execution: float
    mean_p_execution: float
    predictions: list[str]


class SwapPairResult(BaseModel):
    """One swap pair, block by block, with what the classifier was trained on."""

    model_config = ConfigDict(extra="forbid")

    system_prompt_mode: str
    execution_mode: str
    n_swap: int = Field(gt=0)
    n_training_per_mode: dict[str, int]
    per_block: dict[str, BlockSwapResult] = Field(default_factory=dict)


class SwapAggregate(BaseModel):
    """One block's reading pooled over every swap pair, and the direction it names."""

    model_config = ConfigDict(extra="forbid")

    n_execution: int
    n_system: int
    n_total: int
    pct_execution: float
    mean_p_execution: float
    signal_type: str


class PromptSwapResult(BaseModel):
    """The whole test for one run: per pair, and pooled."""

    model_config = ConfigDict(extra="forbid")

    run_name: str
    n_swap_samples: int
    swap_types: list[str]
    per_swap_type: dict[str, SwapPairResult] = Field(default_factory=dict)
    aggregate: dict[str, SwapAggregate] = Field(default_factory=dict)


def _test_blocks(core: Run4Data, swap_features: dict[str, list[F32]]) -> list[str]:
    """Blocks present on both sides, individual blocks first and then composites."""
    blocks = [
        block for block in list(CORE_BLOCKS) + list(FAMILY_BLOCKS)
        if block in core.block_features and block in swap_features
    ]
    blocks += [g for g in BLOCK_UNIONS if g in core.group_features and g in swap_features]
    return blocks


def classify_swap_pair(
    core: Run4Data,
    swap_features: dict[str, list[F32]],
    *,
    indices: Sequence[int],
    system_prompt_mode: str,
    execution_mode: str,
    blocks: Sequence[str],
) -> SwapPairResult:
    """Train on the pair's two pure modes, predict its swap samples, per block.

    The scaler is fitted on the two-mode training slice and applied to the swap
    samples, because a scaler fitted over both would let the swap samples inform
    the standardization they are then judged under.
    """
    mask = (core.modes == system_prompt_mode) | (core.modes == execution_mode)
    y = core.modes[mask]
    result = SwapPairResult(
        system_prompt_mode=system_prompt_mode,
        execution_mode=execution_mode,
        n_swap=len(indices),
        n_training_per_mode={
            system_prompt_mode: int(np.sum(y == system_prompt_mode)),
            execution_mode: int(np.sum(y == execution_mode)),
        },
    )
    for block in blocks:
        try:
            all_rows = core.get_block(block)
        except KeyError:
            continue
        X_train = all_rows[mask]
        X_swap = np.stack([swap_features[block][i] for i in indices], axis=0)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_swap_scaled = scaler.transform(X_swap)

        classifier = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, random_state=SEED, n_jobs=1
        )
        classifier.fit(X_train_scaled, y)
        predictions = classifier.predict(X_swap_scaled)
        probabilities = classifier.predict_proba(X_swap_scaled)
        execution_column = list(classifier.classes_).index(execution_mode)

        n_execution = int(np.sum(predictions == execution_mode))
        n_system = int(np.sum(predictions == system_prompt_mode))
        result.per_block[block] = BlockSwapResult(
            n_execution=n_execution,
            n_system=n_system,
            n_total=len(predictions),
            pct_execution=float(n_execution / len(predictions)),
            mean_p_execution=float(np.mean(probabilities[:, execution_column])),
            predictions=[str(p) for p in predictions],
        )
        verdict = "EXEC" if n_execution > n_system else ("SYS" if n_system > n_execution else "TIE")
        logger.info(
            f"    {block:<25} {n_execution}/{len(predictions)} exec  "
            f"P(exec)={result.per_block[block].mean_p_execution:.3f}  [{verdict}]"
        )
    return result


def signal_type(n_execution: int, n_system: int) -> str:
    """Which axis a pooled count names, at the 1.5:1 bar."""
    if n_execution > n_system * DIRECTION_RATIO:
        return EXECUTION_BASED
    if n_system > n_execution * DIRECTION_RATIO:
        return PROMPT_BASED
    return AMBIGUOUS


def aggregate_swaps(
    per_pair: dict[str, SwapPairResult], blocks: Sequence[str]
) -> dict[str, SwapAggregate]:
    """Pool each block over every swap pair.

    Counts are summed and the probability is averaged over pairs rather than over
    samples: a pair is the unit of the experiment, and one pair with more swap
    samples than another should not weigh more on the mean probability.
    """
    out: dict[str, SwapAggregate] = {}
    for block in blocks:
        n_execution = n_system = n_total = 0
        probabilities: list[float] = []
        for pair in per_pair.values():
            block_result = pair.per_block.get(block)
            if block_result is None:
                continue
            n_execution += block_result.n_execution
            n_system += block_result.n_system
            n_total += block_result.n_total
            probabilities.append(block_result.mean_p_execution)
        if n_total == 0:
            continue
        out[block] = SwapAggregate(
            n_execution=n_execution,
            n_system=n_system,
            n_total=n_total,
            pct_execution=float(n_execution / n_total),
            mean_p_execution=float(np.mean(probabilities)),
            signal_type=signal_type(n_execution, n_system),
        )
        logger.info(
            f"    {block:<25} {n_execution}/{n_total} exec  "
            f"P(exec)={out[block].mean_p_execution:.3f}  [{out[block].signal_type.upper()}]"
        )
    return out


def run_binary_prompt_swap(
    run_name: str,
    signature_dir: Path,
    addon_dirs: Sequence[Path] | None = None,
) -> PromptSwapResult:
    """The whole test for one run: every swap pair, every block, then the pool."""
    core = load_run4(
        signature_dir=signature_dir, core_only=True, addon_dirs=list(addon_dirs or []) or None
    )
    samples, swap_features = load_swap_samples(signature_dir, addon_dirs)

    by_swap: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        by_swap.setdefault(sample.swap_name, []).append(index)

    blocks = _test_blocks(core, swap_features)
    logger.info(f"swap types: {sorted(by_swap)}; blocks under test: {blocks}")

    result = PromptSwapResult(
        run_name=run_name,
        n_swap_samples=len(samples),
        swap_types=sorted(by_swap),
    )
    for swap_name, indices in sorted(by_swap.items()):
        parsed = parse_swap_mode(swap_name)
        if parsed is None:
            continue
        system_prompt_mode, execution_mode = parsed
        logger.info(
            f"  {swap_name} (sys={system_prompt_mode}, exec={execution_mode}, n={len(indices)})"
        )
        result.per_swap_type[swap_name] = classify_swap_pair(
            core,
            swap_features,
            indices=indices,
            system_prompt_mode=system_prompt_mode,
            execution_mode=execution_mode,
            blocks=blocks,
        )
    logger.info(f"  aggregate over {len(samples)} swap generations")
    result.aggregate = aggregate_swaps(result.per_swap_type, blocks)
    return result


def swap_report(results: dict[str, PromptSwapResult]) -> dict[str, Any]:
    """Several runs' results as the banked document."""
    return {run: result.model_dump() for run, result in results.items()}


__all__ = [
    "AMBIGUOUS",
    "DIRECTION_RATIO",
    "EXECUTION_BASED",
    "PROMPT_BASED",
    "PromptSwapResult",
    "SwapAggregate",
    "SwapPairResult",
    "SwapSample",
    "BlockSwapResult",
    "aggregate_swaps",
    "classify_swap_pair",
    "load_swap_samples",
    "parse_swap_mode",
    "run_binary_prompt_swap",
    "signal_type",
    "swap_report",
]
