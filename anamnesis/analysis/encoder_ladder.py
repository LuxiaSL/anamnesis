"""The extraction ladder: hand-features against the raw state they summarize.

A signature is a projection of a forward pass, and a hand-built feature set is one
choice of projection. So there is a question the hand features cannot answer about
themselves: when they show nothing, is there nothing there, or did the projection
drop it?

The ladder answers it by putting three readouts on the identical split:

1. **hand features** — the banked signature vector, as computed,
2. **raw linear** — a linear readout on the raw state, which is the floor,
3. **raw encoder** — one small nonlinear readout on the same raw state.

Read together they separate two findings that look the same from the hand side. If
the raw readouts find a difference the hand features miss, the whole vector carries
structure the projection does not, and the hierarchy holds: a stake is a lossy read
of the pass, not the pass. If the raw readouts miss it too, the hand features were
not the limitation, and that is the result worth stopping on.

**Matched by construction.** The two arms are the same continuations replayed with
and without the intervention, so the token sequences are identical and only the
computation differs. That is what makes a classifier's success readable as reading
the intervention's imprint rather than the content — and it is why only generations
present in *both* arms enter the comparison.

Everything measured here runs through :mod:`anamnesis.analysis.audit_lib`: its
surface sampling, its per-fold residualize-standardize-reduce, its readout pair.
Reusing that machinery rather than restating it is what makes these numbers
comparable to the other ladders measured against it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from sklearn.model_selection import GroupKFold

from anamnesis.analysis.audit_lib import (
    DEEP,
    DEEP_EPOCHS_REDUCED,
    LBFGS_L2,
    LOGIT,
    DEEP_BOTTLENECK,
    preprocess_fold_gpu,
    sample_positions,
    surface_vector,
    train_eval,
)

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]

DEFAULT_SURFACES: tuple[str, ...] = ("residual", "attention")
"""The two surfaces the source ranking puts first. Attention is the slower one to
resample, which is the only reason a caller would narrow to residual alone."""

N_SEEDS = 3
N_SPLITS = 5
CHANCE_BINARY = 0.5

CATCH_MARGIN = 0.08
"""How far the raw encoder has to clear the best hand readout before the difference
is read as the projection having missed something."""

HAND_FLOOR_BAR = 0.62
"""Above this, the hand features are not at the floor, so the contrast is not the
clean case the ladder was built to adjudicate and the numbers are read directly."""

CATCH = (
    "ENCODER CATCHES WHAT THE PROJECTION MISSED: raw encoder clears the hand "
    "readouts, which are at the floor — the whole vector carries the stake"
)
SEE_NUMBERS = "SEE THE NUMBERS: not the clean pattern either way — read them, do not summarize"
MISS = (
    "ENCODER MISSES IT TOO: the raw state is at the floor as well — the hand "
    "features were not the limitation"
)


def raw_vector(
    npz_path: Path, surfaces: Sequence[str] = DEFAULT_SURFACES
) -> F32 | None:
    """One generation's raw state as a fixed-width vector, surfaces concatenated.

    Returns None where the capture cannot supply the surfaces asked for — a bank
    written without one of them, or a generation with no positions to sample. A
    missing row is dropped by the caller rather than filled, because a zero row is a
    claim about the state and an absent one is not.
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
    except (OSError, ValueError) as exc:
        logger.warning(f"{npz_path.name}: unreadable ({exc})")
        return None
    length = int(data["actual_lengths"].shape[0]) if "actual_lengths" in data.files else 0
    if length <= 0:
        return None
    positions = sample_positions(length)
    parts: list[F32] = []
    for surface in surfaces:
        vector = surface_vector(data, surface, positions, length)
        if vector is None:
            return None
        parts.append(np.asarray(vector, dtype=np.float32))
    return np.concatenate(parts)


def hand_vector(run_dir: Path, gen_id: int, *, signatures_subdir: str = "signatures_v3") -> F32 | None:
    """The banked signature vector for one generation, or None where it is absent.

    The vector lives in the npz beside the metadata, not in the JSON: the JSON
    carries the metadata and the slice table, and reading the features from it would
    read nothing.
    """
    path = Path(run_dir) / signatures_subdir / f"gen_{gen_id:03d}.npz"
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=True) as data:
            if "features" not in data.files:
                return None
            return np.asarray(data["features"], dtype=np.float32)
    except (OSError, ValueError) as exc:
        logger.warning(f"{path.name}: unreadable ({exc})")
        return None


class Arm(BaseModel):
    """One side of the contrast: its raw captures, its signatures, its label."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(min_length=1)
    raw_dir: Path = Field(description="Directory of gen_*.npz raw captures")
    run_dir: Path = Field(description="Run directory holding the banked signatures")
    y: int = Field(ge=0, description="Class label this arm's rows carry")


class ArmRows:
    """What one arm contributed, keyed by generation id so arms can be matched."""

    def __init__(self) -> None:
        self.raw: dict[int, F32] = {}
        self.hand: dict[int, F32] = {}
        self.topic: dict[int, str] = {}
        self.covariates: dict[int, tuple[float, float]] = {}

    @property
    def gen_ids(self) -> list[int]:
        return sorted(self.raw)


def load_arm(
    arm: Arm,
    source_metadata: dict[int, dict[str, Any]],
    *,
    surfaces: Sequence[str] = DEFAULT_SURFACES,
    signatures_subdir: str = "signatures_v3",
) -> ArmRows:
    """Read one arm's raw and hand vectors, with the topic and length covariates.

    A generation enters only when the raw capture, the banked signature *and* the
    source record are all present: the three are what the comparison needs, and a
    row missing one of them would be compared on a different footing from its
    neighbours.
    """
    rows = ArmRows()
    files = sorted(
        Path(arm.raw_dir).glob("gen_*.npz"), key=lambda p: int(p.stem.split("_")[1])
    )
    for path in files:
        gen_id = int(path.stem.split("_")[1])
        record = source_metadata.get(gen_id)
        if record is None:
            continue
        raw = raw_vector(path, surfaces)
        hand = hand_vector(arm.run_dir, gen_id, signatures_subdir=signatures_subdir)
        if raw is None or hand is None:
            continue
        rows.raw[gen_id] = raw
        rows.hand[gen_id] = hand
        rows.topic[gen_id] = str(record.get("topic_idx", record.get("topic", gen_id)))
        rows.covariates[gen_id] = (
            float(record.get("prompt_length", 0) or 0),
            float(record.get("num_generated_tokens", record.get("gen_length", 0)) or 0),
        )
    logger.info(f"{arm.label}: {len(rows.raw)} generations with raw, signature and record")
    return rows


class LadderRung(BaseModel):
    """One readout's accuracy over every fold and seed."""

    model_config = ConfigDict(extra="forbid")

    test: float
    std: float


class RungPair(BaseModel):
    """A feature set read by both architectures on the identical split."""

    model_config = ConfigDict(extra="forbid")

    logit: LadderRung
    deep: LadderRung

    @property
    def best(self) -> float:
        return max(self.logit.test, self.deep.test)


def cross_validated_pair(
    X: NDArray[Any],
    y: NDArray[np.int_],
    topic: NDArray[np.int_],
    covariates: NDArray[Any],
    *,
    device: str,
    name: str,
    n_seeds: int = N_SEEDS,
    n_splits: int = N_SPLITS,
    deep_epochs: int = DEEP_EPOCHS_REDUCED,
    k: int = DEEP_BOTTLENECK,
) -> RungPair:
    """Both architectures over GroupKFold-by-topic, several seeds, one preprocessing.

    The fold's residualize-standardize-reduce runs **once** and both architectures
    read its output, which is what makes their difference a property of the readout
    rather than of two preprocessings.
    """
    accuracies: dict[str, list[float]] = {LOGIT: [], DEEP: []}
    for seed in range(n_seeds):
        for train, test in GroupKFold(n_splits=n_splits).split(X, y, topic):
            Z_train, Z_test = preprocess_fold_gpu(
                X[train], X[test], covariates[train], covariates[test], True, device
            )
            for arch in (LOGIT, DEEP):
                test_acc, _train_acc = train_eval(
                    Z_train, y[train], Z_test, y[test], arch, seed, device,
                    deep_epochs=deep_epochs, lbfgs_l2=LBFGS_L2, nclass=2, k=k,
                )
                accuracies[arch].append(test_acc)
    pair = RungPair(
        logit=LadderRung(
            test=round(float(np.mean(accuracies[LOGIT])), 4),
            std=round(float(np.std(accuracies[LOGIT])), 4),
        ),
        deep=LadderRung(
            test=round(float(np.mean(accuracies[DEEP])), 4),
            std=round(float(np.std(accuracies[DEEP])), 4),
        ),
    )
    logger.info(
        f"  {name}: logit {pair.logit.test:.1%}+/-{pair.logit.std:.1%}  "
        f"deep {pair.deep.test:.1%}+/-{pair.deep.std:.1%}  (chance {CHANCE_BINARY:.0%})"
    )
    return pair


def read_ladder(hand: RungPair, raw: RungPair) -> str:
    """Which of the three readings the two pairs support.

    The order of the checks is the asymmetry: the catch is only claimable while the
    hand side is genuinely at the floor, so a raw readout above the floor bar is
    reported as numbers rather than as a verdict either way.
    """
    if raw.deep.test - hand.best > CATCH_MARGIN and hand.best < HAND_FLOOR_BAR:
        return CATCH
    if raw.deep.test > HAND_FLOOR_BAR:
        return SEE_NUMBERS
    return MISS


class LadderResult(BaseModel):
    """The ladder as banked: both pairs, the shapes behind them, the reading."""

    model_config = ConfigDict(extra="forbid")

    n_matched_pairs: int = Field(gt=0)
    n_topics: int = Field(gt=0)
    n_raw_features: int = Field(gt=0)
    n_hand_features: int = Field(gt=0)
    surfaces_raw: list[str]
    chance: float = CHANCE_BINARY
    hand_features: RungPair
    raw_linear_and_encoder: RungPair
    reading: str
    notes: list[str] = Field(default_factory=list)


def run_ladder(
    positive: Arm,
    negative: Arm,
    source_metadata: dict[int, dict[str, Any]],
    *,
    device: str = "cpu",
    surfaces: Sequence[str] = DEFAULT_SURFACES,
    signatures_subdir: str = "signatures_v3",
    n_seeds: int = N_SEEDS,
    n_splits: int = N_SPLITS,
    deep_epochs: int = DEEP_EPOCHS_REDUCED,
) -> LadderResult:
    """The whole ladder over two matched arms.

    Only generations present in both arms are used, and they enter in one order on
    both sides, so a row of the positive arm and the corresponding row of the
    negative arm are the same continuation. Topics are re-indexed over the matched
    set, because ``GroupKFold`` groups by position and the arms' own topic labels are
    strings.
    """
    positive_rows = load_arm(
        positive, source_metadata, surfaces=surfaces, signatures_subdir=signatures_subdir
    )
    negative_rows = load_arm(
        negative, source_metadata, surfaces=surfaces, signatures_subdir=signatures_subdir
    )
    matched = sorted(set(positive_rows.gen_ids) & set(negative_rows.gen_ids))
    if not matched:
        raise ValueError(
            f"{positive.label} and {negative.label} share no generation with raw, "
            f"signature and record on both sides — there is nothing matched to compare"
        )
    logger.info(f"matched generations in both arms: {len(matched)}")

    X_raw = np.stack(
        [positive_rows.raw[g] for g in matched] + [negative_rows.raw[g] for g in matched]
    ).astype(np.float32)
    X_hand = np.stack(
        [positive_rows.hand[g] for g in matched] + [negative_rows.hand[g] for g in matched]
    ).astype(np.float32)
    topic_labels = [positive_rows.topic[g] for g in matched] + [
        negative_rows.topic[g] for g in matched
    ]
    index_of = {label: i for i, label in enumerate(sorted(set(topic_labels)))}
    topic = np.array([index_of[label] for label in topic_labels], dtype=np.int64)
    covariates = np.asarray(
        [positive_rows.covariates[g] for g in matched]
        + [negative_rows.covariates[g] for g in matched],
        dtype=np.float64,
    )
    y = np.array([positive.y] * len(matched) + [negative.y] * len(matched), dtype=np.int64)
    logger.info(
        f"X_raw {X_raw.shape}  X_hand {X_hand.shape}  topics={len(index_of)}"
    )

    hand_pair = cross_validated_pair(
        X_hand, y, topic, covariates, device=device, name="hand features (banked signature)",
        n_seeds=n_seeds, n_splits=n_splits, deep_epochs=deep_epochs,
    )
    raw_pair = cross_validated_pair(
        X_raw, y, topic, covariates, device=device,
        name=f"raw ({'+'.join(surfaces)})",
        n_seeds=n_seeds, n_splits=n_splits, deep_epochs=deep_epochs,
    )
    reading = read_ladder(hand_pair, raw_pair)
    logger.info(f"reading: {reading}")
    return LadderResult(
        n_matched_pairs=len(matched),
        n_topics=len(index_of),
        n_raw_features=int(X_raw.shape[1]),
        n_hand_features=int(X_hand.shape[1]),
        surfaces_raw=list(surfaces),
        hand_features=hand_pair,
        raw_linear_and_encoder=raw_pair,
        reading=reading,
        notes=[
            f"matched-token: the same continuations on both sides, "
            f"{positive.label} against {negative.label}",
            "raw surfaces are Gram-reduced per fold by audit_lib.preprocess_fold_gpu",
            "one split for all three rungs: hand, raw-linear, raw-encoder",
        ],
    )


__all__ = [
    "Arm",
    "ArmRows",
    "CATCH",
    "CATCH_MARGIN",
    "DEFAULT_SURFACES",
    "HAND_FLOOR_BAR",
    "LadderResult",
    "LadderRung",
    "MISS",
    "RungPair",
    "SEE_NUMBERS",
    "cross_validated_pair",
    "hand_vector",
    "load_arm",
    "raw_vector",
    "read_ladder",
    "run_ladder",
]
