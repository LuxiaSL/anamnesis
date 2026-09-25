"""A synthetic signature bank, so a checkout has a corpus to read before it has a GPU.

Everything on the reading side of this instrument — the gauntlet's eleven sections, the
leak gate, the subfamily decomposition, the transfer comparisons — takes a directory of
banked signatures and returns claims about them. Producing a real one costs a model, a
device and hours. That is the right cost for a result and the wrong cost for finding out
whether the thing runs, so this writes a bank of the same shape from a seed.

**The numbers here are about nothing.** They are drawn from a generator, not measured
from a forward pass, and no reading of them is a finding about any model. What they are
is *structured*: each mode gets a fixed offset in feature space, each topic gets a
smaller offset shared across every mode, and a generation is a mode offset plus a topic
offset plus noise. So a classifier separates the modes, which is what makes the output
legible as a worked example rather than a page of chance. Two consequences follow from
that construction and both matter:

* the topic axis is present in every mode equally, so it is a nuisance direction rather
  than a label the classifier can cheat with — a synthetic bank that leaked topic would
  teach the opposite of what the leak gate is for;
* generation length is drawn independently of mode, so the length controls have nothing
  to find. A real corpus is not so obliging, which is why those controls exist.

Three more things the construction is careful about, all so that reading the demo teaches
the instrument rather than an artefact of the fixture:

* **no block is staged as more informative than another.** Each block's slice of the mode
  offsets is rescaled so the separation between mode centroids comes out the same in every
  block whatever its width. That is deliberately not the same as equal strength per
  column: separation accumulates over a block's columns, so equalizing per column would
  hand the widest block the strongest readout, and a per-block ranking would then be a
  statement about the widths — which are an arbitrary property of this fixture. What
  survives is the draw, which moves the strongest block from seed to seed. The remaining
  spread and the absence of a width trend are pinned in `tests/test_synthetic_bank.py`,
  because a change to the widths or the noise could quietly reintroduce the bias.
  **Read any block ordering the demo prints as a property of the fixture**: the generator
  has no substrates for an ordering to be about, and the claim this project had to revise
  was exactly a claim that one bin was load-bearing.
* every block is written, so no union is short and no section has to state an absence
  it would only be stating about the fixture.
* **the text says nothing about the mode.** Every mode on a topic writes the same text,
  so the semantic section's text baseline has nothing to read and the signatures are set
  against a text channel at chance. A fixture whose text named the mode would score that
  baseline at ceiling, and the demo would then teach that the text gives the mode away.

The bank carries a lane identifier naming itself as synthetic, because the read side
gates on lane identity and an unstamped bank is indistinguishable from one whose
provenance was lost.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from anamnesis.analysis.gauntlet.signature_io import (
    ATTENTION_AND_DELTAS,
    ATTENTION_FLOW,
    BLOCK_NPZ_KEYS,
    CACHE_AND_KEYS,
    GATE_FEATURES,
    NORMS_AND_OUTPUT_STATS,
    RESIDUAL_PCA,
    RESIDUAL_TRAJECTORY,
)
from anamnesis.extraction.state_extractor import STORED_BLOCK_SLICES_KEY
from anamnesis.modes import CORE_MODE_SET, mode_set

SYNTHETIC_LANE = "synthetic-bank-v1"
"""The lane identity a synthetic bank carries, so it is never mistaken for a measurement.

It carries :data:`anamnesis.analysis.lane_guard.DRAWN_LANE_MARKER`, which is how a
reader states that what it reports is about a generator: a bank of drawn vectors is
the same shape as a measured one, and the lane is the only place the difference is
written down.
"""

DEFAULT_BLOCK_WIDTHS: dict[str, int] = {
    NORMS_AND_OUTPUT_STATS: 16,
    ATTENTION_AND_DELTAS: 24,
    CACHE_AND_KEYS: 24,
    RESIDUAL_PCA: 12,
    RESIDUAL_TRAJECTORY: 18,
    ATTENTION_FLOW: 24,
    GATE_FEATURES: 20,
}
"""Every block the loader knows, narrow.

All of them, rather than a readable subset, because a union is built only when every
one of its members is present — so a bank missing one block is a bank on which several
sections state an absence instead of reading. A fixture whose purpose is to exercise
the reading side should not be the reason a section cannot run. The widths are small
because the shape is the point, not the size.
"""

FEATURES_KEY_PREFIX = "features_"


class SyntheticBankSpec(BaseModel):
    """What to draw. Every field has a default, so the zero-argument call is the demo."""

    model_config = ConfigDict(frozen=True)

    modes: tuple[str, ...] = Field(
        default_factory=lambda: mode_set(CORE_MODE_SET).names(),
        description="Mode labels, in the instrument's own vocabulary rather than invented ones",
        min_length=2,
    )
    # The cross-condition generalization section cuts topic folds at 4, 5, 10 and 20,
    # and it cuts them by integer division: a topic count that is not a multiple of a
    # fold count leaves the remainder in no test fold at all. Twenty is the smallest
    # count divisible by all four, which is why it is the default and why the banked
    # corpora use it. Choosing another number is allowed and quietly costs coverage.
    topics: int = Field(default=20, ge=2, description="Distinct topics, shared across modes")
    repetitions: int = Field(default=2, ge=1, description="Generations per mode-topic pair")
    block_widths: dict[str, int] = Field(default_factory=lambda: dict(DEFAULT_BLOCK_WIDTHS))
    mode_effect: float = Field(
        default=0.45, gt=0.0, description="Offset scale that separates modes"
    )
    topic_effect: float = Field(
        default=0.30, ge=0.0, description="Offset scale of the shared nuisance axis"
    )
    noise: float = Field(default=1.0, gt=0.0, description="Per-generation noise scale")
    seed: int = Field(default=0, description="Two people with the same seed get the same bank")
    lane_id: str = Field(default=SYNTHETIC_LANE)

    @property
    def width(self) -> int:
        """Total feature width, the sum of the block widths."""
        return sum(self.block_widths.values())

    @property
    def generations(self) -> int:
        """How many generation pairs the bank will hold."""
        return len(self.modes) * self.topics * self.repetitions


class SyntheticBank(BaseModel):
    """Where a bank was written and what is in it."""

    model_config = ConfigDict(frozen=True)

    directory: Path
    generations: int
    width: int
    modes: tuple[str, ...]
    topics: int
    lane_id: str


def _block_layout(block_widths: dict[str, int]) -> tuple[dict[str, tuple[int, int]], list[str]]:
    """Block label -> half-open column span, and the feature names across all blocks.

    Raises
    ------
    KeyError
        When a label is not one the loader knows, which would write a block no reader
        discovers.
    """
    spans: dict[str, tuple[int, int]] = {}
    names: list[str] = []
    cursor = 0
    for block, width in block_widths.items():
        if block not in BLOCK_NPZ_KEYS:
            known = ", ".join(sorted(BLOCK_NPZ_KEYS))
            raise KeyError(f"unknown block label {block!r}; the loader knows: {known}")
        spans[block] = (cursor, cursor + width)
        names.extend(f"{block}_f{i}" for i in range(width))
        cursor += width
    return spans, names


def _mode_offsets(
    rng: np.random.Generator,
    n_modes: int,
    spans: dict[str, tuple[int, int]],
    scale: float,
) -> np.ndarray:
    """Per-mode offsets that separate the modes equally well in every block.

    What a classifier reads is a block's *total* separation, not its strength per
    column: the distance between two mode centroids grows as the square root of the
    number of columns it is measured over. So equalizing per column would hand the
    widest block the strongest readout — and a per-block ranking would then be a
    statement about the block widths, which are an arbitrary property of this
    fixture.

    Each block's slice is therefore rescaled so that its centroid separation is the
    same in every block, whatever its width. The narrow blocks carry more signal per
    column and the wide ones less, which is the trade that buys a flat readout.

    The target is the separation a block of the *mean* width would have had under
    per-column normalization, rather than a bare constant, so that equalizing does
    not also change how strongly the bank separates overall. ``scale`` therefore
    keeps meaning roughly what it meant, and the remaining spread across blocks is
    the draw alone.
    """
    width = max(stop for _, stop in spans.values())
    mean_columns = sum(stop - start for start, stop in spans.values()) / len(spans)
    offsets = rng.normal(size=(n_modes, width))
    for start, stop in spans.values():
        block = offsets[:, start:stop]
        rms = np.sqrt(np.mean(block**2))
        offsets[:, start:stop] = block / rms * scale * np.sqrt(mean_columns / (stop - start))
    return offsets


def write_synthetic_bank(
    directory: Path, spec: SyntheticBankSpec | None = None
) -> SyntheticBank:
    """Write a synthetic bank into ``directory`` and return what was written.

    The directory is created if absent and written into if present; a name already
    taken by a real corpus is the caller's to avoid, since this function cannot tell
    a banked signature from one it wrote itself — which is the reason every file it
    writes carries the synthetic lane identifier.
    """
    spec = spec or SyntheticBankSpec()
    directory.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(spec.seed)
    spans, feature_names = _block_layout(spec.block_widths)
    width = spec.width

    mode_offsets = _mode_offsets(rng, len(spec.modes), spans, spec.mode_effect)
    topic_offsets = rng.normal(scale=spec.topic_effect, size=(spec.topics, width))
    names_array = np.array(feature_names)
    # The slice table is keyed the way the loader keys it: the npz array name with its
    # ``features_`` prefix removed. Reading the key off the loader's own map is what
    # keeps a bank written here discoverable by the reader that will open it.
    slices = {
        BLOCK_NPZ_KEYS[block].removeprefix(FEATURES_KEY_PREFIX): [start, stop]
        for block, (start, stop) in spans.items()
    }

    index = 0
    for mode_idx, mode in enumerate(spec.modes):
        for topic_idx in range(spec.topics):
            for _ in range(spec.repetitions):
                vector = (
                    mode_offsets[mode_idx]
                    + topic_offsets[topic_idx]
                    + rng.normal(scale=spec.noise, size=width)
                ).astype(np.float32)

                arrays = {
                    BLOCK_NPZ_KEYS[block]: vector[start:stop]
                    for block, (start, stop) in spans.items()
                }
                np.savez(
                    directory / f"gen_{index:03d}.npz",
                    feature_names=names_array,
                    **arrays,
                )

                topic = f"topic_{topic_idx:02d}"
                metadata = {
                    "generation_id": index,
                    "topic": topic,
                    "topic_idx": topic_idx,
                    "mode": mode,
                    "mode_idx": mode_idx,
                    # Drawn independently of mode, so the length controls find nothing.
                    "num_generated_tokens": int(rng.integers(180, 420)),
                    # Names the topic and not the mode, and is the same text for every
                    # mode on a topic, so the text baseline the semantic section measures
                    # signatures against finds nothing to read: a fixture whose text
                    # carried the label would score the text channel at ceiling.
                    "generated_text": f"synthetic generation on {topic}",
                    "system_prompt": f"synthetic system prompt for {mode}",
                    "user_prompt": f"synthetic user prompt for {topic}",
                    STORED_BLOCK_SLICES_KEY: slices,
                    "lane_id": spec.lane_id,
                }
                (directory / f"gen_{index:03d}.json").write_text(json.dumps(metadata, indent=2))
                index += 1

    return SyntheticBank(
        directory=directory,
        generations=index,
        width=width,
        modes=spec.modes,
        topics=spec.topics,
        lane_id=spec.lane_id,
    )
