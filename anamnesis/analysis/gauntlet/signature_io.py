"""Reading banked signatures off disk — the gauntlet's one entry point for data.

A signature directory holds one ``gen_NNN.npz`` of feature vectors beside one
``gen_NNN.json`` of stimulus metadata per generation. This module turns a
directory into the two objects the gauntlet's sections consume:

``Run4Data``
    Feature matrices partitioned by block, plus the unions over them, plus the
    mode and topic labels aligned to the matrix rows. Both the four core blocks
    and the v2 families are supported, and which ones are present is discovered
    from the npz contents rather than declared, so a directory extracted with a
    narrower suite loads as itself.
``AnalysisData``
    The same, plus the generated text and prompt fields that the semantic
    section reads. Every section takes this type; the narrower one exists
    because loading text is work a run that skips section 9 need not do.

Both live here, in one module, because they are one operation read at two
widths: a directory in, a matrix out, with the join rules enforced once. Two
modules whose names both say "load" would leave a reader guessing which to call.

Those join rules are the reason this is not a bare ``np.load`` loop. Addon
directories merge extra feature families into an existing row order, matched by
file stem, and a concatenation is as much a scientific join as a comparison is —
so the lane guard runs over the primary metadata and again over every addon,
and an incomplete addon is dropped whole rather than stacked with holes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from anamnesis.analysis.lane_guard import MixedLaneError, require_single_lane
from anamnesis.config.paths import legacy_data_root
from anamnesis.extraction.state_extractor import (
    STORED_ATTENTION_AND_DELTAS,
    STORED_BLOCK_SLICES_KEY,
    STORED_CACHE_AND_KEYS,
    STORED_NORMS_AND_OUTPUT_STATS,
    STORED_RESIDUAL_PCA,
)

logger = logging.getLogger(__name__)


# ── Block labels ──────────────────────────────────────────────────────────────
# A feature vector is addressed in contiguous blocks, each reported under a label.
# A label is what a reader of a results file sees, so it says what its block reads;
# it is not the name the block is stored under. BLOCK_STORED_NAMES below is where
# the two meet, and `anamnesis/analysis/gauntlet/schemas/compat.py` maps the labels
# of an older results file onto the current ones.
#
# The four core blocks read, in order: residual activation norms with output
# statistics; attention distributions with cross-layer residual deltas; cache-read
# profiles with pre-RoPE key geometry; residual-stream PCA. The third spans two
# substrates, which is why a block is an address and not a finding —
# `anamnesis/feature_map.py` is what says which substrate a feature reads.

NORMS_AND_OUTPUT_STATS = "norms_and_output_stats"
ATTENTION_AND_DELTAS = "attention_and_deltas"
CACHE_AND_KEYS = "cache_and_keys"
RESIDUAL_PCA = "residual_pca"

RESIDUAL_TRAJECTORY = "residual_trajectory"
ATTENTION_FLOW = "attention_flow"
GATE_FEATURES = "gate_features"

# Union labels: a block built by concatenating others, addressed as one.
ATTENTION_AND_CACHE = "attention_and_cache"
ALL_CORE = "combined"
ALL_FAMILIES = "engineered"
EVERYTHING = "every_block"
ATTENTION_AND_CACHE_WITH_FAMILIES = "attention_and_cache+engineered"

# Label → the name the block is stored under. The four core blocks were banked
# under names that say nothing about what they read, and every signature ever
# written indexes into its vector with exactly those names, so they are held as
# constants in `anamnesis/extraction/state_extractor.py` and read from there:
# this table is the only place a label meets a stored name. A family is stored
# under its own label.
BLOCK_STORED_NAMES: dict[str, str] = {
    NORMS_AND_OUTPUT_STATS: STORED_NORMS_AND_OUTPUT_STATS,
    ATTENTION_AND_DELTAS: STORED_ATTENTION_AND_DELTAS,
    CACHE_AND_KEYS: STORED_CACHE_AND_KEYS,
    RESIDUAL_PCA: STORED_RESIDUAL_PCA,
    RESIDUAL_TRAJECTORY: RESIDUAL_TRAJECTORY,
    ATTENTION_FLOW: ATTENTION_FLOW,
    GATE_FEATURES: GATE_FEATURES,
}

# An npz holds a block's columns under `features_<stored name>`, and the JSON
# sidecar holds its bounds under the stored name alone.
NPZ_KEY_PREFIX = "features_"

# Label → npz array key. The loader skips any block absent from the data.
BLOCK_NPZ_KEYS: dict[str, str] = {
    label: NPZ_KEY_PREFIX + stored for label, stored in BLOCK_STORED_NAMES.items()
}

# The blocks the numeric anchor builds, in vector order, and the families beside them.
CORE_BLOCKS = [NORMS_AND_OUTPUT_STATS, ATTENTION_AND_DELTAS, CACHE_AND_KEYS, RESIDUAL_PCA]
FAMILY_BLOCKS = [RESIDUAL_TRAJECTORY, ATTENTION_FLOW, GATE_FEATURES]

# A union is built only when every member it names is present; a union missing a
# member is omitted rather than built short, because a short union would be a
# different feature set reported under the same label — a label naming blocks the
# numbers do not contain.
#
# ``ALL_FAMILIES`` spans every family block, so it is written as that list rather
# than as a second enumeration of the same members.
BLOCK_UNIONS: dict[str, list[str]] = {
    ATTENTION_AND_CACHE: [ATTENTION_AND_DELTAS, CACHE_AND_KEYS],
    ALL_CORE: list(CORE_BLOCKS),
    ALL_FAMILIES: list(FAMILY_BLOCKS),
    EVERYTHING: [*CORE_BLOCKS, *FAMILY_BLOCKS],
    ATTENTION_AND_CACHE_WITH_FAMILIES: [
        ATTENTION_AND_DELTAS, CACHE_AND_KEYS, *FAMILY_BLOCKS,
    ],
}

# Every label a block or a union can be reported under. Derived rather than
# listed, so a label added to either table is covered by whatever reads this —
# the compatibility table's completeness check among them.
ALL_LABELS: frozenset[str] = frozenset(BLOCK_NPZ_KEYS) | frozenset(BLOCK_UNIONS)


def default_signature_dir() -> Path:
    """The Phase-0 run-4 signature directory, resolved when asked.

    The legacy root is an environment variable, so the location is read at call
    time: a process that repoints ``ANAMNESIS_LEGACY_DATA`` does not have to
    reload the package, and a test can redirect it with a temporary directory.
    """
    return legacy_data_root() / "outputs" / "runs" / "run4_format_controlled" / "signatures"


@dataclass
class SampleMeta:
    """Metadata for a single generation sample."""
    generation_id: int
    topic: str
    topic_idx: int
    mode: str
    mode_idx: int
    num_generated_tokens: int
    file_stem: str


@dataclass
class Run4Data:
    """Loaded Run 4 data with feature matrices and metadata."""
    # Per-block feature matrices: {block_name: (N, D_block)}
    block_features: dict[str, NDArray[np.float32]]
    # Composite group matrices: {group_name: (N, D_group)}
    group_features: dict[str, NDArray[np.float32]]
    # Every present block concatenated, in block order: (N, sum of block widths)
    all_features: NDArray[np.float32]
    # Feature names per block
    block_feature_names: dict[str, NDArray]
    # Sample metadata (ordered to match matrix rows)
    samples: list[SampleMeta]
    # Convenience arrays
    modes: NDArray  # (N,) string array of mode labels
    topics: NDArray  # (N,) string array of topic labels
    mode_indices: NDArray[np.int64]  # (N,) integer mode indices
    topic_indices: NDArray[np.int64]  # (N,) integer topic indices
    lane_id: str | None = None  # None denotes legacy untagged, not certification.

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def unique_modes(self) -> list[str]:
        return sorted(set(self.modes))

    @property
    def unique_topics(self) -> list[str]:
        return sorted(set(self.topics))

    def mode_mask(self, mode: str) -> NDArray[np.bool_]:
        """Boolean mask for samples of a given mode."""
        return self.modes == mode

    def topic_mask(self, topic: str) -> NDArray[np.bool_]:
        """Boolean mask for samples of a given topic."""
        return self.topics == topic

    def has_block(self, block_or_group: str) -> bool:
        """Whether this corpus holds that block or union.

        What a caller asks before reading one: a union is absent whenever any
        member it names is absent, so a corpus narrower than the suite that
        wrote it has fewer of them, and asking is how a section reports that
        rather than dying on it.
        """
        return block_or_group in self.block_features or block_or_group in self.group_features

    def get_block(self, block_or_group: str) -> NDArray[np.float32]:
        """Get feature matrix for a block name or group name.

        Raises
        ------
        KeyError
            When this corpus holds no such block or union. Callers that can
            proceed without it test :meth:`has_block` first and report the
            absence; this refusal is for the ones that cannot.
        """
        if block_or_group in self.block_features:
            return self.block_features[block_or_group]
        if block_or_group in self.group_features:
            return self.group_features[block_or_group]
        raise KeyError(f"Unknown block/group: {block_or_group}. "
                       f"Available: {list(self.block_features) + list(self.group_features)}")


def load_run4(
    signature_dir: Path | str | None = None,
    core_only: bool = True,
    addon_dirs: list[Path | str] | None = None,
    mode_filter: list[str] | None = None,
) -> Run4Data:
    """
    Load signature data with optional addon directories for split feature sets.

    Parameters
    ----------
    signature_dir : Path, optional
        Primary directory containing gen_NNN.npz and gen_NNN.json files.
        Defaults to :func:`default_signature_dir`.
    core_only : bool
        If True (default), load only the balanced core set:
        20 shared topics × 5 modes = 100 samples.
        Excludes multi-repetition extras and supplementary linear samples.
    addon_dirs : list[Path], optional
        Additional directories with features_* arrays to merge in.
        Files must match gen_NNN.npz naming. Extra blocks are added
        alongside those from the primary directory.
    mode_filter : list[str], optional
        If provided, only include samples whose mode is in this list.
        Applied after core_only filtering.

    Returns
    -------
    Run4Data with feature matrices and metadata.
    """
    sig_dir = Path(signature_dir) if signature_dir is not None else default_signature_dir()
    if not sig_dir.exists():
        raise FileNotFoundError(f"Signature directory not found: {sig_dir}")

    # Discover all generation files
    npz_files = sorted(sig_dir.glob("gen_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {sig_dir}")

    # First pass: load metadata to determine which samples to include
    all_meta: list[tuple[Path, dict]] = []
    missing_metadata = []
    for npz_path in npz_files:
        json_path = npz_path.with_suffix(".json")
        if not json_path.exists():
            missing_metadata.append(str(json_path))
            continue
        with open(json_path) as f:
            meta = json.load(f)
        all_meta.append((npz_path, meta))

    if core_only:
        # Exclude swap/special modes from the core balance calculation.
        # Swap modes (e.g. "swap_socratic→linear") are auxiliary — they
        # contaminate the shared-topic intersection if included.
        regular_meta = [(p, m) for p, m in all_meta if not m["mode"].startswith("swap_")]
        swap_meta = [(p, m) for p, m in all_meta if m["mode"].startswith("swap_")]

        # Find shared topics across regular modes only
        mode_topics: dict[str, set[str]] = {}
        for _, meta in regular_meta:
            mode = meta["mode"]
            topic = meta["topic"]
            mode_topics.setdefault(mode, set()).add(topic)

        shared_topics = set.intersection(*mode_topics.values()) if mode_topics else set()

        # For shared topics with multiple repetitions, take only the first
        seen: set[tuple[str, str]] = set()
        filtered: list[tuple[Path, dict]] = []
        for npz_path, meta in regular_meta:
            key = (meta["mode"], meta["topic"])
            if meta["topic"] in shared_topics and key not in seen:
                seen.add(key)
                filtered.append((npz_path, meta))
        all_meta = filtered

    # Apply mode filter if specified
    if mode_filter is not None:
        allowed = set(mode_filter)
        all_meta = [(p, m) for p, m in all_meta if m["mode"] in allowed]
        if not all_meta:
            raise ValueError(
                f"No samples remain after mode_filter={mode_filter}. "
                f"Check that mode names match the data."
            )
        logger.info(f"Mode filter applied: {len(all_meta)} samples for modes {sorted(allowed)}")

    # Sort by (mode_idx, topic_idx) for consistent ordering
    lane_id = require_single_lane([meta for _, meta in all_meta])
    if lane_id is not None and missing_metadata:
        raise MixedLaneError("tagged signature directory contains files without lane metadata")
    all_meta.sort(key=lambda x: (x[1]["mode_idx"], x[1]["topic_idx"]))

    # Second pass: load features. Block discovery happens on the first file of
    # this loop rather than in a pass of its own, so the first npz is opened and
    # decompressed once.
    present_blocks: dict[str, str] = {}
    blocks_discovered = False
    block_arrays: dict[str, list[NDArray]] = {}
    samples: list[SampleMeta] = []
    all_feature_names: dict[str, NDArray] | None = None

    for npz_path, meta in all_meta:
        data = np.load(npz_path, allow_pickle=True)

        if not blocks_discovered:
            blocks_discovered = True
            # ── Discover available blocks from the first npz file ──
            available_npz_keys = set(data.files)
            for block_name, npz_key in BLOCK_NPZ_KEYS.items():
                if npz_key in available_npz_keys:
                    present_blocks[block_name] = npz_key
            block_arrays = {k: [] for k in present_blocks}
            logger.info(
                f"Discovered {len(present_blocks)} blocks: {list(present_blocks.keys())}"
            )
            missing = set(BLOCK_NPZ_KEYS) - set(present_blocks)
            if missing:
                logger.info(f"  Missing (skipped): {sorted(missing)}")

        for block_name, npz_key in present_blocks.items():
            block_arrays[block_name].append(data[npz_key])

        samples.append(SampleMeta(
            generation_id=meta["generation_id"],
            topic=meta["topic"],
            topic_idx=meta["topic_idx"],
            mode=meta["mode"],
            mode_idx=meta["mode_idx"],
            num_generated_tokens=meta["num_generated_tokens"],
            file_stem=npz_path.stem,
        ))

        # Grab feature names once
        if all_feature_names is None:
            names = data.get("feature_names")
            slices = meta.get(STORED_BLOCK_SLICES_KEY, {})
            all_feature_names = {}
            if names is not None and slices:
                for block_name in present_blocks:
                    slice_key = BLOCK_STORED_NAMES[block_name]
                    if slice_key in slices:
                        start, end = slices[slice_key]
                        all_feature_names[block_name] = names[start:end]

    # Stack into matrices
    block_features = {
        name: np.stack(arrays, axis=0)
        for name, arrays in block_arrays.items()
        if arrays  # skip empty
    }

    # ── Merge addon directories ──
    if addon_dirs:
        # Build a mapping from file stem to sample index for matching
        stem_to_idx = {s.file_stem: i for i, s in enumerate(samples)}

        for addon_dir in addon_dirs:
            addon_path = Path(addon_dir)
            if not addon_path.exists():
                logger.warning(f"Addon dir not found: {addon_path}")
                continue

            # Discover blocks in addon
            addon_files = sorted(addon_path.glob("gen_*.npz"))
            if not addon_files:
                logger.warning(f"No npz files in addon dir: {addon_path}")
                continue

            first_addon = np.load(addon_files[0], allow_pickle=True)
            addon_blocks: dict[str, str] = {}
            for block_name, npz_key in BLOCK_NPZ_KEYS.items():
                if npz_key in first_addon.files and block_name not in block_features:
                    addon_blocks[block_name] = npz_key

            if not addon_blocks:
                logger.info(f"  Addon {addon_path.name}: no new blocks (all duplicates)")
                continue

            logger.info(
                f"  Addon {addon_path.name}: merging {list(addon_blocks.keys())}"
            )

            # Load addon features in sample order
            addon_arrays: dict[str, list[NDArray | None]] = {
                k: [None] * len(samples) for k in addon_blocks
            }
            matched = 0
            for npz_file in addon_files:
                stem = npz_file.stem
                if stem not in stem_to_idx:
                    continue
                idx = stem_to_idx[stem]
                addon_metadata_path=npz_file.with_suffix(".json")
                try:
                    addon_metadata=json.loads(addon_metadata_path.read_text()) if addon_metadata_path.exists() else {}
                except (json.JSONDecodeError,OSError) as error:
                    if lane_id is not None:
                        raise MixedLaneError("tagged signature addon has unreadable metadata") from error
                    addon_metadata={}
                # Feature concatenation is also a scientific join; an untagged
                # addon must not silently enter a tagged lane's vector.
                require_single_lane([all_meta[idx][1],addon_metadata])
                data = np.load(npz_file, allow_pickle=True)
                for block_name, npz_key in addon_blocks.items():
                    addon_arrays[block_name][idx] = data[npz_key]
                matched += 1

                # Grab feature names
                if all_feature_names is not None:
                    addon_names = data.get("feature_names")
                    addon_meta_path = npz_file.with_suffix(".json")
                    if addon_names is not None and addon_meta_path.exists():
                        try:
                            with open(addon_meta_path) as f:
                                addon_meta = json.load(f)
                            addon_slices = addon_meta.get(STORED_BLOCK_SLICES_KEY, {})
                            for tn in addon_blocks:
                                if tn not in all_feature_names:
                                    sk = BLOCK_STORED_NAMES[tn]
                                    if sk in addon_slices:
                                        s, e = addon_slices[sk]
                                        all_feature_names[tn] = addon_names[s:e]
                        except (json.JSONDecodeError, OSError):
                            pass

            if matched < len(samples):
                logger.warning(
                    f"  Addon {addon_path.name}: only {matched}/{len(samples)} "
                    f"samples matched — skipping incomplete addon"
                )
                continue

            # Stack and add to block_features
            for block_name, arrays in addon_arrays.items():
                if any(a is None for a in arrays):
                    logger.warning(f"  Addon block {block_name}: has None entries, skipping")
                    continue
                block_features[block_name] = np.stack(arrays, axis=0)

    # Build the unions — a union whose every member is present, and no other.
    group_features: dict[str, NDArray[np.float32]] = {}
    for group_name, block_list in BLOCK_UNIONS.items():
        missing = [t for t in block_list if t not in block_features]
        if missing:
            logger.info(
                f"  Union '{group_name}' not built: {sorted(missing)} absent from this "
                f"corpus, and a union built short would name blocks it does not hold"
            )
            continue
        group_features[group_name] = np.concatenate(
            [block_features[t] for t in block_list], axis=1
        )

    # Full combined — all present individual blocks
    all_present_individual = [
        t for t in list(CORE_BLOCKS) + list(FAMILY_BLOCKS)
        if t in block_features
    ]
    all_features = np.concatenate(
        [block_features[t] for t in all_present_individual], axis=1
    ) if all_present_individual else np.array([], dtype=np.float32)

    # Convenience arrays
    modes = np.array([s.mode for s in samples])
    topics = np.array([s.topic for s in samples])
    mode_indices = np.array([s.mode_idx for s in samples], dtype=np.int64)

    # Build stable topic indices (alphabetical)
    unique_topics_sorted = sorted(set(topics))
    topic_to_idx = {t: i for i, t in enumerate(unique_topics_sorted)}
    topic_indices = np.array([topic_to_idx[s.topic] for s in samples], dtype=np.int64)

    return Run4Data(
        block_features=block_features,
        group_features=group_features,
        all_features=all_features,
        block_feature_names=all_feature_names or {},
        samples=samples,
        modes=modes,
        topics=topics,
        mode_indices=mode_indices,
        topic_indices=topic_indices,
        lane_id=lane_id,
    )


def check_data_quality(data: Run4Data) -> dict[str, object]:
    """Quick sanity checks on loaded data."""
    report: dict[str, object] = {
        "n_samples": data.n_samples,
        "n_modes": len(data.unique_modes),
        "n_topics": len(data.unique_topics),
        "modes": data.unique_modes,
        "samples_per_mode": {
            m: int(np.sum(data.mode_mask(m)))
            for m in data.unique_modes
        },
    }

    # Check for NaN/Inf
    nan_counts: dict[str, int] = {}
    inf_counts: dict[str, int] = {}
    for name, feat in {**data.block_features, **data.group_features}.items():
        nan_counts[name] = int(np.sum(np.isnan(feat)))
        inf_counts[name] = int(np.sum(np.isinf(feat)))

    report["nan_counts"] = nan_counts
    report["inf_counts"] = inf_counts

    # Feature dimensions
    report["block_dims"] = {
        name: feat.shape[1]
        for name, feat in data.block_features.items()
    }
    report["group_dims"] = {
        name: feat.shape[1]
        for name, feat in data.group_features.items()
    }

    return report


@dataclass
class AnalysisData:
    """A loaded run plus the generated text, for sections that read both."""

    run4: Run4Data
    run_name: str
    generated_texts: list[str] | None = None
    system_prompts: list[str] | None = None
    user_prompts: list[str] | None = None
    generation_lengths: NDArray[np.int64] | None = None

    # Delegate common accessors
    @property
    def n_samples(self) -> int:
        return self.run4.n_samples

    @property
    def modes(self) -> NDArray:
        return self.run4.modes

    @property
    def topics(self) -> NDArray:
        return self.run4.topics

    @property
    def unique_modes(self) -> list[str]:
        return self.run4.unique_modes

    @property
    def unique_topics(self) -> list[str]:
        return self.run4.unique_topics

    def has_block(self, name: str) -> bool:
        return self.run4.has_block(name)

    def get_block(self, name: str) -> NDArray[np.float32]:
        return self.run4.get_block(name)

    def mode_mask(self, mode: str) -> NDArray[np.bool_]:
        return self.run4.mode_mask(mode)

    def topic_mask(self, topic: str) -> NDArray[np.bool_]:
        return self.run4.topic_mask(topic)


def load_analysis_data(
    signature_dir: Path | str,
    run_name: str,
    core_only: bool = True,
    load_text: bool = True,
    addon_dirs: list[Path | str] | None = None,
    mode_filter: list[str] | None = None,
) -> AnalysisData:
    """Load signature data with optional text fields for semantic analysis.

    Parameters
    ----------
    signature_dir : Path
        Directory containing gen_NNN.npz and gen_NNN.json files.
    run_name : str
        Label for this run (e.g. "8b_baseline").
    core_only : bool
        If True, load only one rep per topic-mode pair.
    load_text : bool
        If True, also load generated text from JSON metadata.
    addon_dirs : list[Path], optional
        Additional directories with features_* arrays to merge.
    mode_filter : list[str], optional
        If provided, only include samples whose mode is in this list.
    """
    sig_dir = Path(signature_dir)
    run4 = load_run4(
        signature_dir=sig_dir,
        core_only=core_only,
        addon_dirs=addon_dirs,
        mode_filter=mode_filter,
    )

    generated_texts: list[str] | None = None
    system_prompts: list[str] | None = None
    user_prompts: list[str] | None = None
    gen_lengths: NDArray[np.int64] | None = None

    if load_text:
        texts = []
        sys_prompts = []
        usr_prompts = []
        lengths = []

        for sample in run4.samples:
            json_path = sig_dir / f"{sample.file_stem}.json"
            if json_path.exists():
                with open(json_path) as f:
                    meta = json.load(f)
                texts.append(meta.get("generated_text", ""))
                sys_prompts.append(meta.get("system_prompt", ""))
                usr_prompts.append(meta.get("user_prompt", ""))
                lengths.append(meta.get("num_generated_tokens", 0))
            else:
                texts.append("")
                sys_prompts.append("")
                usr_prompts.append("")
                lengths.append(0)

        generated_texts = texts
        system_prompts = sys_prompts
        user_prompts = usr_prompts
        gen_lengths = np.array(lengths, dtype=np.int64)

    return AnalysisData(
        run4=run4,
        run_name=run_name,
        generated_texts=generated_texts,
        system_prompts=system_prompts,
        user_prompts=user_prompts,
        generation_lengths=gen_lengths,
    )
