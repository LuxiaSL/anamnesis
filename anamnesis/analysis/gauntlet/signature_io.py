"""Reading banked signatures off disk — the gauntlet's one entry point for data.

A signature directory holds one ``gen_NNN.npz`` of feature vectors beside one
``gen_NNN.json`` of stimulus metadata per generation. This module turns a
directory into the two objects the gauntlet's sections consume:

``Run4Data``
    Feature matrices partitioned by tier, plus the composite groups, plus the
    mode and topic labels aligned to the matrix rows. Both the baseline tiers
    (T1/T2/T2.5/T3) and the v2 families are supported, and which ones are
    present is discovered from the npz contents rather than declared, so a
    directory extracted with a narrower suite loads as itself.
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

logger = logging.getLogger(__name__)


# ── Tier definitions ──────────────────────────────────────────────────────────
# Maps human-readable tier names to npz array keys.
# All known tiers — loader gracefully skips any that are absent from the data.

TIER_KEYS: dict[str, str] = {
    # Baseline tiers
    "T1": "features_tier1",
    "T2": "features_tier2",
    "T2.5": "features_tier2_5",
    "T3": "features_tier3",
    # v2 feature families
    "residual_trajectory": "features_residual_trajectory",
    "attention_flow": "features_attention_flow",
    "gate_features": "features_gate_features",
    "temporal_dynamics": "features_temporal_dynamics",
    "contrastive_projection": "features_contrastive_projection",
}

# Tier categories for structured ablation
BASELINE_TIERS = ["T1", "T2", "T2.5", "T3"]
ENGINEERED_TIERS = [
    "residual_trajectory", "attention_flow", "gate_features",
    "temporal_dynamics", "contrastive_projection",
]

# Composite tier groups — built from whichever individual tiers are present.
# Groups with missing members are silently omitted.
TIER_GROUPS: dict[str, list[str]] = {
    "T2+T2.5": ["T2", "T2.5"],
    "combined": ["T1", "T2", "T2.5", "T3"],
    "engineered": [
        "residual_trajectory", "attention_flow", "gate_features",
        "temporal_dynamics",
    ],
    "combined_v2": [
        "T1", "T2", "T2.5", "T3",
        "residual_trajectory", "attention_flow", "gate_features",
        "temporal_dynamics", "contrastive_projection",
    ],
    "T2+T2.5+engineered": [
        "T2", "T2.5",
        "residual_trajectory", "attention_flow", "gate_features",
        "temporal_dynamics",
    ],
}

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
    # Per-tier feature matrices: {tier_name: (N, D_tier)}
    tier_features: dict[str, NDArray[np.float32]]
    # Composite group matrices: {group_name: (N, D_group)}
    group_features: dict[str, NDArray[np.float32]]
    # Full combined matrix (N, 1837)
    all_features: NDArray[np.float32]
    # Feature names per tier
    tier_feature_names: dict[str, NDArray]
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

    def get_tier(self, tier_or_group: str) -> NDArray[np.float32]:
        """Get feature matrix for a tier name or group name."""
        if tier_or_group in self.tier_features:
            return self.tier_features[tier_or_group]
        if tier_or_group in self.group_features:
            return self.group_features[tier_or_group]
        raise KeyError(f"Unknown tier/group: {tier_or_group}. "
                       f"Available: {list(self.tier_features) + list(self.group_features)}")


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
        Files must match gen_NNN.npz naming. Extra tiers are added
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

    # Second pass: load features. Tier discovery happens on the first file of
    # this loop rather than in a pass of its own, so the first npz is opened and
    # decompressed once.
    present_tiers: dict[str, str] = {}
    tiers_discovered = False
    tier_arrays: dict[str, list[NDArray]] = {}
    samples: list[SampleMeta] = []
    all_feature_names: dict[str, NDArray] | None = None

    for npz_path, meta in all_meta:
        data = np.load(npz_path, allow_pickle=True)

        if not tiers_discovered:
            tiers_discovered = True
            # ── Discover available tiers from the first npz file ──
            available_npz_keys = set(data.files)
            for tier_name, npz_key in TIER_KEYS.items():
                if npz_key in available_npz_keys:
                    present_tiers[tier_name] = npz_key
            tier_arrays = {k: [] for k in present_tiers}
            logger.info(
                f"Discovered {len(present_tiers)} tiers: {list(present_tiers.keys())}"
            )
            missing = set(TIER_KEYS) - set(present_tiers)
            if missing:
                logger.info(f"  Missing (skipped): {sorted(missing)}")

        for tier_name, npz_key in present_tiers.items():
            tier_arrays[tier_name].append(data[npz_key])

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
            slices = meta.get("tier_slices", {})
            all_feature_names = {}
            if names is not None and slices:
                for tier_name in present_tiers:
                    # Map tier display name to slice key
                    slice_key = TIER_KEYS[tier_name].replace("features_", "")
                    if slice_key in slices:
                        start, end = slices[slice_key]
                        all_feature_names[tier_name] = names[start:end]

    # Stack into matrices
    tier_features = {
        name: np.stack(arrays, axis=0)
        for name, arrays in tier_arrays.items()
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

            # Discover tiers in addon
            addon_files = sorted(addon_path.glob("gen_*.npz"))
            if not addon_files:
                logger.warning(f"No npz files in addon dir: {addon_path}")
                continue

            first_addon = np.load(addon_files[0], allow_pickle=True)
            addon_tiers: dict[str, str] = {}
            for tier_name, npz_key in TIER_KEYS.items():
                if npz_key in first_addon.files and tier_name not in tier_features:
                    addon_tiers[tier_name] = npz_key

            if not addon_tiers:
                logger.info(f"  Addon {addon_path.name}: no new tiers (all duplicates)")
                continue

            logger.info(
                f"  Addon {addon_path.name}: merging {list(addon_tiers.keys())}"
            )

            # Load addon features in sample order
            addon_arrays: dict[str, list[NDArray | None]] = {
                k: [None] * len(samples) for k in addon_tiers
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
                for tier_name, npz_key in addon_tiers.items():
                    addon_arrays[tier_name][idx] = data[npz_key]
                matched += 1

                # Grab feature names
                if all_feature_names is not None:
                    addon_names = data.get("feature_names")
                    addon_meta_path = npz_file.with_suffix(".json")
                    if addon_names is not None and addon_meta_path.exists():
                        try:
                            with open(addon_meta_path) as f:
                                addon_meta = json.load(f)
                            addon_slices = addon_meta.get("tier_slices", {})
                            for tn in addon_tiers:
                                if tn not in all_feature_names:
                                    sk = TIER_KEYS[tn].replace("features_", "")
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

            # Stack and add to tier_features
            for tier_name, arrays in addon_arrays.items():
                if any(a is None for a in arrays):
                    logger.warning(f"  Addon tier {tier_name}: has None entries, skipping")
                    continue
                tier_features[tier_name] = np.stack(arrays, axis=0)

    # Build composite groups — only include groups where all members are present
    group_features: dict[str, NDArray[np.float32]] = {}
    for group_name, tier_list in TIER_GROUPS.items():
        available_members = [t for t in tier_list if t in tier_features]
        if not available_members:
            continue
        if len(available_members) < len(tier_list):
            # Partial group — still useful, note which members are present
            logger.debug(
                f"  Group '{group_name}': {len(available_members)}/{len(tier_list)} "
                f"members present ({available_members})"
            )
        group_features[group_name] = np.concatenate(
            [tier_features[t] for t in available_members], axis=1
        )

    # Full combined — all present individual tiers
    all_present_individual = [
        t for t in list(BASELINE_TIERS) + list(ENGINEERED_TIERS)
        if t in tier_features
    ]
    all_features = np.concatenate(
        [tier_features[t] for t in all_present_individual], axis=1
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
        tier_features=tier_features,
        group_features=group_features,
        all_features=all_features,
        tier_feature_names=all_feature_names or {},
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
    for name, feat in {**data.tier_features, **data.group_features}.items():
        nan_counts[name] = int(np.sum(np.isnan(feat)))
        inf_counts[name] = int(np.sum(np.isinf(feat)))

    report["nan_counts"] = nan_counts
    report["inf_counts"] = inf_counts

    # Feature dimensions
    report["tier_dims"] = {
        name: feat.shape[1]
        for name, feat in data.tier_features.items()
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

    def get_tier(self, name: str) -> NDArray[np.float32]:
        return self.run4.get_tier(name)

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
