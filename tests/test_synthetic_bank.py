"""The synthetic bank: is it loadable, is it reproducible, and does it say nothing extra.

A fixture that stands in for a corpus is only worth shipping if the reader that will open
a real bank opens this one by the same path. So the load here goes through
`anamnesis.analysis.gauntlet.signature_io`, not through a bespoke parse.

The rest of these are about what the bank must *not* contain. It carries no retired
vocabulary, it separates modes without letting topic stand in for them, and it does not
quietly favour one feature family — each of which would teach a reader something false
about the instrument rather than nothing at all.

CPU only; no model, no GPU, no network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from anamnesis.analysis.gauntlet.signature_io import (
    ALL_FAMILIES,
    BLOCK_STORED_NAMES,
    BLOCK_UNIONS,
    EVERYTHING,
    load_run4,
)
from anamnesis.synthetic_bank import (
    DEFAULT_BLOCK_WIDTHS,
    SYNTHETIC_LANE,
    SyntheticBankSpec,
    _block_layout,
    _mode_offsets,
    write_synthetic_bank,
)

# The fold counts the cross-condition section cuts topics at. The default topic count
# must be a multiple of every one of them or the remainder lands in no test fold.
CCGP_FOLD_COUNTS = (4, 5, 10, 20)

RETIRED_BIN_VOCABULARY = re.compile(r"\bT1\b|\bT2\b|\bT2\.5\b|\bT3\b|\btier[0-9]", re.IGNORECASE)


@pytest.fixture
def small_spec() -> SyntheticBankSpec:
    """Two modes, four topics: enough to load and stack, fast enough to use everywhere."""
    return SyntheticBankSpec(modes=("linear", "socratic"), topics=4, repetitions=1, seed=7)


def test_the_gauntlet_loader_opens_it(tmp_path: Path, small_spec: SyntheticBankSpec) -> None:
    bank = write_synthetic_bank(tmp_path / "signatures", small_spec)
    data = load_run4(bank.directory, core_only=False)

    assert set(data.block_features) == set(DEFAULT_BLOCK_WIDTHS)
    for block, width in DEFAULT_BLOCK_WIDTHS.items():
        assert data.block_features[block].shape == (small_spec.generations, width)

    # Every union is built, because every member is written — which is the property that
    # lets a section read rather than state an absence. Each is checked at the width its
    # own members sum to, since a union reported narrower than that would be a label
    # overstating its contents.
    for union, members in BLOCK_UNIONS.items():
        expected = sum(DEFAULT_BLOCK_WIDTHS[m] for m in members)
        assert data.group_features[union].shape == (small_spec.generations, expected), union
    assert data.group_features[EVERYTHING].shape == (small_spec.generations, bank.width)


def test_the_same_seed_writes_the_same_bank(
    tmp_path: Path, small_spec: SyntheticBankSpec
) -> None:
    first = write_synthetic_bank(tmp_path / "one", small_spec)
    second = write_synthetic_bank(tmp_path / "two", small_spec)

    for path in sorted(first.directory.glob("*.npz")):
        mine = np.load(path)
        theirs = np.load(second.directory / path.name)
        for key in mine.files:
            np.testing.assert_array_equal(mine[key], theirs[key])


def test_a_different_seed_writes_a_different_bank(tmp_path: Path) -> None:
    base = SyntheticBankSpec(topics=4, repetitions=1, seed=1)
    other = base.model_copy(update={"seed": 2})
    first = write_synthetic_bank(tmp_path / "one", base)
    second = write_synthetic_bank(tmp_path / "two", other)

    mine = np.load(first.directory / "gen_000.npz")
    theirs = np.load(second.directory / "gen_000.npz")
    assert not np.array_equal(mine["features_attention_flow"], theirs["features_attention_flow"])


def test_no_retired_bin_vocabulary_reaches_a_reader(
    tmp_path: Path, small_spec: SyntheticBankSpec
) -> None:
    """The fixture names every block for what it reads, never for a retired bin.

    Two things carry a bin name and neither is this fixture's to rename: the npz array
    keys and the slice-table keys inside the sidecar. Those are the frozen on-disk format
    — bytes in banked files that will never be rewritten — and they reach a reader only
    through the ``STORED_*`` constants. So they are exempted **by derivation from those
    constants**, not by a hand-written list, which is what stops the exemption from
    widening quietly. Everything else is checked: feature names, and every string the
    sidecar carries.
    """
    bank = write_synthetic_bank(tmp_path / "signatures", small_spec)
    wire_format = set(BLOCK_STORED_NAMES.values())

    for npz_path in bank.directory.glob("*.npz"):
        for name in np.load(npz_path)["feature_names"]:
            assert not RETIRED_BIN_VOCABULARY.search(str(name)), name

    for json_path in bank.directory.glob("*.json"):
        metadata = json.loads(json_path.read_text())
        for key, value in metadata.items():
            if isinstance(value, str):
                assert not RETIRED_BIN_VOCABULARY.search(value), (key, value)
            if isinstance(value, dict):
                for inner in value:
                    if inner in wire_format:
                        continue
                    assert not RETIRED_BIN_VOCABULARY.search(inner), (key, inner)


def test_every_generation_carries_the_synthetic_lane(
    tmp_path: Path, small_spec: SyntheticBankSpec
) -> None:
    bank = write_synthetic_bank(tmp_path / "signatures", small_spec)
    for json_path in bank.directory.glob("*.json"):
        assert json.loads(json_path.read_text())["lane_id"] == SYNTHETIC_LANE
    assert bank.lane_id == SYNTHETIC_LANE


def test_the_text_carries_the_topic_and_not_the_mode(
    tmp_path: Path, small_spec: SyntheticBankSpec
) -> None:
    """The semantic section measures signatures against the text, so the text holds no label."""
    bank = write_synthetic_bank(tmp_path / "signatures", small_spec)
    texts: dict[str, set[str]] = {}
    for json_path in bank.directory.glob("*.json"):
        meta = json.loads(json_path.read_text())
        assert meta["mode"] not in meta["generated_text"]
        texts.setdefault(meta["topic"], set()).add(meta["generated_text"])
    assert all(len(on_topic) == 1 for on_topic in texts.values()), texts


def test_mode_is_recoverable_and_topic_is_not_a_stand_in(tmp_path: Path) -> None:
    """Structure exists, and it is not the topic axis wearing the mode's label.

    Both halves matter. Without the first the fixture teaches nothing; without the second
    it would demonstrate the leak the leak gate exists to catch.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import cross_val_score

    spec = SyntheticBankSpec(topics=20, repetitions=1, seed=3)
    bank = write_synthetic_bank(tmp_path / "signatures", spec)
    data = load_run4(bank.directory, core_only=False)
    features = data.group_features[ALL_FAMILIES]

    # Shrinkage is a requirement here, not a preference: this bank offers 62 features
    # over 100 rows, so an unregularized discriminant is fitting more directions than
    # the folds support and lands near chance whatever structure is present. Covariance
    # shrinkage is what makes the readout a statement about the bank rather than about
    # the estimator, and it holds across draws — an unregularized fit does not, which
    # would make this assertion a property of the seed.
    by_mode = cross_val_score(
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        features,
        data.modes,
        cv=4,
    ).mean()
    assert by_mode > 0.7, by_mode

    # Every topic appears once under every mode, so the topic label carries no
    # information about the mode: the design, not the draw, is what rules the leak out.
    pairs = sorted(zip(data.modes, data.topics, strict=True))
    assert len(pairs) == len(set(pairs)) == len(spec.modes) * spec.topics


def _centroid_separations(offsets: np.ndarray, spans: dict[str, tuple[int, int]]) -> dict[str, float]:
    """Mean pairwise distance between mode centroids, per block."""
    out: dict[str, float] = {}
    for block, (start, stop) in spans.items():
        slice_ = offsets[:, start:stop]
        pairs = [
            float(np.linalg.norm(slice_[i] - slice_[j]))
            for i in range(len(slice_))
            for j in range(i + 1, len(slice_))
        ]
        out[block] = float(np.mean(pairs))
    return out


def test_no_block_is_staged_as_the_load_bearing_one() -> None:
    """Every block separates the modes equally, averaged over the draw.

    The quantity that has to match is *centroid separation*, not strength per column.
    Separation accumulates over a block's columns, so equal per-column strength gives
    the widest block the strongest readout — and the block widths here are arbitrary,
    so a ranking that followed them would read as a statement about substrates. This
    asserts the invariant the construction actually needs.
    """
    spans, _ = _block_layout(DEFAULT_BLOCK_WIDTHS)
    per_seed = [
        _centroid_separations(_mode_offsets(np.random.default_rng(seed), 5, spans, scale=0.4), spans)
        for seed in range(30)
    ]
    mean = {block: float(np.mean([s[block] for s in per_seed])) for block in spans}

    widest = max(spans, key=lambda b: spans[b][1] - spans[b][0])
    narrowest = min(spans, key=lambda b: spans[b][1] - spans[b][0])
    spread = max(mean.values()) - min(mean.values())

    # Normalizing per column instead would spread these separations by roughly 0.95 on
    # these widths and put the narrowest block last in most draws. A tenth of that, with
    # no ordering by width, is what separates a flat fixture from one staging a substrate.
    assert spread < 0.15, mean
    assert mean[narrowest] == pytest.approx(mean[widest], rel=0.05), mean


def test_block_separation_does_not_track_width() -> None:
    """The residual spread is the draw, so it must not correlate with width.

    A width trend is the specific failure this construction exists to avoid: it is how
    a fixture would teach a reader that one substrate carries the signal. Pearson
    correlation over the seven blocks, averaged across draws, has no reason to be
    large unless the normalization has regressed.
    """
    spans, _ = _block_layout(DEFAULT_BLOCK_WIDTHS)
    widths = np.array([spans[b][1] - spans[b][0] for b in spans], dtype=float)
    correlations: list[float] = []
    for seed in range(30):
        sep = _centroid_separations(
            _mode_offsets(np.random.default_rng(seed), 5, spans, scale=0.4), spans
        )
        values = np.array([sep[b] for b in spans], dtype=float)
        correlations.append(float(np.corrcoef(widths, values)[0, 1]))

    assert abs(float(np.mean(correlations))) < 0.35, float(np.mean(correlations))


def test_an_unknown_block_label_is_refused() -> None:
    with pytest.raises(KeyError, match="unknown block label"):
        _block_layout({"not_a_block": 4})


def test_the_default_topic_count_covers_every_fold_count() -> None:
    """Pins the reason for the default: integer division drops the remainder."""
    topics = SyntheticBankSpec().topics
    for folds in CCGP_FOLD_COUNTS:
        assert topics % folds == 0, (topics, folds)
