"""What a calibration directory is called, and how the two artifacts in it are read.

Positional decomposition needs per-position means to subtract, and the residual
PCA needs a fitted basis to project onto. Both are model-specific files written
once per model by :mod:`anamnesis.extraction.calibration_fit`, and both are read
by every path that computes features: the in-process extraction pass, the replay
worker, the fast lane, and the offline recompute.

The filenames themselves are :data:`anamnesis.config.experiment.PCA_MODEL_NAME` and
its siblings, imported here rather than spelled again. They sit in the
configuration layer because that layer builds the paths as well, and because it is
the base of this package and imports nothing else in it — the other direction would
put numpy behind every import of a run's description. What lives here is the
resolution: which of the accepted names a directory answers with, and what each file
turns into.

Two on-disk shapes for the PCA model are both accepted, because both are banked:
a plain mapping with ``components`` and ``mean`` keys, and a pickled scikit-learn
estimator with ``components_`` and ``mean_`` attributes. They are read to the
same pair of float32 arrays. What is *not* read here is the per-layer basis a
corrected fit produces, keyed by layer index: that form is consumed only by the
offline recompute over banked tensors, and
:mod:`anamnesis.extraction.feature_pipeline` reads it there — the module whose
arithmetic the feature receipts pin, and therefore the module that keeps its own
reader. A per-layer basis reaching the reader below is refused by name rather
than reduced to whichever layer a key happened to hold.

Absence is returned, not raised. A caller that cannot proceed without a
calibration says so itself — :mod:`anamnesis.scripts.run_gpu_replay` refuses a
partial one outright, while a pass whose features do not touch the residual PCA
runs without it. Silently handing back uncorrected features under the name of
corrected ones is the failure this split avoids, so a missing positional means
is logged as a warning at the point of reading.

This module is pure numpy and pickle. Fitting a basis needs scikit-learn, a model
runtime and weights on a device, so it sits in a sibling module that the CPU
recompute lane and the fast-lane readers never import.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from anamnesis.config.experiment import (
    PCA_MODEL_NAME,
    PCA_MODEL_NAMES,
    POSITIONAL_MEANS_NAME,
)

F32 = NDArray[np.float32]

logger = logging.getLogger(__name__)

POSITIONAL_MEANS_KEY = "positional_means"
"""The array key inside the positional-means archive."""

POSITION_COUNTS_KEY = "pos_counts"
"""How many states each position's mean was taken over, in the same archive."""

CALIBRATION_ARTIFACT_NAMES: tuple[str, ...] = (POSITIONAL_MEANS_NAME, PCA_MODEL_NAME)
"""The pair a provenance digest over a calibration directory hashes.

The written names, not :data:`anamnesis.config.experiment.PCA_MODEL_NAMES`: a digest
names the bytes it read, so a directory that answers :func:`resolve_pca_model` under
the other accepted spelling has no digest under this one and the pass that wanted a
provenance stamp fails on the missing file rather than stamping a different set.
"""


def resolve_pca_model(calib_dir: Path) -> Path | None:
    """The basis in a calibration directory, or ``None`` when it holds none.

    Tries :data:`PCA_MODEL_NAMES` in order and returns the first that exists, so a
    directory written by a fit and a banked directory both resolve without the
    caller naming a file.
    """
    directory = Path(calib_dir)
    for name in PCA_MODEL_NAMES:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def load_positional_means(calib_dir: Path) -> F32 | None:
    """Per-position means from a calibration directory, or ``None`` if absent."""
    path = Path(calib_dir) / POSITIONAL_MEANS_NAME
    if not path.exists():
        logger.warning(
            f"no positional means at {path}; features that subtract them would be wrong"
        )
        return None
    means: F32 = np.load(path)[POSITIONAL_MEANS_KEY].astype(np.float32)
    logger.info(f"positional_means {means.shape}")
    return means


def load_pca_model(path: Path) -> tuple[F32, F32]:
    """One pooled PCA basis and its mean, from either banked shape.

    Raises
    ------
    FileNotFoundError
        When the file is absent — a caller that asked for a specific PCA by path
        meant that one.
    KeyError
        When the object is a mapping of bases keyed by layer index. A pass that
        loads a model projects onto one basis, so a per-layer basis is refused
        here rather than reduced to whichever layer a key happened to hold; the
        reader for that shape is :mod:`anamnesis.extraction.feature_pipeline`, and
        the pooled shape comes from a pooled fit.
    AttributeError
        When the object is neither a mapping nor an estimator carrying
        ``components_`` and ``mean_``.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"no PCA model at {target}")
    with open(target, "rb") as f:
        model = pickle.load(f)
    if isinstance(model, dict):
        if "components" not in model:
            raise KeyError(
                f"{target} holds a basis per layer, keyed {sorted(model)}, and this reader "
                "projects onto one basis; the recompute path reads the per-layer shape, and "
                "a pooled fit writes the shape this reader takes"
            )
        return (
            np.asarray(model["components"], dtype=np.float32),
            np.asarray(model["mean"], dtype=np.float32),
        )
    return (
        np.asarray(model.components_, dtype=np.float32),
        np.asarray(model.mean_, dtype=np.float32),
    )


def load_calibration(
    calib_dir: Path, enable_pca: bool = True
) -> tuple[F32 | None, F32 | None, F32 | None]:
    """Positional means and, when the residual PCA is on, its basis and mean.

    Returns the triple every feature-computing entry point takes: ``(positional
    means, PCA components, PCA mean)``, each of which is ``None`` when the
    artifact is absent or, for the PCA pair, when it is switched off.

    The basis is resolved by :func:`resolve_pca_model`, so :data:`PCA_MODEL_NAME`
    wins over the banked spelling beside it when a directory holds both.

    Raises
    ------
    KeyError, AttributeError
        From :func:`load_pca_model`, when the resolved file holds a shape this
        reader cannot project onto. A directory that holds no basis at all is the
        ``None`` case; one that holds an unusable basis is a refusal, because a
        pass that asked for corrected features would otherwise compute
        uncorrected ones under their name.
    """
    directory = Path(calib_dir)
    positional_means = load_positional_means(directory)
    components: F32 | None = None
    mean: F32 | None = None
    pca_path = resolve_pca_model(directory) if enable_pca else None
    if pca_path is not None:
        components, mean = load_pca_model(pca_path)
        logger.info(f"pca components {components.shape} from {pca_path.name}")
    return positional_means, components, mean
