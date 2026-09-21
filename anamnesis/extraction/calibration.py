"""Reading the two calibration artifacts a corrected feature depends on.

Positional decomposition needs per-position means to subtract, and the residual
PCA needs a fitted basis to project onto. Both are model-specific files written
once per model by the calibration pass, and both are read by every path that
computes features: the in-process extraction pass, the replay worker, the fast
lane, and the offline recompute. This module is where a model-loading pass reads
them, so that the in-process, replay and fast-lane paths cannot disagree about
what a calibration directory contains.

Two on-disk shapes for the PCA model are both accepted, because both are banked:
a plain mapping with ``components`` and ``mean`` keys, and a pickled scikit-learn
estimator with ``components_`` and ``mean_`` attributes. They are read to the
same pair of float32 arrays. What is *not* read here is the per-layer basis a
corrected refit produces, keyed by layer index: that form is consumed only by the
offline recompute over banked tensors, and
:mod:`anamnesis.extraction.feature_pipeline` reads it there — the module whose
arithmetic the feature receipts pin, and therefore the module that keeps its own
reader.

Absence is returned, not raised. A caller that cannot proceed without a
calibration says so itself — :mod:`anamnesis.scripts.run_gpu_replay` refuses a
partial one outright, while a pass whose features do not touch the residual PCA
runs without it. Silently handing back uncorrected features under the name of
corrected ones is the failure this split avoids, so a missing positional means
is logged as a warning at the point of reading.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

F32 = NDArray[np.float32]

logger = logging.getLogger(__name__)

POSITIONAL_MEANS_NAME = "positional_means.npz"
"""Per-position, per-layer means, under the ``positional_means`` array key."""

PCA_MODEL_NAME = "pca_model.pkl"
"""The fitted residual-stream PCA: components and the mean they are taken about."""

POSITIONAL_MEANS_KEY = "positional_means"
"""The array key inside the positional-means archive."""


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
    KeyError, AttributeError
        When the object is neither of the two accepted shapes, which includes the
        per-layer form: a pass that loads a model projects onto one basis, and a
        mapping of bases would be silently reduced to whichever one a key happened
        to hold.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"no PCA model at {target}")
    with open(target, "rb") as f:
        model = pickle.load(f)
    if isinstance(model, dict):
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
    """
    directory = Path(calib_dir)
    positional_means = load_positional_means(directory)
    components: F32 | None = None
    mean: F32 | None = None
    pca_path = directory / PCA_MODEL_NAME
    if enable_pca and pca_path.exists():
        components, mean = load_pca_model(pca_path)
        logger.info(f"pca components {components.shape}")
    return positional_means, components, mean
