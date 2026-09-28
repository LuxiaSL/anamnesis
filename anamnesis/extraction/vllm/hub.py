"""The calibrations the vLLM lane's fixtures were reduced with, and where to fetch them.

A fixture vector depends on the calibration it was reduced with, so the install
check can only compare a host with the fixtures under that exact calibration:
its positional means and residual basis, byte for byte. A calibration fitted on
the host would differ in its last digits and fail every row. The files are too
large to ship inside the package, so they are published in a Hugging Face
dataset, :data:`CALIBRATION_REPO`, and pinned here by size and sha256: a
download that does not match its pin is deleted and refused, never used.

:data:`CALIBRATIONS` is the pin table. Each model's two digests combine, through
:func:`anamnesis.provenance.digest_of_shas`, into the calibration digest its
shipped fixture set records, which a test checks.
"""

from __future__ import annotations

from pathlib import Path

from anamnesis.provenance import file_sha

CALIBRATION_REPO = "LuxiaSL/anamnesis-vllm-lane"
"""The Hugging Face dataset holding ``calibration/<model>/<file>`` for every lane."""

CALIBRATIONS: dict[str, dict[str, tuple[int, str]]] = {
    "3b": {
        "positional_means.npz": (
            171859234, "f5c238aa085140fc9a44783a98a5d7fa820687d8d23b934322736f3bb5053893"),
        "pca_model.pkl": (
            627403, "6de2adbca1946002863233fc1fcbc7466ce2b0158267ee66a204911f345c749f"),
    },
    "8b": {
        "positional_means.npz": (
            239115982, "ada9c9b8a9148dd081bc33b36275bc073ad7fabf4f91f97b4a86570e2668c514"),
        "pca_model.pkl": (
            836299, "6bc09c70f8ef2a6faa49495913bd6b6927cb9cf78722607e361dfaf41be8a648"),
    },
    "70b": {
        "positional_means.npz": (
            1637774442, "bd13dafc98e730544329e8a2e664215d06636486a382efbc2cf12b3df9091681"),
        "pca_model.pkl": (
            1671883, "1c0321ab8f5ce1b4b106822a70985f03fa7af678c4f86eec41f602640e7bab40"),
    },
}
"""Per model, each calibration file's size in bytes and sha256."""


def default_calibration_dir(model: str) -> Path:
    """Where a fetched calibration is kept: under the output root, per model."""
    from anamnesis.config import outputs_root

    return outputs_root() / "vllm_calibration" / model


def verify_calibration(model: str, directory: Path) -> None:
    """Refuse a calibration directory whose files are not ``model``'s pinned ones.

    Raises
    ------
    ValueError
        Naming each missing file, and each file whose size or sha256 differs.
    """
    if model not in CALIBRATIONS:
        raise ValueError(f"no calibration is pinned for {model!r}")
    wrong = []
    for name, (size, digest) in CALIBRATIONS[model].items():
        path = Path(directory) / name
        if not path.is_file():
            wrong.append(f"{name} missing")
        elif path.stat().st_size != size or file_sha(path) != digest:
            wrong.append(f"{name} differs from its pin")
    if wrong:
        raise ValueError(f"{directory} is not the {model} lane's calibration: "
                         f"{', '.join(wrong)}")


def fetch_calibration(model: str, directory: Path | None = None) -> Path:
    """``model``'s calibration, downloaded once into ``directory`` and verified.

    Files already present and matching their pins are kept; a missing one is
    downloaded from :data:`CALIBRATION_REPO` into the Hugging Face cache and copied
    here; a copy that does not match its pin is deleted before the refusal. Returns
    the directory.

    Raises
    ------
    ValueError
        When the model has no pinned calibration, or a file does not match its pin.
    """
    if model not in CALIBRATIONS:
        raise ValueError(f"no calibration is pinned for {model!r}")
    import shutil

    from huggingface_hub import hf_hub_download

    directory = Path(directory) if directory is not None else default_calibration_dir(model)
    directory.mkdir(parents=True, exist_ok=True)
    for name, (size, digest) in CALIBRATIONS[model].items():
        path = directory / name
        if path.is_file() and path.stat().st_size == size and file_sha(path) == digest:
            continue
        cached = hf_hub_download(CALIBRATION_REPO, f"calibration/{model}/{name}",
                                 repo_type="dataset")
        shutil.copyfile(cached, path)
        if path.stat().st_size != size or file_sha(path) != digest:
            path.unlink()
            raise ValueError(f"the downloaded {model} {name} does not match its pin")
    return directory
