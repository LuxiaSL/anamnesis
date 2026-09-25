"""How many residual-basis components a calibration's samples actually determine.

The per-layer basis is fitted over states sampled from the calibration's own
sequences. Whether its fiftieth component means anything depends on how many
samples stand behind it, and that is measurable: fit the basis twice over disjoint
halves of the samples, against the same positional means, and count the directions
the two share (:func:`anamnesis.extraction.calibration_fit.principal_cosines` at or
above a threshold). Those are determined; the rest are properties of the sample
rather than of the model. The receipt also carries the leading-run curve
(:func:`anamnesis.extraction.calibration_fit.subspace_agreement`), which says how far
the components' *order* is stable — less than what is determined, because
components of nearly equal variance trade ranks between fits.

The sequences are the ones the calibration was fitted over — the replay manifest
written beside its basis — so nothing is sampled here: each is replayed once and split
one of two ways. ``--split prompts`` puts even-indexed prompts in one half and odd in
the other, so the halves differ in content and agreement measures what the ruler
determines. ``--split positions`` gives both halves every prompt, alternating its
sampled positions, so the halves share their content and agreement measures sampling
noise alone. A basis that agrees under the second and not the first is a property of
which prompts it was fitted over. Half the samples understates what the whole set
determines, so the counts reported are floors.

Writes a JSON receipt with, per layer, the agreement curves and the determined count
at the stated threshold, and prints those counts as a ``pca_components_by_layer`` entry
for the model's registry row. A floor measured on halves is a conservative count; the
row is where it is decided, so the extraction's schema moves only when the row does.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from anamnesis.config import preset_names, resolve_preset
from anamnesis.extraction import calibration_fit
from anamnesis.extraction.calibration import (
    PCA_MODEL_NAME,
    load_positional_means,
)
from anamnesis.extraction.replay.manifest import load_replay_manifest

logger = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=list(preset_names()), required=True)
    p.add_argument("--model-path", required=True, help="Local checkpoint directory")
    p.add_argument(
        "--calib-dir", type=Path, required=True,
        help="A calibration directory holding positional means and the replay manifest "
             "its basis was fitted over",
    )
    p.add_argument("--pca-name", default=PCA_MODEL_NAME, help="The basis whose manifest to read")
    p.add_argument("--steps-per-prompt", type=int, default=calibration_fit.BASIS_STEPS_PER_PROMPT)
    p.add_argument("--n-components", type=int, default=None, help="Default: the extraction default")
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--split", choices=("prompts", "positions"), default="prompts")
    p.add_argument("--json", type=Path, required=True, help="Where the receipt is written")
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args(argv)
    preset = resolve_preset(args.model)
    means = load_positional_means(args.calib_dir)
    if means is None:
        raise SystemExit(f"no positional means in {args.calib_dir}")
    manifest_path = calibration_fit.tokens_path(args.calib_dir / args.pca_name)
    try:
        manifest = load_replay_manifest(manifest_path)
    except FileNotFoundError as exc:
        raise SystemExit(f"{exc}; a calibration written before its sequences were kept has none") from exc

    from anamnesis.config import ExtractionConfig, ModelConfig
    from anamnesis.extraction.model_loader import load_model

    n_components = args.n_components or ExtractionConfig.from_preset(preset).pca_components
    loaded = load_model(
        ModelConfig.from_preset(preset, model_id=args.model_path), sampled_layers=[]
    )
    loaded.disable_hooks()
    halves: tuple[list, list] = ([], [])
    try:
        for index, prompt in zip(
            manifest.gen_ids(), calibration_fit.replay_prompt_states(loaded, manifest)
        ):
            samples = calibration_fit.basis_samples(
                prompt, preset.pca_layers, False, args.steps_per_prompt
            )
            if args.split == "prompts":
                halves[index % 2].extend(samples)
            else:
                positions = sorted({absolute for _, absolute, _ in samples})
                side = {absolute: rank % 2 for rank, absolute in enumerate(positions)}
                for sample in samples:
                    halves[side[sample[1]]].append(sample)
    finally:
        loaded.remove_hooks()

    fits = [
        calibration_fit.fit_per_layer_basis(half, means, preset.pca_layers, n_components)
        for half in halves
    ]
    layers = {}
    for layer in preset.pca_layers:
        curve = calibration_fit.subspace_agreement(
            fits[0][layer]["components"], fits[1][layer]["components"]
        )
        cosines = calibration_fit.principal_cosines(
            fits[0][layer]["components"], fits[1][layer]["components"]
        )
        layers[int(layer)] = dict(
            determined=calibration_fit.determined_components(cosines, args.threshold),
            agreement=[round(float(value), 6) for value in curve],
            principal_cosines=[round(float(value), 6) for value in cosines],
        )
        logger.info(
            f"layer {layer}: the halves share {layers[int(layer)]['determined']} of "
            f"{len(curve)} directions at cosine >= {args.threshold}"
        )
    receipt = dict(
        model=args.model,
        calib_dir=str(args.calib_dir),
        sequences=len(manifest.gen_ids()),
        steps_per_prompt=args.steps_per_prompt,
        samples_per_half={
            str(layer): [sum(1 for sample in half if sample[0] == layer) for half in halves]
            for layer in preset.pca_layers
        },
        n_components=n_components,
        threshold=args.threshold,
        split=args.split,
        layers=layers,
        suggested_pca_components_by_layer={
            str(layer): max(1, entry["determined"]) for layer, entry in layers.items()
        },
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"pca_components_by_layer": receipt["suggested_pca_components_by_layer"]}))


if __name__ == "__main__":
    main()
