"""Build, screen and gate a steering vector — the entry point for golden path 6.

The path is: contrast → vectors → screens → on-policy gate → lever readout →
judge. This script is every leg up to the judge, one subcommand each, and it is a
shim: it parses arguments and calls :mod:`anamnesis.steering`. Every decision
about what a construction means lives in the package, so a different front end
over the same capability cannot disagree with this one about the science. The
judge leg is :mod:`anamnesis.judging`, which lands separately.

``sweep``
    Per-layer held-out Cohen's d between two banks of per-prompt mean residuals.
    Read the site off the peak before building anything at it.

``build``
    A vector bank at chosen sites. ``--stage contrast`` is the paired
    contrastive-prompt and replay-contrast construction with isotropic controls
    and the dose currency; ``--stage whitened`` adds the ``Σ⁻¹Δ`` direction with
    its raw control at the same capture.

``screen``
    The residual covariance at each site, the Mahalanobis and eigenmass of every
    vector that lives there, and the band mass of a named vector. Banks the
    eigendecomposition so the band and null work runs later without a model.

``gate``
    The matched-token on-policy pilot, with its ``alpha = 0`` baseline. A cell
    that fails here does not enter a grid.

``null``
    A vector's spectral position against the null for its own construction, from
    a banked eigendecomposition. Refuses, rather than answers, where no matched
    null exists.

``lever``
    What the grid did: movement along the axis, movement off it, and the lever
    ratio against the random controls at each dose. The last leg before the judge.

Sites, doses and the covariance are per-model throughout. A bank built for one
checkpoint says so in its stamps and is not injectable into another.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from anamnesis.config import MODEL_PRESETS
from anamnesis.steering import gates, readouts, screens, vectors

logger = logging.getLogger("steer_vectors")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sweep = sub.add_parser("sweep", help="per-layer held-out Cohen's d; read the site off the peak")
    sweep.add_argument("--positive-means", type=Path, required=True,
                       help="npz with 'means' [n, L, d] and 'prompt_ids'")
    sweep.add_argument("--negative-means", type=Path, required=True)
    sweep.add_argument("--k-splits", type=int, default=50)
    sweep.add_argument("--out-json", type=Path, required=True)

    build = sub.add_parser("build", help="bank steering vectors at the chosen sites")
    build.add_argument("--model", choices=sorted(MODEL_PRESETS), required=True)
    build.add_argument("--model-path", required=True)
    build.add_argument("--sites", required=True, help="comma-separated injection layer indices")
    build.add_argument("--stage", choices=("contrast", "whitened"), default="contrast")
    build.add_argument("--floor-run", type=Path, required=True,
                       help="run whose replay manifest gives the dose currency")
    build.add_argument("--runs-root", type=Path, required=True)
    build.add_argument("--positive-run", required=True, help="run directory name of the + corpus")
    build.add_argument("--negative-run", required=True, help="run directory name of the − corpus")
    build.add_argument("--topics", type=Path, default=None,
                       help="JSON list of topics; enables the contrastive-prompt construction")
    build.add_argument("--contrast-max-new-tokens", type=int, default=160)
    build.add_argument("--limit", type=int, default=None, help="cap generations per corpus")
    build.add_argument("--shrink-scale", type=float, default=None,
                       help="scale the automatic Ledoit-Wolf shrinkage (whitened stage)")
    build.add_argument("--out-dir", type=Path, required=True)

    screen = sub.add_parser("screen", help="covariance screen, band mass, and a banked eigenbasis")
    screen.add_argument("--model", choices=sorted(MODEL_PRESETS), required=True)
    screen.add_argument("--model-path", required=True)
    screen.add_argument("--floor-run", type=Path, required=True)
    screen.add_argument("--vectors", type=Path, required=True, help="vector bank directory")
    screen.add_argument("--sites", required=True)
    screen.add_argument("--n-gens", type=int, default=60)
    screen.add_argument("--ridge-rel", type=float, default=screens.DEFAULT_RIDGE_REL)
    screen.add_argument("--band", default="16,256")
    screen.add_argument("--band-mass-key", default=None,
                        help="also report the band mass of this bank key")
    screen.add_argument("--out-dir", type=Path, required=True)

    gate = sub.add_parser("gate", help="the matched-token on-policy pilot (bar 0.85)")
    gate.add_argument("--model", choices=sorted(MODEL_PRESETS), required=True)
    gate.add_argument("--model-path", required=True)
    gate.add_argument("--floor-run", type=Path, required=True)
    gate.add_argument("--vectors", type=Path, required=True, help="vector bank directory")
    gate.add_argument("--map-site", type=int, required=True)
    gate.add_argument("--alpha-fractions", default="0.03,0.1")
    gate.add_argument("--n-pilot", type=int, default=20)
    gate.add_argument("--gate-bar", type=float, default=gates.GATE_BAR)
    gate.add_argument("--out-json", type=Path, required=True)

    null = sub.add_parser("null", help="a spectral position against its own construction's null")
    null.add_argument("--sigma", type=Path, required=True, help="banked eigendecomposition npz")
    null.add_argument("--vectors", type=Path, required=True, help="vector bank directory")
    null.add_argument("--key", required=True)
    null.add_argument("--construction", required=True,
                      choices=sorted(gates.NULL_AVAILABILITY), help="how the vector was built")
    null.add_argument("--k", type=int, default=256)
    null.add_argument("--n-draws", type=int, default=200)
    null.add_argument("--out-json", type=Path, default=None)

    lever = sub.add_parser("lever", help="target shift, off-target movement and the lever ratio")
    lever.add_argument("--run-dir", type=Path, required=True, help="root of the steered cells")
    lever.add_argument("--pole-a-dir", type=Path, required=True)
    lever.add_argument("--pole-b-dir", type=Path, required=True)
    lever.add_argument("--floor-dir", type=Path, required=True, help="the model's floor signatures")
    lever.add_argument("--pole-a-name", default="pole_a")
    lever.add_argument("--map-site", type=int, required=True)
    lever.add_argument("--baseline-cell", default=None,
                       help="unsteered cell the displacements are measured from")
    lever.add_argument("--lever-vector", default="V3")
    lever.add_argument("--sig-subdir", default="signatures_v3")
    lever.add_argument("--out-json", type=Path, required=True)
    return p


def capture_model(args: argparse.Namespace) -> object:
    """The model a capture leg reads residuals out of, in its preset's dtype."""
    from anamnesis.extraction.model_loader import load_unhooked_model

    return load_unhooked_model(args.model_path, MODEL_PRESETS[args.model].torch_dtype)


def run_sweep(args: argparse.Namespace) -> None:
    positive = np.load(args.positive_means, allow_pickle=True)
    negative = np.load(args.negative_means, allow_pickle=True)
    pos = vectors.per_prompt_average(positive["means"], [str(x) for x in positive["prompt_ids"]])
    neg = vectors.per_prompt_average(negative["means"], [str(x) for x in negative["prompt_ids"]])
    A, B, shared = vectors.pair_on_prompts(pos, neg)
    d_mean, d_sd = vectors.half_split_sweep(A, B, k_splits=args.k_splits)
    peak = int(np.argmax(np.abs(d_mean)))
    logger.info(f"peak L{peak} ({100 * peak / len(d_mean):.0f}% depth) d={d_mean[peak]:+.3f}")
    readouts.write_json(args.out_json, {
        "n_prompts": len(shared), "k_splits": int(args.k_splits),
        "d_mean": d_mean.tolist(), "d_sd": d_sd.tolist(),
        "peak_layer": peak, "peak_d": float(d_mean[peak]),
        "law": (
            "held-out Cohen's d along the training-half mean difference, per-prompt averaged, "
            "K random half-splits over prompts"
        ),
    })


def run_build(args: argparse.Namespace) -> None:
    preset = MODEL_PRESETS[args.model]
    sites = [int(s) for s in args.sites.split(",")]
    model = capture_model(args)
    bank: dict[str, object] = {}
    stamps: dict[str, object] = {"model": args.model, "sites": sites}

    floor_entries = vectors.replay_entries(args.floor_run / "replay_manifest.json")
    stamps["median_resid_norms"] = vectors.capture_median_residual_norms(model, floor_entries, sites)

    pos_dir = args.runs_root / args.positive_run
    neg_dir = args.runs_root / args.negative_run
    if args.stage == "contrast":
        bank.update(vectors.random_unit_vectors(int(preset.hidden_dim)))
        stamps.update({
            key: {"trait": "random", "seed": vectors.RANDOM_SEED} for key in ("R1", "R2", "R3")
        })
        if args.topics is not None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            topics = json.loads(args.topics.read_text())
            built, built_stamps = vectors.build_contrastive_prompt_vectors(
                model, tokenizer, topics, sites,
                max_new_tokens=args.contrast_max_new_tokens,
                temperature=float(preset.temperature),
                eos_token_ids=tuple(preset.eos_token_ids),
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else preset.eos_token_ids[0]
                ),
            )
            bank.update(built)
            stamps.update(built_stamps)
        built, built_stamps = vectors.build_replay_contrast_vectors(
            model,
            {
                args.positive_run: (pos_dir / "replay_manifest.json", pos_dir / "metadata.json"),
                args.negative_run: (neg_dir / "replay_manifest.json", neg_dir / "metadata.json"),
            },
            (args.positive_run, args.negative_run),
            sites,
        )
        bank.update(built)
        stamps.update(built_stamps)
    else:
        pos = vectors.capture_mean_residuals(
            model, vectors.replay_entries(pos_dir / "replay_manifest.json"), sites, args.limit
        )
        neg = vectors.capture_mean_residuals(
            model, vectors.replay_entries(neg_dir / "replay_manifest.json"), sites, args.limit
        )
        built, diagnostics = vectors.build_whitened_vectors(pos, neg, shrink_scale=args.shrink_scale)
        bank.update(built)
        dim = int(np.asarray(next(iter(built.values()))).shape[0])
        bank.update(vectors.random_unit_vectors(dim))
        stamps["diagnostics"] = diagnostics
        stamps["pair"] = [args.positive_run, args.negative_run]

    vectors.save_vector_bank(args.out_dir, bank, stamps)


def run_screen(args: argparse.Namespace) -> None:
    sites = [int(s) for s in args.sites.split(",")]
    band = tuple(int(x) for x in args.band.split(","))
    model = capture_model(args)
    entries = vectors.replay_entries(args.floor_run / "replay_manifest.json")
    all_ids = sorted(int(k) for k in entries)
    gen_ids = all_ids[:: max(1, len(all_ids) // max(args.n_gens, 1))][: args.n_gens]
    bank, _stamps = vectors.load_vector_bank(args.vectors)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"model": args.model, "n_gens": len(gen_ids), "sites": {}}
    for site in sites:
        rows = screens.capture_site_inputs(model, entries, gen_ids, site)
        spectrum = screens.Spectrum.from_rows(rows, ridge_rel=args.ridge_rel)
        sigma_path = args.out_dir / f"sigma_L{site}_{args.model}.npz"
        np.savez(
            sigma_path, evals=spectrum.evals, evecs=spectrum.evecs,
            mean=rows.mean(axis=0), ridge=np.float64(spectrum.ridge),
            n_positions=np.int64(len(rows)),
        )
        site_report: dict[str, object] = {
            "n_positions": int(len(rows)),
            "mean_eigenvalue": float(spectrum.evals.mean()),
            "ridge": spectrum.ridge,
            "median_site_input_norm": vectors.median_row_norm(rows),
            "sigma": str(sigma_path),
            "vectors": screens.screen_bank(bank, spectrum, site),
        }
        if args.band_mass_key and args.band_mass_key in bank:
            site_report["band_mass"] = screens.band_mass(bank[args.band_mass_key], spectrum, band)
        report["sites"][str(site)] = site_report
        logger.info(f"L{site}: screened {len(site_report['vectors'])} vectors over {len(rows)} positions")
    readouts.write_json(args.out_dir / f"covariance_screen_{args.model}.json", report)


def run_gate(args: argparse.Namespace) -> None:
    model = capture_model(args)
    bank, stamps = vectors.load_vector_bank(args.vectors)
    entries = vectors.replay_entries(args.floor_run / "replay_manifest.json")
    pilots = gates.pilot_generations(entries, count=args.n_pilot)
    report = gates.on_policy_gate(
        model, bank, stamps["median_resid_norms"], pilots,
        map_site=args.map_site,
        alpha_fractions=[float(x) for x in args.alpha_fractions.split(",")],
        gate_bar=args.gate_bar,
    )
    report["model"] = args.model
    passed = sum(1 for cell in report["cells"].values() if cell["PASS"])
    logger.info(f"{passed}/{len(report['cells'])} cells PASS")
    readouts.write_json(args.out_json, report)


def run_null(args: argparse.Namespace) -> None:
    spectrum = gates.Spectrum.from_npz(args.sigma)
    bank, _stamps = vectors.load_vector_bank(args.vectors)
    vector = vectors.load_vector(bank, args.key, str(args.vectors))
    verdict = gates.assert_against_own_null(
        vector, args.construction, spectrum, k=args.k, n_draws=args.n_draws
    )
    print(verdict)
    if args.out_json is not None:
        readouts.write_json(args.out_json, {"key": args.key, **verdict.__dict__})


def run_lever(args: argparse.Namespace) -> None:
    report = readouts.lever_readout(
        args.run_dir, args.pole_a_dir, args.pole_b_dir, args.floor_dir,
        map_site=args.map_site, baseline_cell=args.baseline_cell,
        lever_vector=args.lever_vector, sig_subdir=args.sig_subdir,
        pole_a_name=args.pole_a_name,
    )
    for name, row in report["lever_ratio_by_dose"].items():
        logger.info(f"{name}: lever_ratio={row['lever_ratio']}")
    readouts.write_json(args.out_json, report)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args(argv)
    {
        "sweep": run_sweep, "build": run_build, "screen": run_screen,
        "gate": run_gate, "null": run_null, "lever": run_lever,
    }[args.command](args)


if __name__ == "__main__":
    main()
