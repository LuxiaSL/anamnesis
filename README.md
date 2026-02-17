# Semantically Independent Computational Signatures in Transformer Internal States

**Evidence from Tier Inversion, Mechanistic Dissociation, and Format-Controlled Generation in a 3B Model**

> Can transformer internal states fingerprint *how* something was processed, orthogonal to *what* was processed?

**Paper:** [arXiv preprint (coming soon)]()

---

## Abstract

We investigate whether transformer internal states carry structured information about *how* something is processed that is distinct from *what* is processed — and whether these two axes are separable. Using Llama 3.2 3B Instruct, we extract 1,837 features per generation across four tiers: logit statistics (T1), attention routing dynamics (T2), KV cache geometry (T2.5), and residual stream projections (T3). Across three experimental iterations with progressively stronger confound controls, we find that computational signatures of processing and semantic text features measure approximately orthogonal properties of generation: ridge regression from sentence-transformer embeddings to compute features yields median R² = -1.11, and adding semantic information to compute features does not improve mode discrimination (McNemar p = 1.000).

Three principal findings emerge. First, under format control — where five processing modes produce visually identical paragraph prose — all ten pairwise mode comparisons are distinguishable (72.5–100% binary accuracy), with 5-way classification at 70% (p < 0.001), 4-way excluding a trivially separable mode at 55% (p = 0.02), and topic-heldout evaluation at 78%. Second, discriminative signal concentrates in a specific architectural neighborhood: 366 features from attention routing and KV cache dynamics outperform all 1,837 combined features, with the dominant tier shifting from output-level statistics (T1) to cache dynamics (T2.5) as surface confounds are removed. Third, the signal is execution-based: prompt-swap texts produce computational signatures matching their executed processing, not their prompted mode (classification at chance).

---

## Reproduction

All analyses can be reproduced from stored artifacts without a GPU:

```bash
# Full reproduction pipeline
bash scripts/reproduce_all.sh

# Or step by step
export ANAMNESIS_RUN_NAME=run4_format_controlled
python scripts/run_analysis.py               # Main analysis + figures
python scripts/run_4way_no_analogical.py     # 4-way excluding analogical
python scripts/run_permutation_test.py       # 1000-permutation null (~18 min)
python scripts/run_cv_stability.py           # 100-seed CV distribution
python scripts/run_supplementary_analysis.py # Temperature + prompt-swap controls
python scripts/build_review_pack.py          # Consolidated data + surface baseline
```

To regenerate from scratch (requires GPU + ~2 hours):

```bash
bash scripts/run_all.sh --run-name run4_format_controlled
```

Calibration data (`outputs/calibration/`) is not included in the repository (164MB). Regenerate with:

```bash
python scripts/run_calibration.py
```

### Dependencies

```bash
pip install -e .
```

Requires Python 3.10+, PyTorch, HuggingFace Transformers, scikit-learn, numpy, and sentence-transformers. See `pyproject.toml` for full dependency list.

---

## Repository Structure

```
anamnesis/
├── extraction/                  # Core pipeline (3-layer separation)
│   ├── model_loader.py          #   Model loading + k_proj hooks + generate()
│   ├── state_extractor.py       #   Pure numpy feature extraction (no torch)
│   └── generation_runner.py     #   Orchestration: generate → collect → extract → save
├── analysis/                    # Analysis modules
│   ├── clustering.py
│   ├── distance_matrices.py
│   ├── feature_importance.py
│   ├── noise_floor.py
│   └── ...
├── scripts/                     # Runnable experiment + analysis scripts
│   ├── run_all.sh               #   Full pipeline
│   ├── reproduce_all.sh         #   Reproduction from artifacts
│   ├── run_experiment.py        #   Generation + feature extraction
│   ├── run_analysis.py          #   Main analysis
│   ├── run_contrastive_projection.py  # Signal geometry experiments
│   ├── run_tier_ablation.py
│   ├── run_semantic_disambiguation.py
│   └── ...
├── outputs/
│   ├── runs/                    # Per-run data (metadata, results, signatures)
│   │   ├── run2_epistemic_modes/    # Run 1 in paper
│   │   ├── run3_process_modes/      # Run 2 in paper
│   │   └── run4_format_controlled/  # Run 3 in paper (primary)
│   ├── phase1/                  # Signal geometry experiments
│   │   ├── contrastive_projection/
│   │   ├── cross_run_transfer/
│   │   └── tier_ablation/
│   ├── phase05/                 # Semantic disambiguation
│   └── paper_figures/           # Publication-ready figures
├── review_pack/                 # Consolidated review artifacts
│   ├── protocol.md              #   Experiment protocol
│   ├── samples.parquet          #   All samples with metadata
│   ├── features.npz             #   Feature matrix (N×1837) + labels
│   └── surface_baseline.json    #   TF-IDF confound check
├── research/                    # Paper, notes, and research record
│   └── phase0_report.md         #   Full technical report
├── config.py                    # Mode prompts, layer selections, feature config
├── prompts/prompt_sets.json     # Topic × mode prompt specifications
└── pyproject.toml
```

### Run Naming

The paper refers to experimental iterations as **Run 1, 2, 3**. These map to repository directories as follows:

| Paper name | Directory | Modes | Key property |
|------------|-----------|-------|-------------|
| Run 1 | `run2_epistemic_modes` | analytical, creative, uncertain, confident, emotional | Epistemic stance variation |
| Run 2 | `run3_process_modes` | structured, associative, deliberative, compressed, pedagogical | Process-prescriptive, format-free |
| Run 3 | `run4_format_controlled` | linear, analogical, socratic, contrastive, dialectical | Format-controlled (primary) |

The directory numbering reflects development history; the paper numbering reflects the progressive confound removal narrative.

---

## Key Results

| Metric | Value |
|--------|-------|
| 5-way RF (format-controlled) | 70%, p < 0.001 |
| 100-seed CV median | 63.7% |
| Topic-heldout (GroupKFold) | 78% |
| All 10 pairwise comparisons | 72.5–100% |
| T2+T2.5 alone vs. all 1,837 features | 73% vs. 63% |
| Semantic → compute R² | median -1.11 |
| Prompt-swap accuracy | 50% (chance) |
| Temperature → T1 accuracy | 90% |

---

## Citation

```
@misc{anamnesis2026,
  title={Semantically Independent Computational Signatures in Transformer Internal States},
  author={[Luxia]},
  year={2026},
  note={arXiv preprint (forthcoming)}
}
```

---

## License

[TBD]
