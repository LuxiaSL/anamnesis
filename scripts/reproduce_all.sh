#!/usr/bin/env bash
# reproduce_all.sh — Regenerate all Phase 0 results from existing artifacts.
#
# This script reproduces every analysis result, figure, and confound check
# from the stored signatures and metadata. No GPU needed.
#
# Usage:
#   bash scripts/reproduce_all.sh                          # defaults to run4_format_controlled
#   bash scripts/reproduce_all.sh --run-name run3_process_modes
#   bash scripts/reproduce_all.sh --skip-main              # skip the slow main analysis
#   bash scripts/reproduce_all.sh --skip-permutation       # skip the ~18min permutation test
#
# Prerequisites:
#   pip install numpy scipy scikit-learn pandas pyarrow matplotlib sentence-transformers

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Defaults ──────────────────────────────────────────────────────────────────
RUN_NAME="${ANAMNESIS_RUN_NAME:-run4_format_controlled}"
SKIP_MAIN=false
SKIP_PERMUTATION=false

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --skip-main)
            SKIP_MAIN=true
            shift
            ;;
        --skip-permutation)
            SKIP_PERMUTATION=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--run-name NAME] [--skip-main] [--skip-permutation]"
            echo ""
            echo "Reproduces all analysis results from stored artifacts."
            echo "Defaults to run4_format_controlled."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

export ANAMNESIS_RUN_NAME="$RUN_NAME"

echo "============================================================"
echo "  Anamnesis Phase 0 — Full Reproduction"
echo "  Run: $RUN_NAME"
echo "  Data: $PROJECT_DIR/outputs/runs/$RUN_NAME/"
echo "============================================================"
echo ""

# Verify data exists
DATA_DIR="$PROJECT_DIR/outputs/runs/$RUN_NAME"
if [ ! -d "$DATA_DIR/signatures" ]; then
    echo "ERROR: No signatures found at $DATA_DIR/signatures"
    echo "       Run the experiment first, or check the run name."
    exit 1
fi

NSIGS=$(ls "$DATA_DIR/signatures/"gen_*.npz 2>/dev/null | wc -l)
echo "Found $NSIGS signature files."
echo ""

# Track timing
SECONDS=0

# ── Step 1: Main analysis ────────────────────────────────────────────────────
if [ "$SKIP_MAIN" = false ]; then
    echo "── Step 1/6: Main analysis (noise floor, distances, clustering, retrieval, features) ──"
    python "$SCRIPT_DIR/run_analysis.py"
    echo ""
else
    echo "── Step 1/6: Main analysis — SKIPPED ──"
    echo ""
fi

# ── Step 2: 4-way excluding analogical ───────────────────────────────────────
echo "── Step 2/6: 4-way analysis (excluding analogical) ──"
python "$SCRIPT_DIR/run_4way_no_analogical.py"
echo ""

# ── Step 3: Permutation test ─────────────────────────────────────────────────
if [ "$SKIP_PERMUTATION" = false ]; then
    echo "── Step 3/6: Permutation test (1000 shuffles, ~18 min) ──"
    python "$SCRIPT_DIR/run_permutation_test.py"
    echo ""
else
    echo "── Step 3/6: Permutation test — SKIPPED (use --skip-permutation to skip) ──"
    echo ""
fi

# ── Step 4: CV stability ─────────────────────────────────────────────────────
echo "── Step 4/6: CV stability (100 seeds) ──"
python "$SCRIPT_DIR/run_cv_stability.py"
echo ""

# ── Step 5: Supplementary analysis ───────────────────────────────────────────
SUPP_META="$DATA_DIR/supplementary_metadata.json"
if [ -f "$SUPP_META" ]; then
    echo "── Step 5/6: Supplementary analysis (temperature + prompt-swap) ──"
    python "$SCRIPT_DIR/run_supplementary_analysis.py"
    echo ""
else
    echo "── Step 5/6: Supplementary analysis — SKIPPED (no supplementary_metadata.json) ──"
    echo ""
fi

# ── Step 6: Build review pack ────────────────────────────────────────────────
echo "── Step 6/6: Building review pack (parquet, npz, surface baseline) ──"
python "$SCRIPT_DIR/build_review_pack.py"
echo ""

# ── Summary ──────────────────────────────────────────────────────────────────
ELAPSED=$SECONDS
MINS=$((ELAPSED / 60))
SECS=$((ELAPSED % 60))

echo "============================================================"
echo "  Reproduction complete in ${MINS}m ${SECS}s"
echo "============================================================"
echo ""
echo "Outputs:"
echo "  Analysis:     $DATA_DIR/results.json"
echo "  Figures:      $DATA_DIR/figures/"
echo "  4-way:        $DATA_DIR/4way_no_analogical.json"
echo "  Permutation:  $DATA_DIR/permutation_test.json"
echo "  CV stability: $DATA_DIR/cv_stability.json"
echo "  Supplementary:$DATA_DIR/supplementary_analysis.json"
echo ""
echo "Review pack:    $PROJECT_DIR/review_pack/"
echo "  samples.parquet    — consolidated samples"
echo "  features.npz       — feature matrix + labels"
echo "  surface_baseline.json — TF-IDF confound check"
echo "  protocol.md        — experiment protocol"
