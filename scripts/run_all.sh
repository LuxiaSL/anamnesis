#!/usr/bin/env bash
# Full pipeline: calibration → smoke test → experiment → analysis
# Usage: bash scripts/run_all.sh [--run-name NAME] [--skip-calibration] [--skip-clean]
#
# Run name controls output directory: outputs/runs/<NAME>/
# Calibration data is shared across runs in outputs/calibration/
#
# All output goes to /tmp/anamnesis_pipeline.log
# Each step also gets its own log in /tmp/

set -euo pipefail

LOGDIR="/tmp"
MASTER_LOG="$LOGDIR/anamnesis_pipeline.log"
PROJECT_DIR="/anamnesis"

cd "$PROJECT_DIR"

# Parse flags
SKIP_CALIB=false
SKIP_CLEAN=false
RUN_NAME=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-calibration) SKIP_CALIB=true; shift ;;
        --skip-clean) SKIP_CLEAN=true; shift ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# Export run name so config.py picks it up
if [ -n "$RUN_NAME" ]; then
    export ANAMNESIS_RUN_NAME="$RUN_NAME"
fi

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" | tee -a "$MASTER_LOG"
}

run_step() {
    local name="$1"
    local cmd="$2"
    local step_log="$LOGDIR/${name}.log"

    log "=== START: $name ==="
    if eval "$cmd" > "$step_log" 2>&1; then
        log "=== DONE: $name (see $step_log) ==="
        return 0
    else
        log "=== FAILED: $name (see $step_log) ==="
        log "Last 20 lines of $step_log:"
        tail -20 "$step_log" | tee -a "$MASTER_LOG"
        return 1
    fi
}

# ── Start ──
echo "" > "$MASTER_LOG"
log "Pipeline started"
log "Run name: ${ANAMNESIS_RUN_NAME:-$(python3 -c 'from config import RUN_NAME; print(RUN_NAME)')}"
log "Flags: skip_calib=$SKIP_CALIB, skip_clean=$SKIP_CLEAN"

# ── 0. Git pull ──
log "Pulling latest code..."
git pull | tee -a "$MASTER_LOG"

# ── 1. Clean old outputs for this run ──
if [ "$SKIP_CLEAN" = false ]; then
    log "Cleaning old run outputs..."
    # Only clean the run-specific directory, not calibration
    RUN_DIR="outputs/runs/${ANAMNESIS_RUN_NAME:-run3_process_modes}"
    rm -rf "$RUN_DIR"
    log "Clean done (removed $RUN_DIR)"
else
    log "Skipping clean (--skip-clean)"
fi

# ── 2. Calibration ──
if [ "$SKIP_CALIB" = false ]; then
    run_step "calibration" "uv run python scripts/run_calibration.py"
else
    log "Skipping calibration (--skip-calibration)"
fi

# ── 3. Smoke test ──
run_step "smoke_test" "uv run python scripts/smoke_test.py"

# ── 4. Set D (positive control, quick validation) ──
run_step "experiment_setD" "uv run python scripts/run_experiment.py --set D"

# ── 5. Full experiment (Sets A, B, C — skips already-done D) ──
run_step "experiment_full" "uv run python scripts/run_experiment.py"

# ── 6. Analysis ──
run_step "analysis" "uv run python scripts/run_analysis.py"

# ── Summary ──
log "=========================================="
log "Pipeline complete!"
log "=========================================="

# Print key results if available
RESULTS_PATH="outputs/runs/${ANAMNESIS_RUN_NAME:-run3_process_modes}/results.json"
if [ -f "$RESULTS_PATH" ]; then
    log "Key results:"
    uv run python -c "
import json
with open('$RESULTS_PATH') as f:
    r = json.load(f)
d = r.get('decision', {})
print(f\"  Verdict: {d.get('verdict', 'N/A')}\")
print(f\"  Mode silhouette: {d.get('mode_silhouette', 'N/A')}\")
print(f\"  Topic silhouette: {d.get('topic_silhouette', 'N/A')}\")
print(f\"  Mantel r: {d.get('mantel_r', 'N/A')}\")
gl = r.get('generation_lengths', {})
pm = gl.get('per_mode', {})
if pm:
    for mode, stats in sorted(pm.items()):
        print(f\"  {mode}: mean={stats['mean']:.0f} tokens, range=[{stats['min']}, {stats['max']}]\")
print(f\"  Justification: {d.get('justification', 'N/A')}\")
" 2>&1 | tee -a "$MASTER_LOG"
fi

log "All logs in $LOGDIR: calibration.log, smoke_test.log, experiment_setD.log, experiment_full.log, analysis.log"
