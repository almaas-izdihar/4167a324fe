#!/usr/bin/env bash
# Full experiment run — sequential sessions, no manual steps.
# Session lifecycle: baseline → emaskd → analysis (each start/stop in order)
#
# Usage: ./colab/run_experiment.sh smoke|full [slug-label]
# Examples:
#   ./colab/run_experiment.sh smoke
#   ./colab/run_experiment.sh full 200ep-resnet18
#
# Slug auto-generated: YYYY-MM-DD-HHMM-<label>

set -e

MODE="${1:?Usage: $0 smoke|full [slug-label]}"
LABEL="${2:-${MODE}}"
SLUG="$(date +%Y-%m-%d-%H%M)-${LABEL}"

if [[ "$MODE" == "smoke" ]]; then
    BASELINE_SCRIPT="colab/colab_baseline_smoke.py"
    EMASKD_SCRIPT="colab/colab_emaskd_smoke.py"
    BASELINE_TIMEOUT=600
    EMASKD_TIMEOUT=600
elif [[ "$MODE" == "full" ]]; then
    BASELINE_SCRIPT="colab/colab_baseline.py"
    EMASKD_SCRIPT="colab/colab_emaskd.py"
    BASELINE_TIMEOUT=18000
    EMASKD_TIMEOUT=36000
else
    echo "ERROR: mode must be 'smoke' or 'full'"; exit 1
fi

echo "========================================"
echo " EMA-SKD Experiment Run"
echo " mode   : ${MODE}"
echo " slug   : ${SLUG}"
echo " started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# ── Session 1: Baseline ───────────────────
echo ""
echo "[1/3] BASELINE SESSION"
colab --auth=adc new --gpu T4 -s baseline
colab --auth=adc exec --timeout "$BASELINE_TIMEOUT" -s baseline -f "$BASELINE_SCRIPT"
bash colab/collect_baseline.sh baseline
colab --auth=adc stop -s baseline
echo "[1/3] baseline done"

# ── Session 2: EMA-SKD ────────────────────
echo ""
echo "[2/3] EMASKD SESSION"
colab --auth=adc new --gpu T4 -s emaskd
colab --auth=adc exec --timeout "$EMASKD_TIMEOUT" -s emaskd -f "$EMASKD_SCRIPT"
bash colab/collect_emaskd.sh emaskd
colab --auth=adc stop -s emaskd
echo "[2/3] emaskd done"

# ── Session 3: Analysis ───────────────────
echo ""
echo "[3/3] ANALYSIS SESSION"
bash colab/collect_results.sh "$SLUG"
colab --auth=adc stop -s analysis
echo "[3/3] analysis done"

echo ""
echo "========================================"
echo " All done: ${SLUG}"
echo " finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo " results : results/logs/${SLUG}/"
echo "========================================"
