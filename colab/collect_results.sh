#!/usr/bin/env bash
# Phase 3 of 3 — run after collect_baseline.sh and collect_emaskd.sh.
# Provisions a fresh analysis session, uploads both artifact sets,
# runs notebook, downloads outputs, commits to results/logs/<slug>/.
# Does NOT stop the analysis session — caller stops it after verifying outputs.
#
# Usage: ./colab/collect_results.sh <slug>
# Example: ./colab/collect_results.sh 2026-06-08-1430-smoke
#
# Requires:
#   /tmp/baseline-collect/  populated by collect_baseline.sh
#   /tmp/emaskd-collect/    populated by collect_emaskd.sh

set -e

SLUG="${1:?Usage: $0 <slug>}"
ANALYSIS_SESSION="analysis"
REMOTE_DIR="/content/ema-skd/results"
LOCAL_DIR="results/logs/${SLUG}"
BASELINE_TMP="/tmp/baseline-collect"
EMASKD_TMP="/tmp/emaskd-collect"
REPO="https://github.com/almaas-izdihar/4167a324fe"
BRANCH="experiment/confidence-filter"

echo "[collect_results] slug=${SLUG}"

# 0. Verify artifacts exist locally
for f in "${BASELINE_TMP}/baseline_log.txt" "${EMASKD_TMP}/emaskd_log.txt"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found — run collect_baseline.sh and collect_emaskd.sh first"
        exit 1
    fi
done

# 1. Provision fresh analysis session
echo "[collect_results] provisioning analysis session..."
colab --auth=adc new --gpu T4 -s "$ANALYSIS_SESSION"

# 2. Clone repo on analysis session (sets up dir structure for notebook)
cat > /tmp/setup_analysis.py << PYEOF
import subprocess, os
REPO   = "${REPO}"
BRANCH = "${BRANCH}"
DIR    = "/content/ema-skd"
if not os.path.exists(DIR):
    subprocess.run(f"git clone {REPO} {DIR}", shell=True, check=True)
subprocess.run(f"git -C {DIR} fetch origin {BRANCH}", shell=True, check=True)
subprocess.run(f"git -C {DIR} checkout {BRANCH}", shell=True, check=True)
subprocess.run(f"git -C {DIR} pull origin {BRANCH}", shell=True, check=True)
os.makedirs(f"{DIR}/results", exist_ok=True)
print(f"repo ready at {DIR}")
PYEOF
colab --auth=adc exec -s "$ANALYSIS_SESSION" -f /tmp/setup_analysis.py

# 3. Upload all artifacts to analysis session
echo "[collect_results] uploading artifacts to analysis session..."
colab --auth=adc upload -s "$ANALYSIS_SESSION" "${BASELINE_TMP}/baseline_log.txt"             "${REMOTE_DIR}/baseline_log.txt"
colab --auth=adc upload -s "$ANALYSIS_SESSION" "${EMASKD_TMP}/emaskd_log.txt"                 "${REMOTE_DIR}/emaskd_log.txt"
[ -f "${BASELINE_TMP}/baseline_report.json"         ] && colab --auth=adc upload -s "$ANALYSIS_SESSION" "${BASELINE_TMP}/baseline_report.json"         "${REMOTE_DIR}/baseline_report.json"
[ -f "${BASELINE_TMP}/baseline_gpu_timeseries.json" ] && colab --auth=adc upload -s "$ANALYSIS_SESSION" "${BASELINE_TMP}/baseline_gpu_timeseries.json" "${REMOTE_DIR}/baseline_gpu_timeseries.json"
[ -f "${EMASKD_TMP}/emaskd_report.json"             ] && colab --auth=adc upload -s "$ANALYSIS_SESSION" "${EMASKD_TMP}/emaskd_report.json"             "${REMOTE_DIR}/emaskd_report.json"
[ -f "${EMASKD_TMP}/emaskd_gpu_timeseries.json"     ] && colab --auth=adc upload -s "$ANALYSIS_SESSION" "${EMASKD_TMP}/emaskd_gpu_timeseries.json"     "${REMOTE_DIR}/emaskd_gpu_timeseries.json"

# 4. Run analysis notebook
echo "[collect_results] running analyze_logs.ipynb..."
colab --auth=adc exec --timeout 120 -s "$ANALYSIS_SESSION" -f colab/analyze_logs.ipynb

# 5. Download all outputs
mkdir -p "$LOCAL_DIR"
echo "[collect_results] downloading outputs..."
colab --auth=adc download -s "$ANALYSIS_SESSION" "${REMOTE_DIR}/eval_curves.png"                "${LOCAL_DIR}/eval_curves.png"
colab --auth=adc download -s "$ANALYSIS_SESSION" "${REMOTE_DIR}/gate_curriculum.png"             "${LOCAL_DIR}/gate_curriculum.png"
colab --auth=adc download -s "$ANALYSIS_SESSION" "${REMOTE_DIR}/train_dynamics.png"              "${LOCAL_DIR}/train_dynamics.png"
colab --auth=adc download -s "$ANALYSIS_SESSION" "${REMOTE_DIR}/gpu_timeseries.png"              "${LOCAL_DIR}/gpu_timeseries.png"              2>/dev/null || echo "[skip] gpu_timeseries.png"
colab --auth=adc download -s "$ANALYSIS_SESSION" "${REMOTE_DIR}/baseline_report.json"            "${LOCAL_DIR}/baseline_report.json"            2>/dev/null || echo "[skip] baseline_report.json"
colab --auth=adc download -s "$ANALYSIS_SESSION" "${REMOTE_DIR}/emaskd_report.json"              "${LOCAL_DIR}/emaskd_report.json"              2>/dev/null || echo "[skip] emaskd_report.json"
colab --auth=adc download -s "$ANALYSIS_SESSION" "${REMOTE_DIR}/baseline_gpu_timeseries.json"    "${LOCAL_DIR}/baseline_gpu_timeseries.json"    2>/dev/null || echo "[skip] baseline_gpu_timeseries.json"
colab --auth=adc download -s "$ANALYSIS_SESSION" "${REMOTE_DIR}/emaskd_gpu_timeseries.json"      "${LOCAL_DIR}/emaskd_gpu_timeseries.json"      2>/dev/null || echo "[skip] emaskd_gpu_timeseries.json"
cp colab/analyze_logs_output.ipynb "${LOCAL_DIR}/analyze_logs_output.ipynb"
echo "[collect_results] notebook copied to ${LOCAL_DIR}/"

# 6. Commit + push
git add "${LOCAL_DIR}/"
git commit -m "results: ${SLUG} — plots + executed notebook"
git push origin HEAD
echo "[collect_results] pushed to $(git branch --show-current)"
echo "[collect_results] done — verify results/logs/${SLUG}/, then stop analysis session manually"
