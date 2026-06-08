#!/usr/bin/env bash
# Phase 2 of 2 — run after emaskd training completes AND collect_baseline.sh has run.
# Uploads baseline artifacts to emaskd session, runs notebook, downloads outputs,
# commits to results/logs/<slug>/, stops session.
#
# Usage: ./colab/collect_results.sh <emaskd-session> <slug>
# Example: ./colab/collect_results.sh emaskd 2026-06-08-1430-emaskd-smoke
#
# Requires: /tmp/baseline-collect/ populated by collect_baseline.sh

set -e

SESSION="${1:?Usage: $0 <emaskd-session> <slug>}"
SLUG="${2:?Usage: $0 <emaskd-session> <slug>}"
REMOTE_DIR="/content/ema-skd/results"
LOCAL_DIR="results/logs/${SLUG}"
LOCAL_TMP="/tmp/baseline-collect"

echo "[collect_results] session=${SESSION}  slug=${SLUG}"

# 1. Verify baseline artifacts exist locally
if [ ! -f "${LOCAL_TMP}/baseline_log.txt" ]; then
    echo "ERROR: ${LOCAL_TMP}/baseline_log.txt not found"
    echo "Run collect_baseline.sh first."
    exit 1
fi

# 2. Copy emaskd log to results/ on remote
cat > /tmp/prep_emaskd_logs.py << 'PYEOF'
import glob, shutil, os
os.makedirs("/content/ema-skd/results", exist_ok=True)
logs = sorted(glob.glob("/content/ema-skd/models/*EHSKD_True*/log/log.txt"))
if logs:
    shutil.copy(logs[-1], "/content/ema-skd/results/emaskd_log.txt")
    print(f"emaskd log: {logs[-1]}")
else:
    print("WARNING: no emaskd log found"); exit(1)
PYEOF
colab --auth=adc exec -s "$SESSION" -f /tmp/prep_emaskd_logs.py

# 3. Upload baseline artifacts to emaskd session
echo "[collect_results] uploading baseline artifacts to session..."
colab --auth=adc upload -s "$SESSION" "${LOCAL_TMP}/baseline_log.txt"             "${REMOTE_DIR}/baseline_log.txt"
[ -f "${LOCAL_TMP}/baseline_report.json"         ] && colab --auth=adc upload -s "$SESSION" "${LOCAL_TMP}/baseline_report.json"         "${REMOTE_DIR}/baseline_report.json"
[ -f "${LOCAL_TMP}/baseline_gpu_timeseries.json" ] && colab --auth=adc upload -s "$SESSION" "${LOCAL_TMP}/baseline_gpu_timeseries.json" "${REMOTE_DIR}/baseline_gpu_timeseries.json"

# 4. Run analysis notebook
echo "[collect_results] running analyze_logs.ipynb..."
colab --auth=adc exec --timeout 120 -s "$SESSION" -f colab/analyze_logs.ipynb

# 5. Download all artifacts
mkdir -p "$LOCAL_DIR"
echo "[collect_results] downloading outputs..."
colab --auth=adc download -s "$SESSION" "${REMOTE_DIR}/eval_curves.png"                "${LOCAL_DIR}/eval_curves.png"
colab --auth=adc download -s "$SESSION" "${REMOTE_DIR}/gate_curriculum.png"             "${LOCAL_DIR}/gate_curriculum.png"
colab --auth=adc download -s "$SESSION" "${REMOTE_DIR}/train_dynamics.png"              "${LOCAL_DIR}/train_dynamics.png"
colab --auth=adc download -s "$SESSION" "${REMOTE_DIR}/gpu_timeseries.png"              "${LOCAL_DIR}/gpu_timeseries.png"              2>/dev/null || echo "[skip] gpu_timeseries.png"
colab --auth=adc download -s "$SESSION" "${REMOTE_DIR}/baseline_report.json"            "${LOCAL_DIR}/baseline_report.json"            2>/dev/null || echo "[skip] baseline_report.json"
colab --auth=adc download -s "$SESSION" "${REMOTE_DIR}/emaskd_report.json"              "${LOCAL_DIR}/emaskd_report.json"              2>/dev/null || echo "[skip] emaskd_report.json"
colab --auth=adc download -s "$SESSION" "${REMOTE_DIR}/baseline_gpu_timeseries.json"    "${LOCAL_DIR}/baseline_gpu_timeseries.json"    2>/dev/null || echo "[skip] baseline_gpu_timeseries.json"
colab --auth=adc download -s "$SESSION" "${REMOTE_DIR}/emaskd_gpu_timeseries.json"      "${LOCAL_DIR}/emaskd_gpu_timeseries.json"      2>/dev/null || echo "[skip] emaskd_gpu_timeseries.json"
cp colab/analyze_logs_output.ipynb "${LOCAL_DIR}/analyze_logs_output.ipynb"
echo "[collect_results] notebook copied to ${LOCAL_DIR}/"

# 6. Commit + push
git add "${LOCAL_DIR}/"
git commit -m "results: ${SLUG} — plots + executed notebook"
git push origin HEAD
echo "[collect_results] pushed to $(git branch --show-current)"

# 7. Stop session
colab --auth=adc stop -s "$SESSION"
echo "[collect_results] session '${SESSION}' stopped"
