#!/usr/bin/env bash
# Phase 1 of 2 — run after baseline training completes.
# Downloads baseline artifacts to /tmp/baseline-collect/, stops session.
# Must run before collect_results.sh (phase 2).
#
# Usage: ./colab/collect_baseline.sh <session>
# Example: ./colab/collect_baseline.sh baseline

set -e

SESSION="${1:?Usage: $0 <session>}"
REMOTE_DIR="/content/ema-skd/results"
LOCAL_TMP="/tmp/baseline-collect"

echo "[collect_baseline] session=${SESSION}"

# 1. Copy baseline log to results/ on remote
cat > /tmp/prep_baseline_logs.py << 'PYEOF'
import glob, shutil, os
os.makedirs("/content/ema-skd/results", exist_ok=True)
logs = sorted(glob.glob("/content/ema-skd/models/*EHSKD_False*/log/log.txt"))
if logs:
    shutil.copy(logs[-1], "/content/ema-skd/results/baseline_log.txt")
    print(f"baseline log: {logs[-1]}")
else:
    print("WARNING: no baseline log found"); exit(1)
PYEOF
colab --auth=adc exec -s "$SESSION" -f /tmp/prep_baseline_logs.py

# 2. Download artifacts
mkdir -p "$LOCAL_TMP"
colab --auth=adc download -s "$SESSION" "${REMOTE_DIR}/baseline_log.txt"              "${LOCAL_TMP}/baseline_log.txt"
colab --auth=adc download -s "$SESSION" "${REMOTE_DIR}/baseline_report.json"          "${LOCAL_TMP}/baseline_report.json"          2>/dev/null || echo "[skip] baseline_report.json"
colab --auth=adc download -s "$SESSION" "${REMOTE_DIR}/baseline_gpu_timeseries.json"  "${LOCAL_TMP}/baseline_gpu_timeseries.json"  2>/dev/null || echo "[skip] baseline_gpu_timeseries.json"

echo "[collect_baseline] artifacts saved to ${LOCAL_TMP}/"
ls -lh "$LOCAL_TMP/"

# 3. Stop session
colab --auth=adc stop -s "$SESSION"
echo "[collect_baseline] session '${SESSION}' stopped"
echo "[collect_baseline] done — run collect_results.sh next"
