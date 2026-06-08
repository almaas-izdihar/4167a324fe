#!/usr/bin/env bash
# Phase 2 of 3 — run after emaskd training completes.
# Downloads emaskd artifacts to /tmp/emaskd-collect/, stops session.
# Must run before collect_results.sh (phase 3).
#
# Usage: ./colab/collect_emaskd.sh <session>
# Example: ./colab/collect_emaskd.sh emaskd

set -e

SESSION="${1:?Usage: $0 <session>}"
REMOTE_DIR="/content/ema-skd/results"
LOCAL_TMP="/tmp/emaskd-collect"

echo "[collect_emaskd] session=${SESSION}"

# 1. Copy emaskd log to results/ on remote
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

# 2. Download artifacts
mkdir -p "$LOCAL_TMP"
colab --auth=adc download -s "$SESSION" "${REMOTE_DIR}/emaskd_log.txt"              "${LOCAL_TMP}/emaskd_log.txt"
colab --auth=adc download -s "$SESSION" "${REMOTE_DIR}/emaskd_report.json"          "${LOCAL_TMP}/emaskd_report.json"          2>/dev/null || echo "[skip] emaskd_report.json"
colab --auth=adc download -s "$SESSION" "${REMOTE_DIR}/emaskd_gpu_timeseries.json"  "${LOCAL_TMP}/emaskd_gpu_timeseries.json"  2>/dev/null || echo "[skip] emaskd_gpu_timeseries.json"

echo "[collect_emaskd] artifacts saved to ${LOCAL_TMP}/"
ls -lh "$LOCAL_TMP/"

# 3. Stop session
colab --auth=adc stop -s "$SESSION"
echo "[collect_emaskd] session '${SESSION}' stopped"
echo "[collect_emaskd] done — run collect_results.sh next"
