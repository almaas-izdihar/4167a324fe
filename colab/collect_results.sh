#!/usr/bin/env bash
# Usage: ./colab/collect_results.sh <session> <slug>
# Example: ./colab/collect_results.sh smoke 2026-06-08-smoke
#
# Downloads notebook output + plots from remote VM into results/logs/<slug>/
# then commits and pushes to current branch.

set -e

SESSION="${1:?Usage: $0 <session> <slug>}"
SLUG="${2:?Usage: $0 <session> <slug>}"
REMOTE_DIR="/content/ema-skd/results"
LOCAL_DIR="results/logs/${SLUG}"

echo "[collect] session=${SESSION}  slug=${SLUG}"
echo "[collect] destination: ${LOCAL_DIR}"

# 1. Copy model logs to results/ on remote (smoke scripts skip this)
echo "[collect] prep: copying model logs to remote results/"
cat > /tmp/prep_logs.py << 'PYEOF'
import glob, shutil, os
os.makedirs("/content/ema-skd/results", exist_ok=True)
base = sorted(glob.glob("/content/ema-skd/models/*EHSKD_False*/log/log.txt"))
ema  = sorted(glob.glob("/content/ema-skd/models/*EHSKD_True*/log/log.txt"))
if base: shutil.copy(base[-1], "/content/ema-skd/results/baseline_log.txt"); print(f"baseline: {base[-1]}")
else:    print("WARNING: no baseline log found")
if ema:  shutil.copy(ema[-1],  "/content/ema-skd/results/emaskd_log.txt");  print(f"emaskd:   {ema[-1]}")
else:    print("WARNING: no emaskd log found")
PYEOF
colab --auth=adc exec -s "$SESSION" -f /tmp/prep_logs.py

# 2. Run analysis notebook
echo "[collect] running analyze_logs.ipynb..."
colab --auth=adc exec --timeout 120 -s "$SESSION" -f colab/analyze_logs.ipynb

# 3. Create local log dir
mkdir -p "$LOCAL_DIR"

# 4. Download plots
echo "[collect] downloading plots..."
colab --auth=adc download -s "$SESSION" "${REMOTE_DIR}/eval_curves.png"    "${LOCAL_DIR}/eval_curves.png"
colab --auth=adc download -s "$SESSION" "${REMOTE_DIR}/gate_curriculum.png" "${LOCAL_DIR}/gate_curriculum.png"
colab --auth=adc download -s "$SESSION" "${REMOTE_DIR}/train_dynamics.png"  "${LOCAL_DIR}/train_dynamics.png"

# 5. Copy executed notebook (saved locally by colab exec)
cp colab/analyze_logs_output.ipynb "${LOCAL_DIR}/analyze_logs_output.ipynb"
echo "[collect] notebook copied to ${LOCAL_DIR}/"

# 6. Commit + push
git add "${LOCAL_DIR}/"
git commit -m "results: ${SLUG} — plots + executed notebook"
git push origin HEAD
echo "[collect] pushed to $(git branch --show-current)"

# 7. Stop session
colab --auth=adc stop -s "$SESSION"
echo "[collect] session '${SESSION}' stopped"
