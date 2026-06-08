#!/usr/bin/env bash
# Usage: ./monitor.sh <session-name> [interval_seconds]
# Run in a separate terminal while colab exec is running in another.
# Uses colab ls (file op) — safe, does not interrupt running kernel.

SESSION="${1:-smoke}"
INTERVAL="${2:-20}"
START=$(date +%s)

echo "[monitor] session=${SESSION}  interval=${INTERVAL}s  (Ctrl+C to stop)"
echo ""

while true; do
    NOW=$(date +%s)
    ELAPSED=$(( NOW - START ))
    MINS=$(( ELAPSED / 60 ))
    SECS=$(( ELAPSED % 60 ))

    echo "=== $(date '+%H:%M:%S')  +${MINS}m${SECS}s ==="

    echo "-- sessions --"
    colab sessions 2>&1

    echo "-- models dir --"
    colab ls "/content/ema-skd/models/" -s "$SESSION" 2>&1 || echo "(not available yet)"

    echo "-- log tail --"
    colab ls "/content/ema-skd/models/" -s "$SESSION" 2>/dev/null \
        | grep -oE '[^ ]+' \
        | head -1 \
        | xargs -I{} colab ls "/content/ema-skd/models/{}/log/" -s "$SESSION" 2>/dev/null \
        && echo "(log dir found — download with: colab download -s $SESSION /content/ema-skd/models/<dir>/log/log.txt ./log.txt)" \
        || echo "(log not found yet)"

    echo ""
    sleep "$INTERVAL"
done
