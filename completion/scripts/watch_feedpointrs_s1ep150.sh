#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
LOG="logs/complete/train_feedpointrs_s1ep150_seed42.log"
echo "=== $(date '+%F %T') Stage-2 monitor ==="
if ! pgrep -f "main.py.*feedpointrs_ft_complete.yaml" >/dev/null 2>&1; then
  echo "STATUS: NOT RUNNING"
else
  echo "STATUS: RUNNING (DDP)"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/  GPU /' || true
fi
[[ -f "$LOG" ]] || { echo "LOG missing: $LOG"; exit 0; }
echo "--- latest train ---"
grep -E '\[Epoch [0-9]+/50\]\[Batch|Training\] EPOCH:' "$LOG" | tail -2 || true
echo "--- val history ---"
grep '\[Validation\] EPOCH:' "$LOG" || true
echo "--- ckpt-best saves ---"
grep 'Save checkpoint.*ckpt-best' "$LOG" | tail -2 || true
grep -E 'Early Stop' "$LOG" | tail -1 || true
echo "--- ref: S1 ep150 test CDL1=6.6540 F=0.8390 ---"
