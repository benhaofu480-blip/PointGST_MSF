#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONUNBUFFERED=1
PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
LOG="/tmp/mine_hard_pc_samples.log"
echo "Mining hard samples -> data/PCN_hard/  log=$LOG"
exec "$PYTHON" -u scripts/mine_hard_pc_samples.py "$@" 2>&1 | tee "$LOG"
