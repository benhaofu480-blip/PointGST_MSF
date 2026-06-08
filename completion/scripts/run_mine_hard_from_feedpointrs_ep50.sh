#!/usr/bin/env bash
# Mine hard PCN samples from FeedPoinTrS ep50 ckpt (do NOT overwrite data/PCN_hard/).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
GPU="${GPU:-0}"
SEED="${SEED:-42}"

CFG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft.yaml"
CKPT="experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft/PCN_models/exp_MSF_Pure_Group_sigmoid_feedpointrs_ft_crop02_04_seed42/ckpt-epoch-050.pth"
OUT_DIR="data/PCN_hard_from_feedpointrs_ep50"
LOG="/tmp/mine_hard_from_feedpointrs_ep50_seed${SEED}.log"

if [[ ! -f "$CKPT" ]]; then
  echo "Missing ckpt: $CKPT" >&2
  exit 1
fi

: > "$LOG"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u scripts/mine_hard_pc_samples.py \
  --config "$CFG" \
  --ckpts "$CKPT" \
  --out-dir "$OUT_DIR" \
  --top-ratio 0.30 \
  --sigma-margin 0.5 \
  --hard-repeat 3 \
  --random-ratio 0.20 \
  --full-train-list data/PCN_Core/pcn_core_train.txt \
  --seed "$SEED" \
  2>&1 | tee -a "$LOG"

echo "Done. mix list: ${OUT_DIR}/hard_train_ft_mix.txt"
echo "log: $LOG"
