#!/usr/bin/env bash
# Stage-1 纯组级 Sigmoid @ ShapeNet-34，从 AdaPoinTr_ps55 微调
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$(dirname "$0")/_logs_dir.sh"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU:-0,1}"
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"

CONFIG="cfgs/ShapeNet34_models/AdaPoinTr_MSF_Pure_Group_sigmoid.yaml"
EXP="exp_MSF_Pure_Group_sigmoid_sn34_seed42"
LOG="${LOG_ROOT}/train_shapenet34_sigmoid_seed42.log"
EXP_DIR="experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_sn34/ShapeNet34_models/${EXP}"

[[ -f "${ROOT}/data/ShapeNet55-34/ShapeNet-34/train.txt" ]] || {
  echo "ERROR: ShapeNet-34 未就绪，先运行: bash scripts/download_shapenet34.sh" >&2
  exit 1
}

if pgrep -f "main.py.*AdaPoinTr_MSF_Pure_Group_sigmoid_sn34" >/dev/null 2>&1 \
   || pgrep -f "main.py.*ShapeNet34_models/AdaPoinTr_MSF_Pure_Group_sigmoid.yaml" >/dev/null 2>&1; then
  echo "ERROR: ShapeNet-34 训练已在运行." >&2
  exit 1
fi

mkdir -p "$EXP_DIR"
: >"$LOG"

echo "=== $(date) START ShapeNet-34 Stage-1 Sigmoid GPU=${CUDA_VISIBLE_DEVICES} ===" | tee "$LOG"
nohup "$PYTHON" -u main.py \
  --launcher none \
  --config "$CONFIG" \
  --exp_name "$EXP" \
  --num_workers 4 \
  --model pgst \
  --seed 42 \
  >>"$LOG" 2>&1 &
echo "pid=$! log=$LOG"
echo "Monitor: tail -f $LOG"
