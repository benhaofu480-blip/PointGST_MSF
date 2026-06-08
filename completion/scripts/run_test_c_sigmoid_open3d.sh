#!/usr/bin/env bash
# C sigmoid ckpt-best + Open3D statistical outlier post-process on dense pred, then PCN test metrics.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export OPEN3D_CPU_RENDERING=TRUE
export LIBGL_ALWAYS_SOFTWARE=1
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"

CKPT="${ROOT}/experiments/AdaPoinTr_MSF_Pure_Group_sigmoid/PCN_models/exp_MSF_Pure_Group_sigmoid/ckpt-best.pth"
CFG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_test_o3d.yaml"
LOG_DIR="${ROOT}/experiments/MSF_postprocess_eval/c_sigmoid_o3d"
mkdir -p "$LOG_DIR"

if [[ ! -f "$CKPT" ]]; then
  echo "Missing checkpoint: $CKPT"
  exit 1
fi

echo "Test C sigmoid + Open3D filter"
echo "  ckpt: $CKPT"
echo "  cfg:  $CFG"
echo "  log:  $LOG_DIR/nohup.out"

CUDA_VISIBLE_DEVICES="${GPU:-0}" nohup "$PYTHON" -u main.py --test \
  --ckpts "$CKPT" \
  --config "$CFG" \
  --exp_name test_c_sigmoid_open3d \
  --model pgst \
  --num_workers 4 \
  > "${LOG_DIR}/nohup.out" 2>&1 &

echo $! > "${LOG_DIR}/pid"
echo "pid=$(cat "${LOG_DIR}/pid")"
echo "watch: tail -f ${LOG_DIR}/nohup.out"
