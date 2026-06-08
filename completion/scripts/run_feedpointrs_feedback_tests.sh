#!/usr/bin/env bash
# FeedPoinTrS-style double-pass test on official PCN test (1200).
# All ckpts use MSF Pure Group sigmoid (C) adapter weights.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
CFG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid.yaml"
GPU="${GPU:-0}"
CROP_RATIO="${CROP_RATIO:-0.30}"
SEED=42

run_one() {
  local tag="$1"
  local ckpt="$2"
  local gpu="$3"
  local log="/tmp/feedpointrs_${tag}.log"
  if [[ ! -f "$ckpt" ]]; then
    echo "SKIP $tag: missing $ckpt" | tee -a /tmp/feedpointrs_launch.log
    return 1
  fi
  echo "START $tag GPU=$gpu -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON" -u scripts/test_feedpointrs_feedback.py \
    --config "$CFG" \
    --ckpts "$ckpt" \
    --tag "$tag" \
    --crop_ratio "$CROP_RATIO" \
    --seed "$SEED" \
    --num_workers 4 \
    > "$log" 2>&1 &
  echo $! >> /tmp/feedpointrs_pids.txt
  echo "  pid=$! log=$log"
}

: > /tmp/feedpointrs_pids.txt
: > /tmp/feedpointrs_launch.log

# 1) C sigmoid 一阶段
run_one "sigmoid_s1" \
  "${ROOT}/experiments/AdaPoinTr_MSF_Pure_Group_sigmoid/PCN_models/exp_MSF_Pure_Group_sigmoid/ckpt-best.pth" \
  "$GPU"

# 2) hard_ft cov005 seed42（从 sigmoid 初始化微调）
run_one "hardft_cov005_s42" \
  "${ROOT}/experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_hard_ft_cov005/PCN_models/exp_MSF_Pure_Group_sigmoid_hard_ft_cov005_seed42/ckpt-best.pth" \
  "${GPU2:-1}"

# 3) D1 短训 ep5 best — 等前两个之一结束后再跑，避免三进程抢显存
(
  while pgrep -f "test_feedpointrs_feedback.py.*sigmoid_s1" >/dev/null 2>&1 \
     || pgrep -f "test_feedpointrs_feedback.py.*hardft_cov005" >/dev/null 2>&1; do
    sleep 120
  done
  run_one "hardft_diag_ep10_s42" \
    "${ROOT}/experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_hard_ft_diag_static_ep10/PCN_models/exp_MSF_Pure_Group_sigmoid_hard_ft_diag_static_ep10_seed42/ckpt-best.pth" \
    "$GPU"
) &

echo "Launched. Monitor:"
echo "  tail -f /tmp/feedpointrs_sigmoid_s1.log"
echo "  grep -E 'Overall|OPENLOOP|FEEDBACK' /tmp/feedpointrs_*.log"
