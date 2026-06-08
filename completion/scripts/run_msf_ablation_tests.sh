#!/usr/bin/env bash
# A/B/C 测试集评测：三路后台并行。main.py 正文日志在 experiment_path 下带时间戳的 .log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
EVAL="${ROOT}/experiments/MSF_ablation_eval"
mkdir -p "$EVAL"
STAMP="$(date '+%F %T')"
echo "[$STAMP] launch A/B/C tests" >> "${EVAL}/test_launch.log"

launch_one() {
  local tag="$1" gpu="$2" ckpt="$3" cfg="$4" ename="$5" exp_subdir="$6"
  local out="${EVAL}/${tag}"
  mkdir -p "$out" "${ROOT}/experiments/${exp_subdir}"
  if [[ ! -f "$ckpt" ]]; then
    echo "[$STAMP] SKIP ${tag}: no $ckpt" >> "${EVAL}/test_launch.log"
    return
  fi
  echo "[$STAMP] ${tag} gpu=${gpu} -> experiments/${exp_subdir}/*.log" >> "${EVAL}/test_launch.log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON" -u main.py --test \
    --ckpts "$ckpt" --config "$cfg" --exp_name "$ename" --model pgst --num_workers 4 \
    >> "${out}/nohup.out" 2>&1 &
  echo $! > "${out}/pid"
  echo "[$STAMP] ${tag} pid=$(cat "${out}/pid")" >> "${EVAL}/test_launch.log"
}

# 双卡：A=0, B=1, C=0（与 A 同卡，若 OOM 可改 MSF_TEST_GPU_C=1 后重跑 C）
GPU_A="${MSF_TEST_GPU_A:-0}"
GPU_B="${MSF_TEST_GPU_B:-1}"
GPU_C="${MSF_TEST_GPU_C:-0}"

launch_one "A-softmax" "$GPU_A" \
  "${ROOT}/experiments/AdaPoinTr_MSF_Pure_Group/PCN_models/exp_MSF_Pure_Group/ckpt-best.pth" \
  "cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group.yaml" "ablation_eval_softmax" \
  "AdaPoinTr_MSF_Pure_Group/PCN_models/test_ablation_eval_softmax"

launch_one "B-tanh" "$GPU_B" \
  "${ROOT}/experiments/AdaPoinTr_MSF_Pure_Group_tanh/PCN_models/exp_MSF_Pure_Group_tanh/ckpt-best.pth" \
  "cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_tanh.yaml" "ablation_eval_tanh" \
  "AdaPoinTr_MSF_Pure_Group_tanh/PCN_models/test_ablation_eval_tanh"

launch_one "C-sigmoid" "$GPU_C" \
  "${ROOT}/experiments/AdaPoinTr_MSF_Pure_Group_sigmoid/PCN_models/exp_MSF_Pure_Group_sigmoid/ckpt-best.pth" \
  "cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid.yaml" "ablation_eval_sigmoid" \
  "AdaPoinTr_MSF_Pure_Group_sigmoid/PCN_models/test_ablation_eval_sigmoid"

echo "[$STAMP] all launched; see ${EVAL}/test_launch.log" >> "${EVAL}/test_launch.log"
