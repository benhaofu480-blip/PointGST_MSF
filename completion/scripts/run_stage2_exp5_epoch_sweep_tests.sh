#!/usr/bin/env bash
# Exp5: official PCN test on available Exp1-ON ckpts (no retrain).
# Disk only has ckpt-best (~ep5), ckpt-epoch-049, ckpt-epoch-050 (=last).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$(dirname "$0")/_logs_dir.sh"

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"

EXP_DIR="${ROOT}/experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft/PCN_models/exp_MSF_Pure_Group_sigmoid_feedpointrs_ft_crop02_04_seed42"
CFG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft.yaml"
OUT="${LOG_ROOT}/exp5_epoch_sweep_summary.txt"
GPU="${GPU:-0}"

run_one() {
  local tag="$1" ckpt="$2"
  local log="${LOG_ROOT}/test_exp5_${tag}.log"
  if [[ ! -f "$ckpt" ]]; then
    echo "SKIP $tag missing $ckpt" | tee -a "$OUT"
    return 0
  fi
  echo "=== $(date) TEST $tag GPU=$GPU ===" | tee -a "$OUT"
  : >"$log"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u main.py --test \
    --ckpts "$ckpt" \
    --config "$CFG" \
    --exp_name "test_exp5_${tag}" \
    --model pgst \
    --num_workers 4 \
    >>"$log" 2>&1
  ep=$("$PYTHON" -c "import torch; print(torch.load('${ckpt}', map_location='cpu')['epoch'])")
  ov=$(grep -E '^\S+.*Overall' "$log" | tail -1 || true)
  echo "epoch=$ep  $ov" | tee -a "$OUT"
  echo "  log=$log" | tee -a "$OUT"
}

: >"$OUT"
echo "Exp5 epoch sweep (official test, seed42 Exp1-ON)" | tee -a "$OUT"
echo "--- Val curve (PCN val split, from train log) ---" | tee -a "$OUT"
"$PYTHON" scripts/parse_exp5_val_from_log.py 2>>"$OUT" | tee -a "$OUT"
echo "--- Official test on saved ckpts ---" | tee -a "$OUT"

run_one "ckpt_best" "${EXP_DIR}/ckpt-best.pth"
run_one "epoch049" "${EXP_DIR}/ckpt-epoch-049.pth"
run_one "epoch050" "${EXP_DIR}/ckpt-epoch-050.pth"

echo "Done. Summary: $OUT"
