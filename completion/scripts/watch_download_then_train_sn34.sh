#!/usr/bin/env bash
# 等待 ShapeNet-34 下载完成后自动启动训练
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$(dirname "$0")/_logs_dir.sh"

WATCH="${LOG_ROOT}/watch_sn34_download_train.log"
: >"$WATCH"

echo "=== $(date) watch download -> train ===" | tee "$WATCH"
for i in $(seq 1 720); do
  if [[ -f "${ROOT}/data/ShapeNet55-34/ShapeNet-34/train.txt" ]] \
     && [[ -d "${ROOT}/data/ShapeNet55-34/shapenet_pc" ]]; then
    echo "=== $(date) data ready, start training ===" | tee -a "$WATCH"
    bash scripts/train_shapenet34_sigmoid.sh >>"$WATCH" 2>&1
    exit 0
  fi
  if ! pgrep -f "download_shapenet34.sh" >/dev/null 2>&1 \
     && ! pgrep -f "gdown.*1jUB5yD7DP97" >/dev/null 2>&1; then
    if [[ $i -gt 3 ]]; then
      echo "=== $(date) download process gone but data missing ===" | tee -a "$WATCH"
      tail -20 "${LOG_ROOT}/download_shapenet34.log" >>"$WATCH" 2>&1 || true
      exit 1
    fi
  fi
  sleep 30
done
echo "ERROR: timeout waiting for ShapeNet-34" | tee -a "$WATCH"
exit 1
