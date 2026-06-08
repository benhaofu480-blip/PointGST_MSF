#!/usr/bin/env bash
# 下载并解压 ShapeNet-34 数据集（PoinTr 官方 Google Drive 镜像）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$(dirname "$0")/_logs_dir.sh" 2>/dev/null || LOG_ROOT="${ROOT}/logs"
mkdir -p "$LOG_ROOT"

PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
PIP="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/pip"
DATA_DIR="${ROOT}/data"
ZIP="${DATA_DIR}/ShapeNet55-34.zip"
TARGET="${DATA_DIR}/ShapeNet55-34"
LOG="${LOG_ROOT}/download_shapenet34.log"

marker_ok() {
  [[ -f "${TARGET}/ShapeNet-34/train.txt" ]] \
    && [[ -f "${TARGET}/ShapeNet-34/test.txt" ]] \
    && [[ -d "${TARGET}/shapenet_pc" ]]
}

if marker_ok; then
  echo "ShapeNet-34 已存在: ${TARGET}"
  exit 0
fi

echo "=== $(date) download ShapeNet55-34 ===" | tee "$LOG"
"$PIP" install -q gdown

mkdir -p "$DATA_DIR"
if [[ ! -f "$ZIP" ]]; then
  echo "Downloading ${ZIP} ..." | tee -a "$LOG"
  "$PYTHON" -m gdown 1jUB5yD7DP97-EqqU2A9mmr61JpNwZBVK -O "$ZIP" >>"$LOG" 2>&1
fi

TMP="${DATA_DIR}/_shapenet34_unpack"
rm -rf "$TMP"
mkdir -p "$TMP"
echo "Unpacking ..." | tee -a "$LOG"
unzip -q "$ZIP" -d "$TMP" >>"$LOG" 2>&1

if [[ -d "${TMP}/ShapeNet55-34" ]]; then
  rm -rf "$TARGET"
  mv "${TMP}/ShapeNet55-34" "$TARGET"
elif [[ -d "${TMP}/ShapeNet-34" ]]; then
  rm -rf "$TARGET"
  mkdir -p "$TARGET"
  mv "${TMP}/ShapeNet-34" "$TARGET/ShapeNet-34"
  [[ -d "${TMP}/shapenet_pc" ]] && mv "${TMP}/shapenet_pc" "$TARGET/shapenet_pc"
else
  rm -rf "$TARGET"
  mv "$TMP" "$TARGET"
fi
rm -rf "$TMP"

if ! marker_ok; then
  echo "ERROR: unpack failed, check ${LOG}" | tee -a "$LOG"
  find "$DATA_DIR" -maxdepth 3 -type d | head -30 | tee -a "$LOG"
  exit 1
fi

echo "Done. train=$(wc -l < "${TARGET}/ShapeNet-34/train.txt") test=$(wc -l < "${TARGET}/ShapeNet-34/test.txt")" | tee -a "$LOG"
