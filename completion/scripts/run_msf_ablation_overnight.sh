#!/usr/bin/env bash
# Overnight pipeline: wait for Softmax (A) if running -> train Tanh (B) -> train Sigmoid (C) -> test A/B/C.
# 请用 nohup 启动本脚本（SSH 断开不影响）。
# 日志: experiments/overnight_orchestrator.log
# 评测: experiments/MSF_ablation_eval/（不写入 exp_MSF_Pure_Group）
#
# 韧性（可环境变量覆盖）:
#   MAX_TRAIN_START_ATTEMPTS=5     每阶段（B/C）启动失败后的重试次数
#   STARTUP_MAX_WAIT=1800        等待首条 [Epoch x/150] 的最长时间（秒）
#   RETRY_BACKOFF_SEC=60         重试前休眠
#   POLL_TRAIN_SEC / POLL_STARTUP_SEC  轮询间隔
#   MSF_DDP_GPUS=0,1             双卡可见设备（勿继承单卡 CUDA_VISIBLE_DEVICES）

set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${MSF_DDP_GPUS:-0,1}"
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"

ORCH_LOG="${ROOT}/experiments/overnight_orchestrator.log"
POLL_TRAIN_SEC="${POLL_TRAIN_SEC:-1800}"
POLL_STARTUP_SEC="${POLL_STARTUP_SEC:-120}"
MAX_TRAIN_START_ATTEMPTS="${MAX_TRAIN_START_ATTEMPTS:-5}"
STARTUP_MAX_WAIT="${STARTUP_MAX_WAIT:-1800}"
RETRY_BACKOFF_SEC="${RETRY_BACKOFF_SEC:-60}"
mkdir -p "$(dirname "$ORCH_LOG")"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$ORCH_LOG"; }

is_training() {
  local cfg="$1"
  pgrep -f "main.py.*${cfg}" >/dev/null 2>&1
}

any_msf_training() {
  is_training "AdaPoinTr_MSF_Pure_Group.yaml" \
    || is_training "AdaPoinTr_MSF_Pure_Group_tanh.yaml" \
    || is_training "AdaPoinTr_MSF_Pure_Group_sigmoid.yaml"
}

log_has_training_started() {
  local logfile="$1"
  [[ -f "$logfile" ]] && grep -qE '\[Epoch [0-9]+/150\]' "$logfile"
}

# 日志里出现典型「已崩溃、不会再出 Epoch」的信号时提前结束等待，进入重试
log_suggests_startup_crash() {
  local logfile="$1"
  [[ -f "$logfile" ]] || return 1
  if grep -qE 'invalid device ordinal|ChildFailedError|CUDA out of memory|RuntimeError:.*CUDA|nccl.*error|NCCL.*error|PytorchStreamReader failed|Address already in use' "$logfile" 2>/dev/null; then
    return 0
  fi
  return 1
}

training_finished() {
  local cfg="$1" logfile="$2"
  if is_training "$cfg"; then
    return 1
  fi
  [[ -f "$logfile" ]] || return 1
  if grep -qE '\[Early Stop\]' "$logfile"; then
    return 0
  fi
  if grep -qE '\[Training\] EPOCH: 150 ' "$logfile"; then
    return 0
  fi
  if grep -qE '\[Epoch 150/150\]' "$logfile"; then
    return 0
  fi
  return 1
}

wait_until_training_done() {
  local cfg="$1" logfile="$2" label="$3"
  while ! training_finished "$cfg" "$logfile"; do
    if is_training "$cfg"; then
      local epoch_line
      epoch_line="$(grep -E '\[Epoch [0-9]+/150\]' "$logfile" 2>/dev/null | tail -1 || true)"
      log "${label}: running. Latest: ${epoch_line:-no epoch line yet}. Sleep ${POLL_TRAIN_SEC}s"
    else
      log "${label}: process gone but finish marker not in log yet. Sleep ${POLL_STARTUP_SEC}s"
      sleep "$POLL_STARTUP_SEC"
      if ! training_finished "$cfg" "$logfile"; then
        log "WARN ${label}: training died without finish marker. Check ${logfile}"
        return 1
      fi
      return 0
    fi
    sleep "$POLL_TRAIN_SEC"
  done
  log "${label} training finished (log + no process)."
}

launch_train() {
  local variant="$1"
  bash "${ROOT}/scripts/train_msf_ablation_bc.sh" "$variant"
}

log_tail_snippet() {
  local logfile="$1" n="${2:-25}"
  if [[ -f "$logfile" ]]; then
    log "--- tail -${n} ${logfile} ---"
    tail -n "$n" "$logfile" | while IFS= read -r line; do
      printf '[%s]   %s\n' "$(date '+%F %T')" "$line" >> "$ORCH_LOG"
    done
  else
    log "(no log file yet: ${logfile})"
  fi
}

# 返回值: 0=已看到 Epoch；1=超时；2=检测到崩溃日志且训练进程已不在（应重试）
wait_for_startup_log() {
  local logfile="$1" label="$2" cfg="$3" max_wait="${4:-1800}"
  local elapsed=0
  while [[ "$elapsed" -lt "$max_wait" ]]; do
    if log_has_training_started "$logfile"; then
      log "${label}: training log active (${logfile})"
      return 0
    fi
    if ! is_training "$cfg" && log_suggests_startup_crash "$logfile"; then
      log "${label}: startup crash pattern in log (no running process) — will retry. Snippet:"
      log_tail_snippet "$logfile" 30
      return 2
    fi
    log "${label}: waiting for first Epoch line (${elapsed}s / ${max_wait}s)..."
    sleep "$POLL_STARTUP_SEC"
    elapsed=$((elapsed + POLL_STARTUP_SEC))
  done
  log "ERROR ${label}: no Epoch line in ${logfile} within ${max_wait}s"
  log_tail_snippet "$logfile" 40
  return 1
}

# 带重试的训练阶段：启动 -> 等首 Epoch；失败则退避再启，最多 MAX_TRAIN_START_ATTEMPTS 次
run_train_stage_with_retries() {
  local variant="$1" cfg="$2" logfile="$3" label="$4"
  local attempt=0
  while [[ "$attempt" -lt "$MAX_TRAIN_START_ATTEMPTS" ]]; do
    attempt=$((attempt + 1))
    if [[ "$attempt" -gt 1 ]]; then
      log "${label}: clearing stale main.py for ${cfg} (if any) before retry..."
      pkill -f "main.py.*${cfg}" 2>/dev/null || true
      sleep 3
    fi
    log "${label}: startup attempt ${attempt}/${MAX_TRAIN_START_ATTEMPTS} (MSF_DDP_GPUS=${CUDA_VISIBLE_DEVICES})"

    if is_training "$cfg"; then
      log "${label}: main.py already running for ${cfg}, skip launch"
    else
      launch_train "$variant"
    fi

    local wret=0
    wait_for_startup_log "$logfile" "$label" "$cfg" "$STARTUP_MAX_WAIT" || wret=$?
    if [[ "$wret" -eq 0 ]]; then
      if wait_until_training_done "$cfg" "$logfile" "$label"; then
        return 0
      fi
      log "${label}: training aborted mid-run — will retry from scratch after backoff"
      log_tail_snippet "$logfile" 35
    else
      log "${label}: startup wait failed (code ${wret})"
    fi

    if [[ "$attempt" -lt "$MAX_TRAIN_START_ATTEMPTS" ]]; then
      log "${label}: backoff ${RETRY_BACKOFF_SEC}s before retry..."
      sleep "$RETRY_BACKOFF_SEC"
    fi
  done
  log "FATAL ${label}: exceeded MAX_TRAIN_START_ATTEMPTS=${MAX_TRAIN_START_ATTEMPTS}"
  return 1
}

# 评测并行：后台启动；日志出现 Test[200/...]（可配）后启动下一个。默认 GPU 0,1,0。
TEST_PROGRESS_RE="${TEST_PROGRESS_RE:-Test\\[200/}"
TEST_POLL_SEC="${TEST_POLL_SEC:-30}"
TEST_TRIGGER_MAX_WAIT="${TEST_TRIGGER_MAX_WAIT:-14400}"

launch_test_background() {
  local label="$1" config="$2" ckpt="$3" exp_name="$4" gpu="${5:-0}"
  if [[ ! -f "$ckpt" ]]; then
    log "SKIP test ${label}: missing ${ckpt}"
    return 1
  fi
  local eval_root="${ROOT}/experiments/MSF_ablation_eval"
  mkdir -p "${eval_root}/${label}"
  local test_log="${eval_root}/${label}/test.log"
  log "Test ${label} launch (bg, gpu=${gpu}) -> ${test_log}"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON" -u main.py \
    --test \
    --ckpts "$ckpt" \
    --config "$config" \
    --exp_name "${exp_name}" \
    --model pgst \
    --num_workers 4 \
    > "$test_log" 2>&1 &
  local pid=$!
  echo "$pid" > "${eval_root}/${label}/test.pid"
  log "Test ${label} pid=${pid}"
  return 0
}

wait_test_trigger() {
  local label="$1" logfile="$2" pid="${3:-}"
  local elapsed=0
  while [[ "$elapsed" -lt "$TEST_TRIGGER_MAX_WAIT" ]]; do
    if [[ -f "$logfile" ]] && grep -qE "$TEST_PROGRESS_RE" "$logfile" 2>/dev/null; then
      log "${label}: saw progress pattern (${TEST_PROGRESS_RE}) — start next test"
      return 0
    fi
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      if [[ -f "$logfile" ]] && grep -qE "$TEST_PROGRESS_RE" "$logfile" 2>/dev/null; then
        return 0
      fi
      log "WARN ${label}: test exited before ${TEST_PROGRESS_RE}; see ${logfile}"
      return 1
    fi
    sleep "$TEST_POLL_SEC"
    elapsed=$((elapsed + TEST_POLL_SEC))
  done
  log "WARN ${label}: no ${TEST_PROGRESS_RE} within ${TEST_TRIGGER_MAX_WAIT}s — start next anyway"
  return 1
}

wait_all_test_jobs() {
  local eval_root="${ROOT}/experiments/MSF_ablation_eval"
  local label pid tret
  for label in A-softmax B-tanh C-sigmoid; do
    local pid_file="${eval_root}/${label}/test.pid"
    [[ -f "$pid_file" ]] || continue
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      log "Waiting for test ${label} pid=${pid}..."
      if wait "$pid"; then
        tret=0
      else
        tret=$?
      fi
      if [[ "$tret" -ne 0 ]]; then
        log "WARN test ${label} exited code ${tret}"
        log_tail_snippet "${eval_root}/${label}/test.log" 25
      else
        log "Test ${label} finished OK"
      fi
    fi
  done
}

# 交错并行：A 后台 -> 见 Test[200/ -> B 后台 -> 见 Test[200/ -> C 后台 -> wait 全部
run_tests_staggered_parallel() {
  local gpu_a gpu_b gpu_c
  IFS=',' read -r gpu_a gpu_b gpu_c _ <<< "${MSF_TEST_GPUS:-0,1,0},,"
  gpu_a="${gpu_a:-0}"
  gpu_b="${gpu_b:-1}"
  gpu_c="${gpu_c:-0}"

  local eval_root="${ROOT}/experiments/MSF_ablation_eval"
  local log_a="${eval_root}/A-softmax/test.log"
  local log_b="${eval_root}/B-tanh/test.log"
  local log_c="${eval_root}/C-sigmoid/test.log"
  local pid_a="" pid_b="" pid_c=""

  log "Tests: staggered parallel (GPUs ${gpu_a},${gpu_b},${gpu_c}; trigger ${TEST_PROGRESS_RE})"

  if launch_test_background "A-softmax" "$1" "$2" "$3" "$gpu_a"; then
    pid_a="$(cat "${eval_root}/A-softmax/test.pid")"
    wait_test_trigger "A-softmax" "$log_a" "$pid_a" || true
  else
    log "A-softmax skipped — launch B immediately"
  fi

  if launch_test_background "B-tanh" "$4" "$5" "$6" "$gpu_b"; then
    pid_b="$(cat "${eval_root}/B-tanh/test.pid")"
    wait_test_trigger "B-tanh" "$log_b" "$pid_b" || true
  else
    log "B-tanh skipped — launch C immediately"
  fi

  launch_test_background "C-sigmoid" "$7" "$8" "$9" "$gpu_c" || true
  pid_c="$(cat "${eval_root}/C-sigmoid/test.pid" 2>/dev/null || true)"

  wait_all_test_jobs
}

# --- main ---
log "===== MSF ablation overnight orchestrator start (resilient mode) ====="
log "Config: MAX_TRAIN_START_ATTEMPTS=${MAX_TRAIN_START_ATTEMPTS} STARTUP_MAX_WAIT=${STARTUP_MAX_WAIT} RETRY_BACKOFF_SEC=${RETRY_BACKOFF_SEC} MSF_DDP_GPUS=${CUDA_VISIBLE_DEVICES}"

A_CFG="AdaPoinTr_MSF_Pure_Group.yaml"
A_LOG="${ROOT}/experiments/AdaPoinTr_MSF_Pure_Group/PCN_models/exp_MSF_Pure_Group/train_from_ps55.log"
[[ -f "$A_LOG" ]] || A_LOG="${ROOT}/experiments/AdaPoinTr_MSF_Pure_Group/PCN_models/exp_MSF_Pure_Group/train.log"

B_CFG="AdaPoinTr_MSF_Pure_Group_tanh.yaml"
B_LOG="${ROOT}/experiments/AdaPoinTr_MSF_Pure_Group_tanh/PCN_models/exp_MSF_Pure_Group_tanh/train.log"

C_CFG="AdaPoinTr_MSF_Pure_Group_sigmoid.yaml"
C_LOG="${ROOT}/experiments/AdaPoinTr_MSF_Pure_Group_sigmoid/PCN_models/exp_MSF_Pure_Group_sigmoid/train.log"

if is_training "$A_CFG"; then
  log "Softmax (A) is running — wait until GPUs are free before B."
  wait_until_training_done "$A_CFG" "$A_LOG" "Softmax-A" || true
else
  log "Softmax (A) not running (assume finished or not started)."
fi

while any_msf_training; do
  log "Another MSF job still active — sleep ${POLL_TRAIN_SEC}s"
  sleep "$POLL_TRAIN_SEC"
done

log "===== Stage Tanh (B) ====="
run_train_stage_with_retries tanh "$B_CFG" "$B_LOG" "Tanh-B" || log "Stage B failed after retries — continuing to C"

log "===== Stage Sigmoid (C) ====="
run_train_stage_with_retries sigmoid "$C_CFG" "$C_LOG" "Sigmoid-C" || log "Stage C failed after retries — continuing to tests"

log "===== Test A / B / C (staggered parallel, ckpt-best) ====="
A_CKPT="${ROOT}/experiments/AdaPoinTr_MSF_Pure_Group/PCN_models/exp_MSF_Pure_Group/ckpt-best.pth"
B_CKPT="${ROOT}/experiments/AdaPoinTr_MSF_Pure_Group_tanh/PCN_models/exp_MSF_Pure_Group_tanh/ckpt-best.pth"
C_CKPT="${ROOT}/experiments/AdaPoinTr_MSF_Pure_Group_sigmoid/PCN_models/exp_MSF_Pure_Group_sigmoid/ckpt-best.pth"

run_tests_staggered_parallel \
  "cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group.yaml" "$A_CKPT" "ablation_eval_softmax" \
  "cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_tanh.yaml" "$B_CKPT" "ablation_eval_tanh" \
  "cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid.yaml" "$C_CKPT" "ablation_eval_sigmoid"

log "===== All stages finished (check SKIP/FATAL/WARN above) ====="
