#!/usr/bin/env bash

set -uo pipefail

export TERM=xterm
export PYTHONUNBUFFERED=1
export TZ='CST-8'

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

PYTHON=/isaac-sim/python.sh
CONFIG="${CONFIG:-configs/tad_hierarchical_config.yaml}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ENV_TYPE=$(grep -oE "^[[:space:]]*env_type:[[:space:]]+[a-z_]+" "$CONFIG" | head -n1 | awk '{print $NF}')
if [[ -z "${LOG_SUBDIR:-}" ]]; then
  LOG_SUBDIR="tad_hierarchical"
fi

ALGO=$(grep -oE "^[[:space:]]*algo:[[:space:]]+[a-z0-9]+" "$CONFIG" | head -n1 | awk '{print $NF}')
ALGO="${ALGO:-mappo}"
RUN_DIR="logs/${LOG_SUBDIR}/$(date +%Y-%m-%d_%H-%M-%S)_${ALGO}"
mkdir -p "$RUN_DIR"
RUN_LOG="$RUN_DIR/run_train.log"

if [[ -n "${RUN_REDIRECT:-}" && -f "${RUN_REDIRECT:-}" ]]; then
  cat "$RUN_REDIRECT" >> "$RUN_LOG" 2>/dev/null || true
  rm -f "$RUN_REDIRECT"
  exec >> "$RUN_LOG" 2>&1
else
  exec > >(tee -a "$RUN_LOG") 2>&1
fi
echo "[run] 本次目录：$RUN_DIR"
echo "[run] 日志：$RUN_LOG"

echo "[run] ===== 1/2 训练 ====="
"$PYTHON" scripts/train_tad.py --config "$CONFIG" --log_dir "$RUN_DIR" --headless --livestream 0
train_rc=$?
if [[ $train_rc -ne 0 ]]; then
  echo "[run] ✗ 训练失败（退出码 $train_rc），跳过测试。日志见 $RUN_LOG"
  exit $train_rc
fi
sleep 5

ckpt_dir=$(find "$RUN_DIR" -type d -name checkpoints 2>/dev/null | head -n1)
ckpt=$(find "$ckpt_dir" -maxdepth 1 -name 'agent_*.pt' 2>/dev/null \
       | sed -E 's|.*/agent_([0-9]+)\.pt|\1\t&|' | sort -n -k1 | tail -n1 | cut -f2-)
if [[ -z "${ckpt:-}" || ! -f "$ckpt" ]]; then
  echo "[run] ✗ 未找到 checkpoint（在 $RUN_DIR），跳过测试。"
  exit 1
fi
echo "[run] checkpoint：$ckpt"

echo "[run] ===== 2/2 测试（play）====="

PLAY_SCRIPT="scripts/play_tad_hierarchical.py"
echo "[run] play 脚本：$PLAY_SCRIPT  (env_type=$ENV_TYPE)"
"$PYTHON" "$PLAY_SCRIPT" --config "$CONFIG" --checkpoint "$ckpt" --num_envs 1 --device cpu --livestream 0 --headless --max_duration 300
