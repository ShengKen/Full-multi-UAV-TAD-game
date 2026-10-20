#!/usr/bin/env bash
export TZ='CST-8'
export TZ='CST-8'
/isaac-sim/IsaacLab/isaaclab.sh
/isaac-sim/python.sh

export TERM=xterm PYTHONUNBUFFERED=1 LIVESTREAM=0
mkdir -p logs
export CONFIG=configs/tad_hierarchical_10v12v10.yaml
export LOG_SUBDIR=tad_hierarchical
PARAMS="$(/isaac-sim/python.sh scripts/get_yaml_val.py "$CONFIG" rl.algo rl.critic_mode scenario.train_side scenario.train_layer)"
echo "[run] CONFIG=$CONFIG  $PARAMS"
export RUN_REDIRECT="logs/.run_tmp_$$.log"
nohup bash run_train_test.sh > "$RUN_REDIRECT" 2>&1 &
RUN_PID=$!
disown
echo "[run] 外层 run_train_test.sh PID=$RUN_PID （kill 它无效，见下）"
echo "[run] 等待训练真身 python3 启动……"
TRAIN_PID=""
for _ in $(seq 1 40); do
  sh_pid=$(pgrep -P "$RUN_PID" | head -1)
  [[ -n "$sh_pid" ]] && TRAIN_PID=$(pgrep -P "$sh_pid" -f 'kit/python/bin/python3' | head -1)
  [[ -n "$TRAIN_PID" ]] && break
  sleep 1.5
done
if [[ -n "$TRAIN_PID" ]]; then
  echo "[run] 训练真身 python3 PID=$TRAIN_PID   |   $PARAMS   →   停训练： kill -9 $TRAIN_PID"
else
  echo "[run] [WARN] 约60s未探测到 python3（启动慢或失败）。手动查： pgrep -af 'kit/python/bin/python3.*train_tad' ；失败看日志： logs/tad_hierarchical/*/run_train.log"
fi

/isaac-sim/python.sh scripts/play_tad_hierarchical.py \
  --config logs/tad_hierarchical/2026-08-23_17-01-27_maddpg/run_config.yaml \
  --checkpoint logs/tad_hierarchical/2026-08-23_17-01-27_maddpg/26-08-23_17-01-38-912490_MADDPG/checkpoints/agent_800000.pt \
  --num_envs 1 \
  --livestream 2  \
  --max_duration 18000

/isaac-sim/python.sh scripts/play_tad_hierarchical.py \  --config configs/tad_hierarchical_config.yaml \
  --checkpoint logs/tad_hierarchical/2026-07-21_22-25-40_maddpg/26-07-21_22-25-47-349984_MADDPG/checkpoints/agent_200000.pt \
  --num_envs 1 \
  --headless  \
  --max_duration 18000
