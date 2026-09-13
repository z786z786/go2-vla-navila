#!/usr/bin/env bash
set -euo pipefail

source /home/wxh/go2_short_vln/scripts/m1_proxy_env.sh
export OMNI_KIT_ACCEPT_EULA=YES
unset PYTHONPATH PYTHONHOME

ROOT=/mnt/wxh/go2_short_vln
PREFIX="$ROOT/envs/conda/navila-isaac"
LOG_DIR="$ROOT/outputs/m1/logs"
RUN_ID="$(date +%Y%m%dT%H%M%S%z)"
VIDEO_DIR="$ROOT/outputs/m1/video/$RUN_ID"
SCREENSHOT_DIR="$ROOT/outputs/m1/screenshot/$RUN_ID"
PLANNER="$ROOT/third_party/NaVILA-Bench/scripts/demo_planner.py"
CAPTURE=/home/wxh/go2_short_vln/scripts/m1_capture_demo_planner.py
export M1_ISAAC_ASSET_ROOT="$ROOT/assets/isaac_sim_4_1"
GO2_USD="$M1_ISAAC_ASSET_ROOT/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd"
GO2_PROPS="$M1_ISAAC_ASSET_ROOT/Isaac/IsaacLab/Robots/Unitree/Go2/Props/instanceable_meshes.usd"

mkdir -p "$LOG_DIR" "$VIDEO_DIR" "$SCREENSHOT_DIR"
ROLLOUT_LOG="$LOG_DIR/pd-planner-rollout-$RUN_ID.log"
GPU_LOG="$LOG_DIR/gpu-monitor-$RUN_ID.csv"
exec > >(tee -a "$ROLLOUT_LOG") 2>&1

date --iso-8601=seconds
echo "m1_run_id=$RUN_ID"
echo "M1 official PD planner: one Go2 Matterport episode, headless with cameras."
sha256sum "$PLANNER"
printf 'M1 local Isaac asset root: %s\n' "$M1_ISAAC_ASSET_ROOT"
test -f "$GO2_USD"
test -f "$GO2_PROPS"
sha256sum "$GO2_USD" "$GO2_PROPS"
df -h / /mnt
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total --format=csv,noheader,nounits -l 1 > "$GPU_LOG" &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT

export M1_VIDEO_DIR="$VIDEO_DIR"
export M1_SCREENSHOT_DIR="$SCREENSHOT_DIR"
"$PREFIX/bin/python" "$CAPTURE" \
  --task=go2_matterport_vision \
  --history_length=9 \
  --load_run=2024-09-25_23-22-02 \
  --num_envs=1 \
  --episode_index=0 \
  --headless \
  --enable_cameras

awk -F, 'NR==1 || ($3+0)>max {max=$3+0; row=$0} END {print "peak_gpu_sample=" row}' "$GPU_LOG"
find "$VIDEO_DIR" "$SCREENSHOT_DIR" -maxdepth 1 -type f -printf '%p %s bytes\n' | sort
date --iso-8601=seconds
