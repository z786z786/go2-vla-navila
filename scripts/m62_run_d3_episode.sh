#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 SHORT_EPISODE_ID OUTPUT_ROOT SOCKET ISAAC_SEED" >&2
  exit 2
fi

SHORT_ID="$1"
OUTPUT_ROOT="$2"
SOCKET_PATH="$3"
ISAAC_SEED="$4"
PROJECT=/home/wxh/go2_short_vln
ROOT=/mnt/wxh/go2_short_vln
PYTHON="$ROOT/envs/conda/navila-isaac/bin/python"
EPISODE_DIR="$OUTPUT_ROOT/$SHORT_ID"
LOG_DIR="$OUTPUT_ROOT/logs"
RUN_TAG="$(date +%Y%m%dT%H%M%S%z)-$SHORT_ID"
LOG_PATH="$LOG_DIR/$RUN_TAG.log"
GPU_LOG="$LOG_DIR/$RUN_TAG-gpu.csv"

if [[ -e "$EPISODE_DIR" ]]; then
  echo "Refusing to overwrite episode directory: $EPISODE_DIR" >&2
  exit 2
fi

source "$PROJECT/scripts/m1_proxy_env.sh"
export OMNI_KIT_ACCEPT_EULA=YES
export MPLCONFIGDIR="$ROOT/cache/matplotlib"
unset PYTHONPATH PYTHONHOME
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR" "$MPLCONFIGDIR"
exec > >(tee -a "$LOG_PATH") 2>&1
date --iso-8601=seconds
echo "d3_short_episode_id=$SHORT_ID"
echo "d3_episode_dir=$EPISODE_DIR"
echo "d3_isaac_seed=$ISAAC_SEED"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -l 1 > "$GPU_LOG" &
GPU_MONITOR_PID=$!
cleanup() {
  kill "$GPU_MONITOR_PID" 2>/dev/null || true
  wait "$GPU_MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT

cd "$PROJECT"
/usr/bin/time -v "$PYTHON" -m src.inference.isaac_client \
  --short-dataset "$PROJECT/data/short_vln_v1.json" \
  --short-episode-id "$SHORT_ID" \
  --episode-dir "$EPISODE_DIR" \
  --socket "$SOCKET_PATH" \
  --request-timeout 10 \
  --execute-steps 10 \
  --max-steps 1500 \
  --success-streak 10 \
  --jpeg-quality 90 \
  --asset-root "$ROOT/assets/isaac_sim_4_1" \
  --navila-root "$ROOT/third_party/NaVILA-Bench" \
  --task go2_matterport_vision \
  --history_length 9 \
  --load_run 2024-09-25_23-22-02 \
  --num_envs 1 \
  --seed "$ISAAC_SEED" \
  --headless \
  --enable_cameras

find "$EPISODE_DIR" -maxdepth 2 -type f -printf '%p %s bytes\n' | sort
date --iso-8601=seconds
