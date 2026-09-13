#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 SHORT_EPISODE_ID OUTPUT_ROOT" >&2
  exit 2
fi

SHORT_ID="$1"
OUTPUT_ROOT="$2"
PROJECT=/home/wxh/go2_short_vln
ROOT=/mnt/wxh/go2_short_vln
PREFIX="$ROOT/envs/conda/navila-isaac"
EPISODE_DIR="$OUTPUT_ROOT/$SHORT_ID"
LOG_DIR="$ROOT/outputs/m4/logs"
RUN_TAG="$(date +%Y%m%dT%H%M%S%z)-$SHORT_ID"
LOG_PATH="$LOG_DIR/$RUN_TAG.log"
GPU_LOG="$LOG_DIR/$RUN_TAG-gpu.csv"
TERMINAL_HOLD_FRAMES="${M6_2_R7_TERMINAL_HOLD_FRAMES:-0}"

if [[ ! "$TERMINAL_HOLD_FRAMES" =~ ^[0-9]+$ ]] || (( TERMINAL_HOLD_FRAMES > 1000 )); then
  echo "M6_2_R7_TERMINAL_HOLD_FRAMES must be an integer in [0,1000]" >&2
  exit 2
fi

source "$PROJECT/scripts/m1_proxy_env.sh"
export OMNI_KIT_ACCEPT_EULA=YES
export MPLCONFIGDIR="$ROOT/cache/matplotlib"
export M1_ISAAC_ASSET_ROOT="$ROOT/assets/isaac_sim_4_1"
unset PYTHONPATH PYTHONHOME
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR" "$MPLCONFIGDIR"

exec > >(tee -a "$LOG_PATH") 2>&1
date --iso-8601=seconds
echo "m4_short_episode_id=$SHORT_ID"
echo "m4_episode_dir=$EPISODE_DIR"
df -h / /mnt
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total --format=csv,noheader,nounits -l 1 > "$GPU_LOG" &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT

cd "$PROJECT"
"$PREFIX/bin/python" -m src.collector.collect_expert \
  --short-dataset "$PROJECT/data/short_vln_v1.json" \
  --short-episode-id "$SHORT_ID" \
  --episode-dir "$EPISODE_DIR" \
  --official-source "$ROOT/third_party/NaVILA-Bench/scripts/demo_planner.py" \
  --asset-root "$ROOT/assets/isaac_sim_4_1" \
  --jpeg-quality 90 \
  --terminal-hold-frames "$TERMINAL_HOLD_FRAMES" \
  --task=go2_matterport_vision \
  --history_length=9 \
  --load_run=2024-09-25_23-22-02 \
  --num_envs=1 \
  --headless \
  --enable_cameras

if [[ ! -f "$EPISODE_DIR/summary.json" ]]; then
  echo "M4 collector did not produce summary.json; refusing post-processing" >&2
  exit 1
fi

"$PREFIX/bin/python" -m src.collector.check_expert \
  --episode-dirs "$EPISODE_DIR" \
  --report-json "$EPISODE_DIR/sanity_report.json" \
  --report-md "$EPISODE_DIR/sanity_report.md"

awk -F, 'NR==1 || ($3+0)>max {max=$3+0; row=$0} END {print "peak_gpu_sample=" row}' "$GPU_LOG"
find "$EPISODE_DIR" -maxdepth 2 -type f -printf '%p %s bytes\n' | sort
date --iso-8601=seconds
