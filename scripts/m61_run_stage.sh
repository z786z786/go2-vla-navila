#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || "$1" != "smoke" ]]; then
  echo "usage: $0 smoke" >&2
  exit 2
fi

PROJECT_ROOT="/home/wxh/go2_short_vln"
ROOT="/mnt/wxh/go2_short_vln"
DATASET_ROOT="$ROOT/data/lerobot/short_vln_v1"
SOURCE_CHECKPOINT="$ROOT/outputs/m6/smoke_20260831T224205+0800/checkpoints/step_002000"
OUTPUT_ROOT="$ROOT/outputs/m6_1"
PYTHON="$ROOT/envs/conda/smolvla/bin/python"
STAMP="$(date +%Y%m%dT%H%M%S%z)"
RUN_DIR="${OUTPUT_ROOT}/smoke_${STAMP}"
LOG_DIR="${OUTPUT_ROOT}/logs"

export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RUN_DIR" "$LOG_DIR"
GPU_CSV="${LOG_DIR}/smoke_${STAMP}_gpu_samples.csv"
echo "timestamp,index,memory.used,memory.total,utilization.gpu" > "$GPU_CSV"
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits --loop=1 >> "$GPU_CSV" &
MONITOR_PID=$!
cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT

LOG_FILE="${LOG_DIR}/smoke_${STAMP}.log"
cd "$PROJECT_ROOT"
set -o pipefail
/usr/bin/time -v "$PYTHON" -m src.smolvla.m6_training \
  --stage smoke \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$RUN_DIR" \
  --model-id "$SOURCE_CHECKPOINT" \
  --model-revision c83c3163b8ca9b7e67c509fffd9121e66cb96205 \
  --source-checkpoint "$SOURCE_CHECKPOINT" \
  --bounded-actions \
  --codec-epsilon 1e-4 \
  --seed 20260831 \
  --batch-size 1 \
  --num-workers 2 \
  --video-backend pyav \
  --steps 2000 \
  --log-freq 50 \
  --save-freq 500 \
  --probe-count 4 \
  --val-count 64 2>&1 | tee "$LOG_FILE"

echo "$RUN_DIR"
