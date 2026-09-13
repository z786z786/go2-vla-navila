#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 single-batch|tiny-overfit|smoke" >&2
  exit 2
fi

STAGE="$1"
PROJECT_ROOT="/home/wxh/go2_short_vln"
DATASET_ROOT="/mnt/wxh/go2_short_vln/data/lerobot/short_vln_v1"
OUTPUT_ROOT="/mnt/wxh/go2_short_vln/outputs/m6"
PYTHON="/mnt/wxh/go2_short_vln/envs/conda/smolvla/bin/python"
STAMP="$(date +%Y%m%dT%H%M%S%z)"
RUN_DIR="${OUTPUT_ROOT}/${STAGE}_${STAMP}"
LOG_DIR="${OUTPUT_ROOT}/logs"

export HF_HOME="/mnt/wxh/go2_short_vln/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RUN_DIR" "$LOG_DIR"

GPU_CSV="${LOG_DIR}/${STAGE}_${STAMP}_gpu_samples.csv"
echo "timestamp,index,memory.used,memory.total,utilization.gpu" > "$GPU_CSV"
nvidia-smi \
  --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits \
  --loop=1 >> "$GPU_CSV" &
MONITOR_PID=$!
cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT

ARGS=(
  --stage "$STAGE"
  --dataset-root "$DATASET_ROOT"
  --output-dir "$RUN_DIR"
  --model-id /mnt/wxh/go2_short_vln/models/smolvla_base_c83c316
  --model-revision c83c3163b8ca9b7e67c509fffd9121e66cb96205
  --seed 20260831
  --batch-size 1
  --video-backend pyav
)

if [[ "$STAGE" == "single-batch" ]]; then
  ARGS+=(--num-workers 0)
elif [[ "$STAGE" == "tiny-overfit" ]]; then
  ARGS+=(--steps 500 --log-freq 10 --save-freq 500 --num-workers 2 --probe-count 4)
elif [[ "$STAGE" == "smoke" ]]; then
  ARGS+=(--steps 2000 --log-freq 50 --save-freq 500 --num-workers 2 --probe-count 4 --val-count 64)
else
  echo "unknown stage: $STAGE" >&2
  exit 2
fi

LOG_FILE="${LOG_DIR}/${STAGE}_${STAMP}.log"
cd "$PROJECT_ROOT"
set -o pipefail
/usr/bin/time -v "$PYTHON" src/smolvla/m6_training.py "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"

echo "$RUN_DIR"
