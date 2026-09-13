#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/wxh/go2_short_vln
ROOT=/mnt/wxh/go2_short_vln
PYTHON="$ROOT/envs/conda/smolvla/bin/python"
DATASET_ROOT="$ROOT/data/lerobot/short_vln_v1"
CHECKPOINT="$ROOT/outputs/m6_1/smoke_20260901T104846+0800/checkpoints/step_002000"
OUTPUT_PARENT="$ROOT/outputs/m6_2/d1_codec"
STAMP="$(date +%Y%m%dT%H%M%S%z)"
OUTPUT_DIR="$OUTPUT_PARENT/$STAMP"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite D1 output directory: $OUTPUT_DIR" >&2
  exit 2
fi

export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUTPUT_PARENT"
cd "$PROJECT"
exec "$PYTHON" -m src.smolvla.evaluate_action_codec \
  --dataset-root "$DATASET_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --diagnosis-report "$PROJECT/reports/M6_2_DIAGNOSIS.md"
