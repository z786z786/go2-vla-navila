#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/wxh/go2_short_vln
ROOT=/mnt/wxh/go2_short_vln
PYTHON="$ROOT/envs/conda/smolvla/bin/python"
SOURCE_EVIDENCE="$ROOT/outputs/m7/offline_train_action_eval_20260901T120600+0800"
DATASET_ROOT="$ROOT/data/lerobot/short_vln_v1"
CHECKPOINT="$ROOT/outputs/m6_1/smoke_20260901T104846+0800/checkpoints/step_002000"
OUTPUT_PARENT="$ROOT/outputs/m6_2/d2_offline"
STAMP="$(date +%Y%m%dT%H%M%S%z)"
OUTPUT_DIR="$OUTPUT_PARENT/$STAMP"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite D2 output directory: $OUTPUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_PARENT"
cd "$PROJECT"
exec "$PYTHON" -m src.smolvla.analyze_train_actions \
  --source-evidence "$SOURCE_EVIDENCE" \
  --dataset-root "$DATASET_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --diagnosis-report "$PROJECT/reports/M6_2_DIAGNOSIS.md"
