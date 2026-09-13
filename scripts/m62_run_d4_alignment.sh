#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 OUTPUT_PARENT" >&2
  exit 2
fi

PROJECT=/home/wxh/go2_short_vln
ROOT=/mnt/wxh/go2_short_vln
PYTHON="$ROOT/envs/conda/smolvla/bin/python"
STAMP="$(date +%Y%m%dT%H%M%S%z)"
OUTPUT_DIR="$1/$STAMP"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite D4 output: $OUTPUT_DIR" >&2
  exit 2
fi

export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT"
exec "$PYTHON" -m src.smolvla.analyze_temporal_alignment \
  --m4-root "$ROOT/data/expert_v1/m4_gate_final_20260831" \
  --dataset-root "$ROOT/data/lerobot/short_vln_v1" \
  --d2-predictions "$ROOT/outputs/m7/offline_train_action_eval_20260901T120600+0800/predictions.npz" \
  --checkpoint "$ROOT/outputs/m6_1/smoke_20260901T104846+0800/checkpoints/step_002000" \
  --output-dir "$OUTPUT_DIR" \
  --diagnosis-report "$PROJECT/reports/M6_2_DIAGNOSIS.md"
