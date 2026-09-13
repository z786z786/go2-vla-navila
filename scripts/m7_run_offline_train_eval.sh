#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 OUTPUT_ROOT" >&2
  exit 2
fi

OUTPUT_ROOT="$1"
PROJECT=/home/wxh/go2_short_vln
ROOT=/mnt/wxh/go2_short_vln
PYTHON="$ROOT/envs/conda/smolvla/bin/python"
CHECKPOINT="$ROOT/outputs/m6_1/smoke_20260901T104846+0800/checkpoints/step_002000"
DATASET_ROOT="$ROOT/data/lerobot/short_vln_v1"

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to overwrite output root: $OUTPUT_ROOT" >&2
  exit 2
fi
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [[ "$FREE_MIB" -lt 20480 ]]; then
  echo "M7 offline evaluation requires at least 20480 MiB free GPU memory; found $FREE_MIB MiB" >&2
  exit 3
fi

export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT"
exec "$PYTHON" -m src.smolvla.evaluate_train_actions \
  --dataset-root "$DATASET_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_ROOT" \
  --base-seeds 20260831 20260832 20260833
