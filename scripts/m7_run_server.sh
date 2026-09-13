#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RUN_DIR" >&2
  exit 2
fi

RUN_DIR="$1"
PROJECT=/home/wxh/go2_short_vln
ROOT=/mnt/wxh/go2_short_vln
PYTHON="$ROOT/envs/conda/smolvla/bin/python"
CHECKPOINT="${M7_CHECKPOINT:-$ROOT/outputs/m6_1/smoke_20260901T104846+0800/checkpoints/step_002000}"

mkdir -p "$RUN_DIR"
export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT"
exec "$PYTHON" -m src.inference.server \
  --checkpoint "$CHECKPOINT" \
  --socket /tmp/go2_smolvla_m7.sock \
  --request-log "$RUN_DIR/server_requests.jsonl" \
  --ready-file "$RUN_DIR/server_ready.json"
