#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RUN_ROOT" >&2
  exit 2
fi

PROJECT=/home/wxh/go2_short_vln
ROOT=/mnt/wxh/go2_short_vln
POLICY_PY="$ROOT/envs/conda/smolvla/bin/python"
RUN_ROOT="$1"
STAMP="$(basename "$RUN_ROOT")"
DATASET_ROOT="$ROOT/data/lerobot/short_vln_m6_2_d5_$STAMP"
EXPERT_ROOT="$ROOT/data/expert_v1/m6_2_d5_$STAMP"
M61_CHECKPOINT="$ROOT/outputs/m6_1/smoke_20260901T104846+0800/checkpoints/step_002000"
TRAIN_ROOT="$RUN_ROOT/training/data_expanded"

export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

if [[ ! -f "$RUN_ROOT/collection_manifest.json" || ! -f "$RUN_ROOT/dataset_audit/d5_dataset_audit.json" || ! -f "$TRAIN_ROOT/m6_report.json" ]]; then
  echo "D5 post-training resume requires completed collection, audit, and training reports" >&2
  exit 2
fi

read -r FINAL_STEPS TRAIN_PASSED SHIFT < <(
  "$POLICY_PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); a=json.load(open(sys.argv[2])); print(r["training"]["steps"], str(r["passed"]).lower(), str(a["normalizer_shift"]["substantial"]).lower())' \
    "$TRAIN_ROOT/m6_report.json" "$RUN_ROOT/dataset_audit/d5_dataset_audit.json"
)
if [[ "$TRAIN_PASSED" != true || ! -f "$TRAIN_ROOT/extended_training_acceptance.json" ]]; then
  echo "D5 extended training must pass its audited acceptance migration before resume" >&2
  exit 3
fi
if [[ "$SHIFT" == true ]]; then
  echo "This recovery entrypoint refuses an unfinished normalizer ablation" >&2
  exit 4
fi

require_gpu() {
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [[ "$free_mib" -lt 12288 ]]; then
    echo "D5 requires at least 12288 MiB free GPU memory; found $free_mib MiB. No other GPU task was changed." >&2
    exit 3
  fi
}

evaluate_checkpoint() {
  local label="$1" checkpoint="$2" split="$3" output="$RUN_ROOT/offline/$1/$3"
  if [[ -f "$output/metrics.json" ]]; then
    return
  fi
  if [[ -e "$output" ]]; then
    echo "Refusing partial offline output: $output" >&2
    exit 5
  fi
  require_gpu
  "$POLICY_PY" -m src.smolvla.evaluate_train_actions \
    --dataset-root "$DATASET_ROOT" \
    --checkpoint "$checkpoint" \
    --split "$split" \
    --output-dir "$output" \
    --base-seeds 20260831 20260832 20260833 \
    --video-backend pyav
}

cd "$PROJECT"
FINAL_CHECKPOINT="$TRAIN_ROOT/checkpoints/step_$(printf '%06d' "$FINAL_STEPS")"
evaluate_checkpoint m61 "$M61_CHECKPOINT" train
evaluate_checkpoint m61 "$M61_CHECKPOINT" seen_val
evaluate_checkpoint expanded_step_002000 "$TRAIN_ROOT/checkpoints/step_002000" train
evaluate_checkpoint expanded_step_002000 "$TRAIN_ROOT/checkpoints/step_002000" seen_val
evaluate_checkpoint expanded_equal_exposure "$FINAL_CHECKPOINT" train
evaluate_checkpoint expanded_equal_exposure "$FINAL_CHECKPOINT" seen_val

"$PROJECT/scripts/m62_run_d5_closed_loop.sh" "$RUN_ROOT" "$EXPERT_ROOT" "$FINAL_STEPS" false --resume

"$POLICY_PY" -m src.smolvla.d5_report \
  --run-root "$RUN_ROOT" \
  --diagnosis-report "$PROJECT/reports/M6_2_DIAGNOSIS.md"

echo "D5 post-training offline and closed-loop diagnostics completed: $RUN_ROOT"
