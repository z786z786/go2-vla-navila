#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 && ! ( $# -eq 2 && "$1" == "--resume" ) ]]; then
  echo "Usage: $0 OUTPUT_PARENT | $0 --resume RUN_ROOT" >&2
  exit 2
fi

PROJECT=/home/wxh/go2_short_vln
ROOT=/mnt/wxh/go2_short_vln
ISAAC_PY="$ROOT/envs/conda/navila-isaac/bin/python"
POLICY_PY="$ROOT/envs/conda/smolvla/bin/python"
SOURCE_MODEL="$ROOT/outputs/m6/smoke_20260831T224205+0800/checkpoints/step_002000"
M61_CHECKPOINT="$ROOT/outputs/m6_1/smoke_20260901T104846+0800/checkpoints/step_002000"
BASE_DATASET="$ROOT/data/lerobot/short_vln_v1"
BASE_EXPERT="$ROOT/data/expert_v1/m4_gate_final_20260831"
SHORT_DATASET="$PROJECT/data/short_vln_v1.json"
ASSIGNMENT_OVERRIDES="$PROJECT/config/d5_r4_assignment_overrides.json"
RESUME=false
if [[ "$1" == "--resume" ]]; then
  RESUME=true
  RUN_ROOT="$2"
  STAMP="$(basename "$RUN_ROOT")"
else
  OUTPUT_PARENT="$1"
  STAMP="$(date +%Y%m%dT%H%M%S%z)"
  RUN_ROOT="$OUTPUT_PARENT/$STAMP"
fi
EXPERT_ROOT="$ROOT/data/expert_v1/m6_2_d5_$STAMP"
ATTEMPT_ROOT="$ROOT/data/expert_v1/m6_2_d5_attempts_$STAMP"
DATASET_ROOT="$ROOT/data/lerobot/short_vln_m6_2_d5_$STAMP"
SOCKET=/tmp/go2_smolvla_m7.sock

require_gpu() {
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [[ "$free_mib" -lt 12288 ]]; then
    echo "D5 requires at least 12288 MiB free GPU memory; found $free_mib MiB. No other GPU task was changed." >&2
    exit 3
  fi
}

if [[ "$RESUME" == false && ( -e "$RUN_ROOT" || -e "$EXPERT_ROOT" || -e "$ATTEMPT_ROOT" || -e "$DATASET_ROOT" ) ]]; then
  echo "D5 refuses to overwrite an existing timestamped output" >&2
  exit 2
fi
if [[ "$RESUME" == true && ( ! -d "$RUN_ROOT" || ! -d "$EXPERT_ROOT" || ! -d "$ATTEMPT_ROOT" || ! -f "$RUN_ROOT/selection.json" || -e "$DATASET_ROOT" ) ]]; then
  echo "D5 resume requires the existing collection root and no downstream dataset output" >&2
  exit 2
fi

export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
if [[ "$RESUME" == false ]]; then
  mkdir -p "$RUN_ROOT"
fi

cd "$PROJECT"
if [[ "$RESUME" == false ]]; then
  "$ISAAC_PY" -m src.episodes.select_d5_episodes \
    --dataset "$SHORT_DATASET" \
    --output "$RUN_ROOT/selection.json" \
    --seed 20260831
fi

require_gpu
COLLECTION_ARGS=()
if [[ "$RESUME" == true ]]; then
  COLLECTION_ARGS+=(--resume)
fi
"$ISAAC_PY" -m src.collector.d5_collection \
  --dataset "$SHORT_DATASET" \
  --selection "$RUN_ROOT/selection.json" \
  --attempt-root "$ATTEMPT_ROOT" \
  --accepted-root "$EXPERT_ROOT" \
  --manifest "$RUN_ROOT/collection_manifest.json" \
  --runner "$PROJECT/scripts/m4_run_episode.sh" \
  --assignment-overrides "$ASSIGNMENT_OVERRIDES" \
  --seed 20260831 \
  "${COLLECTION_ARGS[@]}"

"$POLICY_PY" -m src.dataset.convert_to_lerobot \
  --expert-root "$EXPERT_ROOT" \
  --output-root "$DATASET_ROOT" \
  --repo-prefix "local/go2_short_vln_m6_2_d5_$STAMP" \
  --splits train seen_val

"$POLICY_PY" -m src.dataset.check_dataset \
  --dataset-root "$DATASET_ROOT" \
  --report-dir "$RUN_ROOT/dataset_check" \
  --seed 20260831 \
  --batch-size 4 \
  --video-backend pyav

"$POLICY_PY" -m src.smolvla.audit_d5_dataset \
  --expert-root "$EXPERT_ROOT" \
  --dataset-root "$DATASET_ROOT" \
  --baseline-dataset-root "$BASE_DATASET" \
  --baseline-checkpoint "$M61_CHECKPOINT" \
  --output-dir "$RUN_ROOT/dataset_audit" \
  --video-backend pyav

FINAL_STEPS=$("$POLICY_PY" -c 'import json,math,sys; a=json.load(open(sys.argv[1])); b=a["baseline_sampler_audit"]["trainable_anchor_count"]; e=a["sampler_audit"]["trainable_anchor_count"]; print(math.ceil((2000*e/b)/500)*500)' "$RUN_ROOT/dataset_audit/d5_dataset_audit.json")
if [[ "$FINAL_STEPS" -le 2000 ]]; then
  echo "D5 equal-exposure final step count must exceed 2000; got $FINAL_STEPS" >&2
  exit 4
fi

require_gpu
TRAIN_ROOT="$RUN_ROOT/training/data_expanded"
"$POLICY_PY" -m src.smolvla.m6_training \
  --stage smoke \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$TRAIN_ROOT" \
  --model-id "$SOURCE_MODEL" \
  --model-revision c83c3163b8ca9b7e67c509fffd9121e66cb96205 \
  --source-checkpoint "$SOURCE_MODEL" \
  --bounded-actions \
  --codec-epsilon 1e-4 \
  --seed 20260831 \
  --batch-size 1 \
  --num-workers 2 \
  --video-backend pyav \
  --steps "$FINAL_STEPS" \
  --checkpoint-steps 2000 "$FINAL_STEPS" \
  --log-freq 50 \
  --probe-count 12 \
  --val-count 128

SHIFT=$("$POLICY_PY" -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["normalizer_shift"]["substantial"]).lower())' "$RUN_ROOT/dataset_audit/d5_dataset_audit.json")
if [[ "$SHIFT" == true ]]; then
  require_gpu
  "$POLICY_PY" -m src.smolvla.m6_training \
    --stage smoke \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$RUN_ROOT/training/m61_normalizer_ablation" \
    --model-id "$SOURCE_MODEL" \
    --model-revision c83c3163b8ca9b7e67c509fffd9121e66cb96205 \
    --source-checkpoint "$SOURCE_MODEL" \
    --action-normalizer-checkpoint "$M61_CHECKPOINT" \
    --bounded-actions \
    --codec-epsilon 1e-4 \
    --seed 20260831 \
    --batch-size 1 \
    --num-workers 2 \
    --video-backend pyav \
    --steps "$FINAL_STEPS" \
    --checkpoint-steps 2000 "$FINAL_STEPS" \
    --log-freq 50 \
    --probe-count 12 \
    --val-count 128
fi

evaluate_checkpoint() {
  local label="$1" checkpoint="$2" split="$3"
  require_gpu
  "$POLICY_PY" -m src.smolvla.evaluate_train_actions \
    --dataset-root "$DATASET_ROOT" \
    --checkpoint "$checkpoint" \
    --split "$split" \
    --output-dir "$RUN_ROOT/offline/$label/$split" \
    --base-seeds 20260831 20260832 20260833 \
    --video-backend pyav
}

evaluate_checkpoint m61 "$M61_CHECKPOINT" train
evaluate_checkpoint m61 "$M61_CHECKPOINT" seen_val
evaluate_checkpoint expanded_step_002000 "$TRAIN_ROOT/checkpoints/step_002000" train
evaluate_checkpoint expanded_step_002000 "$TRAIN_ROOT/checkpoints/step_002000" seen_val
FINAL_CHECKPOINT="$TRAIN_ROOT/checkpoints/step_$(printf '%06d' "$FINAL_STEPS")"
evaluate_checkpoint expanded_equal_exposure "$FINAL_CHECKPOINT" train
evaluate_checkpoint expanded_equal_exposure "$FINAL_CHECKPOINT" seen_val
if [[ "$SHIFT" == true ]]; then
  ABLATION_CHECKPOINT="$RUN_ROOT/training/m61_normalizer_ablation/checkpoints/step_$(printf '%06d' "$FINAL_STEPS")"
  evaluate_checkpoint m61_normalizer_ablation "$ABLATION_CHECKPOINT" train
  evaluate_checkpoint m61_normalizer_ablation "$ABLATION_CHECKPOINT" seen_val
fi

"$PROJECT/scripts/m62_run_d5_closed_loop.sh" "$RUN_ROOT" "$EXPERT_ROOT" "$FINAL_STEPS" "$SHIFT"

"$POLICY_PY" -m src.smolvla.d5_report \
  --run-root "$RUN_ROOT" \
  --diagnosis-report "$PROJECT/reports/M6_2_DIAGNOSIS.md"

echo "D5 data collection, conversion, training, offline, and phased closed-loop diagnostics completed: $RUN_ROOT"
