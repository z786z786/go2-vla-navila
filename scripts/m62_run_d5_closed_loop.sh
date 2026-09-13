#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "Usage: $0 RUN_ROOT EXPERT_ROOT FINAL_STEPS NORMALIZER_SHIFT [--resume] [--comparison-only]" >&2
  exit 2
fi

PROJECT=/home/wxh/go2_short_vln
ROOT=/mnt/wxh/go2_short_vln
POLICY_PY="$ROOT/envs/conda/smolvla/bin/python"
M61_CHECKPOINT="$ROOT/outputs/m6_1/smoke_20260901T104846+0800/checkpoints/step_002000"
RUN_ROOT="$1"
EXPERT_ROOT="$2"
FINAL_STEPS="$3"
NORMALIZER_SHIFT="$4"
TRAIN_ROOT="$RUN_ROOT/training/data_expanded"
ROLL_ROOT="$RUN_ROOT/closed_loop"
SOCKET=/tmp/go2_smolvla_m7.sock
MODELS=(m61 expanded_step_002000 expanded_equal_exposure)
CHECKPOINTS=("$M61_CHECKPOINT" "$TRAIN_ROOT/checkpoints/step_002000" "$TRAIN_ROOT/checkpoints/step_$(printf '%06d' "$FINAL_STEPS")")
EPISODES=(short_vln_v1_0000 short_vln_v1_0001 short_vln_v1_0003 short_vln_v1_0004 short_vln_v1_0006)
RESUME=false
COMPARISON_ONLY=false
for option in "${@:5}"; do
  case "$option" in
    --resume) RESUME=true ;;
    --comparison-only) COMPARISON_ONLY=true ;;
    *) echo "Unknown option: $option" >&2; exit 2 ;;
  esac
done
if [[ "$COMPARISON_ONLY" == true && "$RESUME" != true ]]; then
  echo "--comparison-only requires --resume" >&2
  exit 2
fi

require_gpu() {
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [[ "$free_mib" -lt 12288 ]]; then
    echo "D5 closed-loop requires at least 12288 MiB free GPU memory; found $free_mib MiB. No other GPU task was changed." >&2
    exit 3
  fi
}

if [[ "$NORMALIZER_SHIFT" == true ]]; then
  MODELS+=(m61_normalizer_ablation)
  CHECKPOINTS+=("$RUN_ROOT/training/m61_normalizer_ablation/checkpoints/step_$(printf '%06d' "$FINAL_STEPS")")
fi
if [[ -e "$SOCKET" ]]; then
  echo "D5 closed-loop refuses an existing socket: $SOCKET" >&2
  exit 2
fi
if [[ "$RESUME" == true ]]; then
  [[ -d "$ROLL_ROOT" ]] || { echo "D5 closed-loop resume requires existing output: $ROLL_ROOT" >&2; exit 2; }
elif [[ -e "$ROLL_ROOT" ]]; then
  echo "D5 closed-loop refuses existing output without --resume: $ROLL_ROOT" >&2
  exit 2
fi

export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$ROLL_ROOT"
cd "$PROJECT"

run_model_seed() {
  local label="$1" checkpoint="$2" seed="$3"
  shift 3
  local seed_root="$ROLL_ROOT/$label/seed_$seed"
  local pending=()
  local episode episode_dir
  for episode in "$@"; do
    episode_dir="$seed_root/$episode"
    if [[ -e "$episode_dir" ]]; then
      if "$POLICY_PY" -m src.inference.d5_resume \
        --episode-dir "$episode_dir" --expected-episode-id "$episode" >/dev/null; then
        echo "D5 resume: verified and skipped $label/seed_$seed/$episode"
      else
        echo "D5 resume refuses incomplete or invalid episode: $episode_dir" >&2
        exit 8
      fi
    else
      pending+=("$episode")
    fi
  done
  [[ ${#pending[@]} -gt 0 ]] || return 0
  mkdir -p "$seed_root"
  require_gpu
  local attempt_root="$seed_root/server_attempts/$(date +%Y%m%dT%H%M%S%z)"
  mkdir -p "$attempt_root"
  "$POLICY_PY" -m src.inference.server \
    --checkpoint "$checkpoint" \
    --socket "$SOCKET" \
    --request-log "$attempt_root/server_requests.jsonl" \
    --ready-file "$attempt_root/server_ready.json" \
    --base-seed "$seed" > "$attempt_root/server.log" 2>&1 &
  local server_pid=$!
  cleanup_server() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup_server RETURN EXIT
  for _ in $(seq 1 180); do
    [[ -s "$attempt_root/server_ready.json" && -S "$SOCKET" ]] && break
    if ! kill -0 "$server_pid" 2>/dev/null; then
      wait "$server_pid" || true
      echo "D5 server exited before readiness; see $attempt_root/server.log" >&2
      exit 7
    fi
    sleep 1
  done
  [[ -s "$attempt_root/server_ready.json" && -S "$SOCKET" ]] || { echo "D5 server readiness timeout" >&2; exit 7; }
  if ! "$POLICY_PY" -m src.inference.preflight --episode-dir "$EXPERT_ROOT/short_vln_v1_0004" --socket "$SOCKET" --timeout 10 --output "$attempt_root/preflight.json"; then
    if [[ "$COMPARISON_ONLY" == true ]] && "$POLICY_PY" -c \
      'import json,sys; r=json.load(open(sys.argv[1])); c=r["checks"]; raise SystemExit(0 if c["response_valid"] and c["latency_within_timeout"] and not c["raw_output_range"] else 1)' \
      "$attempt_root/preflight.json"; then
      echo "D5 comparison diagnostic: retaining raw-range preflight failure and continuing through the action safety gate"
    else
      echo "D5 preflight failed; see $attempt_root/preflight.json" >&2
      return 9
    fi
  fi
  for episode in "${pending[@]}"; do
    "$PROJECT/scripts/m62_run_d3_episode.sh" "$episode" "$seed_root" "$SOCKET" 20260831
    "$POLICY_PY" -m src.inference.d5_resume \
      --episode-dir "$seed_root/$episode" --expected-episode-id "$episode" >/dev/null
  done
  cleanup_server
  trap - RETURN EXIT
  [[ ! -e "$SOCKET" ]] || { echo "D5 server left a socket behind" >&2; exit 7; }
}

if [[ "$COMPARISON_ONLY" == true ]]; then
  run_model_seed expanded_step_002000 "$TRAIN_ROOT/checkpoints/step_002000" 20260831 short_vln_v1_0000
  run_model_seed expanded_equal_exposure "$TRAIN_ROOT/checkpoints/step_$(printf '%06d' "$FINAL_STEPS")" 20260831 short_vln_v1_0000
  "$POLICY_PY" -m src.inference.d5_checkpoint_compare \
    --run-root "$RUN_ROOT" \
    --oracle-root "$EXPERT_ROOT" \
    --output-json "$ROLL_ROOT/three_checkpoint_comparison.json"
  echo "D5 three-checkpoint comparison completed: $ROLL_ROOT/three_checkpoint_comparison.json"
  exit 0
fi

for index in "${!MODELS[@]}"; do
  run_model_seed "${MODELS[$index]}" "${CHECKPOINTS[$index]}" 20260831 "${EPISODES[@]}"
done
for index in "${!MODELS[@]}"; do
  run_model_seed "${MODELS[$index]}" "${CHECKPOINTS[$index]}" 20260832 short_vln_v1_0004
  run_model_seed "${MODELS[$index]}" "${CHECKPOINTS[$index]}" 20260833 short_vln_v1_0004
done

"$POLICY_PY" -m src.inference.d5_closed_loop --run-root "$ROLL_ROOT" --oracle-root "$EXPERT_ROOT" --output-json "$ROLL_ROOT/d5_closed_loop_stage1.json" --output-md "$ROLL_ROOT/d5_closed_loop_stage1.md"
mapfile -t FOLLOWUPS < <("$POLICY_PY" -m src.inference.d5_followup --closed-loop-json "$ROLL_ROOT/d5_closed_loop_stage1.json")
for episode in "${FOLLOWUPS[@]}"; do
  [[ "$episode" == short_vln_v1_0004 ]] && continue
  for index in "${!MODELS[@]}"; do
    run_model_seed "${MODELS[$index]}" "${CHECKPOINTS[$index]}" 20260832 "$episode"
    run_model_seed "${MODELS[$index]}" "${CHECKPOINTS[$index]}" 20260833 "$episode"
  done
done
"$POLICY_PY" -m src.inference.d5_closed_loop --run-root "$ROLL_ROOT" --oracle-root "$EXPERT_ROOT" --output-json "$ROLL_ROOT/d5_closed_loop.json" --output-md "$ROLL_ROOT/d5_closed_loop.md"
echo "D5 phased closed-loop diagnostics completed: $ROLL_ROOT"
