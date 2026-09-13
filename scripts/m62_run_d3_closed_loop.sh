#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 OUTPUT_PARENT" >&2
  exit 2
fi

PROJECT=/home/wxh/go2_short_vln
ROOT=/mnt/wxh/go2_short_vln
POLICY_PYTHON="$ROOT/envs/conda/smolvla/bin/python"
CHECKPOINT="$ROOT/outputs/m6_1/smoke_20260901T104846+0800/checkpoints/step_002000"
ORACLE_ROOT="$ROOT/data/expert_v1/m4_gate_final_20260831"
OUTPUT_PARENT="$1"
STAMP="$(date +%Y%m%dT%H%M%S%z)"
RUN_ROOT="$OUTPUT_PARENT/$STAMP"
SOCKET=/tmp/go2_smolvla_m7.sock
ISAAC_SEED=20260831
EXPECTED_MODEL=facc4a73b500e52468a3c57fa985656ca085ed482211e9b97bb13328b58748e7
EXPECTED_CODEC=143c8fce264be7906bd1aea11d3e502a4d67f40e7d8a691fac678f752657af27
EXPECTED_DATASET=d55f2e388c62350a63c55cc7954208d9d77a16b838725965251a28e218f3d402
EXPECTED_ORACLE_DATASET=88255ffb8e175372048c4948531d7ad059a9d8aa9217b7be95e2f60f520b34b6
EXPECTED_PLANNER=20ea09b51287defead44d20a47730a38ba1ecf399e034c8e3acf13196a686432
POLICY_SEEDS=(20260831 20260832 20260833)
EPISODES=(short_vln_v1_0000 short_vln_v1_0004)

if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to overwrite D3 output directory: $RUN_ROOT" >&2
  exit 2
fi

require_gpu() {
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [[ "$free_mib" -lt 20480 ]]; then
    echo "D3 requires at least 20480 MiB free GPU memory; found $free_mib MiB. No other task was changed." >&2
    exit 3
  fi
}

assert_hash() {
  local path="$1" expected="$2" actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "Frozen-input hash mismatch: $path expected=$expected actual=$actual" >&2
    exit 4
  fi
}

mkdir -p "$OUTPUT_PARENT" "$RUN_ROOT"
assert_hash "$CHECKPOINT/model.safetensors" "$EXPECTED_MODEL"
assert_hash "$CHECKPOINT/bounded_action_codec.json" "$EXPECTED_CODEC"
assert_hash "$ROOT/data/lerobot/short_vln_v1/conversion_manifest.json" "$EXPECTED_DATASET"
for episode in "${EPISODES[@]}"; do
  assert_hash "$ORACLE_ROOT/$episode/summary.json" "$(case "$episode" in short_vln_v1_0000) echo 19e620343a110e0afba7c0589d503c8e9f539e2ed4083e05d23c6d481a085b38 ;; short_vln_v1_0004) echo 3d78f84d59a603f077a6df3d2e66774f2ec624839e6e3301f97de414dec87838 ;; esac)"
done
if [[ "$(sha256sum "$PROJECT/data/short_vln_v1.json" | awk '{print $1}')" != "$EXPECTED_ORACLE_DATASET" ]]; then
  echo "Short-task JSON hash mismatch" >&2
  exit 4
fi
if ! rg -q "$EXPECTED_PLANNER" "$ORACLE_ROOT/short_vln_v1_0000/summary.json"; then
  echo "M4 planner hash mismatch" >&2
  exit 4
fi

export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
printf '{\n  "format": "go2-short-vln-m6-2-d3-run-v1",\n  "checkpoint": "%s",\n  "model_sha256": "%s",\n  "codec_sha256": "%s",\n  "isaac_seed": %s,\n  "policy_base_seeds": [%s, %s, %s]\n}\n' "$CHECKPOINT" "$EXPECTED_MODEL" "$EXPECTED_CODEC" "$ISAAC_SEED" "${POLICY_SEEDS[0]}" "${POLICY_SEEDS[1]}" "${POLICY_SEEDS[2]}" > "$RUN_ROOT/run_config.json"

for base_seed in "${POLICY_SEEDS[@]}"; do
  require_gpu
  seed_root="$RUN_ROOT/seed_$base_seed"
  mkdir -p "$seed_root"
  printf '{\n  "format": "go2-short-vln-m6-2-d3-seed-v1",\n  "policy_base_seed": %s,\n  "isaac_seed": %s\n}\n' "$base_seed" "$ISAAC_SEED" > "$seed_root/seed_config.json"
  if [[ -e "$SOCKET" ]]; then
    echo "Refusing to replace pre-existing socket: $SOCKET" >&2
    exit 5
  fi
  cd "$PROJECT"
  "$POLICY_PYTHON" -m src.inference.server \
    --checkpoint "$CHECKPOINT" \
    --socket "$SOCKET" \
    --request-log "$seed_root/server_requests.jsonl" \
    --ready-file "$seed_root/server_ready.json" \
    --base-seed "$base_seed" > "$seed_root/server.log" 2>&1 &
  server_pid=$!
  cleanup_server() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
    if [[ -e "$SOCKET" ]]; then
      echo "Owned D3 server left socket behind: $SOCKET" >&2
      exit 6
    fi
  }
  trap cleanup_server EXIT
  for _ in $(seq 1 180); do
    [[ -s "$seed_root/server_ready.json" && -S "$SOCKET" ]] && break
    if ! kill -0 "$server_pid" 2>/dev/null; then
      wait "$server_pid" || true
      echo "D3 server exited before readiness; see $seed_root/server.log" >&2
      exit 7
    fi
    sleep 1
  done
  [[ -s "$seed_root/server_ready.json" && -S "$SOCKET" ]] || { echo "D3 server readiness timeout" >&2; exit 7; }
  "$POLICY_PYTHON" -m src.inference.preflight \
    --episode-dir "$ORACLE_ROOT/short_vln_v1_0004" \
    --socket "$SOCKET" \
    --timeout 10 \
    --output "$seed_root/preflight.json"
  for episode in "${EPISODES[@]}"; do
    require_gpu
    "$PROJECT/scripts/m62_run_d3_episode.sh" "$episode" "$seed_root" "$SOCKET" "$ISAAC_SEED"
  done
  cleanup_server
  trap - EXIT
done

cd "$PROJECT"
"$POLICY_PYTHON" -m src.inference.d3_closed_loop \
  --run-root "$RUN_ROOT" \
  --oracle-root "$ORACLE_ROOT" \
  --report-json "$RUN_ROOT/d3_report.json" \
  --report-md "$RUN_ROOT/d3_report.md" \
  --diagnosis-report "$PROJECT/reports/M6_2_DIAGNOSIS.md"
find "$RUN_ROOT" -maxdepth 3 -type f -printf '%p %s bytes\n' | sort > "$RUN_ROOT/artifact_manifest.txt"
echo "D3 completed: $RUN_ROOT"
