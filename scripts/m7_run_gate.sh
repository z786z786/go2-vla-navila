#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 OUTPUT_ROOT" >&2
  exit 2
fi

OUTPUT_ROOT="$1"
PROJECT=/home/wxh/go2_short_vln
ROOT=/mnt/wxh/go2_short_vln
SERVER_RUNNER="$PROJECT/scripts/m7_run_server.sh"
EPISODE_RUNNER="$PROJECT/scripts/m7_run_episode.sh"
CHECKER_PYTHON="$ROOT/envs/conda/smolvla/bin/python"
SOCKET_PATH=/tmp/go2_smolvla_m7.sock
ORACLE_ROOT="$ROOT/data/expert_v1/m4_gate_final_20260831"
CHECKPOINT_SHA256=facc4a73b500e52468a3c57fa985656ca085ed482211e9b97bb13328b58748e7

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to overwrite gate output root: $OUTPUT_ROOT" >&2
  exit 2
fi
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [[ "$FREE_MIB" -lt 20480 ]]; then
  echo "M7 requires at least 20480 MiB free GPU memory; found $FREE_MIB MiB" >&2
  exit 3
fi
mkdir -p "$OUTPUT_ROOT"
cd "$PROJECT"

"$SERVER_RUNNER" "$OUTPUT_ROOT" > "$OUTPUT_ROOT/server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 90); do
  if [[ -s "$OUTPUT_ROOT/server_ready.json" && -S "$SOCKET_PATH" ]]; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    cat "$OUTPUT_ROOT/server.log" >&2
    exit 4
  fi
  sleep 2
done
if [[ ! -s "$OUTPUT_ROOT/server_ready.json" ]]; then
  echo "M7 inference server readiness timeout" >&2
  exit 5
fi

"$CHECKER_PYTHON" -m src.inference.preflight \
  --episode-dir "$ORACLE_ROOT/short_vln_v1_0004" \
  --socket "$SOCKET_PATH" \
  --timeout 10 \
  --output "$OUTPUT_ROOT/preflight.json"

"$EPISODE_RUNNER" short_vln_v1_0004 "$OUTPUT_ROOT" "$SOCKET_PATH"
"$CHECKER_PYTHON" -m src.inference.check_rollout \
  --episode-dirs "$OUTPUT_ROOT/short_vln_v1_0004" \
  --oracle-root "$ORACLE_ROOT" \
  --gate-stage first \
  --expected-checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --report-json "$OUTPUT_ROOT/first_gate_report.json" \
  --report-md "$OUTPUT_ROOT/first_gate_report.md"

for SHORT_ID in short_vln_v1_0000 short_vln_v1_0001 short_vln_v1_0003 short_vln_v1_0006; do
  "$EPISODE_RUNNER" "$SHORT_ID" "$OUTPUT_ROOT" "$SOCKET_PATH"
done

"$CHECKER_PYTHON" -m src.inference.check_rollout \
  --episode-dirs \
    "$OUTPUT_ROOT/short_vln_v1_0000" \
    "$OUTPUT_ROOT/short_vln_v1_0001" \
    "$OUTPUT_ROOT/short_vln_v1_0003" \
    "$OUTPUT_ROOT/short_vln_v1_0004" \
    "$OUTPUT_ROOT/short_vln_v1_0006" \
  --oracle-root "$ORACLE_ROOT" \
  --gate-stage full \
  --expected-checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --report-json "$OUTPUT_ROOT/m7_gate_report.json" \
  --report-md "$OUTPUT_ROOT/m7_gate_report.md"
