#!/usr/bin/env bash
# One non-training DT1 coexistence probe, after the standard GPU waiter.
set -euo pipefail
[[ $# -ge 1 && $# -le 2 && "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{2,70}$ ]] || exit 64
run_id="$1"
mode=()
if [[ $# == 2 ]]; then
  [[ "$2" == --shared-coexistence ]] || exit 64
  mode=(--shared-coexistence)
fi
project_root=/mnt/wxh/go2_short_vln
source_root=/home/wxh/go2_short_vln
python_bin="${project_root}/envs/conda/navila-isaac/bin/python"
output_root="${project_root}/outputs/dual_target_v1"
log_path="${output_root}/${run_id}_launcher.log"
[[ ! -e "${output_root}/${run_id}" && ! -e "$log_path" ]] || exit 73
cd "$source_root"
owned_pid=""
owned_pgid=""
cleanup() {
  [[ -n "$owned_pgid" ]] || return 0
  kill -0 -- "-${owned_pgid}" 2>/dev/null || return 0
  kill -TERM -- "-${owned_pgid}" 2>/dev/null || true
  for _ in {1..10}; do
    kill -0 -- "-${owned_pgid}" 2>/dev/null || return 0
    sleep 1
  done
  kill -KILL -- "-${owned_pgid}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM
env -u PYTHONPATH -u PYTHONHOME OMNI_KIT_ACCEPT_EULA=YES \
  MPLCONFIGDIR="${project_root}/cache/matplotlib" setsid \
  timeout --signal=TERM --kill-after=30s 900s "$python_bin" \
  -m src.dual_target.coexistence_probe --live --headless --enable_cameras "${mode[@]}" \
  --run-id "$run_id" --output-dir "$output_root" --gpu-project-root "$project_root" \
  --go2-usd "${project_root}/assets/isaac_sim_4_1/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd" \
  > "$log_path" 2>&1 &
owned_pid=$!
sleep 1
candidate_pgid="$(ps -o pgid= -p "$owned_pid" | tr -d '[:space:]' || true)"
self_pgid="$(ps -o pgid= -p $$ | tr -d '[:space:]')"
if [[ ! "$candidate_pgid" =~ ^[0-9]+$ || "$candidate_pgid" == "$self_pgid" ]]; then
  kill -TERM "$owned_pid" 2>/dev/null || true
  exit 70
fi
owned_pgid="$candidate_pgid"
printf '{"run_id":"%s","pid":%s,"pgid":%s,"timeout_s":900,"training":false}\n' \
  "$run_id" "$owned_pid" "$owned_pgid" > "${output_root}/${run_id}_launcher_receipt.json"
# Out-of-process protection also covers Kit startup and warmup, before the
# in-process model monitor can run. Each query is bounded by timeout.
while kill -0 "$owned_pid" 2>/dev/null; do
  if ! "$python_bin" -m src.dual_target.coexistence_watch \
       --pgid "$owned_pgid" >> "${output_root}/${run_id}_startup_gpu_samples.jsonl"; then
    cleanup
    exit 75
  fi
  sleep 1
done
set +e
wait "$owned_pid"
code=$?
set -e
[[ "$code" == 0 ]] || exit "$code"
# Kit can mask an exception's process exit code. Require the actual result.
"$python_bin" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="COEXISTENCE_PASSED_NOT_DT1_APPROVED" and d["training"] is False and d["execute_model_actions"] is False' \
  "${output_root}/${run_id}/coexistence_result.json"
