#!/usr/bin/env bash
# One predeclared DT2 episode, bounded and monitored; never trains.
set -euo pipefail
[[ $# -ge 2 && $# -le 3 && "$2" =~ ^(8|9|1[0-5])$ ]] || exit 64
plan="$1"
index="$2"
mode=()
if [[ $# == 3 ]]; then
  [[ "$3" == --shared-dt2 ]] || exit 64
  mode=(--shared-dt2)
fi
source_root=/home/wxh/go2_short_vln
data_root=/mnt/wxh/go2_short_vln
python_bin="$data_root/envs/conda/navila-isaac/bin/python"
cd "$source_root"
run_id="$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["slots"][int(sys.argv[2])]["run_id"])' "$plan" "$index")"
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{2,70}$ ]] || exit 64
batch_root="$(dirname "$plan")"
log="$batch_root/${run_id}_launcher.log"
[[ ! -e "$log" && ! -e "$batch_root/runs/$run_id" ]] || exit 73
owned_pgid=""
cleanup() {
  [[ -n "$owned_pgid" ]] || return 0
  kill -0 -- "-$owned_pgid" 2>/dev/null || return 0
  kill -TERM -- "-$owned_pgid" 2>/dev/null || true
  for _ in {1..10}; do
    kill -0 -- "-$owned_pgid" 2>/dev/null || return 0
    sleep 1
  done
  kill -KILL -- "-$owned_pgid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM
env -u PYTHONPATH -u PYTHONHOME OMNI_KIT_ACCEPT_EULA=YES \
  MPLCONFIGDIR="$data_root/cache/matplotlib" setsid timeout --signal=TERM --kill-after=30s 900s \
  "$python_bin" -m src.dual_target.tiny_runtime --live --headless --enable_cameras \
  --tiny-plan "$plan" --task-index "$index" "${mode[@]}" \
  --go2-usd "$data_root/assets/isaac_sim_4_1/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd" \
  --motion-calibration "$data_root/outputs/dual_target_v1/dt1_motion_20260905_r1/motion_precheck/motion_calibration.json" \
  > "$log" 2>&1 &
owned_pid=$!
sleep 1
candidate_pgid="$(ps -o pgid= -p "$owned_pid" | tr -d '[:space:]' || true)"
self_pgid="$(ps -o pgid= -p $$ | tr -d '[:space:]')"
[[ "$candidate_pgid" =~ ^[0-9]+$ && "$candidate_pgid" != "$self_pgid" ]] || { kill -TERM "$owned_pid" 2>/dev/null || true; exit 70; }
owned_pgid="$candidate_pgid"
printf '{"stage":"DT2","run_id":"%s","pid":%s,"pgid":%s,"timeout_s":900}\n' \
  "$run_id" "$owned_pid" "$owned_pgid" > "$batch_root/${run_id}_launcher_receipt.json"
while kill -0 "$owned_pid" 2>/dev/null; do
  if ! "$python_bin" -m src.dual_target.coexistence_watch --pgid "$owned_pgid" \
      >> "$batch_root/${run_id}_gpu_samples.jsonl"; then
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
"$python_bin" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="EPISODE_SUCCEEDED_NOT_DT2_APPROVED" and d["episode_result"]["status"]=="success"' \
  "$batch_root/runs/$run_id/stage_status.json"
