#!/usr/bin/env bash
# Bounded exclusive DT1 launcher for the physical motion calibration and one
# calibrated truth-side expert probe.  No learner, SmolVLA, training, DT2, or
# shared-GPU exception is available through this script.
set -euo pipefail

project_root="/mnt/wxh/go2_short_vln"
source_root="/home/wxh/go2_short_vln"
python_bin="${project_root}/envs/conda/navila-isaac/bin/python"
output_root="${project_root}/outputs/dual_target_v1"
mode=""
run_id=""
go2_usd=""
group_id="dt1_dev_000"
color_configuration="A_red_B_blue"
target_color="red"
repeat="0"
motion_calibration=""
paired_reset=false
launcher_pid=""
launcher_pgid=""
process_group_verified=false

usage() {
  echo "usage: $0 --mode motion|expert --run-id ID --go2-usd /absolute/path/to/go2.usd [--group-id dt1_dev_000|dt1_dev_001 --color-configuration A_red_B_blue|A_blue_B_red --target-color red|blue --repeat 0|1 --motion-calibration /absolute/motion_calibration.json --paired-reset]" >&2
  exit 64
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) [[ $# -ge 2 ]] || usage; mode="$2"; shift 2 ;;
    --run-id) [[ $# -ge 2 ]] || usage; run_id="$2"; shift 2 ;;
    --go2-usd) [[ $# -ge 2 ]] || usage; go2_usd="$2"; shift 2 ;;
    --group-id) [[ $# -ge 2 ]] || usage; group_id="$2"; shift 2 ;;
    --color-configuration) [[ $# -ge 2 ]] || usage; color_configuration="$2"; shift 2 ;;
    --target-color) [[ $# -ge 2 ]] || usage; target_color="$2"; shift 2 ;;
    --repeat) [[ $# -ge 2 ]] || usage; repeat="$2"; shift 2 ;;
    --motion-calibration) [[ $# -ge 2 ]] || usage; motion_calibration="$2"; shift 2 ;;
    --paired-reset) paired_reset=true; shift ;;
    *) usage ;;
  esac
done

[[ "$mode" == motion || "$mode" == expert ]] || usage
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{2,80}$ ]] || usage
[[ "$go2_usd" = /* && -f "$go2_usd" ]] || usage
[[ "$group_id" == dt1_dev_000 || "$group_id" == dt1_dev_001 ]] || usage
[[ "$color_configuration" == A_red_B_blue || "$color_configuration" == A_blue_B_red ]] || usage
[[ "$target_color" == red || "$target_color" == blue ]] || usage
[[ "$repeat" =~ ^[01]$ ]] || usage
[[ -x "$python_bin" && -d "$source_root" ]] || { echo "DT1 Navila Python/source root unavailable" >&2; exit 66; }
if [[ "$mode" == expert ]]; then
  [[ "$motion_calibration" = /* && -f "$motion_calibration" ]] || {
    echo "expert mode requires an existing absolute --motion-calibration artifact" >&2
    exit 64
  }
elif [[ -n "$motion_calibration" ]]; then
  echo "motion mode derives a calibration and must not receive --motion-calibration" >&2
  exit 64
fi
if [[ "$paired_reset" == true && "$mode" != expert ]]; then
  echo "--paired-reset is only valid in expert mode" >&2
  exit 64
fi

mkdir -p "$output_root" "${project_root}/cache/matplotlib"
runner_log="${output_root}/dt1_${mode}_${run_id}_runner.log"
receipt_json="${output_root}/dt1_${mode}_${run_id}_receipt.json"
summary_json="${output_root}/dt1_${mode}_${run_id}_summary.json"
stage_status_path="${output_root}/${run_id}/stage_status.json"
for path in "$runner_log" "$receipt_json" "$summary_json" "${output_root}/${run_id}"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite $path" >&2; exit 73; }
done

stop_owned_group() {
  if [[ "$process_group_verified" != true ]]; then
    [[ -n "$launcher_pid" ]] && kill -TERM "$launcher_pid" 2>/dev/null || true
    return 0
  fi
  [[ -n "$launcher_pgid" ]] || return 0
  kill -0 -- "-${launcher_pgid}" 2>/dev/null || return 0
  kill -TERM -- "-${launcher_pgid}" 2>/dev/null || true
  for _ in {1..10}; do
    kill -0 -- "-${launcher_pgid}" 2>/dev/null || return 0
    sleep 1
  done
  kill -KILL -- "-${launcher_pgid}" 2>/dev/null || true
}

trap 'stop_owned_group || true' EXIT
trap 'exit 130' INT TERM

runner_args=(
  -m src.dual_target.runner --live --headless --enable_cameras
  --go2-usd "$go2_usd" --run-id "$run_id" --output-dir "$output_root"
  --gpu-project-root "$project_root" --checkpoint
  "${project_root}/third_party/NaVILA-Bench/logs/rsl_rl/go2_vision/2024-09-25_23-22-02/model_26499.pt"
  --low-level-device cuda:0 --group-id "$group_id" --color-configuration "$color_configuration"
  --target-color "$target_color" --repeat "$repeat"
)
if [[ "$mode" == motion ]]; then
  runner_args+=(--motion-precheck)
else
  runner_args+=(--max-env-steps 1500 --motion-calibration "$motion_calibration")
  if [[ "$paired_reset" == true ]]; then
    runner_args+=(--paired-reset)
  fi
fi

cd "$source_root"
env -u PYTHONPATH -u PYTHONHOME OMNI_KIT_ACCEPT_EULA=YES \
  MPLCONFIGDIR="${project_root}/cache/matplotlib" setsid \
  timeout --signal=TERM --kill-after=30s 900s \
  "$python_bin" "${runner_args[@]}" > "$runner_log" 2>&1 &
launcher_pid=$!
sleep 1
launcher_pgid="$(ps -o pgid= -p "$launcher_pid" 2>/dev/null | tr -d '[:space:]' || true)"
supervisor_pgid="$(ps -o pgid= -p $$ 2>/dev/null | tr -d '[:space:]' || true)"
if [[ ! "$launcher_pgid" =~ ^[0-9]+$ || "$launcher_pgid" == "$supervisor_pgid" ]]; then
  echo "DT1 launcher did not create an isolated owned process group" >&2
  exit 70
fi
process_group_verified=true

printf '{"format":"go2-dual-target-dt1-exclusive-launcher-v1","stage":"DT1","mode":"%s","run_id":"%s","launcher_pid":%s,"launcher_pgid":%s,"python":"%s","timeout_s":900,"group_id":"%s","color_configuration":"%s","target_color":"%s","repeat":%s,"motion_calibration":"%s","paired_reset":%s,"runner_log":"%s"}\n' \
  "$mode" "$run_id" "$launcher_pid" "$launcher_pgid" "$python_bin" "$group_id" "$color_configuration" "$target_color" "$repeat" "$motion_calibration" "$paired_reset" "$runner_log" > "$receipt_json"

set +e
wait "$launcher_pid"
runner_exit=$?
set -e

stage_status="MISSING"
result_status="MISSING"
if [[ -r "$stage_status_path" ]]; then
  if [[ "$mode" == motion ]]; then
    result_key="motion_precheck_result"
  else
    result_key="episode_result"
  fi
  read_result="$("$python_bin" -c 'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); k=sys.argv[2]; print(d.get("status", "")); print(d.get(k, {}).get("status", ""))' "$stage_status_path" "$result_key" 2>/dev/null || true)"
  stage_status="$(printf '%s\n' "$read_result" | sed -n '1p')"
  result_status="$(printf '%s\n' "$read_result" | sed -n '2p')"
fi
printf '{"format":"go2-dual-target-dt1-exclusive-launcher-summary-v1","stage":"DT1","mode":"%s","run_id":"%s","exit_code":%d,"stage_status":"%s","result_status":"%s","runner_log":"%s","receipt":"%s"}\n' \
  "$mode" "$run_id" "$runner_exit" "$stage_status" "$result_status" "$runner_log" "$receipt_json" > "$summary_json"

if [[ "$mode" == motion ]]; then
  expected_status="MOTION_PRECHECK_PASSED"
else
  expected_status="success"
fi
[[ "$runner_exit" -eq 0 && "$stage_status" == RUNNING && "$result_status" == "$expected_status" ]] || exit 1
