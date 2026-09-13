#!/usr/bin/env bash
# One bounded, exclusive DT1 physical-contact diagnostic.  It runs no
# navigation planner, learner, SmolVLA model, trainer, or DT2 component.
set -euo pipefail

project_root="/mnt/wxh/go2_short_vln"
source_root="/home/wxh/go2_short_vln"
python_bin="${project_root}/envs/conda/navila-isaac/bin/python"
output_root="${project_root}/outputs/dual_target_v1"
run_id=""
target_color=""
go2_usd=""
launcher_pid=""
launcher_pgid=""
process_group_verified=false

usage() {
  echo "usage: $0 --run-id ID --target-color red|blue --go2-usd /absolute/path/to/go2.usd" >&2
  exit 64
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      [[ $# -ge 2 ]] || usage
      run_id="$2"
      shift 2
      ;;
    --target-color)
      [[ $# -ge 2 ]] || usage
      target_color="$2"
      shift 2
      ;;
    --go2-usd)
      [[ $# -ge 2 ]] || usage
      go2_usd="$2"
      shift 2
      ;;
    *) usage ;;
  esac
done

[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{2,80}$ ]] || usage
[[ "$target_color" == red || "$target_color" == blue ]] || usage
[[ "$go2_usd" = /* && -f "$go2_usd" ]] || usage
[[ -x "$python_bin" && -d "$source_root" ]] || {
  echo "DT1 Navila Python/source root unavailable" >&2
  exit 66
}

mkdir -p "$output_root" "${project_root}/cache/matplotlib"
runner_log="${output_root}/dt1_contact_precheck_${run_id}_${target_color}_runner.log"
receipt_json="${output_root}/dt1_contact_precheck_${run_id}_${target_color}_receipt.json"
summary_json="${output_root}/dt1_contact_precheck_${run_id}_${target_color}_summary.json"
stage_status_path="${output_root}/${run_id}/stage_status.json"
for path in "$runner_log" "$receipt_json" "$summary_json" "${output_root}/${run_id}"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite $path" >&2; exit 73; }
done

stop_owned_group() {
  if [[ "$process_group_verified" != true ]]; then
    # Before we establish a private session, only the direct child is known
    # to be ours.  Never send a group signal based on an unverified PGID.
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

cd "$source_root"
env -u PYTHONPATH -u PYTHONHOME OMNI_KIT_ACCEPT_EULA=YES \
  MPLCONFIGDIR="${project_root}/cache/matplotlib" setsid \
  timeout --signal=TERM --kill-after=30s 900s \
  "$python_bin" -m src.dual_target.runner --live --contact-precheck --headless --enable_cameras \
  --go2-usd "$go2_usd" --run-id "$run_id" --output-dir "$output_root" \
  --gpu-project-root "$project_root" --checkpoint \
  "${project_root}/third_party/NaVILA-Bench/logs/rsl_rl/go2_vision/2024-09-25_23-22-02/model_26499.pt" \
  --low-level-device cuda:0 --target-color "$target_color" --max-env-steps 300 \
  --contact-threshold-n 1.0 > "$runner_log" 2>&1 &
launcher_pid=$!
sleep 1
launcher_pgid="$(ps -o pgid= -p "$launcher_pid" 2>/dev/null | tr -d '[:space:]' || true)"
supervisor_pgid="$(ps -o pgid= -p $$ 2>/dev/null | tr -d '[:space:]' || true)"
if [[ ! "$launcher_pgid" =~ ^[0-9]+$ || "$launcher_pgid" == "$supervisor_pgid" ]]; then
  echo "contact-precheck launcher did not create an isolated owned process group" >&2
  exit 70
fi
process_group_verified=true

printf '{"format":"go2-dual-target-dt1-contact-precheck-launcher-v1","stage":"DT1","run_id":"%s","target_color":"%s","launcher_pid":%s,"launcher_pgid":%s,"python":"%s","timeout_s":900,"positive_max_env_steps":300,"contact_threshold_n":1.0,"runner_log":"%s"}\n' \
  "$run_id" "$target_color" "$launcher_pid" "$launcher_pgid" "$python_bin" "$runner_log" > "$receipt_json"

set +e
wait "$launcher_pid"
runner_exit=$?
set -e
# Keep the verified PGID through EXIT cleanup.  A timeout parent can exit
# before an Isaac child; the final group check is the ownership proof that no
# local GPU process was left behind.

stage_status="MISSING"
precheck_status="MISSING"
if [[ -r "$stage_status_path" ]]; then
  read_result="$("$python_bin" -c 'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); print(d.get("status", "")); print(d.get("contact_precheck_result", {}).get("status", ""))' "$stage_status_path" 2>/dev/null || true)"
  stage_status="$(printf '%s\n' "$read_result" | sed -n '1p')"
  precheck_status="$(printf '%s\n' "$read_result" | sed -n '2p')"
fi
printf '{"format":"go2-dual-target-dt1-contact-precheck-summary-v1","stage":"DT1","run_id":"%s","target_color":"%s","exit_code":%d,"stage_status":"%s","contact_precheck_status":"%s","runner_log":"%s","receipt":"%s"}\n' \
  "$run_id" "$target_color" "$runner_exit" "$stage_status" "$precheck_status" "$runner_log" "$receipt_json" > "$summary_json"

[[ "$runner_exit" -eq 0 && "$stage_status" == RUNNING && "$precheck_status" == CONTACT_PRECHECK_PASSED ]] || exit 1
