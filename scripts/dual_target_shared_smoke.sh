#!/usr/bin/env bash
# The only user-authorized shared-GPU launcher: one DT1, 50-step Isaac/Go2
# smoke.  It is deliberately not a general low-memory stage entry.
set -euo pipefail

project_root="/mnt/wxh/go2_short_vln"
source_root="/home/wxh/go2_short_vln"
python_bin="${project_root}/envs/conda/navila-isaac/bin/python"
output_root="${project_root}/outputs/dual_target_v1"
min_free_mib=2048
run_id=""
go2_usd=""

usage() {
  echo "usage: $0 --run-id ID --go2-usd /absolute/path/to/go2.usd" >&2
  exit 64
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      [[ $# -ge 2 ]] || usage
      run_id="$2"
      shift 2
      ;;
    --go2-usd)
      [[ $# -ge 2 ]] || usage
      go2_usd="$2"
      shift 2
      ;;
    *)
      usage
      ;;
  esac
done

[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{2,80}$ ]] || usage
[[ "$go2_usd" = /* && -f "$go2_usd" ]] || usage
[[ -x "$python_bin" && -d "$source_root" ]] || { echo "DT1 navila Python/source root unavailable" >&2; exit 66; }

mkdir -p "$output_root"
mkdir -p "${project_root}/cache/matplotlib"
runner_log="${output_root}/dt1_shared_smoke_${run_id}_runner.log"
samples_jsonl="${output_root}/dt1_shared_smoke_${run_id}_supervisor_gpu_samples.jsonl"
receipt_json="${output_root}/dt1_shared_smoke_${run_id}_supervisor_receipt.json"
summary_json="${output_root}/dt1_shared_smoke_${run_id}_supervisor_summary.json"
stage_status_path="${output_root}/${run_id}/stage_status.json"
for path in "$runner_log" "$samples_jsonl" "$receipt_json" "$summary_json"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite $path" >&2; exit 73; }
done

launcher_pid=""
launcher_pgid=""
process_group_verified=false
sample_count=0
min_observed_free_mib=""
owned_peak_mib=0
abort_reason=""
stage_status="MISSING"

write_summary() {
  local runner_exit="$1"
  printf '{"format":"go2-dual-target-dt1-shared-smoke-supervisor-summary-v1","stage":"DT1","run_id":"%s","launcher_pid":%s,"launcher_pgid":%s,"exit_code":%d,"sample_count":%d,"min_free_mib":%s,"owned_process_peak_mib":%d,"abort_reason":"%s","stage_status":"%s","stage_status_path":"%s","samples_jsonl":"%s","runner_log":"%s"}\n' \
    "$run_id" "$launcher_pid" "${launcher_pgid:-null}" "$runner_exit" "$sample_count" "${min_observed_free_mib:-null}" "$owned_peak_mib" "$abort_reason" "$stage_status" "$stage_status_path" "$samples_jsonl" "$runner_log" > "$summary_json"
}

read_stage_status() {
  local detected
  if [[ ! -r "$stage_status_path" ]]; then
    stage_status="MISSING"
    abort_reason="${abort_reason:-stage_status_missing}"
    return 1
  fi
  if ! detected="$("$python_bin" -c 'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")).get("status"); assert isinstance(value, str); print(value)' "$stage_status_path" 2>/dev/null)"; then
    stage_status="INVALID"
    abort_reason="${abort_reason:-stage_status_invalid}"
    return 1
  fi
  case "$detected" in
    RUNNING|FAILED|INTERRUPTED_RESUMABLE)
      stage_status="$detected"
      return 0
      ;;
    *)
      stage_status="INVALID"
      abort_reason="${abort_reason:-stage_status_invalid}"
      return 1
      ;;
  esac
}

stop_owned_group() {
  [[ -n "$launcher_pid" ]] || return 0
  if [[ "$process_group_verified" != true || -z "$launcher_pgid" ]]; then
    # A direct child is still ours; never risk signaling an unverified group.
    kill -TERM "$launcher_pid" 2>/dev/null || true
    return 0
  fi
  if ! kill -0 -- "-${launcher_pgid}" 2>/dev/null; then
    return 1
  fi
  kill -TERM -- "-${launcher_pgid}" 2>/dev/null || true
  for _ in {1..10}; do
    # `timeout` may exit before its Isaac child.  Only the whole verified
    # session/process group demonstrates that owned GPU work is gone.
    kill -0 -- "-${launcher_pgid}" 2>/dev/null || return 0
    sleep 1
  done
  kill -KILL -- "-${launcher_pgid}" 2>/dev/null || true
}

cleanup_owned_group_on_exit() {
  trap - EXIT
  stop_owned_group || true
}

trap 'abort_reason="supervisor_interrupted"; stop_owned_group; exit 130' INT TERM
trap cleanup_owned_group_on_exit EXIT

sample_gpu() {
  local timestamp gpu_line process_lines total_mib used_mib free_mib
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
  if ! gpu_line="$(timeout 5s nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null)"; then
    printf '{"timestamp_utc":"%s","sample_index":%d,"monitor_error":"gpu_query_failed","launcher_pid":%s}\n' \
      "$timestamp" "$((sample_count + 1))" "$launcher_pid" >> "$samples_jsonl"
    abort_reason="gpu_query_failed"
    return 1
  fi
  [[ "$(printf '%s\n' "$gpu_line" | sed '/^[[:space:]]*$/d' | wc -l | tr -d '[:space:]')" == "1" ]] || {
    abort_reason="gpu_query_must_have_one_row"; return 1;
  }
  IFS=',' read -r total_mib used_mib <<< "$gpu_line"
  total_mib="${total_mib//[[:space:]]/}"
  used_mib="${used_mib//[[:space:]]/}"
  [[ "$total_mib" =~ ^[0-9]+$ && "$used_mib" =~ ^[0-9]+$ && "$used_mib" -le "$total_mib" ]] || {
    abort_reason="gpu_memory_parse_failed"; return 1;
  }
  free_mib=$((total_mib - used_mib))
  if ! process_lines="$(timeout 5s nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null)"; then
    abort_reason="compute_process_query_failed"
    return 1
  fi
  local processes_json="" pid memory_mib process_pgid owned owned_mib=0 separator=""
  while IFS=',' read -r pid memory_mib; do
    pid="${pid//[[:space:]]/}"
    memory_mib="${memory_mib//[[:space:]]/}"
    [[ -z "$pid" ]] && continue
    [[ "$pid" =~ ^[0-9]+$ && "$memory_mib" =~ ^[0-9]+$ ]] || { abort_reason="compute_process_parse_failed"; return 1; }
    process_pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
    owned=false
    if [[ "$process_pgid" == "$launcher_pgid" ]]; then
      owned=true
      owned_mib=$((owned_mib + memory_mib))
    fi
    processes_json+="${separator}{\"pid\":${pid},\"used_mib\":${memory_mib},\"pgid\":\"${process_pgid}\",\"owned_process_group\":${owned}}"
    separator="," 
  done <<< "$process_lines"
  sample_count=$((sample_count + 1))
  if [[ -z "$min_observed_free_mib" || "$free_mib" -lt "$min_observed_free_mib" ]]; then
    min_observed_free_mib="$free_mib"
  fi
  if [[ "$owned_mib" -gt "$owned_peak_mib" ]]; then
    owned_peak_mib="$owned_mib"
  fi
  printf '{"timestamp_utc":"%s","sample_index":%d,"total_mib":%d,"used_mib":%d,"free_mib":%d,"launcher_pid":%s,"launcher_pgid":%s,"owned_process_peak_mib_so_far":%d,"compute_processes":[%s]}\n' \
    "$timestamp" "$sample_count" "$total_mib" "$used_mib" "$free_mib" "$launcher_pid" "$launcher_pgid" "$owned_peak_mib" "$processes_json" >> "$samples_jsonl"
  if [[ "$free_mib" -lt "$min_free_mib" ]]; then
    abort_reason="free_memory_below_${min_free_mib}_MiB"
    return 1
  fi
  return 0
}

cd "$source_root"
env -u PYTHONPATH -u PYTHONHOME OMNI_KIT_ACCEPT_EULA=YES \
  MPLCONFIGDIR="${project_root}/cache/matplotlib" setsid \
  timeout --signal=TERM --kill-after=30s 900s \
  "$python_bin" -m src.dual_target.runner --live --shared-smoke --headless --enable_cameras \
  --go2-usd "$go2_usd" --run-id "$run_id" --output-dir "$output_root" \
  --gpu-project-root "$project_root" --checkpoint \
  "${project_root}/third_party/NaVILA-Bench/logs/rsl_rl/go2_vision/2024-09-25_23-22-02/model_26499.pt" \
  --low-level-device cuda:0 --max-env-steps 50 > "$runner_log" 2>&1 &
launcher_pid=$!
sleep 1
launcher_pgid="$(ps -o pgid= -p "$launcher_pid" 2>/dev/null | tr -d '[:space:]' || true)"
supervisor_pgid="$(ps -o pgid= -p $$ 2>/dev/null | tr -d '[:space:]' || true)"
if [[ "$launcher_pgid" =~ ^[0-9]+$ && "$launcher_pgid" != "$supervisor_pgid" ]]; then
  process_group_verified=true
fi
launcher_pgid_json="${launcher_pgid:-null}"
printf '{"format":"go2-dual-target-dt1-shared-smoke-supervisor-v1","stage":"DT1","run_id":"%s","purpose":"one_50_step_isaac_go2_smoke_no_smolvla_no_training","launcher_pid":%s,"launcher_pgid":%s,"process_group_verified":%s,"python":"%s","timeout_s":900,"term_grace_s":30,"min_free_mib_during_run":%d,"runner_log":"%s"}\n' \
  "$run_id" "$launcher_pid" "$launcher_pgid_json" "$process_group_verified" "$python_bin" "$min_free_mib" "$runner_log" > "$receipt_json"
[[ "$process_group_verified" == true ]] || {
  echo "shared-smoke launcher did not create an isolated owned process group" >&2
  abort_reason="launcher_process_group_not_isolated"
  stop_owned_group || true
  set +e
  wait "$launcher_pid"
  runner_exit=$?
  set -e
  write_summary "$runner_exit"
  exit 70
}

while kill -0 "$launcher_pid" 2>/dev/null; do
  if ! sample_gpu; then
    stop_owned_group || true
    break
  fi
  sleep 1
done
set +e
wait "$launcher_pid"
runner_exit=$?
set -e
if ! read_stage_status; then
  # A missing/invalid stage record is never a successful smoke even if Kit
  # teardown returned zero.  Preserve an existing monitor abort reason.
  runner_exit=1
elif [[ "$stage_status" == "FAILED" ]]; then
  # The runner writes this before app.close() with a full traceback.  Kit may
  # still return zero from close, so stage evidence is authoritative here.
  abort_reason="${abort_reason:-runner_stage_status_failed}"
  runner_exit=1
fi
write_summary "$runner_exit"
exit "$runner_exit"
