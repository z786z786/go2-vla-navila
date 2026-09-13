#!/usr/bin/env bash
# Run the predeclared DT1 16-slot expert matrix with one fresh exclusive GPU
# admission per slot.  This launcher contains no retry/best-of selection,
# learner, SmolVLA, trainer, DT2 action, or approval path.
set -euo pipefail

project_root="/mnt/wxh/go2_short_vln"
source_root="/home/wxh/go2_short_vln"
python_bin="${project_root}/envs/conda/navila-isaac/bin/python"
output_root="${project_root}/outputs/dual_target_v1"
matrix_run_id=""
go2_usd=""
motion_calibration=""

usage() {
  echo "usage: $0 --matrix-run-id ID --go2-usd /absolute/go2.usd --motion-calibration /absolute/motion_calibration.json" >&2
  exit 64
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --matrix-run-id) [[ $# -ge 2 ]] || usage; matrix_run_id="$2"; shift 2 ;;
    --go2-usd) [[ $# -ge 2 ]] || usage; go2_usd="$2"; shift 2 ;;
    --motion-calibration) [[ $# -ge 2 ]] || usage; motion_calibration="$2"; shift 2 ;;
    *) usage ;;
  esac
done

[[ "$matrix_run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{2,48}$ ]] || usage
[[ "$go2_usd" = /* && -f "$go2_usd" ]] || usage
[[ "$motion_calibration" = /* && -f "$motion_calibration" ]] || usage
[[ -x "$python_bin" && -d "$source_root" ]] || { echo "DT1 Navila Python/source root unavailable" >&2; exit 66; }

cd "$source_root"
plan_log="${output_root}/dt1_matrix_${matrix_run_id}_wait_and_launch.log"
receipt_json="${output_root}/dt1_matrix_${matrix_run_id}_receipt.json"
events_jsonl="${output_root}/dt1_matrix_${matrix_run_id}_events.jsonl"
for path in "$plan_log" "$receipt_json" "$events_jsonl"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite $path" >&2; exit 73; }
done
"$python_bin" -m src.dual_target.expert_matrix_launch --output-dir "$output_root" \
  --matrix-run-id "$matrix_run_id" --launcher-script scripts/dual_target_dt1_matrix.sh --create-plan
plan_path="${output_root}/${matrix_run_id}/expert_matrix_plan.json"
printf '{"format":"go2-dual-target-dt1-matrix-launcher-v1","stage":"DT1","matrix_run_id":"%s","pid":%s,"plan":"%s","plan_sha256":"%s","launcher":"%s","launcher_sha256":"%s","per_slot_admission":"exclusive_3_samples_30s_then_locked_recheck","retry_policy":"no_per_slot_retry_or_best_of_selection"}\n' \
  "$matrix_run_id" "$$" "$plan_path" "$(shasum -a 256 "$plan_path" | awk '{print $1}')" \
  "${source_root}/scripts/dual_target_dt1_matrix.sh" "$(shasum -a 256 scripts/dual_target_dt1_matrix.sh | awk '{print $1}')" > "$receipt_json"

current_child_pid=""
current_slot=""
stop_current_child() {
  [[ -n "$current_child_pid" ]] || return 0
  kill -TERM "$current_child_pid" 2>/dev/null || true
  # The child may be the reviewed expert launcher, which itself sends TERM to
  # a verified Isaac process group and grants it ten seconds.  Do not race
  # that cleanup from this parent; retain a longer bounded grace period.
  for _ in {1..35}; do
    kill -0 "$current_child_pid" 2>/dev/null || {
      printf '{"event":"child_cleanup","slot_run_id":"%s","child_pid":%s,"fallback_kill":false}\n' \
        "$current_slot" "$current_child_pid" >> "$events_jsonl"
      current_child_pid=""
      return 0
    }
    sleep 1
  done
  kill -KILL "$current_child_pid" 2>/dev/null || true
  printf '{"event":"child_cleanup","slot_run_id":"%s","child_pid":%s,"fallback_kill":true}\n' \
    "$current_slot" "$current_child_pid" >> "$events_jsonl"
}
on_signal() {
  printf '{"event":"matrix_interrupted","slot_run_id":"%s","child_pid":%s}\n' \
    "$current_slot" "${current_child_pid:-null}" >> "$events_jsonl"
  stop_current_child
  exit 130
}
trap on_signal INT TERM
trap 'stop_current_child || true' EXIT

run_current() {
  local kind="$1"
  shift
  "$@" &
  current_child_pid=$!
  printf '{"event":"child_started","kind":"%s","slot_run_id":"%s","child_pid":%s}\n' \
    "$kind" "$current_slot" "$current_child_pid" >> "$events_jsonl"
  set +e
  wait "$current_child_pid"
  local code=$?
  set -e
  printf '{"event":"child_exited","kind":"%s","slot_run_id":"%s","child_pid":%s,"exit_code":%d}\n' \
    "$kind" "$current_slot" "$current_child_pid" "$code" >> "$events_jsonl"
  current_child_pid=""
  return "$code"
}

# Each line was written before GPU work begins.  Do not reorder, regenerate,
# skip, or retry a slot here: a diagnostic revision needs a new matrix ID.
failed_slot=""
while IFS=$'\t' read -r slot_run_id group_id color_configuration target_color repeat; do
  current_slot="$slot_run_id"
  printf '{"event":"slot_started","slot_run_id":"%s","group_id":"%s","color_configuration":"%s","target_color":"%s","repeat":%s}\n' \
    "$slot_run_id" "$group_id" "$color_configuration" "$target_color" "$repeat" >> "$events_jsonl"
  if ! run_current "gpu_waiter" env -u PYTHONPATH -u PYTHONHOME "$python_bin" -m src.dual_target.gpu_wait \
      --project-root "$project_root" --run-id "$slot_run_id" >> "$plan_log" 2>&1; then
    failed_slot="$slot_run_id"
    break
  fi
  if ! run_current "expert_launcher" env PATH="$(dirname "$python_bin"):${PATH}" bash scripts/dual_target_dt1_expert.sh \
      --mode expert --run-id "$slot_run_id" --go2-usd "$go2_usd" \
      --group-id "$group_id" --color-configuration "$color_configuration" \
      --target-color "$target_color" --repeat "$repeat" \
      --motion-calibration "$motion_calibration" --paired-reset >> "$plan_log" 2>&1; then
    failed_slot="$slot_run_id"
    break
  fi
  printf '{"event":"slot_completed","slot_run_id":"%s"}\n' "$slot_run_id" >> "$events_jsonl"
done < <("$python_bin" -m src.dual_target.expert_matrix_launch --output-dir "$output_root" \
  --matrix-run-id "$matrix_run_id" --print-slots)

set +e
"$python_bin" -m src.dual_target.expert_matrix_launch --output-dir "$output_root" \
  --matrix-run-id "$matrix_run_id" --collect >> "$plan_log" 2>&1
collect_code=$?
set -e
printf '{"event":"matrix_collection_finished","failed_slot":"%s","collect_exit_code":%d}\n' \
  "$failed_slot" "$collect_code" >> "$events_jsonl"
[[ -z "$failed_slot" && "$collect_code" -eq 0 ]] || exit 1
