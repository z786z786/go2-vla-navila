#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/mnt/wxh/go2_short_vln
SOURCE_ROOT=/home/wxh/go2_short_vln
PYTHON_BIN="$PROJECT_ROOT/envs/conda/navila-isaac/bin/python"
CANDIDATES="$PROJECT_ROOT/outputs/navila_dual_target_v4/layout_candidates_pending_visibility.json"
OUT_ROOT="$PROJECT_ROOT/outputs/navila_dual_target_v4/live_visibility"
LOG="$OUT_ROOT/collection.log"
mkdir -p "$OUT_ROOT"

echo "$(date -Is) waiting for GPU" >> "$LOG"
while true; do
  used=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')
  # One-env visibility collection is allowed to coexist with the user's
  # CARLA process; the validated smoke run stayed within the remaining VRAM.
  if [[ "$used" -lt 12000 ]]; then break; fi
  sleep 30
done
echo "$(date -Is) GPU available (compute MiB: ${used:-0})" >> "$LOG"

for scene in 2azQ1b91cZZ zsNo4HB9uLZ Z6MFQCViBuw x8F5xyUWy9e QUCTc6BB5sX TbHJrupSAjP EU6Fwq7SyZv X7HyMhZNoso; do
  output="$OUT_ROOT/${scene}.json"
  image_dir="$OUT_ROOT/images/${scene}"
  if [[ -e "$output" ]]; then
    echo "$(date -Is) skip existing $scene" >> "$LOG"
    continue
  fi
  echo "$(date -Is) start $scene" >> "$LOG"
  cd "$SOURCE_ROOT"
  env -u PYTHONHOME PYTHONPATH="$SOURCE_ROOT" OMNI_KIT_ACCEPT_EULA=YES \
    timeout --signal=TERM --kill-after=30s 7200s "$PYTHON_BIN" \
    scripts/navila_dual_target_visibility.py --headless --enable_cameras \
    --candidates "$CANDIDATES" --scene "$scene" --output "$output" --image-dir "$image_dir" \
    >> "$LOG" 2>&1
  echo "$(date -Is) done $scene" >> "$LOG"
done
MERGED="$PROJECT_ROOT/outputs/navila_dual_target_v4/live_visibility_merged.json"
MANIFEST="$PROJECT_ROOT/outputs/navila_dual_target_v4/navila_dual_target_split_manifest.json"
REPORT="$PROJECT_ROOT/outputs/navila_dual_target_v4/navila_dual_target_split_audit.md"
if [[ ! -e "$MERGED" ]]; then
  env -u PYTHONHOME PYTHONPATH="$SOURCE_ROOT" "$PYTHON_BIN" "$SOURCE_ROOT/scripts/navila_merge_visibility.py" \
    --input-dir "$OUT_ROOT" --output "$MERGED" >> "$LOG" 2>&1
fi
if [[ ! -e "$MANIFEST" ]]; then
  env -u PYTHONHOME PYTHONPATH="$SOURCE_ROOT" "$PYTHON_BIN" -m src.dual_target.navila_ood_split \
    --candidates "$MERGED" --output "$MANIFEST" --report "$REPORT" >> "$LOG" 2>&1
fi
echo "$(date -Is) collection complete" >> "$LOG"
