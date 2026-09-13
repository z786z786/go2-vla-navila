#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "Usage: $0 OUTPUT_ROOT EPISODE_ID_1 EPISODE_ID_2 EPISODE_ID_3 EPISODE_ID_4 EPISODE_ID_5 EPISODE_ID_6" >&2
  exit 2
fi

OUTPUT_ROOT="$1"
shift
PROJECT=/home/wxh/go2_short_vln
ROOT=/mnt/wxh/go2_short_vln
PREFIX="$ROOT/envs/conda/navila-isaac"
RUNNER="$PROJECT/scripts/m4_run_episode.sh"
EXPECTED_STATE_DIM="${M6_2_R7_EXPECTED_STATE_DIM:-31}"
HOLD_FRAMES=50

if [[ ! "$EXPECTED_STATE_DIM" =~ ^[0-9]+$ ]] || (( EXPECTED_STATE_DIM < 1 )); then
  echo "M6_2_R7_EXPECTED_STATE_DIM must be a positive integer" >&2
  exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to reuse an existing canary output root: $OUTPUT_ROOT" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
episode_dirs=()
for short_id in "$@"; do
  M6_2_R7_TERMINAL_HOLD_FRAMES="$HOLD_FRAMES" "$RUNNER" "$short_id" "$OUTPUT_ROOT"
  "$PREFIX/bin/python" -c '
import json
import pathlib
import sys
sanity = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if sanity.get("passed") is not True:
    failed = [key for key, value in sanity.get("checks", {}).items() if value is not True]
    raise SystemExit("R7 canary fail-fast sanity failure: " + ",".join(failed))
' "$OUTPUT_ROOT/$short_id/sanity.json"
  episode_dirs+=("$OUTPUT_ROOT/$short_id")
done

cd "$PROJECT"
"$PREFIX/bin/python" -m src.collector.r7_canary \
  --episode-dirs "${episode_dirs[@]}" \
  --expected-state-dim "$EXPECTED_STATE_DIM" \
  --expected-terminal-hold-frames "$HOLD_FRAMES" \
  --output "$OUTPUT_ROOT/r7_canary_quality.json"

echo "R7 canary quality report: $OUTPUT_ROOT/r7_canary_quality.json"
