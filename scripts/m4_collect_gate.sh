#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 OUTPUT_ROOT EPISODE_ID_1 ... EPISODE_ID_5 [MORE_IDS...]" >&2
  exit 2
fi

OUTPUT_ROOT="$1"
shift
PROJECT=/home/wxh/go2_short_vln
ROOT=/mnt/wxh/go2_short_vln
PREFIX="$ROOT/envs/conda/navila-isaac"
RUNNER="$PROJECT/scripts/m4_run_episode.sh"

mkdir -p "$OUTPUT_ROOT"
episode_dirs=()
for short_id in "$@"; do
  "$RUNNER" "$short_id" "$OUTPUT_ROOT"
  episode_dirs+=("$OUTPUT_ROOT/$short_id")
done

"$PREFIX/bin/python" "$PROJECT/src/collector/check_expert.py" \
  --episode-dirs "${episode_dirs[@]}" \
  --report-json "$OUTPUT_ROOT/m4_gate_report.json" \
  --report-md "$OUTPUT_ROOT/m4_gate_report.md"

echo "M4 gate report: $OUTPUT_ROOT/m4_gate_report.md"
