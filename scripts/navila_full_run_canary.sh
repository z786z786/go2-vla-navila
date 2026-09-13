#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 CANARY_MANIFEST OUTPUT_ROOT" >&2
  exit 2
fi

MANIFEST="$1"
OUTPUT_ROOT="$2"
PROJECT=/home/wxh/go2_short_vln
ROOT=/mnt/wxh/go2_short_vln
PREFIX="$ROOT/envs/conda/navila-isaac"
DATASET="$ROOT/assets/vln_ce_isaac/vln_ce_isaac_v1.json.gz"

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to reuse an existing canary output root: $OUTPUT_ROOT" >&2
  exit 2
fi
if [[ ! -f "$MANIFEST" || ! -f "$DATASET" ]]; then
  echo "Missing canary manifest or official dataset" >&2
  exit 2
fi

source "$PROJECT/scripts/m1_proxy_env.sh"
export OMNI_KIT_ACCEPT_EULA=YES
export MPLCONFIGDIR="$ROOT/cache/matplotlib"
export M1_ISAAC_ASSET_ROOT="$ROOT/assets/isaac_sim_4_1"
unset PYTHONPATH PYTHONHOME
mkdir -p "$OUTPUT_ROOT" "$MPLCONFIGDIR"

mapfile -t ROUTES < <(
  "$PREFIX/bin/python" -c '
import json, sys
for item in json.load(open(sys.argv[1], encoding="utf-8"))["routes"]:
    print("\t".join((item["source_episode_id"], item["route_id"])))
' "$MANIFEST"
)
if [[ ${#ROUTES[@]} -ne 6 ]]; then
  echo "Canary manifest must contain exactly six routes" >&2
  exit 2
fi

cd "$PROJECT"
for row in "${ROUTES[@]}"; do
  IFS=$'\t' read -r SOURCE_EPISODE_ID ROUTE_ID <<< "$row"
  "$PREFIX/bin/python" -m src.collector.collect_full_episode \
    --dataset "$DATASET" \
    --source-episode-id "$SOURCE_EPISODE_ID" \
    --expected-route-id "$ROUTE_ID" \
    --episode-dir "$OUTPUT_ROOT/$ROUTE_ID" \
    --official-source "$ROOT/third_party/NaVILA-Bench/scripts/demo_planner.py" \
    --asset-root "$ROOT/assets/isaac_sim_4_1" \
    --jpeg-quality 90 \
    --terminal-hold-frames 50 \
    --max-pd-frames 4000 \
    --task=go2_matterport_vision \
    --history_length=9 \
    --load_run=2024-09-25_23-22-02 \
    --num_envs=1 \
    --headless \
    --enable_cameras
done

"$PREFIX/bin/python" -m src.collector.full_episode_canary \
  --canary-manifest "$MANIFEST" \
  --collection-root "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/canary_quality.json"
