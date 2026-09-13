#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 BULK_QUEUE OUTPUT_ROOT COLLECTION_EVIDENCE_ROOT FINAL_MANIFEST" >&2
  exit 2
fi

QUEUE="$1"
OUTPUT_ROOT="$2"
COLLECTION_EVIDENCE_ROOT="$3"
FINAL_MANIFEST="$4"
PROJECT=/home/wxh/go2_short_vln
ROOT=/mnt/wxh/go2_short_vln
PREFIX="$ROOT/envs/conda/navila-isaac"
DATASET="$ROOT/assets/vln_ce_isaac/vln_ce_isaac_v1.json.gz"

if [[ ! -f "$QUEUE" || ! -f "$DATASET" ]]; then
  echo "Missing bulk queue or official dataset" >&2
  exit 2
fi
if [[ -e "$FINAL_MANIFEST" ]]; then
  echo "Refusing to overwrite an existing final manifest: $FINAL_MANIFEST" >&2
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
    print("\t".join((item["source_episode_id"], item["route_id"], item["collection_scope"], item["category"])))
' "$QUEUE"
)

cd "$PROJECT"
if "$PREFIX/bin/python" -m src.navila_full.try_finalize \
  --bulk-queue "$QUEUE" --collection-root "$COLLECTION_EVIDENCE_ROOT" --output "$FINAL_MANIFEST"; then
  exit 0
fi

for row in "${ROUTES[@]}"; do
  IFS=$'\t' read -r SOURCE_EPISODE_ID ROUTE_ID SCOPE CATEGORY <<< "$row"
  EPISODE_DIR="$OUTPUT_ROOT/$ROUTE_ID"
  if [[ -d "$EPISODE_DIR" ]]; then
    echo "BULK skip existing $SCOPE/$CATEGORY source=$SOURCE_EPISODE_ID route=$ROUTE_ID"
  else
    echo "BULK collect $SCOPE/$CATEGORY source=$SOURCE_EPISODE_ID route=$ROUTE_ID"
    if "$PREFIX/bin/python" -m src.collector.collect_full_episode \
      --dataset "$DATASET" \
      --source-episode-id "$SOURCE_EPISODE_ID" \
      --expected-route-id "$ROUTE_ID" \
      --episode-dir "$EPISODE_DIR" \
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
      --enable_cameras; then
      echo "BULK accepted route=$ROUTE_ID"
    else
      echo "BULK rejected route=$ROUTE_ID"
    fi
  fi
  if "$PREFIX/bin/python" -m src.navila_full.try_finalize \
    --bulk-queue "$QUEUE" --collection-root "$COLLECTION_EVIDENCE_ROOT" --output "$FINAL_MANIFEST"; then
    exit 0
  fi
done

echo "Bulk queue exhausted before a valid 30/12/12 split was available" >&2
exit 1
