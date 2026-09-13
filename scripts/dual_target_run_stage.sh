#!/usr/bin/env bash
# Single-stage entry. DT0 remains CPU-only; DT1 has an explicit CPU self-check
# and a separately explicit live Isaac path guarded by the DT1 GPU waiter.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 DT0 [--gpu-wait-dry-run] | DT1 --cpu-self-check | DT1 --live --go2-usd PATH" >&2
  exit 64
fi

stage="$1"
shift
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

case "$stage" in
  DT0)
    exec python3 -m src.dual_target.stage "$stage" "$@"
    ;;
  DT1)
    exec python3 -m src.dual_target.runner "$@"
    ;;
  *)
    echo "refusing ${stage}: DT2+ require a later parent-approved implementation" >&2
    exit 64
    ;;
esac
