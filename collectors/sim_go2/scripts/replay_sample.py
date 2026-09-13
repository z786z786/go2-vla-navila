#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import argparse
import random
from pathlib import Path

from collectors.sim_go2.utils.io import iter_jsonl


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print a few random packed samples.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_path = args.input if args.input.is_file() else args.input / "dataset.jsonl"
    rows = list(iter_jsonl(dataset_path))
    rng = random.Random(args.seed)
    for row in rng.sample(rows, min(args.count, len(rows))):
        print(f"image={row['image']}")
        print(f"instruction={row['instruction']}")
        print(f"state={row['state']}")
        print(f"action_chunk={row['action_chunk']}")
        print(f"action_mask={row['action_mask']}")
        print(
            f"episode={row['episode_id']} step={row['step_id']} split={row.get('split','')} "
            f"visibility={row.get('visibility_bucket','')} visible3={row.get('target_visible_within_3f', False)}"
        )
        print("-")


if __name__ == "__main__":
    main()
