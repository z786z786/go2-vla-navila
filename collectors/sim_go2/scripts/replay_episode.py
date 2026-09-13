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
import json
from pathlib import Path

from collectors.sim_go2.utils.io import iter_jsonl, load_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay one raw episode in text form.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--episode-id", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    meta = load_json(args.raw_root / "episodes" / args.episode_id / "meta.json")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    for index, step in enumerate(iter_jsonl(args.raw_root / "episodes" / args.episode_id / "steps.jsonl")):
        if index >= args.max_steps:
            break
        print(
            f"step={step['step_id']:04d} img={step['image_path']} state={step['state']} train={step['raw_cmd_train']} full={step['raw_cmd_full']}"
        )


if __name__ == "__main__":
    main()
