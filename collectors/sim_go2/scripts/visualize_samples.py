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
import html
import json
import random
from pathlib import Path

from collectors.sim_go2.utils.io import iter_jsonl

CSS = "body{font-family:sans-serif;background:#f7f3ea;color:#222;padding:24px} .card{background:#fff;border:1px solid #ddd;border-radius:10px;padding:14px;margin:14px 0} img{max-width:480px;border-radius:8px;display:block;margin-bottom:8px} code{background:#f2f2f2;padding:2px 4px;border-radius:4px}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a lightweight HTML gallery for packed samples.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_root = args.input if args.input.is_dir() else args.input.parent
    dataset_path = args.input if args.input.is_file() else args.input / "dataset.jsonl"
    rows = list(iter_jsonl(dataset_path))
    rng = random.Random(args.seed)
    rows = rng.sample(rows, min(args.count, len(rows)))
    out_path = args.output or dataset_root / "sample_gallery.html"
    cards = []
    for row in rows:
        image_rel = str(row.get("image", ""))
        cards.append(
            "<div class='card'>"
            f"<img src='{html.escape(image_rel)}' alt='sample' />"
            f"<div><strong>{html.escape(str(row.get('instruction', '')))}</strong></div>"
            f"<div>visibility={html.escape(str(row.get('visibility_bucket', '')))} "
            f"visible3={html.escape(str(row.get('target_visible_within_3f', False)))}</div>"
            f"<div>state={html.escape(json.dumps(row.get('state', [])))}</div>"
            f"<div>action_chunk={html.escape(json.dumps(row.get('action_chunk', [])))}</div>"
            f"<div>episode={html.escape(str(row.get('episode_id', '')))} step={html.escape(str(row.get('step_id', '')))}</div>"
            "</div>"
        )
    out_path.write_text(f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{''.join(cards)}</body></html>", encoding='utf-8')
    print(out_path)


if __name__ == "__main__":
    main()
