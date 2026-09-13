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

from collectors.sim_go2.utils.action_utils import is_valid_action_mask
from collectors.sim_go2.utils.io import dump_json, iter_jsonl


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate packed Isaac Sim datasets.")
    parser.add_argument("--input", type=Path, required=True, help="Packed dataset root or dataset.jsonl path.")
    parser.add_argument("--report", type=Path, default=None, help="Optional output report path.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_path = args.input if args.input.is_file() else args.input / "dataset.jsonl"
    image_root = args.input.parent if args.input.is_file() else args.input
    issues: list[dict[str, object]] = []
    sample_count = 0
    for record in iter_jsonl(dataset_path):
        sample_count += 1
        state = record.get("state", [])
        chunk = record.get("action_chunk", [])
        mask = record.get("action_mask", [])
        if len(state) != 3:
            issues.append({"sample": sample_count, "issue": "state_len_not_3"})
        if len(chunk) != 4:
            issues.append({"sample": sample_count, "issue": "chunk_len_not_4"})
        if len(mask) != 4 or not is_valid_action_mask(mask):
            issues.append({"sample": sample_count, "issue": "invalid_action_mask"})
        if not record.get("visibility_bucket"):
            issues.append({"sample": sample_count, "issue": "missing_visibility_bucket"})
        if "target_visible_within_3f" not in record or "target_visible_within_10f" not in record:
            issues.append({"sample": sample_count, "issue": "missing_visibility_metadata"})
        image_path = image_root / str(record.get("image", ""))
        if not image_path.exists():
            issues.append({"sample": sample_count, "issue": "image_missing", "image": str(image_path)})
        for step in chunk:
            if not isinstance(step, list) or len(step) != 2:
                issues.append({"sample": sample_count, "issue": "action_step_not_2d"})
                break
    report = {"dataset_path": str(dataset_path), "sample_count": sample_count, "issue_count": len(issues), "issues": issues[:200]}
    if args.report:
        dump_json(args.report, report)
    print(json.dumps({k: report[k] for k in ("sample_count", "issue_count")}, ensure_ascii=False))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
