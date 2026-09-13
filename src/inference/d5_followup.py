#!/usr/bin/env python3
"""Select D5 phased closed-loop repeat routes from first-seed evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


PRIMARY_SEED = 20260831
ALWAYS_REPEAT = "short_vln_v1_0004"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closed-loop-json", type=Path, required=True)
    return parser.parse_args()


def select_followups(rows: list[dict]) -> list[str]:
    by_episode: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if int(row["policy_seed"]) == PRIMARY_SEED:
            by_episode[str(row["episode_id"])].append(row)
    output = {ALWAYS_REPEAT}
    for episode, entries in by_episode.items():
        if episode == ALWAYS_REPEAT or len(entries) < 3:
            continue
        success = {bool(item["success"]) for item in entries}
        errors = [float(item["final_navigation_error_m"]) for item in entries]
        if len(success) > 1 or (max(errors) - min(errors)) / max(1e-9, min(errors)) >= 0.20:
            output.add(episode)
    return sorted(output)


def main() -> None:
    args = parse_args()
    report = json.loads(args.closed_loop_json.read_text(encoding="utf-8"))
    for episode in select_followups(report["rows"]):
        print(episode)


if __name__ == "__main__":
    main()
