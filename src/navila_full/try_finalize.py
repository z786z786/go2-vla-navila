#!/usr/bin/env python3
"""Attempt an atomic final split; report pending without a traceback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.navila_full.bulk_collection import validate_bulk_queue
from src.navila_full.finalize_selection import final_manifest, validate_final_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bulk-queue", type=Path, required=True)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    queue = json.loads(args.bulk_queue.read_text(encoding="utf-8"))
    errors = validate_bulk_queue(queue)
    if errors:
        raise ValueError("invalid bulk queue: " + "; ".join(errors))
    try:
        manifest = final_manifest(
            Path(queue["candidate_manifest_path"]),
            args.collection_root,
            Path(queue["canary_manifest_path"]),
            seed=int(queue["seed"]),
        )
    except (ValueError, RuntimeError) as error:
        print(json.dumps({"status": "pending", "reason": str(error)}, ensure_ascii=False))
        raise SystemExit(1)
    validation = validate_final_manifest(manifest)
    if validation:
        raise RuntimeError("invalid final manifest: " + "; ".join(validation))
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"status": "complete", "output": str(args.output), "final_routes": len(manifest["splits"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
