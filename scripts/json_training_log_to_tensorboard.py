#!/usr/bin/env python3
"""Backfill and follow JSON training rows into TensorBoard scalar events.

The source log may contain arbitrary non-JSON output around the training rows.
Progress is stored as a byte offset so restarting the bridge does not duplicate
already imported steps. A non-blocking file lock prevents multiple bridges from
writing the same run concurrently.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import time
from typing import Any


SCALARS = {
    "loss": "Loss/train",
    "val_loss": "Loss/seen_val",
    "gradient_norm": "Optimization/gradient_norm",
    "lr": "Optimization/learning_rate",
    "step_time_s": "Performance/step_time_s",
    "samples_per_s": "Performance/samples_per_s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-log", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--follow", action="store_true")
    return parser.parse_args()


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"device": None, "inode": None, "offset": 0, "last_step": None}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("TensorBoard bridge state must be an object")
    return value


def save_state(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def training_row(line: str) -> dict[str, Any] | None:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("step"), int):
        return None
    if not any(value.get(key) is not None for key in SCALARS):
        return None
    return value


def main() -> None:
    args = parse_args()
    source = args.input_log.resolve()
    log_dir = args.log_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    log_dir.mkdir(parents=True, exist_ok=True)
    state_path = log_dir / "bridge_state.json"
    lock_stream = (log_dir / "bridge.lock").open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"another TensorBoard bridge already owns {log_dir}") from error

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(log_dir))
    state = load_state(state_path)
    while True:
        stat = source.stat()
        identity = (int(stat.st_dev), int(stat.st_ino))
        recorded = (state.get("device"), state.get("inode"))
        offset = int(state.get("offset", 0))
        if recorded != identity or stat.st_size < offset:
            offset = 0
            state = {"device": identity[0], "inode": identity[1], "offset": 0, "last_step": None}

        imported = 0
        with source.open("r", encoding="utf-8", errors="replace") as stream:
            stream.seek(offset)
            while True:
                position = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith("\n"):
                    stream.seek(position)
                    break
                offset = stream.tell()
                row = training_row(line)
                if row is not None:
                    step = int(row["step"])
                    for key, tag in SCALARS.items():
                        value = row.get(key)
                        if value is not None:
                            writer.add_scalar(tag, float(value), step)
                    state["last_step"] = step
                    imported += 1

        state.update({"device": identity[0], "inode": identity[1], "offset": offset})
        save_state(state_path, state)
        if imported:
            writer.flush()
            print(json.dumps({"imported_rows": imported, **state}), flush=True)
        if not args.follow:
            break
        time.sleep(args.poll_seconds)

    writer.close()


if __name__ == "__main__":
    main()
