#!/usr/bin/env python3
"""Compose synchronized, labeled rollout videos for visual diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def compose(videos: Sequence[Path], labels: Sequence[str], output: Path) -> dict[str, Any]:
    import cv2
    import numpy as np

    if len(videos) < 2 or len(videos) != len(labels):
        raise ValueError("videos and labels must have the same length of at least two")
    captures = [cv2.VideoCapture(str(path)) for path in videos]
    try:
        if not all(capture.isOpened() for capture in captures):
            raise ValueError("one or more input videos are unreadable")
        widths = [int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) for capture in captures]
        heights = [int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) for capture in captures]
        rates = [float(capture.get(cv2.CAP_PROP_FPS)) for capture in captures]
        counts = [int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) for capture in captures]
        if len(set(widths)) != 1 or len(set(heights)) != 1:
            raise ValueError("input videos must have identical dimensions")
        if not all(math.isfinite(rate) and math.isclose(rate, rates[0], abs_tol=1e-3) for rate in rates):
            raise ValueError("input videos must have identical finite frame rates")
        if any(count <= 0 for count in counts):
            raise ValueError("input videos must contain frames")

        width, height, fps, output_frames = widths[0], heights[0], rates[0], max(counts)
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width * len(videos), height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot open output video writer: {output}")
        last_frames: list[Any | None] = [None] * len(captures)
        try:
            for _ in range(output_frames):
                panels = []
                for index, capture in enumerate(captures):
                    ok, frame = capture.read()
                    if ok:
                        last_frames[index] = frame
                    elif last_frames[index] is None:
                        raise ValueError(f"video {videos[index]} ended before its first frame")
                    panel = last_frames[index].copy()
                    cv2.rectangle(panel, (0, height - 42), (width, height), (0, 0, 0), -1)
                    scale = 0.47
                    while scale > 0.25:
                        text_width = cv2.getTextSize(labels[index], cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
                        if text_width <= width - 16:
                            break
                        scale -= 0.02
                    cv2.putText(
                        panel,
                        labels[index],
                        (max(8, (width - text_width) // 2), height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        scale,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    panels.append(panel)
                writer.write(np.hstack(panels))
        finally:
            writer.release()
    finally:
        for capture in captures:
            capture.release()

    decoded = cv2.VideoCapture(str(output))
    try:
        decoded_frames = int(decoded.get(cv2.CAP_PROP_FRAME_COUNT)) if decoded.isOpened() else 0
        decoded_width = int(decoded.get(cv2.CAP_PROP_FRAME_WIDTH)) if decoded.isOpened() else 0
        decoded_height = int(decoded.get(cv2.CAP_PROP_FRAME_HEIGHT)) if decoded.isOpened() else 0
    finally:
        decoded.release()
    if decoded_frames < output_frames - 1:
        raise ValueError(f"output decoded {decoded_frames} frames; expected {output_frames}")
    return {
        "output": str(output),
        "input_frame_counts": counts,
        "decoded_frames": decoded_frames,
        "fps": fps,
        "width": decoded_width,
        "height": decoded_height,
    }


def main() -> None:
    args = parse_args()
    result = compose(args.videos, args.labels, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
