#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collectors.sim_go2.utils.io import dump_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute depth statistics and preview renders for raw sim sessions.")
    parser.add_argument("--input", type=Path, required=True, help="Raw sim session root containing index.json and episodes/")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--preview-dir", type=Path, default=None, help="Optional directory for RGB/depth preview panels")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--nav-depth-min", type=float, default=0.2, help="Lower bound for navigation-focused preview normalization")
    parser.add_argument("--nav-depth-max", type=float, default=3.0, help="Upper bound for navigation-focused preview normalization")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_frames(session_root: Path) -> Iterable[dict[str, Any]]:
    index = load_json(session_root / "index.json")
    for episode in index.get("episodes", []):
        episode_id = str(episode.get("episode_id") or "")
        if not episode_id:
            continue
        payload = load_json(session_root / "episodes" / f"{episode_id}.json")
        for frame in payload.get("frames", []):
            if isinstance(frame, dict):
                yield frame


def load_depth_array(path: Path, depth_format: str, depth_scale: float) -> np.ndarray:
    if depth_format == "npy":
        return np.asarray(np.load(path), dtype=np.float32)
    image = Image.open(path)
    array = np.asarray(image, dtype=np.float32)
    return array / float(depth_scale)


def normalize_depth_for_preview(depth: np.ndarray, *, low: float | None = None, high: float | None = None) -> np.ndarray:
    finite = np.isfinite(depth) & (depth > 0.0)
    if not finite.any():
        image = np.zeros((*depth.shape, 3), dtype=np.uint8)
        image[:, :] = np.array([255, 0, 255], dtype=np.uint8)
        return image
    values = depth[finite]
    if low is None:
        low = float(np.percentile(values, 2))
    if high is None:
        high = float(np.percentile(values, 98))
    if high <= low:
        high = low + 1.0
    normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
    invalid_mask = ~finite | (depth <= 0.0)
    normalized[invalid_mask] = 0.0
    heat = np.stack(
        [
            (normalized * 255.0),
            ((1.0 - np.abs(normalized - 0.5) * 2.0) * 255.0),
            ((1.0 - normalized) * 255.0),
        ],
        axis=-1,
    )
    heat[invalid_mask] = np.array([255, 0, 255], dtype=np.uint8)
    return heat.astype(np.uint8)


def build_preview(
    rgb_path: Path,
    depth_array: np.ndarray,
    *,
    global_output_path: Path,
    nav_output_path: Path,
    nav_depth_min: float,
    nav_depth_max: float,
) -> None:
    rgb = Image.open(rgb_path).convert("RGB")
    depth_preview = Image.fromarray(normalize_depth_for_preview(depth_array), mode="RGB").resize(rgb.size)
    nav_preview = Image.fromarray(
        normalize_depth_for_preview(depth_array, low=float(nav_depth_min), high=float(nav_depth_max)),
        mode="RGB",
    ).resize(rgb.size)
    panel = Image.new("RGB", (rgb.width * 3, rgb.height))
    panel.paste(rgb, (0, 0))
    panel.paste(depth_preview, (rgb.width, 0))
    panel.paste(nav_preview, (rgb.width * 2, 0))
    global_output_path.parent.mkdir(parents=True, exist_ok=True)
    nav_output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(global_output_path, format="PNG")

    nav_panel = Image.new("RGB", (rgb.width * 2, rgb.height))
    nav_panel.paste(rgb, (0, 0))
    nav_panel.paste(nav_preview, (rgb.width, 0))
    nav_panel.save(nav_output_path, format="PNG")


def main() -> None:
    args = parse_args()
    session_root = args.input.resolve()
    stats = {
        "session_root": str(session_root),
        "frames_seen": 0,
        "frames_with_depth": 0,
        "depth_format_counts": {},
        "depth_min": None,
        "depth_max": None,
        "depth_median": None,
        "depth_p95": None,
        "depth_p99": None,
        "invalid_ratio": None,
        "zero_ratio": None,
        "nav_depth_min": float(args.nav_depth_min),
        "nav_depth_max": float(args.nav_depth_max),
    }

    values: list[np.ndarray] = []
    invalid_pixels = 0
    zero_pixels = 0
    total_pixels = 0
    depth_format_counts: dict[str, int] = {}
    preview_written = 0

    for frame_idx, frame in enumerate(iter_frames(session_root)):
        stats["frames_seen"] += 1
        depth_ref = frame.get("depth_image")
        if not depth_ref:
            continue
        depth_path = session_root / str(depth_ref)
        meta = frame.get("meta") if isinstance(frame.get("meta"), dict) else {}
        depth_format = str(meta.get("depth_format") or depth_path.suffix.lstrip(".") or "png").lower()
        depth_scale = float(meta.get("depth_scale") or 1000.0)
        if not depth_path.exists():
            continue
        depth = load_depth_array(depth_path, depth_format=depth_format, depth_scale=depth_scale)
        stats["frames_with_depth"] += 1
        depth_format_counts[depth_format] = depth_format_counts.get(depth_format, 0) + 1

        finite = np.isfinite(depth)
        invalid_pixels += int((~finite).sum())
        zero_pixels += int((finite & (depth <= 0.0)).sum())
        total_pixels += int(depth.size)
        positive = depth[finite & (depth > 0.0)]
        if positive.size:
            values.append(positive.astype(np.float32, copy=False))

        if args.preview_dir is not None and preview_written < args.preview_count:
            rgb_path = session_root / str(frame.get("image"))
            if rgb_path.exists():
                build_preview(
                    rgb_path,
                    depth,
                    global_output_path=args.preview_dir / f"preview_{preview_written:03d}.png",
                    nav_output_path=args.preview_dir / f"preview_nav_{preview_written:03d}.png",
                    nav_depth_min=float(args.nav_depth_min),
                    nav_depth_max=float(args.nav_depth_max),
                )
                preview_written += 1

        if stats["frames_with_depth"] >= args.max_frames:
            break

    if values:
        merged = np.concatenate(values)
        stats["depth_min"] = float(np.min(merged))
        stats["depth_max"] = float(np.max(merged))
        stats["depth_median"] = float(np.median(merged))
        stats["depth_p95"] = float(np.percentile(merged, 95))
        stats["depth_p99"] = float(np.percentile(merged, 99))
    if total_pixels > 0:
        stats["invalid_ratio"] = float(invalid_pixels / total_pixels)
        stats["zero_ratio"] = float(zero_pixels / total_pixels)
    stats["depth_format_counts"] = depth_format_counts
    stats["preview_count"] = preview_written

    if args.output is not None:
        dump_json(args.output, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
