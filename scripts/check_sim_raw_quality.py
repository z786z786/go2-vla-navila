#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageChops, ImageStat


def _image_stats(image_path: Path) -> tuple[float, float]:
    image = Image.open(image_path).convert("RGB")
    stat = ImageStat.Stat(image)
    mean = sum(stat.mean) / max(len(stat.mean), 1) / 255.0
    std = sum(stat.stddev) / max(len(stat.stddev), 1) / 255.0
    return mean, std


def _image_delta(previous_path: Path, current_path: Path) -> float:
    previous = Image.open(previous_path).convert("RGB")
    current = Image.open(current_path).convert("RGB")
    diff = ImageChops.difference(previous, current)
    stat = ImageStat.Stat(diff)
    return sum(stat.mean) / max(len(stat.mean), 1) / 255.0


def _frame_image_abs(session_root: Path, frame: dict[str, Any]) -> Path:
    return session_root / str(frame.get("image") or "")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quality gate for Isaac Go2 raw sim sessions.")
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--check-first-frames", type=int, default=3)
    parser.add_argument("--min-first-frame-mean", type=float, default=0.05)
    parser.add_argument("--min-first-frame-std", type=float, default=0.02)
    parser.add_argument("--max-first-frame-delta", type=float, default=0.18)
    parser.add_argument("--tail-frames", type=int, default=3)
    parser.add_argument("--max-tail-command-vx", type=float, default=0.08)
    parser.add_argument("--max-tail-command-wz", type=float, default=0.18)
    parser.add_argument("--max-tail-state-vx", type=float, default=0.08)
    parser.add_argument("--max-tail-state-wz", type=float, default=0.18)
    parser.add_argument("--expect-local-scene", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    session_root = args.session_root.resolve()
    index_path = session_root / "index.json"
    episodes_dir = session_root / "episodes"
    if not index_path.exists() or not episodes_dir.exists():
        raise SystemExit(f"invalid session root: {session_root}")

    issues: list[str] = []
    summary: dict[str, Any] = {
        "session_root": str(session_root),
        "episodes": 0,
        "local_scene_ok": None,
        "bad_first_frame_count": 0,
        "unstable_initial_frame_count": 0,
        "moving_success_tail_count": 0,
        "missing_image_count": 0,
        "short_episode_count": 0,
    }

    resolved_config_path = session_root / "resolved_config.yaml"
    if resolved_config_path.exists():
        resolved_cfg = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8")) or {}
        scene_cfg = resolved_cfg.get("scene", {}) if isinstance(resolved_cfg, dict) else {}
        usd_path = str(scene_cfg.get("usd_path", "")).strip()
        asset_source = str(scene_cfg.get("asset_source", "")).strip().lower()
        scene_kind = str(scene_cfg.get("kind", "")).strip().lower()
        static_props = list(scene_cfg.get("static_props") or [])
        local_scene_ok = bool(
            (asset_source == "local" and usd_path and not usd_path.startswith("/Isaac") and not usd_path.startswith("omniverse://"))
            or (scene_kind == "plane" and static_props)
        )
        summary["local_scene_ok"] = local_scene_ok
        if args.expect_local_scene and not local_scene_ok:
            issues.append("scene asset is not pinned to a local USD path")

    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    episodes = list(index_payload.get("episodes") or [])
    summary["episodes"] = len(episodes)

    for episode_meta in episodes:
        episode_id = str(episode_meta.get("episode_id") or "")
        if not episode_id:
            continue
        episode_path = episodes_dir / f"{episode_id}.json"
        if not episode_path.exists():
            issues.append(f"missing episode json: {episode_path}")
            continue
        payload = json.loads(episode_path.read_text(encoding="utf-8"))
        frames = list(payload.get("frames") or [])
        if len(frames) < max(1, args.check_first_frames):
            summary["short_episode_count"] += 1
            issues.append(f"{episode_id}: too few frames ({len(frames)})")
            continue

        first_frame_paths: list[Path] = []
        first_frame_bad = False
        initial_unstable = False
        for frame in frames[: args.check_first_frames]:
            image_path = _frame_image_abs(session_root, frame)
            if not image_path.exists():
                summary["missing_image_count"] += 1
                issues.append(f"{episode_id}: missing image {image_path}")
                first_frame_bad = True
                continue
            mean, std = _image_stats(image_path)
            if mean < args.min_first_frame_mean or std < args.min_first_frame_std:
                first_frame_bad = True
            first_frame_paths.append(image_path)

        for previous_path, current_path in zip(first_frame_paths[:-1], first_frame_paths[1:]):
            delta = _image_delta(previous_path, current_path)
            if delta > args.max_first_frame_delta:
                initial_unstable = True
                break

        if first_frame_bad:
            summary["bad_first_frame_count"] += 1
            issues.append(f"{episode_id}: initial frames appear blank/uninitialized")
        if initial_unstable:
            summary["unstable_initial_frame_count"] += 1
            issues.append(f"{episode_id}: initial frames change too abruptly, scene may still be streaming")

        if str(payload.get("success") or episode_meta.get("success") or "").strip().lower() == "success":
            tail_frames = frames[-max(1, args.tail_frames) :]
            max_tail_command_vx = max(abs(float((frame.get("control_action") or {}).get("vx", 0.0))) for frame in tail_frames)
            max_tail_command_wz = max(abs(float((frame.get("control_action") or {}).get("wz", 0.0))) for frame in tail_frames)
            max_tail_state_vx = max(abs(float((frame.get("state") or {}).get("vx", 0.0))) for frame in tail_frames)
            max_tail_state_wz = max(abs(float((frame.get("state") or {}).get("wz", 0.0))) for frame in tail_frames)
            if (
                max_tail_command_vx > args.max_tail_command_vx
                or max_tail_command_wz > args.max_tail_command_wz
                or max_tail_state_vx > args.max_tail_state_vx
                or max_tail_state_wz > args.max_tail_state_wz
            ):
                summary["moving_success_tail_count"] += 1
                issues.append(
                    f"{episode_id}: success tail still moving "
                    f"(cmd_vx={max_tail_command_vx:.3f}, cmd_wz={max_tail_command_wz:.3f}, "
                    f"state_vx={max_tail_state_vx:.3f}, state_wz={max_tail_state_wz:.3f})"
                )

    summary["ok"] = not issues
    summary["issues"] = issues[:200]
    output_path = args.output.resolve() if args.output is not None else session_root / "quality_gate_summary.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
