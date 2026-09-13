#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import random
import sys
from dataclasses import dataclass
from html import escape
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlencode, urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from llada_vla_common import (
    ACTION_FIELDS,
    ACTION_ACTIVITY_EPS,
    CONTROLLED_INSTRUCTIONS,
    FRAME_SELECTION_MODE_ALL,
    FRAME_SELECTION_MODE_RETAIN_ONLY,
    QUALITY_LABEL_EXCLUDED,
    QUALITY_REVIEW_FILENAME,
    QUALITY_REVIEW_SCHEMA_VERSION,
    VISUAL_TASK_FAMILIES,
    control_action_from_frame,
    discover_session_roots,
    episode_frame_processing,
    episode_task_metadata,
    infer_task_family,
    load_episode_derived_labels,
    load_json,
    load_jsonl,
    resolved_episode_quality,
    summarize_trajectory_actions,
    summarize_trajectory_metric_series,
)
from data_pipeline_common import (
    summarize_instruction_distribution,
    summarize_phase_distribution,
    summarize_window_integrity,
)


@dataclass
class FrameRecord:
    frame_index: int
    timestamp: float
    image_rel: str
    image_disk_path: Path
    instruction: str
    state: Dict[str, Any]
    control_action: Dict[str, float]
    retain_for_processing: bool
    raw_payload: Dict[str, Any]


@dataclass
class EpisodeRecord:
    session_id: str
    episode_id: str
    instruction: str
    capture_mode: str
    source_type: str
    task_family: str
    target_type: str
    target_label: str
    target_description: str
    d435i_capture_path: str
    derived_target_side: str
    derived_target_distance: str
    derived_label_source: str
    collector_notes: str
    scene_id: str
    operator_id: str
    original_quality_label: int
    quality_label: int
    quality_overridden: bool
    quality_notes: str
    exclude_from_processing: bool
    frame_selection_mode: str
    selected_frame_count: int
    total_frame_count: int
    frame_keep_map: Dict[int, bool]
    frames: List[FrameRecord]
    warnings: List[str]
    info: List[str]
    trajectory_metrics: Dict[str, Any]


@dataclass
class ProcessedPlaybackFrame:
    step_index: int
    frame_id: int
    timestamp: float
    image_disk_path: Optional[Path]


@dataclass
class ProcessedWindowRecord:
    sample_id: str
    session_id: str
    episode_id: str
    start_frame: int
    image_disk_path: Optional[Path]
    instruction: str
    raw_instruction: str
    normalized_instruction: str
    phase_derived_instruction: str
    phase: str
    phase_seq: List[str]
    state: List[float]
    state_dict: Dict[str, Any]
    action_fields: List[str]
    actions_continuous: List[List[float]]
    action_mask: List[int]
    frame_ids: List[int]
    timestamps: List[float]
    scene_id: str
    operator_id: str
    task_family: str
    target_label: str
    target_type: str
    target_description: str
    playback_frames: List[ProcessedPlaybackFrame]
    has_padding: bool
    has_nonzero_action: bool


STATE_VELOCITY_FIELDS = ["vx", "vy", "vz", "wz"]
STATE_ORIENTATION_FIELDS = ["roll", "pitch", "yaw"]
STATE_POSITION_FIELDS = ["x", "y", "z", "body_height"]
STATE_AXIS_COLORS = {
    "vx": "#cc5500",
    "vy": "#007f5f",
    "vz": "#8854d0",
    "wz": "#2b59c3",
    "roll": "#8c564b",
    "pitch": "#e67e22",
    "yaw": "#1f77b4",
    "x": "#c0392b",
    "y": "#16a085",
    "z": "#6c5ce7",
    "body_height": "#7f8c8d",
}


CSS = """
html { height: 100%; }
body {
  font-family: ui-sans-serif, sans-serif;
  margin: 0;
  padding: 10px;
  background: #f7f7f7;
  color: #111827;
}
body.detail-page {
  min-height: 100dvh;
  overflow: hidden;
}
a {
  color: #1f2937;
  text-decoration: none;
}
a:hover { text-decoration: underline; }
.page-shell {
  display: grid;
  gap: 8px;
}
body.detail-page .page-shell {
  height: calc(100dvh - 20px);
  grid-template-rows: auto minmax(0, 1fr);
}
.card {
  background: #ffffff;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  padding: 10px;
}
.viewer {
  position: relative;
  width: 100%;
  min-height: 0;
  height: 100%;
  display: grid;
  align-items: center;
  justify-items: center;
  padding: 8px;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  background: #111111;
  box-sizing: border-box;
}
.viewer img {
  width: 100%;
  height: 100%;
  min-height: 0;
  max-height: none;
  object-fit: contain;
  border-radius: 6px;
  display: block;
  background: #111111;
  aspect-ratio: 16 / 9;
}
.overlay {
  position: absolute;
  left: 12px;
  top: 12px;
  max-width: min(58ch, calc(100% - 24px));
  background: rgba(255, 255, 255, 0.92);
  color: #111827;
  padding: 8px 10px;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  font-size: 12px;
  line-height: 1.4;
}
.controls {
  display: flex;
  gap: 6px;
  margin: 0;
  flex-wrap: wrap;
  align-items: center;
}
.controls label { display: inline-flex; align-items: center; gap: 6px; }
.compact-controls {
  padding-top: 2px;
}
.frame-range-wrap { position: relative; display: inline-flex; align-items: center; min-width: min(460px, 100%); flex: 1 1 320px; }
.frame-range-stack { position: relative; width: min(360px, 55vw); max-width: 100%; height: 28px; display: inline-flex; align-items: center; }
.frame-range-stack input[type=range] { position: absolute; inset: 0; width: 100%; margin: 0; background: transparent; pointer-events: none; }
.frame-range-stack input[type=range]::-webkit-slider-thumb { pointer-events: auto; }
.frame-range-stack input[type=range]::-moz-range-thumb { pointer-events: auto; }
.frame-range-stack .frame-range-marker { pointer-events: auto; z-index: 3; }
.frame-range-stack .frame-range-current { z-index: 2; }
.frame-range-stack input[type=range].is-active { z-index: 4; }
.frame-range-stack .frame-range-marker::-webkit-slider-runnable-track { height: 6px; background: transparent; }
.frame-range-stack .frame-range-marker::-moz-range-track { height: 6px; background: transparent; }
.frame-range-stack .frame-range-marker::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 4px; height: 24px; margin-top: -9px; border: 0; border-radius: 0; background: #4b5563; cursor: ew-resize; }
.frame-range-stack .frame-range-marker::-moz-range-thumb { width: 4px; height: 24px; border: 0; border-radius: 0; background: #4b5563; cursor: ew-resize; }
.frame-range-stack .frame-range-current::-webkit-slider-runnable-track { height: 6px; background: #d1d5db; border-radius: 999px; }
.frame-range-stack .frame-range-current::-moz-range-track { height: 6px; background: #d1d5db; border-radius: 999px; }
.frame-range-stack .frame-range-current::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 10px; height: 18px; margin-top: -6px; border: 2px solid #111827; border-radius: 2px; background: #ffffff; cursor: pointer; }
.frame-range-stack .frame-range-current::-moz-range-thumb { width: 10px; height: 18px; border: 2px solid #111827; border-radius: 2px; background: #ffffff; cursor: pointer; }
.frame-range-stack .frame-range-highlight { position: absolute; top: 11px; height: 6px; border-radius: 999px; background: rgba(156, 163, 175, 0.45); pointer-events: none; }
button, input[type=range], select, input[type=text], input[type=file] { font: inherit; }
button, select, input[type=text] {
  min-height: 32px;
  padding: 5px 9px;
  border: 1px solid #cfd4dc;
  border-radius: 6px;
  background: #fff;
  color: #111827;
}
button {
  cursor: pointer;
  font-weight: 600;
}
button:hover, select:hover, input[type=text]:hover { border-color: #93a7b6; }
button:focus-visible, select:focus-visible, input[type=text]:focus-visible, input[type=range]:focus-visible {
  outline: 2px solid #6b7280;
  outline-offset: 2px;
}
.playback-stats { color: #4b5563; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }
.grid img { width: 100%; border-radius: 8px; }
.meta { color: #4b5563; margin: 4px 0; }
table { border-collapse: collapse; width: 100%; }
th, td { border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; }
code {
  background: #f9fafb;
  padding: 2px 4px;
  border-radius: 4px;
  word-break: break-word;
}
svg { width: 100%; height: auto; }
.review-toolbar { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; margin-top: 8px; }
.review-toolbar .meta { margin: 0; }
.review-summary { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }
.review-pill {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 4px 10px;
  background: #eef4f7;
  font-size: 12px;
}
.quality-select, .quality-note, .quality-filter, .quality-score-filter { min-width: 120px; }
.quality-note { width: min(100%, 240px); }
.quality-badge { font-weight: 600; }
.quality-badge.label-0 { color: #6f6658; }
.quality-badge.label-1 { color: #1d6f42; }
.quality-badge.label-2 { color: #8a5b00; }
.quality-badge.label-3 { color: #a12626; }
tr.quality-excluded { background: #fff0ee; }
tr.quality-review { background: #fffaed; }
.card.quality-excluded { border-color: #c96b5a; }
.review-file-status { color: #5f5f5f; }
.page-nav {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  align-items: center;
  padding: 4px 2px 0;
}
.episode-layout {
  display: grid;
  grid-template-columns: minmax(0, 1.45fr) minmax(360px, 1fr);
  gap: 8px;
  align-items: stretch;
  min-height: 0;
}
.episode-stack {
  display: grid;
  gap: 8px;
  min-height: 0;
}
.viewer-panel {
  grid-template-rows: auto minmax(0, 1fr) auto auto;
}
.episode-side {
  display: grid;
  grid-template-rows: auto minmax(220px, 0.9fr) minmax(240px, 1.1fr);
  gap: 8px;
  min-height: 0;
  overflow: hidden;
  padding-right: 2px;
}
.frame-record-card {
  display: grid;
  grid-template-rows: auto auto minmax(0, 1fr);
  min-height: 0;
}
.frame-record-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 4px 10px;
  margin-bottom: 8px;
}
.frame-record-grid .meta {
  margin: 0;
  font-size: 12px;
}
.json-box {
  margin: 0;
  padding: 8px;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  background: #fafafa;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 11px;
  line-height: 1.35;
}
.meta-grid,
.overview-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 4px 10px; margin-top: 6px; }
.meta-grid .meta { margin: 0; font-size: 13px; line-height: 1.35; }
.compact-card h1 { margin: 0 0 6px; font-size: 22px; }
.compact-card h2 { margin-top: 0; margin-bottom: 6px; font-size: 17px; }
.episode-side .card { padding: 8px; }
.episode-side svg { max-height: 190px; }
.viewer-header {
  display: grid;
  gap: 4px;
}
.viewer-header .meta {
  margin: 0;
}
.viewer-help {
  font-size: 12px;
}
.table-compact th, .table-compact td {
  padding: 6px 8px;
  font-size: 13px;
}
details.details-card {
  overflow: hidden;
}
details.details-card > summary {
  cursor: pointer;
  list-style: none;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  font-weight: 700;
  color: #12314a;
}
details.details-card > summary::-webkit-details-marker {
  display: none;
}
details.details-card > summary::after {
  content: "展开";
  font-size: 12px;
  color: #5f6b76;
  font-weight: 600;
}
details.details-card[open] > summary::after {
  content: "收起";
}
.details-body {
  margin-top: 8px;
}
.secondary-grid {
  display: grid;
  gap: 8px;
}
.plot-card svg {
  max-height: none;
}
input[type=file] { max-width: 220px; }
@media (max-width: 1180px) {
  body.detail-page {
    overflow: auto;
  }
  body.detail-page .page-shell {
    height: auto;
  }
  .episode-layout {
    grid-template-columns: 1fr;
  }
  .episode-side {
    overflow: visible;
    padding-right: 0;
  }
}
@media (max-width: 720px) {
  body {
    padding: 8px;
  }
  .meta-grid,
  .overview-grid,
  .grid {
    grid-template-columns: 1fr;
  }
  .viewer img {
    min-height: 240px;
  }
}
"""

QUALITY_LABEL_TEXT = {
    0: "0 未知",
    1: "1 好样本",
    2: "2 可用但不完美",
    3: "3 失败但保留 / 质量差",
}

QUALITY_LABEL_TO_EPISODE_FIELDS = {
    1: {
        "segment_status": "clean",
        "success": "success",
        "termination_reason": "goal_reached",
    },
    2: {
        "segment_status": "usable",
        "success": "partial",
        "termination_reason": "near_goal_stop",
    },
    3: {
        "segment_status": "usable",
        "success": "fail",
        "termination_reason": "operator_stop",
    },
}


def _quality_badge_text(quality_label: int) -> str:
    return QUALITY_LABEL_TEXT.get(int(quality_label), f"{int(quality_label)} 未知")


def _quality_select_html(select_id: str, quality_label: int, extra_attrs: str = "") -> str:
    options = []
    for value in (1, 2, 3):
        selected = " selected" if int(quality_label) == value else ""
        options.append(f'<option value="{value}"{selected}>{escape(_quality_badge_text(value))}</option>')
    attrs = f" {extra_attrs.strip()}" if extra_attrs.strip() else ""
    return f'<select id="{escape(select_id)}" class="quality-select"{attrs}>{"".join(options)}</select>'


def _initial_review_payload(episodes: Sequence[EpisodeRecord]) -> Dict[str, Dict[str, Any]]:
    payload: Dict[str, Dict[str, Any]] = {}
    for episode in episodes:
        if (
            not episode.quality_overridden and
            not episode.quality_notes and
            episode.frame_selection_mode == FRAME_SELECTION_MODE_ALL and
            not episode.frame_keep_map
        ):
            continue
        key = f"{episode.session_id}::{episode.episode_id}"
        payload[key] = {
            "session_id": episode.session_id,
            "episode_id": episode.episode_id,
            "quality_label": int(episode.quality_label),
            "original_quality_label": int(episode.original_quality_label),
            "quality_notes": episode.quality_notes,
            "exclude_from_processing": bool(episode.exclude_from_processing),
            "frame_selection_mode": episode.frame_selection_mode,
            "frame_keep_map": {str(index): value for index, value in episode.frame_keep_map.items()},
            "selected_frame_count": int(episode.selected_frame_count),
            "total_frame_count": int(episode.total_frame_count),
        }
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="为 Go2 采集 session 生成本地 HTML 回放与体检报告")
    parser.add_argument("--mode", choices=["raw", "processed_windows", "pipeline_hub"], default="raw", help="检查原始 session、处理后的 windows 样本，或生成全链路 hub")
    parser.add_argument("--data-root", type=Path, help="raw 模式下的数据集根目录，或包含多个 session 根目录的父目录")
    parser.add_argument("--windows-root", type=Path, help="processed_windows 模式下的 windows 目录，内部需包含 windows.jsonl")
    parser.add_argument("--filtered-root", type=Path, help="可选：processed_windows 模式下的 filtered 目录，用于展示过滤统计")
    parser.add_argument("--canonical-phase-root", type=Path, help="可选：processed_windows 模式下的 canonical_phase 目录，用于补充 frame image/phase 摘要")
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT, help="pipeline_hub 模式下自动发现阶段目录的工作区根目录")
    parser.add_argument("--real-raw-root", type=Path, help="pipeline_hub 模式下显式指定 real raw 根目录")
    parser.add_argument("--sim-raw-root", type=Path, help="pipeline_hub 模式下显式指定 sim raw 根目录")
    parser.add_argument("--model-export-root", type=Path, help="pipeline_hub 模式下显式指定最终导出/数据集根目录")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "sanity_check", help="HTML 报告输出目录")
    parser.add_argument("--num-samples", type=int, default=4, help="每条 episode 页面中展示的随机静态帧数量")
    parser.add_argument("--max-episodes", type=int, help="可选的 episode 数量上限，用于冒烟测试")
    parser.add_argument("--max-samples", type=int, help="可选：processed_windows 模式下的样本数量上限")
    parser.add_argument("--seed", type=int, default=0, help="预览采样随机种子")
    parser.add_argument("--serve", action="store_true", help="生成报告后启动本地 HTTP 服务，页面改分直接写回原始 episode 源文件")
    parser.add_argument("--host", default="127.0.0.1", help="本地 HTTP 服务监听地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="本地 HTTP 服务端口，默认 8000")
    return parser


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _state_from_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
    payload = frame.get("state") or {}
    if not isinstance(payload, dict):
        payload = {}
    return {
        "vx": _optional_float(payload.get("vx")),
        "vy": _optional_float(payload.get("vy")),
        "vz": _optional_float(payload.get("vz")),
        "wz": _optional_float(payload.get("wz")),
        "yaw_speed": _optional_float(payload.get("yaw_speed")),
        "vx_prev": _optional_float(payload.get("vx_prev")),
        "vy_prev": _optional_float(payload.get("vy_prev")),
        "wz_prev": _optional_float(payload.get("wz_prev")),
        "roll": _optional_float(payload.get("roll")),
        "pitch": _optional_float(payload.get("pitch")),
        "yaw": _optional_float(payload.get("yaw")),
        "x": _optional_float(payload.get("x")),
        "y": _optional_float(payload.get("y")),
        "z": _optional_float(payload.get("z")),
        "body_height": _optional_float(payload.get("body_height")),
        "error_code": int(payload.get("error_code") or 0),
        "mode": int(payload.get("mode") or 0),
        "gait_type": int(payload.get("gait_type") or 0),
    }


def _warn_for_episode(payload: Dict[str, Any], frames: Sequence[FrameRecord]) -> List[str]:
    issues: List[str] = []
    instruction = str(payload.get("instruction") or "")
    task_metadata = episode_task_metadata(payload, {})
    task_family = str(task_metadata.get("task_family") or infer_task_family(instruction))
    if not instruction:
        issues.append("missing_instruction")
    elif task_family == "legacy_motion" and instruction not in CONTROLLED_INSTRUCTIONS:
        issues.append("legacy_motion_instruction_unknown")
    if not payload.get("scene_id"):
        issues.append("missing_scene_id")
    if not payload.get("operator_id"):
        issues.append("missing_operator_id")
    if len(frames) < 2:
        issues.append("too_few_frames")
    return sorted(set(issues))


def _info_for_episode(frames: Sequence[FrameRecord]) -> List[str]:
    info: List[str] = []
    for axis in ACTION_FIELDS:
        values = [frame.control_action[axis] for frame in frames]
        if values and max(values) - min(values) < 1e-6:
            info.append(f"flat_{axis}")
    return sorted(set(info))


def load_episodes(
    data_root: Path,
    max_episodes: Optional[int],
    quality_review_index: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
) -> List[EpisodeRecord]:
    session_roots = discover_session_roots(data_root)
    if not session_roots:
        raise FileNotFoundError(f"no session roots found under {data_root}")

    episodes: List[EpisodeRecord] = []
    for session_root in session_roots:
        index_payload = load_json(session_root / "index.json")
        session_id = str(index_payload.get("session_id") or session_root.name)
        for episode_meta in index_payload.get("episodes") or []:
            if max_episodes is not None and len(episodes) >= max_episodes:
                return episodes
            episode_id = str(episode_meta.get("episode_id") or "")
            if not episode_id:
                continue
            payload = load_json(session_root / "episodes" / f"{episode_id}.json")
            task_metadata = episode_task_metadata(payload, episode_meta)
            derived_labels = load_episode_derived_labels(session_root, episode_id)
            quality_review = resolved_episode_quality(payload, episode_meta, quality_review_index, session_id, episode_id)
            frame_processing = episode_frame_processing(payload)
            selected_indices = set(frame_processing["selected_frame_indices"])
            frames: List[FrameRecord] = []
            for frame_index, frame in enumerate(payload.get("frames") or []):
                image_rel = str(frame.get("image") or "")
                if not image_rel:
                    continue
                image_disk_path = session_root / image_rel
                if not image_disk_path.exists():
                    continue
                frames.append(
                    FrameRecord(
                        frame_index=frame_index,
                        timestamp=_safe_float(frame.get("timestamp")),
                        image_rel=image_rel,
                        image_disk_path=image_disk_path,
                        instruction=str(frame.get("instruction") or payload.get("instruction") or ""),
                        state=_state_from_frame(frame),
                        control_action=control_action_from_frame(frame),
                        retain_for_processing=frame_index in selected_indices,
                        raw_payload=dict(frame),
                    )
                )
            episode = EpisodeRecord(
                session_id=session_id,
                episode_id=episode_id,
                instruction=str(payload.get("instruction") or ""),
                capture_mode=str(task_metadata.get("capture_mode") or ""),
                source_type=str(payload.get("source_type") or episode_meta.get("source_type") or ""),
                task_family=str(task_metadata.get("task_family") or ""),
                target_type=str(task_metadata.get("target_type") or ""),
                target_label=str(task_metadata.get("target_label") or ""),
                target_description=str(task_metadata.get("target_description") or ""),
                d435i_capture_path=str(((payload.get("d435i_capture") or {}) if isinstance(payload.get("d435i_capture"), dict) else {}).get("capture_metadata_path") or ""),
                derived_target_side=str(derived_labels.get("target_side_band") or ""),
                derived_target_distance=str(derived_labels.get("target_distance_band") or ""),
                derived_label_source=str(derived_labels.get("label_source") or ""),
                collector_notes=str(task_metadata.get("collector_notes") or ""),
                scene_id=str(payload.get("scene_id") or episode_meta.get("scene_id") or ""),
                operator_id=str(payload.get("operator_id") or episode_meta.get("operator_id") or ""),
                original_quality_label=int(quality_review.get("original_quality_label") or 0),
                quality_label=int(quality_review.get("quality_label") or 0),
                quality_overridden=bool(quality_review.get("quality_overridden")),
                quality_notes=str(quality_review.get("quality_notes") or ""),
                exclude_from_processing=bool(quality_review.get("exclude_from_processing")),
                frame_selection_mode=str(frame_processing["frame_selection_mode"]),
                selected_frame_count=int(frame_processing["selected_frame_count"]),
                total_frame_count=int(frame_processing["total_frame_count"]),
                frame_keep_map=dict(frame_processing["frame_keep_map"]),
                frames=frames,
                warnings=[],
                info=[],
                trajectory_metrics={},
            )
            episode.warnings = _warn_for_episode(payload, frames)
            episode.info = _info_for_episode(frames)
            episode.trajectory_metrics = summarize_trajectory_actions(
                [frame.control_action for frame in frames],
                [frame.timestamp for frame in frames],
            )
            episodes.append(episode)
    return episodes


def _window_primary_image_path(record: Dict[str, Any]) -> Optional[Path]:
    source_image_path = str(record.get("source_image_path") or "").strip()
    if source_image_path:
        candidate = Path(source_image_path)
        if candidate.exists():
            return candidate
    return None


def _load_canonical_frame_index(
    canonical_phase_root: Optional[Path],
    session_id: str,
    episode_id: str,
    cache: Dict[Tuple[str, str], Dict[int, Dict[str, Any]]],
) -> Dict[int, Dict[str, Any]]:
    key = (session_id, episode_id)
    if key in cache:
        return cache[key]
    if canonical_phase_root is None:
        cache[key] = {}
        return cache[key]
    episode_path = canonical_phase_root / "episodes" / session_id / f"{episode_id}.jsonl"
    if not episode_path.exists():
        cache[key] = {}
        return cache[key]
    mapping: Dict[int, Dict[str, Any]] = {}
    for item in load_jsonl(episode_path):
        frame_id = int(item.get("frame_id") or 0)
        mapping[frame_id] = item
    cache[key] = mapping
    return mapping


def _resolve_processed_playback_frames(
    record: Dict[str, Any],
    canonical_phase_root: Optional[Path],
    canonical_cache: Dict[Tuple[str, str], Dict[int, Dict[str, Any]]],
) -> List[ProcessedPlaybackFrame]:
    session_id = str(record.get("session_id") or "")
    episode_id = str(record.get("episode_id") or "")
    frame_ids = [int(value) for value in list(record.get("frame_ids") or [])]
    timestamps = [float(value) for value in list(record.get("timestamps") or [])]
    fallback_image = _window_primary_image_path(record)
    canonical_index = _load_canonical_frame_index(canonical_phase_root, session_id, episode_id, canonical_cache)
    playback_frames: List[ProcessedPlaybackFrame] = []

    for step_index, frame_id in enumerate(frame_ids):
        timestamp = timestamps[step_index] if step_index < len(timestamps) else 0.0
        image_disk_path: Optional[Path] = fallback_image
        canonical_record = canonical_index.get(frame_id) or {}
        source_image_path = str(canonical_record.get("source_image_path") or "").strip()
        if source_image_path:
            candidate = Path(source_image_path)
            if candidate.exists():
                image_disk_path = candidate
        playback_frames.append(
            ProcessedPlaybackFrame(
                step_index=step_index,
                frame_id=frame_id,
                timestamp=timestamp,
                image_disk_path=image_disk_path,
            )
        )
    return playback_frames


def load_processed_windows(
    windows_root: Path,
    max_samples: Optional[int],
    canonical_phase_root: Optional[Path] = None,
) -> List[ProcessedWindowRecord]:
    windows_path = windows_root / "windows.jsonl"
    if not windows_path.exists():
        raise FileNotFoundError(f"windows jsonl not found under {windows_root}")

    records = load_jsonl(windows_path)
    if max_samples is not None:
        records = records[:max_samples]

    canonical_cache: Dict[Tuple[str, str], Dict[int, Dict[str, Any]]] = {}
    samples: List[ProcessedWindowRecord] = []
    for item in records:
        playback_frames = _resolve_processed_playback_frames(item, canonical_phase_root, canonical_cache)
        action_fields = [str(field) for field in list(item.get("action_fields") or ACTION_FIELDS)]
        actions_continuous = [[float(value) for value in list(step or [])] for step in list(item.get("actions_continuous") or [])]
        action_mask = [int(value) for value in list(item.get("action_mask") or [])]
        phase_seq = [str(value or "") for value in list(item.get("phase_seq") or [])]
        samples.append(
            ProcessedWindowRecord(
                sample_id=str(item.get("sample_id") or ""),
                session_id=str(item.get("session_id") or ""),
                episode_id=str(item.get("episode_id") or ""),
                start_frame=int(item.get("start_frame") or 0),
                image_disk_path=_window_primary_image_path(item),
                instruction=str(item.get("instruction") or ""),
                raw_instruction=str(item.get("raw_instruction") or ""),
                normalized_instruction=str(item.get("normalized_instruction") or ""),
                phase_derived_instruction=str(item.get("phase_derived_instruction") or ""),
                phase=str(item.get("phase") or ""),
                phase_seq=phase_seq,
                state=[float(value) for value in list(item.get("state") or [])],
                state_dict=dict(item.get("state_dict") or {}),
                action_fields=action_fields,
                actions_continuous=actions_continuous,
                action_mask=action_mask,
                frame_ids=[int(value) for value in list(item.get("frame_ids") or [])],
                timestamps=[float(value) for value in list(item.get("timestamps") or [])],
                scene_id=str(item.get("scene_id") or ""),
                operator_id=str(item.get("operator_id") or ""),
                task_family=str(item.get("task_family") or ""),
                target_label=str(item.get("target_label") or ""),
                target_type=str(item.get("target_type") or ""),
                target_description=str(item.get("target_description") or ""),
                playback_frames=playback_frames,
                has_padding=any(int(value) == 0 for value in action_mask),
                has_nonzero_action=any(abs(float(value)) > 1e-6 for step in actions_continuous for value in step),
            )
        )
    return samples


def collect_session_stats(data_root: Path) -> Tuple[int, Dict[str, int]]:
    session_roots = discover_session_roots(data_root)
    session_episode_counts: Dict[str, int] = {}
    empty_session_count = 0
    for session_root in session_roots:
        index_payload = load_json(session_root / "index.json")
        session_id = str(index_payload.get("session_id") or session_root.name)
        episode_count = len(index_payload.get("episodes") or [])
        session_episode_counts[session_id] = episode_count
        if episode_count == 0:
            empty_session_count += 1
    return empty_session_count, dict(sorted(session_episode_counts.items()))


def _default_quality_review_payload(data_root: Path) -> Dict[str, Any]:
    return {
        "schema_version": QUALITY_REVIEW_SCHEMA_VERSION,
        "dataset_root": str(data_root),
        "excluded_quality_label": QUALITY_LABEL_EXCLUDED,
        "review_filename_hint": QUALITY_REVIEW_FILENAME,
        "semantics": "quality_label stores an override for the original recording score (1/2/3)",
        "episodes": [],
    }


def _load_quality_review_payload(data_root: Path) -> Dict[str, Any]:
    review_path = data_root / QUALITY_REVIEW_FILENAME
    if not review_path.exists():
        return _default_quality_review_payload(data_root)
    payload = load_json(review_path)
    if not isinstance(payload, dict):
        return _default_quality_review_payload(data_root)
    result = _default_quality_review_payload(data_root)
    result.update(payload)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        result["episodes"] = []
    return result


def _normalize_quality_review_payload(data_root: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _default_quality_review_payload(data_root)
    normalized["dataset_root"] = str(data_root)
    normalized["episodes"] = []
    for item in payload.get("episodes") or []:
        if not isinstance(item, dict):
            continue
        session_id = str(item.get("session_id") or "").strip()
        episode_id = str(item.get("episode_id") or "").strip()
        if not session_id or not episode_id:
            continue
        quality_label = int(item.get("quality_label") or 0)
        if quality_label < 0:
            quality_label = 0
        if quality_label > 3:
            quality_label = 3
        original_quality_label = int(item.get("original_quality_label") or 0)
        if original_quality_label < 0:
            original_quality_label = 0
        if original_quality_label > 3:
            original_quality_label = 3
        normalized_item = {
            "session_id": session_id,
            "episode_id": episode_id,
            "quality_label": quality_label,
            "original_quality_label": original_quality_label,
            "exclude_from_processing": bool(item.get("exclude_from_processing")) or quality_label == QUALITY_LABEL_EXCLUDED,
        }
        quality_notes = str(item.get("quality_notes") or item.get("notes") or "").strip()
        if quality_notes:
            normalized_item["quality_notes"] = quality_notes
        normalized["episodes"].append(normalized_item)
    normalized["episodes"].sort(key=lambda item: (item["session_id"], item["episode_id"]))
    return normalized


def _write_quality_review_payload(data_root: Path, payload: Dict[str, Any]) -> Path:
    review_path = data_root / QUALITY_REVIEW_FILENAME
    review_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_quality_review_payload(data_root, payload)
    review_path.write_text(json.dumps(normalized, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return review_path


def _normalize_source_episode_reviews(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in payload.get("source_episode_reviews") or []:
        if not isinstance(item, dict):
            continue
        session_id = str(item.get("session_id") or "").strip()
        episode_id = str(item.get("episode_id") or "").strip()
        if not session_id or not episode_id:
            continue
        quality_label = int(item.get("quality_label") or 0)
        if quality_label not in QUALITY_LABEL_TO_EPISODE_FIELDS:
            raise ValueError(f"invalid source quality_label for {session_id}/{episode_id}: {quality_label}")
        normalized_item = {
            "session_id": session_id,
            "episode_id": episode_id,
            "quality_label": quality_label,
            "quality_notes": str(item.get("quality_notes") or item.get("notes") or "").strip(),
        }
        frame_selection_mode = str(item.get("frame_selection_mode") or FRAME_SELECTION_MODE_ALL).strip().lower()
        if frame_selection_mode not in {FRAME_SELECTION_MODE_ALL, FRAME_SELECTION_MODE_RETAIN_ONLY}:
            frame_selection_mode = FRAME_SELECTION_MODE_ALL
        normalized_item["frame_selection_mode"] = frame_selection_mode
        frame_keep_map: Dict[int, bool] = {}
        raw_keep_map = item.get("frame_keep_map") or {}
        if isinstance(raw_keep_map, dict):
            for raw_key, raw_value in raw_keep_map.items():
                try:
                    frame_index = int(raw_key)
                except (TypeError, ValueError):
                    continue
                if frame_index < 0:
                    continue
                frame_keep_map[frame_index] = bool(raw_value)
        normalized_item["frame_keep_map"] = frame_keep_map
        normalized.append(normalized_item)
    normalized.sort(key=lambda item: (item["session_id"], item["episode_id"]))
    return normalized


def _session_root_index(data_root: Path) -> Dict[str, Path]:
    session_index: Dict[str, Path] = {}
    for session_root in discover_session_roots(data_root):
        index_payload = load_json(session_root / "index.json")
        session_id = str(index_payload.get("session_id") or session_root.name).strip()
        if not session_id:
            continue
        existing = session_index.get(session_id)
        if existing is not None and existing != session_root:
            raise ValueError(f"duplicate session_id discovered under data root: {session_id}")
        session_index[session_id] = session_root
    return session_index


def _set_quality_fields(target: Dict[str, Any], quality_label: int) -> bool:
    desired_fields = QUALITY_LABEL_TO_EPISODE_FIELDS.get(int(quality_label))
    if not desired_fields:
        raise ValueError(f"unsupported quality label for source sync: {quality_label}")
    changed = False
    for key, value in desired_fields.items():
        if str(target.get(key) or "") != value:
            target[key] = value
            changed = True
    return changed


def _sync_episode_payload_quality(payload: Dict[str, Any], quality_label: int) -> bool:
    changed = _set_quality_fields(payload, quality_label)
    task_block = payload.get("task")
    if isinstance(task_block, dict):
        changed = _set_quality_fields(task_block, quality_label) or changed
    for frame in payload.get("frames") or []:
        if not isinstance(frame, dict):
            continue
        frame_meta = frame.get("meta")
        if not isinstance(frame_meta, dict):
            frame_meta = {}
            frame["meta"] = frame_meta
            changed = True
        changed = _set_quality_fields(frame_meta, quality_label) or changed
    return changed


def _sync_episode_payload_frame_selection(
    payload: Dict[str, Any],
    frame_selection_mode: str,
    frame_keep_map: Dict[int, bool],
) -> bool:
    changed = False
    normalized_mode = frame_selection_mode if frame_selection_mode in {FRAME_SELECTION_MODE_ALL, FRAME_SELECTION_MODE_RETAIN_ONLY} else FRAME_SELECTION_MODE_ALL
    if str(payload.get("frame_selection_mode") or FRAME_SELECTION_MODE_ALL) != normalized_mode:
        payload["frame_selection_mode"] = normalized_mode
        changed = True

    frames = list(payload.get("frames") or [])
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        frame_meta = frame.get("meta")
        if not isinstance(frame_meta, dict):
            frame_meta = {}
            frame["meta"] = frame_meta
            changed = True
        desired = frame_keep_map.get(frame_index)
        if desired is None:
            if "retain_for_processing" in frame_meta:
                frame_meta.pop("retain_for_processing", None)
                changed = True
            continue
        if bool(frame_meta.get("retain_for_processing")) != bool(desired) or "retain_for_processing" not in frame_meta:
            frame_meta["retain_for_processing"] = bool(desired)
            changed = True
    return changed


def _sync_index_episode_quality(index_payload: Dict[str, Any], episode_id: str, quality_label: int) -> bool:
    changed = False
    for entry in index_payload.get("episodes") or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("episode_id") or "") != episode_id:
            continue
        changed = _set_quality_fields(entry, quality_label) or changed
        break
    return changed


def _write_json_file(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sync_source_episode_files(data_root: Path, payload: Dict[str, Any]) -> Dict[str, int]:
    source_episode_reviews = _normalize_source_episode_reviews(payload)
    if not source_episode_reviews:
        return {"updated_episode_files": 0, "updated_index_files": 0}

    session_roots = _session_root_index(data_root)
    reviews_by_session: Dict[str, List[Dict[str, Any]]] = {}
    for item in source_episode_reviews:
        reviews_by_session.setdefault(item["session_id"], []).append(item)

    updated_episode_files = 0
    updated_index_files = 0
    for session_id, session_reviews in reviews_by_session.items():
        session_root = session_roots.get(session_id)
        if session_root is None:
            raise FileNotFoundError(f"session not found under data root: {session_id}")
        index_path = session_root / "index.json"
        index_payload = load_json(index_path)
        index_changed = False
        for item in session_reviews:
            episode_id = item["episode_id"]
            quality_label = int(item["quality_label"])
            episode_path = session_root / "episodes" / f"{episode_id}.json"
            if not episode_path.exists():
                raise FileNotFoundError(f"episode file not found: {episode_path}")
            episode_payload = load_json(episode_path)
            episode_changed = _sync_episode_payload_quality(episode_payload, quality_label)
            episode_changed = _sync_episode_payload_frame_selection(
                episode_payload,
                str(item.get("frame_selection_mode") or FRAME_SELECTION_MODE_ALL),
                dict(item.get("frame_keep_map") or {}),
            ) or episode_changed
            if episode_changed:
                _write_json_file(episode_path, episode_payload)
                updated_episode_files += 1
            index_changed = _sync_index_episode_quality(index_payload, episode_id, quality_label) or index_changed
        if index_changed:
            _write_json_file(index_path, index_payload)
            updated_index_files += 1
    return {
        "updated_episode_files": updated_episode_files,
        "updated_index_files": updated_index_files,
    }


def _build_live_review_payload(data_root: Path) -> Dict[str, Any]:
    review_payload = _load_quality_review_payload(data_root)
    review_notes: Dict[Tuple[str, str], str] = {}
    for item in review_payload.get("episodes") or []:
        if not isinstance(item, dict):
            continue
        session_id = str(item.get("session_id") or "").strip()
        episode_id = str(item.get("episode_id") or "").strip()
        if not session_id or not episode_id:
            continue
        review_notes[(session_id, episode_id)] = str(item.get("quality_notes") or item.get("notes") or "").strip()

    live_payload = _default_quality_review_payload(data_root)
    episodes: List[Dict[str, Any]] = []
    for session_root in discover_session_roots(data_root):
        index_payload = load_json(session_root / "index.json")
        session_id = str(index_payload.get("session_id") or session_root.name).strip()
        for episode_meta in index_payload.get("episodes") or []:
            if not isinstance(episode_meta, dict):
                continue
            episode_id = str(episode_meta.get("episode_id") or "").strip()
            if not episode_id:
                continue
            episode_path = session_root / "episodes" / f"{episode_id}.json"
            if not episode_path.exists():
                continue
            episode_payload = load_json(episode_path)
            quality_label = resolved_episode_quality(episode_payload, episode_meta, None, session_id, episode_id).get("quality_label") or 0
            frame_processing = episode_frame_processing(episode_payload)
            item = {
                "session_id": session_id,
                "episode_id": episode_id,
                "quality_label": int(quality_label),
                "frame_selection_mode": str(frame_processing["frame_selection_mode"]),
                "frame_keep_map": {str(index): value for index, value in dict(frame_processing["frame_keep_map"]).items()},
                "selected_frame_count": int(frame_processing["selected_frame_count"]),
                "total_frame_count": int(frame_processing["total_frame_count"]),
            }
            quality_notes = review_notes.get((session_id, episode_id), "")
            if quality_notes:
                item["quality_notes"] = quality_notes
            episodes.append(item)
    episodes.sort(key=lambda item: (item["session_id"], item["episode_id"]))
    live_payload["episodes"] = episodes
    return live_payload


def serve_report(
    output_dir: Path,
    data_root: Path,
    host: str,
    port: int,
    extra_file_roots: Optional[Sequence[Path]] = None,
) -> None:
    output_dir = output_dir.resolve()
    data_root = data_root.resolve()
    session_roots = discover_session_roots(data_root)
    review_enabled = bool(session_roots)
    allowed_file_roots = {data_root}
    for session_root in session_roots:
        try:
            allowed_file_roots.add(session_root.resolve())
        except FileNotFoundError:
            continue
    for extra_root in extra_file_roots or []:
        try:
            allowed_file_roots.add(extra_root.resolve())
        except FileNotFoundError:
            continue

    class ReportRequestHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(output_dir), **kwargs)

        def _send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path) -> None:
            mime_type, _ = mimetypes.guess_type(str(path))
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", (mime_type or "application/octet-stream"))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/review/load":
                if not review_enabled:
                    self._send_json({"error": "review_api_disabled"}, status=HTTPStatus.NOT_FOUND)
                    return
                self._send_json(_build_live_review_payload(data_root))
                return
            if parsed.path == "/api/file":
                query = parse_qs(parsed.query)
                raw_path = str((query.get("path") or [""])[0]).strip()
                try:
                    requested_path = Path(raw_path).resolve(strict=True)
                    if not any(requested_path.is_relative_to(root) for root in allowed_file_roots):
                        raise PermissionError(f"path is outside allowed roots: {requested_path}")
                    if not requested_path.is_file():
                        raise FileNotFoundError(str(requested_path))
                    self._send_file(requested_path)
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            super().do_GET()

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/review/save":
                self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
                return
            if not review_enabled:
                self._send_json({"error": "review_api_disabled"}, status=HTTPStatus.NOT_FOUND)
                return
            content_length = int(self.headers.get("Content-Length", "0") or 0)
            try:
                raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
                payload = json.loads(raw_body.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("payload must be object")
                sync_result = _sync_source_episode_files(data_root, payload)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"ok": True, **sync_result})

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), ReportRequestHandler)
    url = f"http://{host}:{port}/index.html"
    print(f"本地报告服务已启动：{url}")
    if review_enabled:
        print(f"同时同步更新原始 episode 源文件：{data_root}/*/episodes/*.json")
    else:
        print("当前为只读报告模式，不会写回原始 episode 源文件。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n收到中断，正在停止本地报告服务...")
    finally:
        server.server_close()


def _polyline(values: Sequence[float], width: int, height: int, minimum: Optional[float] = None, maximum: Optional[float] = None) -> str:
    if not values:
        return ""
    minimum = min(values) if minimum is None else minimum
    maximum = max(values) if maximum is None else maximum
    if math.isclose(minimum, maximum):
        minimum -= 1.0
        maximum += 1.0
    points = []
    for index, value in enumerate(values):
        x = 0.0 if len(values) == 1 else (index / (len(values) - 1)) * width
        y = height - ((value - minimum) / (maximum - minimum)) * height
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _plot_svg(episode: EpisodeRecord) -> str:
    width = 720
    height = 220
    plot_left = 42
    plot_right = width - 16
    plot_top = 18
    plot_bottom = height - 24
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top
    colors = {"vx": "#cc5500", "vy": "#6b7280", "wz": "#2b59c3"}
    values = {axis: [frame.control_action[axis] for frame in episode.frames] for axis in ACTION_FIELDS}
    flat_values = [value for axis_values in values.values() for value in axis_values]
    limit = max([abs(value) for value in flat_values], default=0.1)
    limit = max(limit, 0.1)
    minimum = -limit
    maximum = limit

    def x_pos(index: int) -> float:
        if len(episode.frames) <= 1:
            return float(plot_left)
        return plot_left + (index / (len(episode.frames) - 1)) * plot_width

    def y_pos(value: float) -> float:
        return plot_bottom - ((value - minimum) / (maximum - minimum)) * plot_height

    def polyline(axis: str) -> str:
        return " ".join(
            f"{x_pos(index):.1f},{y_pos(value):.1f}"
            for index, value in enumerate(values[axis])
        )

    y_ticks = [minimum, -limit / 2.0, 0.0, limit / 2.0, maximum]
    x_tick_indices = sorted({0, max(0, len(episode.frames) // 4), max(0, len(episode.frames) // 2), max(0, (3 * len(episode.frames)) // 4), max(0, len(episode.frames) - 1)})

    grid_lines = "".join(
        f'<line x1="{plot_left}" y1="{y_pos(tick):.1f}" x2="{plot_right}" y2="{y_pos(tick):.1f}" stroke="#d1d5db" stroke-width="1" />'
        for tick in y_ticks
    )
    y_labels = "".join(
        f'<text x="{plot_left - 8}" y="{y_pos(tick) + 4:.1f}" fill="{"#111111" if not math.isclose(tick, 0.0) else "#9a3412"}" font-size="11" text-anchor="end">{tick:.2f}</text>'
        for tick in y_ticks
    )
    x_labels = "".join(
        f'<line x1="{x_pos(index):.1f}" y1="{plot_bottom}" x2="{x_pos(index):.1f}" y2="{plot_bottom + 6}" stroke="#6b7280" stroke-width="1" />'
        f'<text x="{x_pos(index):.1f}" y="{plot_bottom + 18}" fill="#374151" font-size="11" text-anchor="middle">{episode.frames[index].frame_index + 1}</text>'
        for index in x_tick_indices
        if episode.frames
    )
    lines = []
    for axis in ACTION_FIELDS:
        lines.append(
            f'<polyline fill="none" stroke="{colors[axis]}" stroke-width="2.2" points="{polyline(axis)}" />'
        )
    legend = " ".join(
        f'<text x="{16 + index * 54}" y="14" fill="{colors[axis]}" font-size="12">{axis}</text>'
        for index, axis in enumerate(ACTION_FIELDS)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="action plot">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" stroke="#d0d7de" />'
        f'<rect x="{plot_left}" y="{plot_top}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#d0d7de" />'
        f'{grid_lines}{y_labels}{x_labels}'
        f'<rect id="actionRangeHighlight" x="{plot_left}" y="{plot_top}" width="0" height="{plot_height}" fill="#e5e7eb" fill-opacity="0.65" visibility="hidden" />'
        f'<line id="actionRangeStartLine" x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#9ca3af" stroke-width="1.5" visibility="hidden" />'
        f'<line id="actionRangeEndLine" x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#9ca3af" stroke-width="1.5" visibility="hidden" />'
        f'<line id="actionCurrentLine" x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#111111" stroke-width="2" visibility="hidden" />'
        f'<circle id="actionCurrentVx" cx="{plot_left}" cy="{y_pos(0.0):.1f}" r="4" fill="{colors["vx"]}" visibility="hidden" />'
        f'<circle id="actionCurrentVy" cx="{plot_left}" cy="{y_pos(0.0):.1f}" r="4" fill="{colors["vy"]}" visibility="hidden" />'
        f'<circle id="actionCurrentWz" cx="{plot_left}" cy="{y_pos(0.0):.1f}" r="4" fill="{colors["wz"]}" visibility="hidden" />'
        f'<line x1="{plot_left}" y1="{y_pos(0.0):.1f}" x2="{plot_right}" y2="{y_pos(0.0):.1f}" stroke="#9a3412" stroke-dasharray="6 4" />'
        f'<text x="{plot_right - 4}" y="{y_pos(0.0) - 6:.1f}" fill="#9a3412" font-size="11" text-anchor="end">0.00</text>'
        f'{"".join(lines)}{legend}'
        f'<text id="actionCurrentLabel" x="{plot_left + 4}" y="{plot_top + 12}" fill="#111111" font-size="11">current image frame</text>'
        f'<text x="{plot_right - 4}" y="{height - 6}" fill="#4b5563" font-size="11" text-anchor="end">frame index</text>'
        '</svg>'
    )


def _state_plot_svg(
    episode: EpisodeRecord,
    axes: Sequence[str],
    title: str,
    overlay_axes: Optional[Sequence[str]] = None,
    overlay_dasharray: str = "7 5",
) -> str:
    width = 720
    height = 150
    raw_values = {axis: [frame.state.get(axis) for frame in episode.frames] for axis in axes}
    available: Dict[str, List[float]] = {}
    for axis, values in raw_values.items():
        if any(value is not None for value in values):
            available[axis] = [_safe_float(value) for value in values]
    overlay_available: Dict[str, List[float]] = {}
    for axis in overlay_axes or []:
        overlay_values = [frame.state.get(axis) for frame in episode.frames]
        if any(value is not None for value in overlay_values):
            overlay_available[axis] = [_safe_float(value) for value in overlay_values]
    if not available and not overlay_available:
        return f'<p class="meta">{escape(title)}：当前 episode 没有这些状态字段。</p>'

    all_series = list(available.values()) + list(overlay_available.values())
    minimum = min(min(values) for values in all_series)
    maximum = max(max(values) for values in all_series)
    if math.isclose(minimum, maximum):
        minimum -= 1.0
        maximum += 1.0

    lines = []
    for axis, values in available.items():
        color = STATE_AXIS_COLORS.get(axis, "#444")
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{_polyline(values, width, height, minimum, maximum)}" />'
        )
    for axis, values in overlay_available.items():
        base_axis = axis[:-5] if axis.endswith("_prev") else axis
        color = STATE_AXIS_COLORS.get(base_axis, STATE_AXIS_COLORS.get(axis, "#444"))
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.2" stroke-dasharray="{overlay_dasharray}" '
            f'points="{_polyline(values, width, height, minimum, maximum)}" />'
        )
    zero_line = ""
    if minimum < 0.0 < maximum:
        zero_y = height - ((0.0 - minimum) / (maximum - minimum)) * height
        zero_line = (
            f'<line x1="0" y1="{zero_y:.1f}" x2="{width}" y2="{zero_y:.1f}" '
            f'stroke="#b8b0a0" stroke-dasharray="4 4" />'
        )
    legend = " ".join(
        f'<text x="{20 + index * 120}" y="20" fill="{STATE_AXIS_COLORS.get(axis, "#444")}" font-size="14">{escape(axis)}</text>'
        for index, axis in enumerate(available.keys())
    )
    overlay_legend = " ".join(
        f'<text x="{20 + (len(available) + index) * 120}" y="20" fill="{STATE_AXIS_COLORS.get(axis[:-5] if axis.endswith("_prev") else axis, "#444")}" font-size="14">{escape(axis)} (dashed)</text>'
        for index, axis in enumerate(overlay_available.keys())
    )
    return (
        f'<svg viewBox="0 0 {width} {height + 30}" role="img" aria-label="{escape(title)}">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff" stroke="#d7ccb8" />'
        f'{zero_line}{"".join(lines)}{legend}{overlay_legend}</svg>'
    )


def _histogram_svg(values: Sequence[float], axis: str) -> str:
    width = 180
    height = 96
    bins = 12
    counts = [0] * bins
    for value in values:
        clamped = max(-1.0, min(1.0, value))
        index = min(bins - 1, max(0, int((clamped + 1.0) / 2.0 * bins)))
        counts[index] += 1
    peak = max(counts) if counts else 1
    bars = []
    bar_width = width / bins
    for index, count in enumerate(counts):
        bar_height = 0 if peak == 0 else (count / peak) * height
        x = index * bar_width
        y = height - bar_height
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 2:.1f}" height="{bar_height:.1f}" fill="#7a9e9f" />')
    return (
        f'<div><strong>{escape(axis)}</strong>'
        f'<svg viewBox="0 0 {width} {height + 20}"><rect x="0" y="0" width="{width}" height="{height}" fill="#fff" stroke="#d7ccb8" />'
        f'{"".join(bars)}<text x="0" y="{height + 16}" font-size="12">-1.0</text>'
        f'<text x="{width - 28}" y="{height + 16}" font-size="12">1.0</text></svg></div>'
    )


def _review_shared_script(data_root: Path, episode_refs_payload: Sequence[Dict[str, Any]], initial_reviews: Dict[str, Dict[str, Any]]) -> str:
    return f"""
    const datasetRoot = {json.dumps(str(data_root), ensure_ascii=True)};
    const qualityReviewFilename = {json.dumps(QUALITY_REVIEW_FILENAME, ensure_ascii=True)};
    const qualityReviewSchemaVersion = {json.dumps(QUALITY_REVIEW_SCHEMA_VERSION, ensure_ascii=True)};
    const excludedQualityLabel = {QUALITY_LABEL_EXCLUDED};
    const actionActivityEps = {ACTION_ACTIVITY_EPS};
    const episodeRefs = {json.dumps(list(episode_refs_payload), ensure_ascii=True)};
    const initialReviews = {json.dumps(initial_reviews, ensure_ascii=True)};
    const reviewStorageKey = `go2_quality_review::${{datasetRoot}}`;
    const episodeRefsByKey = Object.fromEntries(
      episodeRefs.map((item) => [`${{item.session_id}}::${{item.episode_id}}`, item]),
    );
    let reviewState = {{}};
    let serverAutosaveAvailable = false;
    let autosaveHandle = null;
    let autosaveReady = false;
    let reviewDirty = false;
    let saveInFlight = false;

    function notifyReviewStateChanged() {{
      window.dispatchEvent(new CustomEvent('go2-review-state-updated'));
    }}

    function setReviewDirty(value) {{
      reviewDirty = Boolean(value);
      window.dispatchEvent(new CustomEvent('go2-review-dirty-changed', {{ detail: {{ dirty: reviewDirty, saving: saveInFlight }} }}));
    }}

    function setSaveInFlight(value) {{
      saveInFlight = Boolean(value);
      window.dispatchEvent(new CustomEvent('go2-review-dirty-changed', {{ detail: {{ dirty: reviewDirty, saving: saveInFlight }} }}));
    }}

    function reviewHandleDb() {{
      return new Promise((resolve, reject) => {{
        if (!window.indexedDB) {{
          resolve(null);
          return;
        }}
        const request = window.indexedDB.open('go2_quality_review_handles', 1);
        request.onupgradeneeded = () => {{
          const db = request.result;
          if (!db.objectStoreNames.contains('handles')) {{
            db.createObjectStore('handles');
          }}
        }};
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      }});
    }}

    async function loadStoredHandle() {{
      const db = await reviewHandleDb();
      if (!db) {{
        return null;
      }}
      return await new Promise((resolve, reject) => {{
        const tx = db.transaction('handles', 'readonly');
        const store = tx.objectStore('handles');
        const request = store.get(reviewStorageKey);
        request.onsuccess = () => resolve(request.result || null);
        request.onerror = () => reject(request.error);
      }});
    }}

    async function storeHandle(handle) {{
      const db = await reviewHandleDb();
      if (!db) {{
        return;
      }}
      await new Promise((resolve, reject) => {{
        const tx = db.transaction('handles', 'readwrite');
        const store = tx.objectStore('handles');
        const request = store.put(handle, reviewStorageKey);
        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
      }});
    }}

    async function clearStoredHandle() {{
      const db = await reviewHandleDb();
      if (!db) {{
        return;
      }}
      await new Promise((resolve, reject) => {{
        const tx = db.transaction('handles', 'readwrite');
        const store = tx.objectStore('handles');
        const request = store.delete(reviewStorageKey);
        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
      }});
    }}

    function makeEpisodeKey(sessionId, episodeId) {{
      return `${{String(sessionId || '').trim()}}::${{String(episodeId || '').trim()}}`;
    }}

    function labelText(label) {{
      switch (Number(label) || 0) {{
        case 1:
          return '1 好样本';
        case 2:
          return '2 可用但不完美';
        case 3:
          return '3 失败但保留 / 质量差';
        default:
          return '0 未知';
      }}
    }}

    function labelClass(label) {{
      return `label-${{Number(label) || 0}}`;
    }}

    function normalizeReview(raw, fallbackRef) {{
      const ref = fallbackRef || {{}};
      const sourceQualityLabel = Math.max(0, Math.min(3, Number((raw && (raw.source_quality_label ?? raw.original_quality_label ?? raw.quality_label)) ?? ref.quality_label ?? ref.original_quality_label) || 0));
      const currentQualityLabel = Math.max(0, Math.min(3, Number((raw && raw.quality_label) ?? ref.quality_label) || 0));
      const qualityNotes = String((raw && (raw.quality_notes || raw.notes)) || '').trim();
      const frameSelectionMode = String((raw && raw.frame_selection_mode) || ref.frame_selection_mode || 'all').trim() === 'retain_only' ? 'retain_only' : 'all';
      const rawFrameKeepMap = (raw && raw.frame_keep_map) || ref.frame_keep_map || {{}};
      const frameKeepMap = {{}};
      if (rawFrameKeepMap && typeof rawFrameKeepMap === 'object') {{
        Object.entries(rawFrameKeepMap).forEach(([key, value]) => {{
          const index = Number(key);
          if (Number.isInteger(index) && index >= 0) {{
            frameKeepMap[String(index)] = Boolean(value);
          }}
        }});
      }}
      const totalFrameCount = Math.max(0, Number((raw && raw.total_frame_count) ?? ref.total_frame_count) || 0);
      const selectedFrameCount = Object.entries(frameKeepMap).reduce((count, [_, keep]) => {{
        if (frameSelectionMode === 'retain_only') {{
          return count + (keep ? 1 : 0);
        }}
        return count + (keep !== false ? 1 : 0);
      }}, frameSelectionMode === 'retain_only' ? 0 : totalFrameCount - Object.values(frameKeepMap).filter((value) => value === false).length);
      return {{
        session_id: String((raw && raw.session_id) || ref.session_id || '').trim(),
        episode_id: String((raw && raw.episode_id) || ref.episode_id || '').trim(),
        source_quality_label: sourceQualityLabel,
        original_quality_label: sourceQualityLabel,
        quality_label: currentQualityLabel,
        quality_overridden: currentQualityLabel !== sourceQualityLabel,
        quality_notes: qualityNotes,
        exclude_from_processing: currentQualityLabel === excludedQualityLabel || Boolean(raw && raw.exclude_from_processing),
        frame_selection_mode: frameSelectionMode,
        frame_keep_map: frameKeepMap,
        total_frame_count: totalFrameCount,
        selected_frame_count: selectedFrameCount,
      }};
    }}

    function mergeReviewSource(source) {{
      if (!source || typeof source !== 'object') {{
        return;
      }}
      Object.entries(source).forEach(([key, value]) => {{
        const normalizedKey = key.includes('::')
          ? key
          : makeEpisodeKey(value && value.session_id, value && value.episode_id);
        if (!normalizedKey) {{
          return;
        }}
        const fallbackRef = episodeRefsByKey[normalizedKey] || value || {{}};
        const normalized = normalizeReview(value || {{}}, fallbackRef);
        if (!normalized.session_id || !normalized.episode_id) {{
          return;
        }}
        fallbackRef.quality_label = normalized.source_quality_label;
        fallbackRef.original_quality_label = normalized.source_quality_label;
        fallbackRef.source_quality_label = normalized.source_quality_label;
        fallbackRef.frame_selection_mode = normalized.frame_selection_mode;
        fallbackRef.frame_keep_map = normalized.frame_keep_map;
        fallbackRef.total_frame_count = normalized.total_frame_count;
        fallbackRef.selected_frame_count = normalized.selected_frame_count;
        if (!normalized.quality_overridden && !normalized.quality_notes) {{
          delete reviewState[normalizedKey];
          return;
        }}
        reviewState[normalizedKey] = {{
          session_id: normalized.session_id,
          episode_id: normalized.episode_id,
          source_quality_label: normalized.source_quality_label,
          quality_label: normalized.quality_label,
          quality_notes: normalized.quality_notes,
          exclude_from_processing: normalized.exclude_from_processing,
          frame_selection_mode: normalized.frame_selection_mode,
          frame_keep_map: normalized.frame_keep_map,
          total_frame_count: normalized.total_frame_count,
          selected_frame_count: normalized.selected_frame_count,
        }};
      }});
    }}

    function reapplySourceFrameSelection(source) {{
      if (!source || typeof source !== 'object') {{
        return;
      }}
      Object.entries(source).forEach(([key, value]) => {{
        const normalizedKey = key.includes('::')
          ? key
          : makeEpisodeKey(value && value.session_id, value && value.episode_id);
        if (!normalizedKey) {{
          return;
        }}
        const fallbackRef = episodeRefsByKey[normalizedKey] || value || {{}};
        const sourceReview = normalizeReview(value || {{}}, fallbackRef);
        if (!sourceReview.session_id || !sourceReview.episode_id) {{
          return;
        }}
        fallbackRef.frame_selection_mode = sourceReview.frame_selection_mode;
        fallbackRef.frame_keep_map = sourceReview.frame_keep_map;
        fallbackRef.total_frame_count = sourceReview.total_frame_count;
        fallbackRef.selected_frame_count = sourceReview.selected_frame_count;

        const existing = reviewState[normalizedKey];
        if (!existing) {{
          return;
        }}
        const merged = normalizeReview({{
          ...existing,
          frame_selection_mode: sourceReview.frame_selection_mode,
          frame_keep_map: sourceReview.frame_keep_map,
          total_frame_count: sourceReview.total_frame_count,
          selected_frame_count: sourceReview.selected_frame_count,
        }}, fallbackRef);
        if (!merged.quality_overridden && !merged.quality_notes) {{
          delete reviewState[normalizedKey];
          return;
        }}
        reviewState[normalizedKey] = {{
          session_id: merged.session_id,
          episode_id: merged.episode_id,
          source_quality_label: merged.source_quality_label,
          quality_label: merged.quality_label,
          quality_notes: merged.quality_notes,
          exclude_from_processing: merged.exclude_from_processing,
          frame_selection_mode: merged.frame_selection_mode,
          frame_keep_map: merged.frame_keep_map,
          total_frame_count: merged.total_frame_count,
          selected_frame_count: merged.selected_frame_count,
        }};
      }});
    }}

    function payloadEpisodesToSource(payload) {{
      const result = {{}};
      const episodes = Array.isArray(payload && payload.episodes) ? payload.episodes : [];
      episodes.forEach((item) => {{
        const key = makeEpisodeKey(item && item.session_id, item && item.episode_id);
        if (!key) {{
          return;
        }}
        result[key] = {{
          session_id: item.session_id,
          episode_id: item.episode_id,
          source_quality_label: Number(item.quality_label) || Number(item.original_quality_label) || 0,
          quality_label: Number(item.quality_label) || Number(item.original_quality_label) || 0,
          quality_notes: item.quality_notes || '',
          frame_selection_mode: item.frame_selection_mode || 'all',
          frame_keep_map: item.frame_keep_map || {{}},
          total_frame_count: Number(item.total_frame_count) || 0,
          selected_frame_count: Number(item.selected_frame_count) || 0,
        }};
      }});
      return result;
    }}

    function commitSavedReviewStateToSource() {{
      episodeRefs.forEach((item) => {{
        const review = getEpisodeReview(item.session_id, item.episode_id);
        const key = makeEpisodeKey(item.session_id, item.episode_id);
        const sourceQualityLabel = Number(review.quality_label) || 0;
        item.quality_label = sourceQualityLabel;
        item.original_quality_label = sourceQualityLabel;
        if (!review.quality_notes) {{
          delete reviewState[key];
          return;
        }}
        reviewState[key] = {{
          session_id: item.session_id,
          episode_id: item.episode_id,
          source_quality_label: sourceQualityLabel,
          quality_label: sourceQualityLabel,
          quality_notes: review.quality_notes,
          exclude_from_processing: sourceQualityLabel === excludedQualityLabel,
          frame_selection_mode: review.frame_selection_mode,
          frame_keep_map: review.frame_keep_map,
          total_frame_count: review.total_frame_count,
          selected_frame_count: review.selected_frame_count,
        }};
      }});
      persistReviews();
      setReviewDirty(false);
      notifyReviewStateChanged();
    }}

    function loadStoredReviews() {{
      try {{
        const raw = window.localStorage.getItem(reviewStorageKey);
        if (!raw) {{
          return {{}};
        }}
        const payload = JSON.parse(raw);
        return payload && typeof payload === 'object' ? payload : {{}};
      }} catch (error) {{
        return {{}};
      }}
    }}

    function persistReviews() {{
      try {{
        window.localStorage.setItem(reviewStorageKey, JSON.stringify(reviewState));
      }} catch (error) {{
        announceStatus(`保存浏览器缓存失败：${{error}}`, true);
      }}
    }}

    function seedReviews() {{
      reviewState = {{}};
      mergeReviewSource(initialReviews);
      mergeReviewSource(loadStoredReviews());
      persistReviews();
      setReviewDirty(false);
    }}

    function getEpisodeReview(sessionId, episodeId) {{
      const key = makeEpisodeKey(sessionId, episodeId);
      const fallbackRef = episodeRefsByKey[key] || {{ session_id: sessionId, episode_id: episodeId }};
      return normalizeReview(reviewState[key] || {{}}, fallbackRef);
    }}

    function upsertEpisodeReview(sessionId, episodeId, qualityLabel, qualityNotes) {{
      const key = makeEpisodeKey(sessionId, episodeId);
      const fallbackRef = episodeRefsByKey[key] || {{ session_id: sessionId, episode_id: episodeId }};
      const sourceQualityLabel = Number(fallbackRef.quality_label ?? fallbackRef.original_quality_label) || 0;
      const currentQualityLabel = Number(qualityLabel) || 0;
      const normalized = normalizeReview({{
        session_id: fallbackRef.session_id,
        episode_id: fallbackRef.episode_id,
        source_quality_label: sourceQualityLabel,
        quality_label: currentQualityLabel,
        quality_notes: qualityNotes,
        frame_selection_mode: fallbackRef.frame_selection_mode || 'all',
        frame_keep_map: fallbackRef.frame_keep_map || {{}},
        total_frame_count: Number(fallbackRef.total_frame_count) || 0,
      }}, fallbackRef);
      if (!normalized.quality_overridden && !normalized.quality_notes) {{
        delete reviewState[key];
      }} else {{
        reviewState[key] = {{
          session_id: normalized.session_id,
          episode_id: normalized.episode_id,
          source_quality_label: normalized.source_quality_label,
          quality_label: normalized.quality_label,
          quality_notes: normalized.quality_notes,
          exclude_from_processing: normalized.exclude_from_processing,
          frame_selection_mode: normalized.frame_selection_mode,
          frame_keep_map: normalized.frame_keep_map,
          total_frame_count: normalized.total_frame_count,
          selected_frame_count: normalized.selected_frame_count,
        }};
      }}
      persistReviews();
      setReviewDirty(true);
      return getEpisodeReview(sessionId, episodeId);
    }}

    function clearEpisodeOverride(sessionId, episodeId) {{
      const key = makeEpisodeKey(sessionId, episodeId);
      delete reviewState[key];
      persistReviews();
      setReviewDirty(true);
      return getEpisodeReview(sessionId, episodeId);
    }}

    function updateEpisodeFrameSelection(sessionId, episodeId, frameSelectionMode, updates) {{
      const key = makeEpisodeKey(sessionId, episodeId);
      const current = getEpisodeReview(sessionId, episodeId);
      const nextMap = {{ ...(current.frame_keep_map || {{}}) }};
      Object.entries(updates || {{}}).forEach(([frameIndex, keep]) => {{
        const index = Number(frameIndex);
        if (!Number.isInteger(index) || index < 0) {{
          return;
        }}
        if (keep === null) {{
          delete nextMap[String(index)];
          return;
        }}
        nextMap[String(index)] = Boolean(keep);
      }});
      const fallbackRef = episodeRefsByKey[key] || {{ session_id: sessionId, episode_id: episodeId }};
      const normalized = normalizeReview({{
        session_id: fallbackRef.session_id,
        episode_id: fallbackRef.episode_id,
        source_quality_label: current.source_quality_label,
        quality_label: current.quality_label,
        quality_notes: current.quality_notes,
        frame_selection_mode: frameSelectionMode || current.frame_selection_mode || 'all',
        frame_keep_map: nextMap,
        total_frame_count: current.total_frame_count || fallbackRef.total_frame_count || 0,
      }}, fallbackRef);
      reviewState[key] = {{
        session_id: normalized.session_id,
        episode_id: normalized.episode_id,
        source_quality_label: normalized.source_quality_label,
        quality_label: normalized.quality_label,
        quality_notes: normalized.quality_notes,
        exclude_from_processing: normalized.exclude_from_processing,
        frame_selection_mode: normalized.frame_selection_mode,
        frame_keep_map: normalized.frame_keep_map,
        total_frame_count: normalized.total_frame_count,
        selected_frame_count: normalized.selected_frame_count,
      }};
      persistReviews();
      setReviewDirty(true);
      return getEpisodeReview(sessionId, episodeId);
    }}

    function exportableReviews() {{
      return Object.values(reviewState)
        .filter((item) => item.session_id && item.episode_id && ((Number(item.quality_label) || 0) > 0 || item.quality_notes))
        .sort((left, right) => {{
          const sessionCmp = left.session_id.localeCompare(right.session_id);
          if (sessionCmp !== 0) {{
            return sessionCmp;
          }}
          return left.episode_id.localeCompare(right.episode_id);
        }});
    }}

    function exportableSourceEpisodeReviews() {{
      return episodeRefs
        .map((item) => {{
          const review = getEpisodeReview(item.session_id, item.episode_id);
          return {{
            session_id: item.session_id,
            episode_id: item.episode_id,
            quality_label: Number(review.quality_label) || 0,
            quality_notes: review.quality_notes,
            frame_selection_mode: review.frame_selection_mode,
            frame_keep_map: review.frame_keep_map,
            total_frame_count: review.total_frame_count,
            selected_frame_count: review.selected_frame_count,
          }};
        }})
        .filter((item) => item.session_id && item.episode_id && (Number(item.quality_label) || 0) > 0);
    }}

    function buildExportPayload() {{
      return {{
        schema_version: qualityReviewSchemaVersion,
        dataset_root: datasetRoot,
        excluded_quality_label: excludedQualityLabel,
        review_filename_hint: qualityReviewFilename,
        semantics: 'quality_label stores the current recording score (1/2/3)',
        episodes: exportableReviews(),
        source_episode_reviews: exportableSourceEpisodeReviews(),
      }};
    }}

    async function loadServerReviewPayload() {{
      const response = await fetch('/api/review/load', {{
        method: 'GET',
        cache: 'no-store',
        headers: {{ 'Accept': 'application/json' }},
      }});
      if (!response.ok) {{
        throw new Error(`server load failed: ${{response.status}}`);
      }}
      return await response.json();
    }}

    async function saveServerReviewPayload(announceMessage = '') {{
      setSaveInFlight(true);
      const response = await fetch('/api/review/save', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(buildExportPayload()),
      }});
      if (!response.ok) {{
        let errorMessage = `server save failed: ${{response.status}}`;
        try {{
          const payload = await response.json();
          if (payload && payload.error) {{
            errorMessage = String(payload.error);
          }}
        }} catch (error) {{
        }}
        throw new Error(errorMessage);
      }}
      commitSavedReviewStateToSource();
      serverAutosaveAvailable = true;
      updateBoundFileStatus();
      if (announceMessage) {{
        announceStatus(announceMessage);
      }}
      setSaveInFlight(false);
      return true;
    }}

    async function ensureHandlePermission(handle) {{
      if (!handle || typeof handle.queryPermission !== 'function') {{
        return false;
      }}
      if ((await handle.queryPermission({{ mode: 'readwrite' }})) === 'granted') {{
        return true;
      }}
      return (await handle.requestPermission({{ mode: 'readwrite' }})) === 'granted';
    }}

    function updateBoundFileStatus() {{
      const node = document.getElementById('reviewFileStatus');
      if (!node) {{
        return;
      }}
      if (serverAutosaveAvailable) {{
        node.textContent = '已连接本地报告服务，点击“保存到源文件”会写回 episodes/*.json 与 index.json';
        return;
      }}
      if (autosaveHandle && autosaveReady) {{
        node.textContent = `已绑定本地文件：${{autosaveHandle.name}}，点击保存后会写回`;
        return;
      }}
      if (!window.showSaveFilePicker) {{
        node.textContent = '当前浏览器不支持直接写回本地文件，只能先缓存到浏览器。';
        return;
      }}
      node.textContent = '当前仅写入浏览器缓存；绑定记录文件后，点击保存可写回本地 quality_review.json。';
    }}

    async function flushReviewsToBoundFile(announceMessage = '') {{
      if (!autosaveHandle) {{
        updateBoundFileStatus();
        return false;
      }}
      setSaveInFlight(true);
      if (!(await ensureHandlePermission(autosaveHandle))) {{
        autosaveReady = false;
        updateBoundFileStatus();
        announceStatus('本地文件写权限未授权，当前只保存在浏览器缓存中。', true);
        setSaveInFlight(false);
        return false;
      }}
      const text = `${{JSON.stringify(buildExportPayload(), null, 2)}}\\n`;
      const writable = await autosaveHandle.createWritable();
      await writable.write(text);
      await writable.close();
      autosaveReady = true;
      updateBoundFileStatus();
      if (announceMessage) {{
        announceStatus(announceMessage);
      }}
      setReviewDirty(false);
      setSaveInFlight(false);
      return true;
    }}

    async function bindReviewFile() {{
      if (serverAutosaveAvailable) {{
        announceStatus('当前已连接本地报告服务，点击“保存到源文件”会写回 episodes/*.json 与 index.json。');
        return;
      }}
      if (!window.showSaveFilePicker) {{
        announceStatus('当前浏览器不支持直接写回本地文件，请改用导出 JSON。', true);
        return;
      }}
      const handle = await window.showSaveFilePicker({{
        suggestedName: qualityReviewFilename,
        types: [{{ description: 'JSON', accept: {{ 'application/json': ['.json'] }} }}],
      }});
      autosaveHandle = handle;
      autosaveReady = false;
      await storeHandle(handle);
      await flushReviewsToBoundFile(`已绑定并写回 ${{handle.name}}`);
    }}

    async function restoreBoundFile() {{
      try {{
        const handle = await loadStoredHandle();
        if (!handle) {{
          updateBoundFileStatus();
          return;
        }}
        autosaveHandle = handle;
        autosaveReady = (await ensureHandlePermission(handle));
        updateBoundFileStatus();
      }} catch (error) {{
        autosaveHandle = null;
        autosaveReady = false;
        await clearStoredHandle();
        updateBoundFileStatus();
      }}
    }}

    async function persistReviewState(announceMessage = '') {{
      if (serverAutosaveAvailable) {{
        try {{
          await saveServerReviewPayload(announceMessage);
          return;
        }} catch (error) {{
          setSaveInFlight(false);
          serverAutosaveAvailable = false;
          updateBoundFileStatus();
          announceStatus(`本地服务写回失败：${{error}}`, true);
        }}
      }}
      if (autosaveHandle) {{
        try {{
          const saved = await flushReviewsToBoundFile(announceMessage);
          if (saved) {{
            return;
          }}
        }} catch (error) {{
          setSaveInFlight(false);
          autosaveReady = false;
          updateBoundFileStatus();
          announceStatus(`自动写回失败：${{error}}`, true);
          return;
        }}
      }}
      if (announceMessage) {{
        announceStatus(`${{announceMessage}}（当前仅保存在浏览器缓存）`);
      }}
    }}

    async function exportReviewFile() {{
      const text = `${{JSON.stringify(buildExportPayload(), null, 2)}}\\n`;
      if (window.showSaveFilePicker) {{
        const handle = await window.showSaveFilePicker({{
          suggestedName: qualityReviewFilename,
          types: [{{ description: 'JSON', accept: {{ 'application/json': ['.json'] }} }}],
        }});
        const writable = await handle.createWritable();
        await writable.write(text);
        await writable.close();
        announceStatus(`已保存当前页面修改记录，建议放到 ${{datasetRoot}}/${{qualityReviewFilename}}`);
        return;
      }}
      const blob = new Blob([text], {{ type: 'application/json' }});
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = qualityReviewFilename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      announceStatus(`已导出当前页面修改记录，请将下载的文件保存到 ${{datasetRoot}}/${{qualityReviewFilename}}`);
    }}

    async function importReviewFile(file) {{
      const text = await file.text();
      const payload = JSON.parse(text);
      const episodes = Array.isArray(payload && payload.episodes) ? payload.episodes : [];
      const imported = {{}};
      episodes.forEach((item) => {{
        const key = makeEpisodeKey(item && item.session_id, item && item.episode_id);
        if (!key) {{
          return;
        }}
        imported[key] = item;
      }});
      mergeReviewSource(imported);
      persistReviews();
      setReviewDirty(true);
      announceStatus(`已导入 ${{episodes.length}} 条本地修改记录`);
    }}

    function clearReviewCache() {{
      reviewState = {{}};
      persistReviews();
      setReviewDirty(true);
      notifyReviewStateChanged();
    }}

    function announceStatus(message, isError = false) {{
      const node = document.getElementById('reviewStatus');
      if (!node) {{
        return;
      }}
      node.textContent = message;
      node.style.color = isError ? '#a12626' : '#5f5f5f';
    }}

    async function initializeServerAutosave() {{
      if (!String(window.location.protocol || '').startsWith('http')) {{
        updateBoundFileStatus();
        return;
      }}
      try {{
        const payload = await loadServerReviewPayload();
        const serverSource = payloadEpisodesToSource(payload);
        reviewState = {{}};
        mergeReviewSource(serverSource);
        mergeReviewSource(loadStoredReviews());
        reapplySourceFrameSelection(serverSource);
        persistReviews();
        serverAutosaveAvailable = true;
        setReviewDirty(false);
        notifyReviewStateChanged();
      }} catch (error) {{
        serverAutosaveAvailable = false;
      }}
      updateBoundFileStatus();
    }}

    window.addEventListener('beforeunload', (event) => {{
      if (!reviewDirty && !saveInFlight) {{
        return;
      }}
      event.preventDefault();
      event.returnValue = '';
    }});

    seedReviews();
    initializeServerAutosave()
      .catch(() => updateBoundFileStatus())
      .finally(() => {{
        if (!serverAutosaveAvailable) {{
          restoreBoundFile().catch(() => updateBoundFileStatus());
        }}
      }});
    """


def _episode_page(
    episode: EpisodeRecord,
    page_path: Path,
    index_path: Path,
    rng: random.Random,
    num_samples: int,
    data_root: Path,
    episode_refs_payload: Sequence[Dict[str, Any]],
    initial_reviews: Dict[str, Dict[str, Any]],
    prev_page_name: Optional[str],
    next_page_name: Optional[str],
) -> str:
    frames_payload = []
    for frame in episode.frames:
        frames_payload.append(
            {
                "image": os.path.relpath(frame.image_disk_path, page_path.parent),
                "disk_path": str(frame.image_disk_path),
                "frame_index": frame.frame_index,
                "timestamp": round(frame.timestamp, 6),
                "instruction": frame.instruction,
                "vx": frame.control_action["vx"],
                "vy": frame.control_action["vy"],
                "wz": frame.control_action["wz"],
                "state": frame.state,
                "retain_for_processing": frame.retain_for_processing,
                "image_rel": frame.image_rel,
                "raw_frame": frame.raw_payload,
            }
        )
    action_plot_limit = max(
        [abs(frame.control_action[axis]) for frame in episode.frames for axis in ACTION_FIELDS],
        default=0.1,
    )
    action_plot_limit = max(action_plot_limit, 0.1)
    histogram_html = "".join(_histogram_svg([frame.control_action[axis] for frame in episode.frames], axis) for axis in ACTION_FIELDS)
    state_velocity_svg = _state_plot_svg(
        episode,
        ("vx", "vy", "yaw_speed"),
        "state velocity vs previous action",
        overlay_axes=("vx_prev", "vy_prev", "wz_prev"),
    )
    state_orientation_svg = _state_plot_svg(episode, STATE_ORIENTATION_FIELDS, "state orientation")
    state_position_svg = _state_plot_svg(episode, STATE_POSITION_FIELDS, "state position")
    warnings = "无" if not episode.warnings else ", ".join(episode.warnings)
    info = "无" if not episode.info else ", ".join(episode.info)
    index_rel = os.path.relpath(index_path, page_path.parent)
    current_quality_text = _quality_badge_text(episode.quality_label)
    current_processing_text = "已排除" if episode.exclude_from_processing else "保留"
    review_script = _review_shared_script(data_root, episode_refs_payload, initial_reviews)
    review_select_html = _quality_select_html("qualityLabel", episode.quality_label)
    review_panel_class = "card quality-excluded" if episode.exclude_from_processing else "card"
    prev_link_html = f'<a href="{escape(prev_page_name)}">Prev Episode</a>' if prev_page_name else '<span class="meta">Prev Episode</span>'
    next_link_html = f'<a href="{escape(next_page_name)}">Next Episode</a>' if next_page_name else '<span class="meta">Next Episode</span>'

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>{escape(episode.session_id)} {escape(episode.episode_id)}</title>
  <style>{CSS}</style>
</head>
<body class=\"detail-page\">
  <div class=\"page-shell\">
    <div class=\"page-nav\">
      <a href=\"{escape(index_rel)}\">Back to summary</a>
      {prev_link_html}
      {next_link_html}
    </div>
    <div class=\"episode-layout\">
      <div class=\"episode-stack viewer-panel\">
        <div class=\"card compact-card viewer-header\">
          <h1>{escape(episode.session_id)} / {escape(episode.episode_id)}</h1>
          <p class=\"meta\"><strong>instruction</strong> <code>{escape(episode.instruction)}</code></p>
          <p class=\"meta viewer-help\">空格播放/暂停，左右方向键逐帧；图里的黑色竖线就是当前图片对应帧，三个圆点分别落在 vx / vy / wz 曲线上。</p>
        </div>
        <div class=\"viewer\">
          <img id=\"frameImage\" src=\"\" alt=\"episode replay frame\" />
          <div class=\"overlay\" id=\"overlay\"></div>
        </div>
        <div class=\"controls compact-controls\">
          <button id=\"playPause\">Play</button>
          <button id=\"prevFrame\">Prev Frame</button>
          <button id=\"nextFrame\">Next Frame</button>
          <div class=\"frame-range-wrap\">
            <label>Frame</label>
            <div class=\"frame-range-stack\">
              <div id=\"frameRangeHighlightBar\" class=\"frame-range-highlight\"></div>
              <input id=\"frameRangeStartSlider\" class=\"frame-range-marker\" type=\"range\" min=\"1\" max=\"{max(1, len(frames_payload))}\" value=\"1\" />
              <input id=\"frameRangeEndSlider\" class=\"frame-range-marker\" type=\"range\" min=\"1\" max=\"{max(1, len(frames_payload))}\" value=\"{max(1, len(frames_payload))}\" />
              <input id=\"frameSlider\" class=\"frame-range-current\" type=\"range\" min=\"0\" max=\"{max(0, len(frames_payload) - 1)}\" value=\"0\" />
            </div>
          </div>
          <label>Speed
            <select id=\"speedSelect\">
              <option value=\"0.5\">0.5x</option>
              <option value=\"0.75\">0.75x</option>
              <option value=\"1\" selected>1.0x</option>
              <option value=\"1.5\">1.5x</option>
              <option value=\"2\">2.0x</option>
            </select>
          </label>
          <span class=\"playback-stats\" id=\"playbackStats\"></span>
        </div>
      </div>
      <div class=\"episode-stack episode-side\">
        <div class=\"card compact-card\">
          <h2>关键概要</h2>
          <div class=\"overview-grid\">
            <p class=\"meta\">capture_mode=<code>{escape(episode.capture_mode or '-')}</code></p>
            <p class=\"meta\">source_type=<code>{escape(episode.source_type or '-')}</code></p>
            <p class=\"meta\">task_family=<code>{escape(episode.task_family or '-')}</code></p>
            <p class=\"meta\">target=<code>{escape(episode.target_label or episode.target_type or '-')}</code></p>
            <p class=\"meta\">scene=<code>{escape(episode.scene_id or '-')}</code></p>
            <p class=\"meta\">operator=<code>{escape(episode.operator_id or '-')}</code></p>
            <p class=\"meta\">d435i_capture=<code>{escape(episode.d435i_capture_path or '-')}</code></p>
            <p class=\"meta\">duration_s=<code>{episode.trajectory_metrics.get('duration_seconds', 0.0):.3f}</code></p>
            <p class=\"meta\">action_changes=<code>{episode.trajectory_metrics.get('action_change_count', 0)}</code></p>
            <p class=\"meta\">turn_ratio=<code>{episode.trajectory_metrics.get('turn_ratio', 0.0):.1%}</code></p>
            <p class=\"meta\">stop_ratio=<code>{episode.trajectory_metrics.get('stop_ratio', 0.0):.1%}</code></p>
            <p class=\"meta\">derived_side=<code>{escape(episode.derived_target_side or '-')}</code></p>
            <p class=\"meta\">derived_distance=<code>{escape(episode.derived_target_distance or '-')}</code></p>
            <p class=\"meta\">warnings=<code>{escape(warnings)}</code></p>
            <p class=\"meta\">info=<code>{escape(info)}</code></p>
            <p class=\"meta\">score=<code>{escape(current_quality_text)}</code></p>
            <p class=\"meta\">processing=<code>{escape(current_processing_text)}</code></p>
          </div>
        </div>
        <div class=\"card compact-card plot-card\">
          <h2>Action vs time</h2>
          {_plot_svg(episode)}
        </div>
        <div class=\"card compact-card frame-record-card\">
          <h2>Current frame record</h2>
          <div class=\"frame-record-grid\">
            <p class=\"meta\" id=\"frameMetaIndex\"></p>
            <p class=\"meta\" id=\"frameMetaTimestamp\"></p>
            <p class=\"meta\" id=\"frameMetaImage\"></p>
            <p class=\"meta\" id=\"frameMetaRetain\"></p>
            <p class=\"meta\" id=\"frameMetaAction\"></p>
            <p class=\"meta\" id=\"frameMetaState\"></p>
          </div>
          <pre id=\"frameJson\" class=\"json-box\"></pre>
        </div>
      </div>
    </div>
    <details class=\"{review_panel_class} compact-card details-card\" id=\"reviewPanel\">
      <summary>录制分数修改 / Frame 保留</summary>
      <div class=\"details-body\">
        <p class=\"meta\">当前页面会直接修改录制分数。<code>3</code> 会从后续处理里排除；如需恢复，只要改回 <code>2</code> 并保存即可。</p>
        <div class=\"review-toolbar\">
          <label>当前分数 {review_select_html}</label>
          <label>备注 <input id=\"qualityNotes\" class=\"quality-note\" type=\"text\" value=\"{escape(episode.quality_notes)}\" placeholder=\"可选：记录修改原因\" /></label>
          <button id=\"saveReview\" disabled>保存到源文件</button>
          <button id=\"bindReviewFile\">绑定记录文件</button>
          <button id=\"exportReview\">导出修改 JSON</button>
          <button id=\"resetReview\">撤销页面修改</button>
          <label>导入修改 JSON <input id=\"importReviewFile\" type=\"file\" accept=\".json,application/json\" /></label>
        </div>
        <div class=\"review-summary\">
          <span class=\"review-pill\">当前分数 <span id=\"currentQualityBadge\" class=\"quality-badge {escape(f'label-{episode.quality_label}')}">{escape(current_quality_text)}</span></span>
          <span class=\"review-pill\">后续处理 <span id=\"currentProcessingBadge\">{escape(current_processing_text)}</span></span>
          <span class=\"review-pill\">保留帧 <span id=\"selectedFrameCountBadge\">{episode.selected_frame_count}/{episode.total_frame_count}</span></span>
          <span class=\"review-pill review-file-status\" id=\"reviewFileStatus\"></span>
        </div>
        <div class=\"review-toolbar\">
          <label>默认模式
            <select id=\"frameSelectionMode\">
              <option value=\"all\"{" selected" if episode.frame_selection_mode == FRAME_SELECTION_MODE_ALL else ""}>默认保留，按删除排除</option>
              <option value=\"retain_only\"{" selected" if episode.frame_selection_mode == FRAME_SELECTION_MODE_RETAIN_ONLY else ""}>仅保留显式保留帧</option>
            </select>
          </label>
          <label>区间起点 <input id=\"frameRangeStart\" type=\"number\" min=\"1\" max=\"{max(1, len(frames_payload))}\" value=\"1\" /></label>
          <label>区间终点 <input id=\"frameRangeEnd\" type=\"number\" min=\"1\" max=\"{max(1, len(frames_payload))}\" value=\"{max(1, len(frames_payload))}\" /></label>
          <button id=\"keepCurrentFrame\">保留当前帧</button>
          <button id=\"dropCurrentFrame\">删除当前帧</button>
          <button id=\"keepFrameRange\">保留区间</button>
          <button id=\"dropFrameRange\">删除区间</button>
          <button id=\"resetFrameRange\">区间恢复默认</button>
        </div>
        <p id=\"frameSelectionStatus\" class=\"meta\">当前帧会按默认模式参与后续处理。</p>
        <p id=\"reviewStatus\" class=\"meta\">修改后先保存在当前页面；点击“保存到源文件”后才会真正写回。未绑定时也会保存在浏览器缓存中。</p>
      </div>
    </details>
    <details class=\"card compact-card details-card\">
      <summary>更多诊断图</summary>
      <div class=\"details-body secondary-grid\">
        <div class=\"card compact-card plot-card\">
          <h2>Action histograms</h2>
          <div class=\"grid\">{histogram_html}</div>
        </div>
        <div class=\"card compact-card plot-card\">
          <h2>State velocity vs time</h2>
          {state_velocity_svg}
        </div>
        <div class=\"card compact-card plot-card\">
          <h2>Orientation vs time</h2>
          {state_orientation_svg}
        </div>
        <div class=\"card compact-card plot-card\">
          <h2>Position / height vs time</h2>
          {state_position_svg}
        </div>
      </div>
    </details>
  </div>
  <script>
    {review_script}
    const frames = {json.dumps(frames_payload, ensure_ascii=True)};
    const currentEpisode = {json.dumps({"session_id": episode.session_id, "episode_id": episode.episode_id}, ensure_ascii=True)};
    const serverFileMode = String(window.location.protocol || '').startsWith('http');
    let index = 0;
    let timer = null;
    let playing = false;
    let playbackSpeed = 1.0;
    const image = document.getElementById('frameImage');
    const overlay = document.getElementById('overlay');
    const slider = document.getElementById('frameSlider');
    const playPause = document.getElementById('playPause');
    const speedSelect = document.getElementById('speedSelect');
    const playbackStats = document.getElementById('playbackStats');
    const qualityLabel = document.getElementById('qualityLabel');
    const qualityNotes = document.getElementById('qualityNotes');
    const saveReviewButton = document.getElementById('saveReview');
    const currentQualityBadge = document.getElementById('currentQualityBadge');
    const currentProcessingBadge = document.getElementById('currentProcessingBadge');
    const selectedFrameCountBadge = document.getElementById('selectedFrameCountBadge');
    const frameSelectionMode = document.getElementById('frameSelectionMode');
    const frameRangeStart = document.getElementById('frameRangeStart');
    const frameRangeEnd = document.getElementById('frameRangeEnd');
    const frameRangeStartSlider = document.getElementById('frameRangeStartSlider');
    const frameRangeEndSlider = document.getElementById('frameRangeEndSlider');
    const frameRangeHighlightBar = document.getElementById('frameRangeHighlightBar');
    const stackedRangeControls = [frameRangeStartSlider, frameRangeEndSlider, slider].filter(Boolean);
    const frameSelectionStatus = document.getElementById('frameSelectionStatus');
    const actionRangeHighlight = document.getElementById('actionRangeHighlight');
    const actionRangeStartLine = document.getElementById('actionRangeStartLine');
    const actionRangeEndLine = document.getElementById('actionRangeEndLine');
    const actionCurrentLine = document.getElementById('actionCurrentLine');
    const actionCurrentVx = document.getElementById('actionCurrentVx');
    const actionCurrentVy = document.getElementById('actionCurrentVy');
    const actionCurrentWz = document.getElementById('actionCurrentWz');
    const actionCurrentLabel = document.getElementById('actionCurrentLabel');
    const frameJson = document.getElementById('frameJson');
    const frameMetaIndex = document.getElementById('frameMetaIndex');
    const frameMetaTimestamp = document.getElementById('frameMetaTimestamp');
    const frameMetaImage = document.getElementById('frameMetaImage');
    const frameMetaRetain = document.getElementById('frameMetaRetain');
    const frameMetaAction = document.getElementById('frameMetaAction');
    const frameMetaState = document.getElementById('frameMetaState');
    const reviewPanel = document.getElementById('reviewPanel');
    const actionPlotOffset = 42;
    const actionPlotWidth = 662;
    const actionPlotTop = 18;
    const actionPlotBottom = 196;
    const actionPlotMin = {(-action_plot_limit):.6f};
    const actionPlotMax = {(action_plot_limit):.6f};

    image.addEventListener('load', () => {{
      if (image.naturalWidth > 0 && image.naturalHeight > 0) {{
        image.style.aspectRatio = `${{image.naturalWidth}} / ${{image.naturalHeight}}`;
        image.style.height = '100%';
      }}
    }});

    function renderCurrentReview() {{
      const review = getEpisodeReview(currentEpisode.session_id, currentEpisode.episode_id);
      qualityLabel.value = String(review.quality_label);
      qualityNotes.value = review.quality_notes;
      frameSelectionMode.value = review.frame_selection_mode || 'all';
      currentQualityBadge.textContent = labelText(review.quality_label);
      currentQualityBadge.className = `quality-badge ${{labelClass(review.quality_label)}}`;
      currentProcessingBadge.textContent = review.exclude_from_processing ? '已排除' : '保留';
      selectedFrameCountBadge.textContent = `${{review.selected_frame_count || 0}}/${{review.total_frame_count || frames.length}}`;
      reviewPanel.classList.toggle('quality-excluded', review.exclude_from_processing);
      renderFrameSelectionStatus();
    }}

    function explicitRetainedRangeFromReview(review) {{
      const keepMap = (review && review.frame_keep_map) || {{}};
      const retained = Object.entries(keepMap)
        .filter(([_, keep]) => keep === true)
        .map(([frameIndex]) => Number(frameIndex))
        .filter((frameIndex) => Number.isInteger(frameIndex) && frameIndex >= 0)
        .sort((left, right) => left - right);
      if (!retained.length) {{
        return null;
      }}
      return {{
        start: retained[0] + 1,
        end: retained[retained.length - 1] + 1,
      }};
    }}

    function syncRangeControlsFromReview() {{
      if (!frames.length) {{
        return;
      }}
      const review = getEpisodeReview(currentEpisode.session_id, currentEpisode.episode_id);
      let resolvedRange = explicitRetainedRangeFromReview(review);
      if (!resolvedRange) {{
        resolvedRange = {{
          start: 1,
          end: frames.length,
        }};
      }}
      if (!resolvedRange) {{
        return;
      }}
      const startValue = String(clamp(resolvedRange.start, 1, Math.max(1, frames.length)));
      const endValue = String(clamp(resolvedRange.end, 1, Math.max(1, frames.length)));
      frameRangeStart.value = startValue;
      frameRangeEnd.value = endValue;
      frameRangeStartSlider.value = startValue;
      frameRangeEndSlider.value = endValue;
      index = clampIndexToSelectedRange(index);
      syncRangeControls();
    }}

    function applyCurrentReviewDraft() {{
      upsertEpisodeReview(
        currentEpisode.session_id,
        currentEpisode.episode_id,
        Number(qualityLabel.value) || 0,
        qualityNotes.value,
      );
      renderCurrentReview();
      announceStatus(`已暂存 ${{currentEpisode.session_id}}/${{currentEpisode.episode_id}} 的修改，点击“保存到源文件”完成写回。`);
    }}

    function saveCurrentReview() {{
      const review = getEpisodeReview(currentEpisode.session_id, currentEpisode.episode_id);
      persistReviewState(`已保存 ${{currentEpisode.session_id}}/${{currentEpisode.episode_id}} -> ${{labelText(review.quality_label)}}`)
        .then(() => renderCurrentReview())
        .catch((error) => announceStatus(`写回失败：${{error}}`, true));
    }}

    function clamp(value, minValue, maxValue) {{
      return Math.min(maxValue, Math.max(minValue, value));
    }}

    function parseRangeValue(node, fallback) {{
      const maxFrames = Math.max(1, frames.length);
      return clamp(Math.floor(Number(node && node.value) || fallback), 1, maxFrames);
    }}

    function currentRangeBounds() {{
      const start = parseRangeValue(frameRangeStart, index + 1);
      const end = parseRangeValue(frameRangeEnd, index + 1);
      return {{
        start: Math.min(start, end),
        end: Math.max(start, end),
      }};
    }}

    function syncRangeControls(source) {{
      const bounds = currentRangeBounds();
      const startValue = String(bounds.start);
      const endValue = String(bounds.end);
      if (source !== 'start-input') {{
        frameRangeStart.value = startValue;
      }}
      if (source !== 'end-input') {{
        frameRangeEnd.value = endValue;
      }}
      if (source !== 'start-slider') {{
        frameRangeStartSlider.value = startValue;
      }}
      if (source !== 'end-slider') {{
        frameRangeEndSlider.value = endValue;
      }}
      updateActionRangeHighlight(bounds.start, bounds.end);
      updateFrameRangeHighlight(bounds.start, bounds.end);
      renderFrameSelectionStatus();
    }}

    function setActiveRangeControl(activeControl) {{
      stackedRangeControls.forEach((control) => {{
        control.classList.toggle('is-active', control === activeControl);
      }});
    }}

    function updateFrameRangeHighlight(startFrame, endFrame) {{
      if (!frameRangeHighlightBar || frames.length <= 0) {{
        return;
      }}
      const total = Math.max(1, frames.length);
      const leftRatio = (clamp(startFrame, 1, total) - 1) / total;
      const rightRatio = clamp(endFrame, 1, total) / total;
      frameRangeHighlightBar.style.left = `${{leftRatio * 100}}%`;
      frameRangeHighlightBar.style.width = `${{Math.max((rightRatio - leftRatio) * 100, 100 / total)}}%`;
    }}

    function updateActionRangeHighlight(startFrame, endFrame) {{
      if (!actionRangeHighlight || !actionRangeStartLine || !actionRangeEndLine || frames.length <= 1) {{
        return;
      }}
      const leftIndex = clamp(startFrame - 1, 0, frames.length - 1);
      const rightIndex = clamp(endFrame - 1, 0, frames.length - 1);
      const xFromIndex = (frameIndex) => frames.length <= 1 ? actionPlotOffset : actionPlotOffset + (frameIndex / (frames.length - 1)) * actionPlotWidth;
      const leftX = xFromIndex(leftIndex);
      const rightX = xFromIndex(rightIndex);
      const highlightWidth = Math.max(2, rightX - leftX);
      actionRangeHighlight.setAttribute('x', `${{leftX}}`);
      actionRangeHighlight.setAttribute('width', `${{highlightWidth}}`);
      actionRangeHighlight.setAttribute('visibility', 'visible');
      actionRangeStartLine.setAttribute('x1', `${{leftX}}`);
      actionRangeStartLine.setAttribute('x2', `${{leftX}}`);
      actionRangeStartLine.setAttribute('visibility', 'visible');
      actionRangeEndLine.setAttribute('x1', `${{rightX}}`);
      actionRangeEndLine.setAttribute('x2', `${{rightX}}`);
      actionRangeEndLine.setAttribute('visibility', 'visible');
    }}

    function updateActionCurrentLine(frameIndex) {{
      if (!actionCurrentLine || !frames.length) {{
        return;
      }}
      const safeIndex = clamp(frameIndex, 0, Math.max(0, frames.length - 1));
      const cursorX = frames.length <= 1 ? actionPlotOffset : actionPlotOffset + (safeIndex / (frames.length - 1)) * actionPlotWidth;
      const frame = frames[safeIndex] || {{}};
      const values = [Number(frame.vx) || 0, Number(frame.vy) || 0, Number(frame.wz) || 0];
      const yFromValue = (value) => {{
        const normalized = (value - actionPlotMin) / (actionPlotMax - actionPlotMin);
        return actionPlotBottom - normalized * (actionPlotBottom - actionPlotTop);
      }};
      actionCurrentLine.setAttribute('x1', `${{cursorX}}`);
      actionCurrentLine.setAttribute('x2', `${{cursorX}}`);
      actionCurrentLine.setAttribute('visibility', 'visible');
      [[actionCurrentVx, values[0]], [actionCurrentVy, values[1]], [actionCurrentWz, values[2]]].forEach(([node, value]) => {{
        if (!node) return;
        node.setAttribute('cx', `${{cursorX}}`);
        node.setAttribute('cy', `${{yFromValue(value)}}`);
        node.setAttribute('visibility', 'visible');
      }});
      if (actionCurrentLabel) {{
        actionCurrentLabel.textContent = `current image frame ${{safeIndex + 1}} | vx=${{values[0].toFixed(3)}} vy=${{values[1].toFixed(3)}} wz=${{values[2].toFixed(3)}}`;
      }}
    }}

    function clampIndexToSelectedRange(rawIndex) {{
      const bounds = currentRangeBounds();
      return clamp(rawIndex, bounds.start - 1, bounds.end - 1);
    }}

    function currentFrameProcessingStatus() {{
      const review = getEpisodeReview(currentEpisode.session_id, currentEpisode.episode_id);
      const frame = frames[index] || null;
      if (!frame) {{
        return '当前没有可用帧。';
      }}
      const explicit = (review.frame_keep_map || {{}})[String(frame.frame_index)];
      const selected = review.frame_selection_mode === 'retain_only' ? explicit === true : explicit !== false;
      const explicitText = explicit === true ? '显式保留' : (explicit === false ? '显式删除' : '未单独标记');
      const modeText = review.frame_selection_mode === 'retain_only' ? '仅保留显式保留帧' : '默认保留，按删除排除';
      const bounds = currentRangeBounds();
      return `当前帧 #${{frame.frame_index + 1}}：${{selected ? '会参与后续处理' : '不会参与后续处理'}}；${{explicitText}}；模式=${{modeText}}；区间=${{bounds.start}}-${{bounds.end}}`;
    }}

    function renderFrameSelectionStatus() {{
      if (!frameSelectionStatus) {{
        return;
      }}
      frameSelectionStatus.textContent = currentFrameProcessingStatus();
    }}

    function applyFrameSelectionUpdates(startIndex, endIndex, keepValue) {{
      const review = getEpisodeReview(currentEpisode.session_id, currentEpisode.episode_id);
      const lower = clamp(Math.min(startIndex, endIndex), 1, Math.max(1, frames.length));
      const upper = clamp(Math.max(startIndex, endIndex), 1, Math.max(1, frames.length));
      const updates = {{}};
      for (let position = lower; position <= upper; position += 1) {{
        const frame = frames[position - 1];
        if (!frame) {{
          continue;
        }}
        updates[String(frame.frame_index)] = keepValue;
      }}
      updateEpisodeFrameSelection(
        currentEpisode.session_id,
        currentEpisode.episode_id,
        frameSelectionMode.value || review.frame_selection_mode || 'all',
        updates,
      );
      renderCurrentReview();
      announceStatus(`已暂存帧区间 ${{lower}}-${{upper}} 的处理标记，点击“保存到源文件”完成写回。`);
    }}

    function frameImageUrl(frame) {{
      if (serverFileMode && frame.disk_path) {{
        return `/api/file?path=${{encodeURIComponent(frame.disk_path)}}`;
      }}
      return frame.image;
    }}

    function computeNominalFrameGapMs() {{
      const gaps = [];
      for (let i = 1; i < frames.length; i += 1) {{
        const gapMs = (frames[i].timestamp - frames[i - 1].timestamp) * 1000;
        if (Number.isFinite(gapMs) && gapMs > 1) {{
          gaps.push(gapMs);
        }}
      }}
      if (!gaps.length) {{
        return 100;
      }}
      gaps.sort((left, right) => left - right);
      return clamp(gaps[Math.floor(gaps.length / 2)], 33, 250);
    }}

    const nominalFrameGapMs = computeNominalFrameGapMs();

    function preloadAround(currentIndex) {{
      for (let offset = 0; offset <= 2; offset += 1) {{
        const nextIndex = currentIndex + offset;
        if (nextIndex >= frames.length) {{
          break;
        }}
        const preload = new Image();
        preload.src = frameImageUrl(frames[nextIndex]);
      }}
    }}

    function nextFrameDelayMs(currentIndex) {{
      const bounds = currentRangeBounds();
      if (bounds.end - bounds.start <= 0) {{
        return nominalFrameGapMs / playbackSpeed;
      }}
      const safeIndex = clamp(currentIndex, bounds.start - 1, bounds.end - 1);
      const nextIndex = safeIndex >= bounds.end - 1 ? bounds.start - 1 : safeIndex + 1;
      const nextFrame = frames[nextIndex];
      if (!nextFrame) {{
        return nominalFrameGapMs / playbackSpeed;
      }}
      const gapMs = (nextFrame.timestamp - frames[safeIndex].timestamp) * 1000;
      const safeGapMs = Number.isFinite(gapMs) && gapMs > 1 ? gapMs : nominalFrameGapMs;
      return clamp(safeGapMs / playbackSpeed, 16, 250);
    }}

    function updatePlaybackStats() {{
      if (!frames.length) {{
        playbackStats.textContent = '0 frames';
        return;
      }}
      const nominalFps = 1000 / nominalFrameGapMs;
      playbackStats.textContent = `frame ${{index + 1}}/${{frames.length}} | nominal ${{nominalFps.toFixed(1)}} FPS | speed ${{playbackSpeed.toFixed(2)}}x`;
    }}

    function stopPlayback() {{
      playing = false;
      if (timer !== null) {{
        clearTimeout(timer);
        timer = null;
      }}
      playPause.textContent = 'Play';
    }}

    function scheduleNextFrame() {{
      if (!playing || !frames.length) {{
        return;
      }}
      if (timer !== null) {{
        clearTimeout(timer);
      }}
      timer = window.setTimeout(() => {{
        timer = null;
        if (!playing) {{
          return;
        }}
        const bounds = currentRangeBounds();
        if (index >= bounds.end - 1) {{
          index = bounds.start - 1;
        }} else {{
          index = clamp(index + 1, bounds.start - 1, bounds.end - 1);
        }}
        render();
        scheduleNextFrame();
      }}, nextFrameDelayMs(index));
    }}

    function render() {{
      if (!frames.length) {{
        overlay.textContent = 'No frames available';
        updatePlaybackStats();
        return;
      }}
      index = clampIndexToSelectedRange(index);
      const frame = frames[index];
      const state = frame.state || {{}};
      const formatValue = (value) => Number.isFinite(Number(value)) ? Number(value).toFixed(3) : '-';
      const formatInt = (value) => Number.isFinite(Number(value)) ? String(Math.trunc(Number(value))) : '-';
      image.src = frameImageUrl(frame);
      overlay.innerHTML =
        `instruction: ${{frame.instruction}}<br />` +
        `frame: ${{frame.frame_index + 1}} / ${{frames.length}} | t: ${{frame.timestamp.toFixed(3)}}<br />` +
        `action: vx ${{frame.vx.toFixed(3)}} vy ${{frame.vy.toFixed(3)}} wz ${{frame.wz.toFixed(3)}}<br />` +
        `state: vx ${{formatValue(state.vx)}} vz ${{formatValue(state.vz)}} wz ${{formatValue(state.wz)}}<br />` +
        `pose/body: r ${{formatValue(state.roll)}} p ${{formatValue(state.pitch)}} y ${{formatValue(state.yaw)}} | h ${{formatValue(state.body_height)}}<br />` +
        `mode ${{formatInt(state.mode)}} gait ${{formatInt(state.gait_type)}} err ${{formatInt(state.error_code)}}`;
      if (frameMetaIndex) frameMetaIndex.innerHTML = `frame_index=<code>${{frame.frame_index + 1}}</code> / ${{frames.length}}`;
      if (frameMetaTimestamp) frameMetaTimestamp.innerHTML = `timestamp=<code>${{frame.timestamp.toFixed(6)}}</code>`;
      if (frameMetaImage) frameMetaImage.innerHTML = `image=<code>${{frame.image_rel || '-'}}</code>`;
      if (frameMetaRetain) frameMetaRetain.innerHTML = `retain_for_processing=<code>${{String(frame.retain_for_processing)}}</code>`;
      if (frameMetaAction) frameMetaAction.innerHTML = `action=<code>vx=${{frame.vx.toFixed(3)}}, vy=${{frame.vy.toFixed(3)}}, wz=${{frame.wz.toFixed(3)}}</code>`;
      if (frameMetaState) frameMetaState.innerHTML = `state=<code>vx=${{formatValue(state.vx)}}, vy=${{formatValue(state.vy)}}, vz=${{formatValue(state.vz)}}, wz=${{formatValue(state.wz)}}</code>`;
      if (frameJson) frameJson.textContent = JSON.stringify(frame.raw_frame || {{}}, null, 2);
      slider.value = String(index);
      preloadAround(index);
      updateActionCurrentLine(index);
      updatePlaybackStats();
      syncRangeControls();
      renderFrameSelectionStatus();
    }}

    function step(delta) {{
      if (!frames.length) {{
        return;
      }}
      const bounds = currentRangeBounds();
      if (delta < 0) {{
        index = index <= bounds.start - 1 ? bounds.end - 1 : index - 1;
      }} else {{
        index = index >= bounds.end - 1 ? bounds.start - 1 : index + 1;
      }}
      index = clamp(index, bounds.start - 1, bounds.end - 1);
      render();
      if (playing) {{
        scheduleNextFrame();
      }}
    }}

    document.getElementById('prevFrame').addEventListener('click', () => step(-1));
    document.getElementById('nextFrame').addEventListener('click', () => step(1));
    qualityLabel.addEventListener('change', applyCurrentReviewDraft);
    qualityNotes.addEventListener('change', applyCurrentReviewDraft);
    saveReviewButton.addEventListener('click', saveCurrentReview);
    frameSelectionMode.addEventListener('change', () => {{
      updateEpisodeFrameSelection(
        currentEpisode.session_id,
        currentEpisode.episode_id,
        frameSelectionMode.value || 'all',
        {{}},
      );
      renderCurrentReview();
      announceStatus('已暂存帧保留模式修改，点击“保存到源文件”完成写回。');
    }});
    document.getElementById('keepCurrentFrame').addEventListener('click', () => applyFrameSelectionUpdates(index + 1, index + 1, true));
    document.getElementById('dropCurrentFrame').addEventListener('click', () => applyFrameSelectionUpdates(index + 1, index + 1, false));
    document.getElementById('keepFrameRange').addEventListener('click', () => applyFrameSelectionUpdates(parseRangeValue(frameRangeStart, index + 1), parseRangeValue(frameRangeEnd, index + 1), true));
    document.getElementById('dropFrameRange').addEventListener('click', () => applyFrameSelectionUpdates(parseRangeValue(frameRangeStart, index + 1), parseRangeValue(frameRangeEnd, index + 1), false));
    document.getElementById('resetFrameRange').addEventListener('click', () => applyFrameSelectionUpdates(parseRangeValue(frameRangeStart, index + 1), parseRangeValue(frameRangeEnd, index + 1), null));
    frameRangeStart.addEventListener('change', () => {{
      syncRangeControls('start-input');
      index = clampIndexToSelectedRange(index);
      render();
    }});
    frameRangeEnd.addEventListener('change', () => {{
      syncRangeControls('end-input');
      index = clampIndexToSelectedRange(index);
      render();
    }});
    frameRangeStartSlider.addEventListener('input', () => {{
      frameRangeStart.value = frameRangeStartSlider.value;
      syncRangeControls('start-slider');
      index = clampIndexToSelectedRange(index);
      render();
    }});
    frameRangeEndSlider.addEventListener('input', () => {{
      frameRangeEnd.value = frameRangeEndSlider.value;
      syncRangeControls('end-slider');
      index = clampIndexToSelectedRange(index);
      render();
    }});
    document.getElementById('bindReviewFile').addEventListener('click', () => {{
      bindReviewFile().catch((error) => announceStatus(`绑定文件失败：${{error}}`, true));
    }});
    document.getElementById('exportReview').addEventListener('click', () => {{
      exportReviewFile().catch((error) => announceStatus(`导出失败：${{error}}`, true));
    }});
    document.getElementById('resetReview').addEventListener('click', () => {{
      clearEpisodeOverride(currentEpisode.session_id, currentEpisode.episode_id);
      syncRangeControlsFromReview();
      renderCurrentReview();
      announceStatus(`已撤销 ${{currentEpisode.session_id}}/${{currentEpisode.episode_id}} 的页面修改。`);
    }});
    document.getElementById('importReviewFile').addEventListener('change', async (event) => {{
      const [file] = Array.from(event.target.files || []);
      if (!file) {{
        return;
      }}
      try {{
        await importReviewFile(file);
        syncRangeControlsFromReview();
        renderCurrentReview();
        announceStatus('已导入修改记录，点击“保存到源文件”完成写回。');
      }} catch (error) {{
        announceStatus(`导入失败：${{error}}`, true);
      }} finally {{
        event.target.value = '';
      }}
    }});
    window.addEventListener('go2-review-dirty-changed', (event) => {{
      const detail = event.detail || {{}};
      const dirty = Boolean(detail.dirty);
      const saving = Boolean(detail.saving);
      saveReviewButton.disabled = saving || !dirty;
      saveReviewButton.textContent = saving ? '正在保存...' : '保存到源文件';
    }});
    setReviewDirty(reviewDirty);
    window.addEventListener('go2-review-state-updated', () => {{
      syncRangeControlsFromReview();
      renderCurrentReview();
    }});
    slider.addEventListener('input', (event) => {{
      index = clampIndexToSelectedRange(Number(event.target.value));
      render();
      if (playing) {{
        scheduleNextFrame();
      }}
    }});
    stackedRangeControls.forEach((control) => {{
      ['pointerdown', 'mousedown', 'touchstart', 'focus'].forEach((eventName) => {{
        control.addEventListener(eventName, () => setActiveRangeControl(control));
      }});
    }});
    speedSelect.addEventListener('change', (event) => {{
      playbackSpeed = Math.max(0.25, Number(event.target.value) || 1.0);
      updatePlaybackStats();
      if (playing) {{
        scheduleNextFrame();
      }}
    }});
    playPause.addEventListener('click', () => {{
      if (playing) {{
        stopPlayback();
        return;
      }}
      if (!frames.length) {{
        return;
      }}
      playing = true;
      playPause.textContent = 'Pause';
      scheduleNextFrame();
    }});
    window.addEventListener('keydown', (event) => {{
      if (event.key === ' ') {{
        event.preventDefault();
        playPause.click();
      }} else if (event.key === 'ArrowLeft') {{
        event.preventDefault();
        step(-1);
      }} else if (event.key === 'ArrowRight') {{
        event.preventDefault();
        step(1);
      }}
    }});
    syncRangeControlsFromReview();
    renderCurrentReview();
    setActiveRangeControl(slider);
    render();
  </script>
</body>
</html>
"""


def _processed_plot_svg(sample: ProcessedWindowRecord) -> str:
    width = 720
    height = 150
    color_palette = ["#cc5500", "#2b59c3", "#007f5f", "#8a3ffc", "#c2410c"]
    values_by_axis: Dict[str, List[float]] = {}
    for axis_index, axis in enumerate(sample.action_fields):
        values_by_axis[axis] = [
            float(step[axis_index]) if axis_index < len(step) else 0.0
            for step in sample.actions_continuous
        ]
    lines = []
    legend_parts = []
    for axis_index, axis in enumerate(sample.action_fields):
        color = color_palette[axis_index % len(color_palette)]
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" '
            f'points="{_polyline(values_by_axis[axis], width, height, -1.0, 1.0)}" />'
        )
        legend_parts.append(
            f'<text x="{20 + axis_index * 140}" y="20" fill="{color}" font-size="14">{escape(axis)}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height + 30}" role="img" aria-label="window action plot">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff" stroke="#d7ccb8" />'
        f'<rect id="actionRangeHighlight" x="0" y="0" width="0" height="{height}" fill="#f3d58d" fill-opacity="0.35" visibility="hidden" />'
        f'<line id="actionRangeStartLine" x1="0" y1="0" x2="0" y2="{height}" stroke="#9a6b00" stroke-width="2" visibility="hidden" />'
        f'<line id="actionRangeEndLine" x1="0" y1="0" x2="0" y2="{height}" stroke="#9a6b00" stroke-width="2" visibility="hidden" />'
        f'<line id="actionCurrentLine" x1="0" y1="0" x2="0" y2="{height}" stroke="#005b7f" stroke-width="2.5" visibility="hidden" />'
        f'<line x1="0" y1="{height / 2:.1f}" x2="{width}" y2="{height / 2:.1f}" stroke="#9a3412" stroke-dasharray="6 4" />'
        f'<text x="{width - 8}" y="{height / 2 - 6:.1f}" fill="#9a3412" font-size="11" text-anchor="end">0.00</text>'
        f'{"".join(lines)}{"".join(legend_parts)}</svg>'
    )


def _processed_sample_frames_payload(sample: ProcessedWindowRecord, page_path: Path) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for frame in sample.playback_frames:
        image_rel = ""
        if frame.image_disk_path is not None:
            image_rel = os.path.relpath(frame.image_disk_path, page_path.parent)
        payload.append(
            {
                "step_index": frame.step_index,
                "frame_id": frame.frame_id,
                "timestamp": round(frame.timestamp, 6),
                "image": image_rel,
                "disk_path": str(frame.image_disk_path) if frame.image_disk_path is not None else "",
                "action": sample.actions_continuous[frame.step_index] if frame.step_index < len(sample.actions_continuous) else [],
                "action_mask": sample.action_mask[frame.step_index] if frame.step_index < len(sample.action_mask) else 0,
                "phase": sample.phase_seq[frame.step_index] if frame.step_index < len(sample.phase_seq) else "",
            }
        )
    return payload


def _processed_sample_page(
    sample: ProcessedWindowRecord,
    page_path: Path,
    index_path: Path,
    prev_page_name: Optional[str],
    next_page_name: Optional[str],
) -> str:
    frames_payload = _processed_sample_frames_payload(sample, page_path)
    index_rel = os.path.relpath(index_path, page_path.parent)
    prev_link_html = f'<a href="{escape(prev_page_name)}">Prev Sample</a>' if prev_page_name else '<span class="meta">Prev Sample</span>'
    next_link_html = f'<a href="{escape(next_page_name)}">Next Sample</a>' if next_page_name else '<span class="meta">Next Sample</span>'
    action_rows = "".join(
        f"<tr><td>t{step_index}</td><td>{escape(', '.join(f'{value:.3f}' for value in step))}</td><td>{sample.action_mask[step_index] if step_index < len(sample.action_mask) else 0}</td><td>{escape(sample.phase_seq[step_index] if step_index < len(sample.phase_seq) else '-')}</td></tr>"
        for step_index, step in enumerate(sample.actions_continuous)
    )
    state_text = ", ".join(f"{value:.4f}" for value in sample.state) if sample.state else "-"
    image_missing = all(not frame.get("disk_path") for frame in frames_payload)
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>{escape(sample.sample_id)}</title>
  <style>{CSS}</style>
</head>
<body class=\"detail-page\">
  <div class=\"page-shell\">
    <div class=\"page-nav\">
      <a href=\"{escape(index_rel)}\">Back to summary</a>
      {prev_link_html}
      {next_link_html}
    </div>
    <div class=\"episode-layout\">
      <div class=\"episode-stack viewer-panel\">
        <div class=\"card compact-card viewer-header\">
          <h1>{escape(sample.sample_id)}</h1>
          <p class=\"meta\"><strong>instruction</strong> <code>{escape(sample.instruction or '-')}</code></p>
          <p class=\"meta viewer-help\">四步窗口优先检查图像源是否正常、phase 是否合理、动作 chunk 与 action mask 是否对齐。</p>
        </div>
        <div class=\"viewer\">
          <img id=\"frameImage\" src=\"\" alt=\"processed sample replay frame\" />
          <div class=\"overlay\" id=\"overlay\"></div>
        </div>
        <div class=\"controls compact-controls\">
          <button id=\"playPause\">Play</button>
          <button id=\"prevFrame\">Prev Step</button>
          <button id=\"nextFrame\">Next Step</button>
          <div class=\"frame-range-wrap\">
            <label>Step</label>
            <div class=\"frame-range-stack\">
              <div id=\"frameRangeHighlightBar\" class=\"frame-range-highlight\"></div>
              <input id=\"frameRangeStartSlider\" class=\"frame-range-marker\" type=\"range\" min=\"1\" max=\"{max(1, len(frames_payload))}\" value=\"1\" />
              <input id=\"frameRangeEndSlider\" class=\"frame-range-marker\" type=\"range\" min=\"1\" max=\"{max(1, len(frames_payload))}\" value=\"{max(1, len(frames_payload))}\" />
              <input id=\"frameSlider\" class=\"frame-range-current\" type=\"range\" min=\"0\" max=\"{max(0, len(frames_payload) - 1)}\" value=\"0\" />
            </div>
          </div>
          <label>Speed
            <select id=\"speedSelect\">
              <option value=\"0.5\">0.5x</option>
              <option value=\"0.75\">0.75x</option>
              <option value=\"1\" selected>1.0x</option>
              <option value=\"1.5\">1.5x</option>
              <option value=\"2\">2.0x</option>
            </select>
          </label>
          <span class=\"playback-stats\" id=\"playbackStats\"></span>
        </div>
        <div class=\"card compact-card\">
          <p class=\"meta\">{"当前窗口没有可回放的源图像，页面会重复显示首帧或空白占位。" if image_missing else "当前窗口优先使用 canonical/source image 回放四步样本。"}</p>
        </div>
      </div>
      <div class=\"episode-stack episode-side\">
        <div class=\"card compact-card\">
          <h2>窗口概要</h2>
          <div class=\"overview-grid\">
            <p class=\"meta\">session=<code>{escape(sample.session_id or '-')}</code></p>
            <p class=\"meta\">episode=<code>{escape(sample.episode_id or '-')}</code></p>
            <p class=\"meta\">start_frame=<code>{sample.start_frame}</code></p>
            <p class=\"meta\">phase=<code>{escape(sample.phase or '-')}</code></p>
            <p class=\"meta\">phase_seq=<code>{escape(', '.join(item or '-' for item in sample.phase_seq) or '-')}</code></p>
            <p class=\"meta\">task_family=<code>{escape(sample.task_family or '-')}</code></p>
            <p class=\"meta\">scene=<code>{escape(sample.scene_id or '-')}</code></p>
            <p class=\"meta\">operator=<code>{escape(sample.operator_id or '-')}</code></p>
            <p class=\"meta\">target=<code>{escape(sample.target_label or sample.target_type or '-')}</code></p>
            <p class=\"meta\">padding=<code>{'yes' if sample.has_padding else 'no'}</code></p>
            <p class=\"meta\">state=<code>{escape(state_text)}</code></p>
            <p class=\"meta\">phase_instruction=<code>{escape(sample.phase_derived_instruction or '-')}</code></p>
          </div>
        </div>
        <div class=\"card compact-card plot-card\">
          <h2>Action vs time</h2>
          {_processed_plot_svg(sample)}
        </div>
        <div class=\"card compact-card\">
          <h2>4-step actions</h2>
          <table class=\"table-compact\">
            <thead><tr><th>Step</th><th>{escape('/'.join(sample.action_fields) or 'action')}</th><th>Mask</th><th>Phase</th></tr></thead>
            <tbody>{action_rows or '<tr><td colspan="4">No actions</td></tr>'}</tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
  <script>
    const frames = {json.dumps(frames_payload, ensure_ascii=True)};
    const sampleMeta = {json.dumps({
        "instruction": sample.instruction,
        "raw_instruction": sample.raw_instruction,
        "normalized_instruction": sample.normalized_instruction,
        "phase_derived_instruction": sample.phase_derived_instruction,
        "action_fields": sample.action_fields,
    }, ensure_ascii=True)};
    const serverFileMode = String(window.location.protocol || '').startsWith('http');
    let index = 0;
    let timer = null;
    let playing = false;
    let playbackSpeed = 1.0;
    const actionPlotWidth = 720;
    const image = document.getElementById('frameImage');
    const overlay = document.getElementById('overlay');
    const slider = document.getElementById('frameSlider');
    const playPause = document.getElementById('playPause');
    const speedSelect = document.getElementById('speedSelect');
    const playbackStats = document.getElementById('playbackStats');
    const frameRangeStartSlider = document.getElementById('frameRangeStartSlider');
    const frameRangeEndSlider = document.getElementById('frameRangeEndSlider');
    const frameRangeHighlightBar = document.getElementById('frameRangeHighlightBar');
    const actionRangeHighlight = document.getElementById('actionRangeHighlight');
    const actionRangeStartLine = document.getElementById('actionRangeStartLine');
    const actionRangeEndLine = document.getElementById('actionRangeEndLine');
    const actionCurrentLine = document.getElementById('actionCurrentLine');
    const stackedRangeControls = [frameRangeStartSlider, frameRangeEndSlider, slider].filter(Boolean);

    function clamp(value, minValue, maxValue) {{
      return Math.min(maxValue, Math.max(minValue, value));
    }}

    function frameImageUrl(frame) {{
      if (!frame) {{
        return '';
      }}
      if (serverFileMode && frame.disk_path) {{
        return `/api/file?path=${{encodeURIComponent(frame.disk_path)}}`;
      }}
      return frame.image || '';
    }}

    function currentRangeBounds() {{
      const total = Math.max(1, frames.length);
      const start = clamp(Number(frameRangeStartSlider.value) || 1, 1, total);
      const end = clamp(Number(frameRangeEndSlider.value) || total, 1, total);
      return {{
        start: Math.min(start, end),
        end: Math.max(start, end),
      }};
    }}

    function updateFrameRangeHighlight(startStep, endStep) {{
      if (!frameRangeHighlightBar || frames.length <= 0) {{
        return;
      }}
      const total = Math.max(1, frames.length);
      const leftRatio = (clamp(startStep, 1, total) - 1) / total;
      const rightRatio = clamp(endStep, 1, total) / total;
      frameRangeHighlightBar.style.left = `${{leftRatio * 100}}%`;
      frameRangeHighlightBar.style.width = `${{Math.max((rightRatio - leftRatio) * 100, 100 / total)}}%`;
    }}

    function updateActionRangeHighlight(startStep, endStep) {{
      if (!actionRangeHighlight || !actionRangeStartLine || !actionRangeEndLine || frames.length <= 1) {{
        return;
      }}
      const leftIndex = clamp(startStep - 1, 0, frames.length - 1);
      const rightIndex = clamp(endStep - 1, 0, frames.length - 1);
      const xFromIndex = (stepIndex) => frames.length <= 1 ? 0 : (stepIndex / (frames.length - 1)) * actionPlotWidth;
      const leftX = xFromIndex(leftIndex);
      const rightX = xFromIndex(rightIndex);
      actionRangeHighlight.setAttribute('x', `${{leftX}}`);
      actionRangeHighlight.setAttribute('width', `${{Math.max(2, rightX - leftX)}}`);
      actionRangeHighlight.setAttribute('visibility', 'visible');
      actionRangeStartLine.setAttribute('x1', `${{leftX}}`);
      actionRangeStartLine.setAttribute('x2', `${{leftX}}`);
      actionRangeStartLine.setAttribute('visibility', 'visible');
      actionRangeEndLine.setAttribute('x1', `${{rightX}}`);
      actionRangeEndLine.setAttribute('x2', `${{rightX}}`);
      actionRangeEndLine.setAttribute('visibility', 'visible');
    }}

    function updateActionCurrentLine(stepIndex) {{
      if (!actionCurrentLine || !frames.length) {{
        return;
      }}
      const safeIndex = clamp(stepIndex, 0, Math.max(0, frames.length - 1));
      const cursorX = frames.length <= 1 ? 0 : (safeIndex / (frames.length - 1)) * actionPlotWidth;
      actionCurrentLine.setAttribute('x1', `${{cursorX}}`);
      actionCurrentLine.setAttribute('x2', `${{cursorX}}`);
      actionCurrentLine.setAttribute('visibility', 'visible');
    }}

    function syncRangeControls(source) {{
      const bounds = currentRangeBounds();
      const startValue = String(bounds.start);
      const endValue = String(bounds.end);
      if (source !== 'start-slider') {{
        frameRangeStartSlider.value = startValue;
      }}
      if (source !== 'end-slider') {{
        frameRangeEndSlider.value = endValue;
      }}
      updateFrameRangeHighlight(bounds.start, bounds.end);
      updateActionRangeHighlight(bounds.start, bounds.end);
    }}

    function setActiveRangeControl(activeControl) {{
      stackedRangeControls.forEach((control) => {{
        control.classList.toggle('is-active', control === activeControl);
      }});
    }}

    function clampIndexToSelectedRange(rawIndex) {{
      const bounds = currentRangeBounds();
      return clamp(rawIndex, bounds.start - 1, bounds.end - 1);
    }}

    function updatePlaybackStats() {{
      if (!frames.length) {{
        playbackStats.textContent = '0 steps';
        return;
      }}
      playbackStats.textContent = `step ${{index + 1}}/${{frames.length}} | speed ${{playbackSpeed.toFixed(2)}}x`;
    }}

    function preloadAround(currentIndex) {{
      for (let offset = 0; offset <= 2; offset += 1) {{
        const nextIndex = currentIndex + offset;
        if (nextIndex >= frames.length) {{
          break;
        }}
        const url = frameImageUrl(frames[nextIndex]);
        if (!url) {{
          continue;
        }}
        const preload = new Image();
        preload.src = url;
      }}
    }}

    function nextFrameDelayMs() {{
      return clamp(300 / playbackSpeed, 60, 800);
    }}

    function stopPlayback() {{
      playing = false;
      if (timer !== null) {{
        clearTimeout(timer);
        timer = null;
      }}
      playPause.textContent = 'Play';
    }}

    function scheduleNextFrame() {{
      if (!playing || !frames.length) {{
        return;
      }}
      if (timer !== null) {{
        clearTimeout(timer);
      }}
      timer = window.setTimeout(() => {{
        timer = null;
        if (!playing) {{
          return;
        }}
        const bounds = currentRangeBounds();
        index = index >= bounds.end - 1 ? bounds.start - 1 : clamp(index + 1, bounds.start - 1, bounds.end - 1);
        render();
        scheduleNextFrame();
      }}, nextFrameDelayMs());
    }}

    function render() {{
      if (!frames.length) {{
        overlay.textContent = 'No steps available';
        updatePlaybackStats();
        return;
      }}
      index = clampIndexToSelectedRange(index);
      const frame = frames[index];
      const url = frameImageUrl(frame);
      if (url) {{
        image.hidden = false;
        image.src = url;
      }} else {{
        image.hidden = true;
        image.removeAttribute('src');
      }}
      const actionText = Array.isArray(frame.action)
        ? frame.action.map((value, idx) => {{
            const axisName = sampleMeta.action_fields[idx] || ('a' + String(idx));
            return `${{axisName}}:${{Number(value).toFixed(3)}}`;
          }}).join(' ')
        : '';
      const imageState = url ? '' : '<br />image: missing source image';
      overlay.innerHTML = `instruction: ${{sampleMeta.instruction || '-'}}<br />step: ${{frame.step_index + 1}} frame_id: ${{frame.frame_id}}<br />${{actionText}}<br />phase: ${{frame.phase || '-'}} mask: ${{frame.action_mask}}<br />t: ${{Number(frame.timestamp).toFixed(3)}}${{imageState}}`;
      slider.value = String(index);
      preloadAround(index);
      updateActionCurrentLine(index);
      updatePlaybackStats();
      syncRangeControls();
    }}

    function step(delta) {{
      if (!frames.length) {{
        return;
      }}
      const bounds = currentRangeBounds();
      if (delta < 0) {{
        index = index <= bounds.start - 1 ? bounds.end - 1 : index - 1;
      }} else {{
        index = index >= bounds.end - 1 ? bounds.start - 1 : index + 1;
      }}
      index = clamp(index, bounds.start - 1, bounds.end - 1);
      render();
      if (playing) {{
        scheduleNextFrame();
      }}
    }}

    document.getElementById('prevFrame').addEventListener('click', () => step(-1));
    document.getElementById('nextFrame').addEventListener('click', () => step(1));
    slider.addEventListener('input', (event) => {{
      index = clampIndexToSelectedRange(Number(event.target.value) || 0);
      render();
      if (playing) {{
        scheduleNextFrame();
      }}
    }});
    frameRangeStartSlider.addEventListener('input', () => {{
      syncRangeControls('start-slider');
      index = clampIndexToSelectedRange(index);
      render();
    }});
    frameRangeEndSlider.addEventListener('input', () => {{
      syncRangeControls('end-slider');
      index = clampIndexToSelectedRange(index);
      render();
    }});
    stackedRangeControls.forEach((control) => {{
      ['pointerdown', 'mousedown', 'touchstart', 'focus'].forEach((eventName) => {{
        control.addEventListener(eventName, () => setActiveRangeControl(control));
      }});
    }});
    speedSelect.addEventListener('change', (event) => {{
      playbackSpeed = Math.max(0.25, Number(event.target.value) || 1.0);
      updatePlaybackStats();
      if (playing) {{
        scheduleNextFrame();
      }}
    }});
    playPause.addEventListener('click', () => {{
      if (playing) {{
        stopPlayback();
        return;
      }}
      if (!frames.length) {{
        return;
      }}
      playing = true;
      playPause.textContent = 'Pause';
      scheduleNextFrame();
    }});
    window.addEventListener('keydown', (event) => {{
      if (event.key === ' ') {{
        event.preventDefault();
        playPause.click();
      }} else if (event.key === 'ArrowLeft') {{
        event.preventDefault();
        step(-1);
      }} else if (event.key === 'ArrowRight') {{
        event.preventDefault();
        step(1);
      }}
    }});
    setActiveRangeControl(slider);
    syncRangeControls();
    render();
  </script>
</body>
</html>
"""


def _processed_summary_payload(
    samples: Sequence[ProcessedWindowRecord],
    windows_root: Path,
    filtered_root: Optional[Path],
    canonical_phase_root: Optional[Path],
) -> Dict[str, Any]:
    windows_stats = load_json(windows_root / "stats.json") if (windows_root / "stats.json").exists() else {}
    filtered_stats = load_json(filtered_root / "stats.json") if filtered_root is not None and (filtered_root / "stats.json").exists() else {}
    canonical_phase_stats = load_json(canonical_phase_root / "stats.json") if canonical_phase_root is not None and (canonical_phase_root / "stats.json").exists() else {}
    window_records = [
        {
            "instruction": sample.instruction,
            "phase": sample.phase,
            "actions_continuous": sample.actions_continuous,
            "action_mask": sample.action_mask,
            "state": sample.state,
        }
        for sample in samples
    ]
    return {
        "mode": "processed_windows",
        "windows_root": str(windows_root),
        "filtered_root": str(filtered_root) if filtered_root is not None else "",
        "canonical_phase_root": str(canonical_phase_root) if canonical_phase_root is not None else "",
        "sample_count": len(samples),
        "session_count": len({sample.session_id for sample in samples}),
        "episode_count": len({(sample.session_id, sample.episode_id) for sample in samples}),
        "padding_sample_count": sum(1 for sample in samples if sample.has_padding),
        "nonzero_action_sample_count": sum(1 for sample in samples if sample.has_nonzero_action),
        "image_missing_sample_count": sum(1 for sample in samples if all(frame.image_disk_path is None for frame in sample.playback_frames)),
        "instruction_counts": summarize_instruction_distribution(window_records, field="instruction"),
        "phase_counts": summarize_phase_distribution(window_records),
        "integrity": summarize_window_integrity(window_records, expected_chunk_len=max(1, int(windows_stats.get("window_config", {}).get("chunk_len", 4) or 4))),
        "windows_stats": windows_stats,
        "filtered_stats": filtered_stats,
        "canonical_phase_stats": canonical_phase_stats,
    }


def write_processed_reports(
    samples: Sequence[ProcessedWindowRecord],
    output_dir: Path,
    windows_root: Path,
    filtered_root: Optional[Path],
    canonical_phase_root: Optional[Path],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.html"
    page_names = [f"{sample.session_id}_{sample.episode_id}_{sample.start_frame:06d}.html" for sample in samples]

    for row_index, sample in enumerate(samples):
        page_name = page_names[row_index]
        page_path = sample_dir / page_name
        page_path.write_text(
            _processed_sample_page(
                sample,
                page_path,
                index_path,
                page_names[row_index - 1] if row_index > 0 else None,
                page_names[row_index + 1] if row_index + 1 < len(page_names) else None,
            ),
            encoding="utf-8",
        )

    summary = _processed_summary_payload(samples, windows_root, filtered_root, canonical_phase_root)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    integrity = dict(summary.get("integrity") or {})
    integrity_issue_counts = dict(integrity.get("issue_counts") or {})
    bad_chunk_len = int(integrity.get("bad_chunk_len") or integrity_issue_counts.get("unexpected_action_chunk_len") or 0)
    bad_mask_len = int(integrity.get("bad_mask_len") or integrity_issue_counts.get("unexpected_action_mask_len") or integrity_issue_counts.get("invalid_action_mask_value") or 0)
    bad_state_len = int(integrity.get("bad_state_len") or integrity_issue_counts.get("state_not_list") or 0)

    instruction_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{count}</td></tr>"
        for name, count in list(summary["instruction_counts"].items())[:20]
    )
    phase_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{count}</td></tr>"
        for name, count in summary["phase_counts"].items()
    )
    rows = []
    for row_index, sample in enumerate(samples):
        page_name = page_names[row_index]
        rows.append(
            f"<tr data-sample-row=\"1\" "
            f"data-instruction=\"{escape(sample.instruction.lower())}\" "
            f"data-phase=\"{escape((sample.phase or '').lower())}\" "
            f"data-session-id=\"{escape(sample.session_id.lower())}\" "
            f"data-episode-id=\"{escape(sample.episode_id.lower())}\" "
            f"data-padding=\"{'yes' if sample.has_padding else 'no'}\" "
            f"data-nonzero=\"{'yes' if sample.has_nonzero_action else 'no'}\">"
            f"<td><a href=\"samples/{escape(page_name)}\">{escape(sample.sample_id)}</a></td>"
            f"<td>{escape(sample.instruction or '-')}</td>"
            f"<td>{escape(sample.phase or '-')}</td>"
            f"<td>{escape(sample.session_id or '-')}</td>"
            f"<td>{escape(sample.episode_id or '-')}</td>"
            f"<td>{sample.start_frame}</td>"
            f"<td>{'yes' if sample.has_padding else 'no'}</td>"
            f"<td>{'yes' if sample.has_nonzero_action else 'no'}</td>"
            f"<td>{escape(', '.join(str(frame_id) for frame_id in sample.frame_ids) or '-')}</td>"
            f"</tr>"
        )

    filtered_summary_html = ""
    filtered_stats = summary.get("filtered_stats") or {}
    if filtered_stats:
        filtered_summary_html = (
            f"<tr><th>Filtered input frames</th><td>{int(filtered_stats.get('input_frame_count') or 0)}</td></tr>"
            f"<tr><th>Filtered output frames</th><td>{int(filtered_stats.get('output_frame_count') or 0)}</td></tr>"
            f"<tr><th>Filtered retention</th><td>{float(filtered_stats.get('retention_ratio') or 0.0):.1%}</td></tr>"
        )

    canonical_phase_html = ""
    canonical_phase_stats = summary.get("canonical_phase_stats") or {}
    if canonical_phase_stats:
        canonical_phase_html = (
            f"<tr><th>Canonical phase frames</th><td>{int(canonical_phase_stats.get('frame_count') or 0)}</td></tr>"
            f"<tr><th>Canonical phase episodes</th><td>{int(canonical_phase_stats.get('episode_count') or 0)}</td></tr>"
        )

    index_html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Go2 处理后数据体检报告</title>
  <style>{CSS}</style>
</head>
<body>
  <div class=\"card\">
    <h1>Go2 处理后数据体检报告</h1>
    <p class=\"meta\">mode=processed_windows samples={summary['sample_count']} sessions={summary['session_count']} episodes={summary['episode_count']}</p>
    <p class=\"meta\">windows_root=<code>{escape(str(windows_root))}</code></p>
  </div>
  <div class=\"grid\">
    <div class=\"card\">
      <h2>Dataset Overview</h2>
      <table>
        <tbody>
          <tr><th>Window samples</th><td>{summary['sample_count']}</td></tr>
          <tr><th>Sessions</th><td>{summary['session_count']}</td></tr>
          <tr><th>Episodes</th><td>{summary['episode_count']}</td></tr>
          <tr><th>Padding samples</th><td>{summary['padding_sample_count']}</td></tr>
          <tr><th>Nonzero-action samples</th><td>{summary['nonzero_action_sample_count']}</td></tr>
          <tr><th>Missing-image samples</th><td>{summary['image_missing_sample_count']}</td></tr>
          <tr><th>Bad chunk len</th><td>{bad_chunk_len}</td></tr>
          <tr><th>Bad mask len</th><td>{bad_mask_len}</td></tr>
          <tr><th>Bad state len</th><td>{bad_state_len}</td></tr>
          {filtered_summary_html}
          {canonical_phase_html}
        </tbody>
      </table>
    </div>
    <div class=\"card\">
      <h2>Top Instructions</h2>
      <table>
        <thead><tr><th>Instruction</th><th>Samples</th></tr></thead>
        <tbody>{instruction_rows or '<tr><td colspan="2">No instructions</td></tr>'}</tbody>
      </table>
    </div>
    <div class=\"card\">
      <h2>Phase Distribution</h2>
      <table>
        <thead><tr><th>Phase</th><th>Samples</th></tr></thead>
        <tbody>{phase_rows or '<tr><td colspan="2">No phase stats</td></tr>'}</tbody>
      </table>
    </div>
  </div>
  <div class=\"card\">
    <h2>Sample Filters</h2>
    <div class=\"controls\">
      <label>Instruction <input id=\"instructionFilter\" type=\"text\" placeholder=\"contains...\" /></label>
      <label>Phase <input id=\"phaseFilter\" type=\"text\" placeholder=\"FORWARD / APPROACH\" /></label>
      <label>Session <input id=\"sessionFilter\" type=\"text\" placeholder=\"20260413_190341\" /></label>
      <label>Episode <input id=\"episodeFilter\" type=\"text\" placeholder=\"ep_000001\" /></label>
      <label>Padding
        <select id=\"paddingFilter\">
          <option value=\"all\">all</option>
          <option value=\"yes\">yes</option>
          <option value=\"no\">no</option>
        </select>
      </label>
      <label>Nonzero action
        <select id=\"nonzeroFilter\">
          <option value=\"all\">all</option>
          <option value=\"yes\">yes</option>
          <option value=\"no\">no</option>
        </select>
      </label>
      <span class=\"playback-stats\" id=\"visibleCount\"></span>
    </div>
  </div>
  <div class=\"card\">
    <table>
      <thead><tr><th>Sample</th><th>Instruction</th><th>Phase</th><th>Session</th><th>Episode</th><th>Start Frame</th><th>Padding</th><th>Nonzero</th><th>Frame IDs</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
  <script>
    const sampleRows = Array.from(document.querySelectorAll('[data-sample-row]'));
    const instructionFilter = document.getElementById('instructionFilter');
    const phaseFilter = document.getElementById('phaseFilter');
    const sessionFilter = document.getElementById('sessionFilter');
    const episodeFilter = document.getElementById('episodeFilter');
    const paddingFilter = document.getElementById('paddingFilter');
    const nonzeroFilter = document.getElementById('nonzeroFilter');
    const visibleCount = document.getElementById('visibleCount');

    function includesValue(haystack, needle) {{
      return !needle || haystack.includes(needle);
    }}

    function applyFilters() {{
      const instructionNeedle = String(instructionFilter.value || '').trim().toLowerCase();
      const phaseNeedle = String(phaseFilter.value || '').trim().toLowerCase();
      const sessionNeedle = String(sessionFilter.value || '').trim().toLowerCase();
      const episodeNeedle = String(episodeFilter.value || '').trim().toLowerCase();
      const paddingNeedle = paddingFilter.value;
      const nonzeroNeedle = nonzeroFilter.value;
      let visible = 0;
      sampleRows.forEach((row) => {{
        let show = true;
        show = show && includesValue(row.dataset.instruction || '', instructionNeedle);
        show = show && includesValue(row.dataset.phase || '', phaseNeedle);
        show = show && includesValue(row.dataset.sessionId || '', sessionNeedle);
        show = show && includesValue(row.dataset.episodeId || '', episodeNeedle);
        if (show && paddingNeedle !== 'all') {{
          show = (row.dataset.padding || 'no') === paddingNeedle;
        }}
        if (show && nonzeroNeedle !== 'all') {{
          show = (row.dataset.nonzero || 'no') === nonzeroNeedle;
        }}
        row.hidden = !show;
        if (show) {{
          visible += 1;
        }}
      }});
      visibleCount.textContent = `visible ${{visible}} / ${{sampleRows.length}}`;
    }}

    [instructionFilter, phaseFilter, sessionFilter, episodeFilter].forEach((node) => {{
      node.addEventListener('input', applyFilters);
    }});
    [paddingFilter, nonzeroFilter].forEach((node) => {{
      node.addEventListener('change', applyFilters);
    }});
    const query = new URLSearchParams(window.location.search);
    if (query.has('instruction')) instructionFilter.value = query.get('instruction') || '';
    if (query.has('phase')) phaseFilter.value = query.get('phase') || '';
    if (query.has('session_id')) sessionFilter.value = query.get('session_id') || '';
    if (query.has('episode_id')) episodeFilter.value = query.get('episode_id') || '';
    if (query.has('padding')) paddingFilter.value = query.get('padding') || 'all';
    if (query.has('nonzero')) nonzeroFilter.value = query.get('nonzero') || 'all';
    applyFilters();
  </script>
</body>
</html>
"""
    index_path.write_text(index_html, encoding="utf-8")


def processed_file_roots(samples: Sequence[ProcessedWindowRecord], windows_root: Path) -> List[Path]:
    roots = {windows_root.resolve()}
    for sample in samples:
        if sample.image_disk_path is not None:
            roots.add(sample.image_disk_path.parent.resolve())
        for frame in sample.playback_frames:
            if frame.image_disk_path is not None:
                roots.add(frame.image_disk_path.parent.resolve())
    return sorted(roots)


def _read_summary_payload(root: Path) -> Dict[str, Any]:
    for name in ("summary.json", "stats.json", "pipeline_summary.json"):
        candidate = root / name
        if candidate.exists():
            payload = load_json(candidate)
            if isinstance(payload, dict):
                return payload
    return {}


def _discover_pipeline_roots(workspace_root: Path, args: argparse.Namespace) -> Dict[str, Optional[Path]]:
    roots: Dict[str, Optional[Path]] = {
        "real_raw": args.real_raw_root.resolve() if args.real_raw_root is not None else None,
        "sim_raw": args.sim_raw_root.resolve() if args.sim_raw_root is not None else None,
        "canonical_phase": args.canonical_phase_root.resolve() if args.canonical_phase_root is not None else None,
        "filtered": args.filtered_root.resolve() if args.filtered_root is not None else None,
        "windows": args.windows_root.resolve() if args.windows_root is not None else None,
        "model_export": args.model_export_root.resolve() if args.model_export_root is not None else None,
    }
    defaults = {
        "real_raw": workspace_root / "data" / "real_sessions",
        "sim_raw": workspace_root / "data" / "sim_sessions",
        "canonical_phase": workspace_root / "outputs" / "canonical_phase",
        "filtered": workspace_root / "outputs" / "filtered",
        "windows": workspace_root / "outputs" / "windows",
        "model_export": workspace_root / "datasets" / "current_v1",
    }
    for key, default_path in defaults.items():
        if roots[key] is None and default_path.exists():
            roots[key] = default_path.resolve()
    return roots


def _write_generic_stage_index(
    title: str,
    stage_root: Path,
    output_dir: Path,
    *,
    hub_rel: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _read_summary_payload(stage_root)
    rows = []
    for key, value in sorted(summary.items()):
        if isinstance(value, (dict, list)):
            continue
        rows.append(f"<tr><th>{escape(str(key))}</th><td><code>{escape(str(value))}</code></td></tr>")
    notable = []
    for name in ("dataset.jsonl", "windows.jsonl", "train.jsonl", "val.jsonl", "test.jsonl", "stats.json", "summary.json", "pipeline_summary.json"):
        candidate = stage_root / name
        if candidate.exists():
            notable.append(f"<li><code>{escape(name)}</code></li>")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{escape(title)}</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="page-nav"><a href="{escape(hub_rel)}">Back to pipeline hub</a></div>
  <div class="card">
    <h1>{escape(title)}</h1>
    <p class="meta">root=<code>{escape(str(stage_root))}</code></p>
  </div>
  <div class="grid">
    <div class="card">
      <h2>Summary</h2>
      <table><tbody>{''.join(rows) or '<tr><td colspan="2">No summary file found.</td></tr>'}</tbody></table>
    </div>
    <div class="card">
      <h2>Notable Files</h2>
      <ul>{''.join(notable) or '<li>No known files discovered.</li>'}</ul>
    </div>
  </div>
</body>
</html>
"""
    index_path = output_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path


def _inject_pipeline_hub_links(report_root: Path) -> None:
    index_path = report_root / "index.html"
    if index_path.exists():
        text = index_path.read_text(encoding="utf-8")
        if 'Back to pipeline hub' not in text:
            text = text.replace("<body>", '<body>\n  <div class="page-nav"><a href="../index.html">Back to pipeline hub</a></div>', 1)
            index_path.write_text(text, encoding="utf-8")

    for detail_dir in ("episodes", "samples"):
        directory = report_root / detail_dir
        if not directory.exists():
            continue
        for page_path in directory.glob("*.html"):
            text = page_path.read_text(encoding="utf-8")
            if 'Pipeline hub' in text:
                continue
            text = text.replace('<div class="page-nav">', '<div class="page-nav"><a href="../../index.html">Pipeline hub</a>', 1)
            page_path.write_text(text, encoding="utf-8")


def write_pipeline_hub(args: argparse.Namespace, output_dir: Path) -> Path:
    workspace_root = args.workspace_root.resolve()
    roots = _discover_pipeline_roots(workspace_root, args)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_links: Dict[str, Path] = {}
    stage_meta: Dict[str, Dict[str, Any]] = {}

    raw_stage_specs = [
        ("real_raw", "Raw / Real", roots["real_raw"]),
        ("sim_raw", "Raw / Sim", roots["sim_raw"]),
    ]
    for key, title, root in raw_stage_specs:
        if root is None or not root.exists():
            stage_meta[key] = {"title": title, "status": "missing", "root": root}
            continue
        stage_output = output_dir / key
        episodes = load_episodes(root, args.max_episodes, None)
        write_reports(episodes, stage_output, root, args.seed, args.num_samples)
        _inject_pipeline_hub_links(stage_output)
        stage_links[key] = stage_output / "index.html"
        stage_meta[key] = {
            "title": title,
            "status": "ready",
            "root": root,
            "summary": _read_summary_payload(stage_output),
        }

    if roots["windows"] is not None and roots["windows"].exists():
        windows_output = output_dir / "windows"
        samples = load_processed_windows(roots["windows"], args.max_samples, roots["canonical_phase"])
        write_processed_reports(samples, windows_output, roots["windows"], roots["filtered"], roots["canonical_phase"])
        _inject_pipeline_hub_links(windows_output)
        stage_links["windows"] = windows_output / "index.html"
        stage_meta["windows"] = {
            "title": "Windows",
            "status": "ready",
            "root": roots["windows"],
            "summary": _read_summary_payload(windows_output),
        }
    else:
        stage_meta["windows"] = {"title": "Windows", "status": "missing", "root": roots["windows"]}

    generic_specs = [
        ("canonical_phase", "Canonical Phase", roots["canonical_phase"]),
        ("filtered", "Filtered", roots["filtered"]),
        ("model_export", "Model Export / Final Dataset", roots["model_export"]),
    ]
    for key, title, root in generic_specs:
        if root is None or not root.exists():
            stage_meta[key] = {"title": title, "status": "missing", "root": root}
            continue
        stage_output = output_dir / key
        stage_links[key] = _write_generic_stage_index(title, root, stage_output, hub_rel="../index.html")
        stage_meta[key] = {
            "title": title,
            "status": "ready",
            "root": root,
            "summary": _read_summary_payload(root),
        }

    cards = []
    order = ["real_raw", "sim_raw", "canonical_phase", "filtered", "windows", "model_export"]
    for key in order:
        item = stage_meta.get(key) or {}
        title = str(item.get("title") or key)
        root = item.get("root")
        summary = item.get("summary") or {}
        lines = []
        for candidate_key in ("episode_count", "sample_count", "frame_count", "session_count", "samples_total"):
            if candidate_key in summary and not isinstance(summary[candidate_key], (dict, list)):
                lines.append(f"<li><code>{escape(candidate_key)}</code> = {escape(str(summary[candidate_key]))}</li>")
        if not lines:
            lines.append("<li class=\"meta\">No summary metrics available.</li>")
        link_html = (
            f'<p><a href="{escape(os.path.relpath(stage_links[key], output_dir))}">Open stage page</a></p>'
            if key in stage_links else '<p class="meta">Stage missing</p>'
        )
        cards.append(
            "<div class=\"card\">"
            f"<h2>{escape(title)}</h2>"
            f"<p class=\"meta\">status={escape(str(item.get('status') or 'missing'))}</p>"
            f"<p class=\"meta\">root=<code>{escape(str(root) if root is not None else '-')}</code></p>"
            f"{link_html}"
            f"<ul>{''.join(lines)}</ul>"
            "</div>"
        )

    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Go2 Pipeline Sanity Hub</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="card">
    <h1>Go2 Pipeline Sanity Hub</h1>
    <p class="meta">workspace_root=<code>{escape(str(workspace_root))}</code></p>
    <p class="meta">从这里跳到 raw / canonical_phase / filtered / windows / model_export 各阶段检查页。</p>
  </div>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    index_path = output_dir / "index.html"
    index_path.write_text(index_html, encoding="utf-8")
    return index_path


def write_reports(episodes: Sequence[EpisodeRecord], output_dir: Path, data_root: Path, seed: int, num_samples: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_dir = output_dir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.html"
    rng = random.Random(seed)
    episode_refs_payload = [
        {
            "session_id": episode.session_id,
            "episode_id": episode.episode_id,
            "quality_label": episode.original_quality_label,
            "original_quality_label": episode.original_quality_label,
            "frame_selection_mode": episode.frame_selection_mode,
            "frame_keep_map": {str(index): value for index, value in episode.frame_keep_map.items()},
            "selected_frame_count": episode.selected_frame_count,
            "total_frame_count": episode.total_frame_count,
        }
        for episode in episodes
    ]
    initial_reviews = _initial_review_payload(episodes)
    page_names = [f"{episode.session_id}_{episode.episode_id}.html" for episode in episodes]

    rows = []
    for row_index, episode in enumerate(episodes):
        page_name = page_names[row_index]
        page_path = episode_dir / page_name
        page_path.write_text(
            _episode_page(
                episode,
                page_path,
                index_path,
                rng,
                num_samples,
                data_root,
                episode_refs_payload,
                initial_reviews,
                page_names[row_index - 1] if row_index > 0 else None,
                page_names[row_index + 1] if row_index + 1 < len(page_names) else None,
            ),
            encoding="utf-8",
        )
        review_row_class = "quality-excluded" if episode.exclude_from_processing else ("quality-review" if episode.quality_overridden else "")
        review_select_attr = 'data-review-select="1"'
        rows.append(
            f"<tr class=\"{review_row_class}\" data-review-row=\"1\" "
            f"data-session-id=\"{escape(episode.session_id)}\" "
            f"data-episode-id=\"{escape(episode.episode_id)}\" "
            f"data-instruction=\"{escape(episode.instruction.lower())}\" "
            f"data-source-type=\"{escape((episode.source_type or '').lower())}\" "
            f"data-scene=\"{escape((episode.scene_id or '').lower())}\" "
            f"data-target=\"{escape((episode.target_label or episode.target_type or '').lower())}\">"
            f"<td><a href=\"episodes/{escape(page_name)}\">{escape(episode.session_id)} / {escape(episode.episode_id)}</a></td>"
            f"<td>{escape(episode.instruction)}</td><td>{escape(episode.capture_mode or '-')}</td><td>{escape(episode.source_type or '-')}</td><td>{escape(episode.task_family or '-')}</td><td>{escape(episode.target_label or episode.target_type or '-')}</td><td>{escape(episode.derived_target_side or '-')}</td><td>{escape(episode.derived_target_distance or '-')}</td><td>{len(episode.frames)}</td>"
            f"<td>{episode.trajectory_metrics.get('duration_seconds', 0.0):.2f}</td><td>{int(episode.trajectory_metrics.get('action_change_count', 0))}</td><td>{episode.trajectory_metrics.get('stop_ratio', 0.0):.1%}</td><td>{episode.trajectory_metrics.get('turn_ratio', 0.0):.1%}</td>"
            f"<td>{escape(episode.scene_id or '-')}</td><td>{escape(episode.operator_id or '-')}</td><td>{escape(episode.d435i_capture_path or '-')}</td>"
            f"<td>{escape(', '.join(episode.warnings) or '无')}</td>"
            f"<td>{escape(', '.join(episode.info) or '无')}</td>"
            f"<td><span class=\"quality-badge label-{episode.quality_label}\" data-review-badge>{escape(_quality_badge_text(episode.quality_label))}</span></td>"
            f"<td>{_quality_select_html(f'quality-select-{row_index}', episode.quality_label, review_select_attr)}</td>"
            f"<td><input class=\"quality-note\" data-review-note=\"1\" type=\"text\" value=\"{escape(episode.quality_notes)}\" placeholder=\"可选：记录排除原因\" /></td></tr>"
        )

    instruction_episode_counts: Dict[str, int] = {}
    instruction_frame_counts: Dict[str, int] = {}
    scene_counts: Dict[str, int] = {}
    operator_counts: Dict[str, int] = {}
    capture_mode_counts: Dict[str, int] = {}
    task_family_counts: Dict[str, int] = {}
    target_type_counts: Dict[str, int] = {}
    target_label_counts: Dict[str, int] = {}
    target_description_counts: Dict[str, int] = {}
    derived_target_side_counts: Dict[str, int] = {}
    derived_target_distance_counts: Dict[str, int] = {}
    quality_label_counts: Dict[str, int] = {}
    instruction_scene_sets: Dict[str, set] = {}
    instruction_target_sets: Dict[str, set] = {}
    zero_action_frame_count = 0
    frame_lengths: List[int] = []
    duration_values: List[float] = []
    action_change_values: List[float] = []
    stop_ratio_values: List[float] = []
    turn_ratio_values: List[float] = []
    trajectory_episodes: List[EpisodeRecord] = []
    for episode in episodes:
        instruction_episode_counts[episode.instruction] = instruction_episode_counts.get(episode.instruction, 0) + 1
        instruction_frame_counts[episode.instruction] = instruction_frame_counts.get(episode.instruction, 0) + len(episode.frames)
        scene_key = episode.scene_id or "-"
        operator_key = episode.operator_id or "-"
        capture_mode_key = episode.capture_mode or "-"
        scene_counts[scene_key] = scene_counts.get(scene_key, 0) + 1
        operator_counts[operator_key] = operator_counts.get(operator_key, 0) + 1
        capture_mode_counts[capture_mode_key] = capture_mode_counts.get(capture_mode_key, 0) + 1
        task_family_key = episode.task_family or "-"
        target_type_key = episode.target_type or "-"
        target_label_key = episode.target_label or "-"
        target_description_key = episode.target_description or "-"
        derived_target_side_key = episode.derived_target_side or "-"
        derived_target_distance_key = episode.derived_target_distance or "-"
        quality_label_key = _quality_badge_text(episode.quality_label)
        task_family_counts[task_family_key] = task_family_counts.get(task_family_key, 0) + 1
        target_type_counts[target_type_key] = target_type_counts.get(target_type_key, 0) + 1
        target_label_counts[target_label_key] = target_label_counts.get(target_label_key, 0) + 1
        target_description_counts[target_description_key] = target_description_counts.get(target_description_key, 0) + 1
        derived_target_side_counts[derived_target_side_key] = derived_target_side_counts.get(derived_target_side_key, 0) + 1
        derived_target_distance_counts[derived_target_distance_key] = derived_target_distance_counts.get(derived_target_distance_key, 0) + 1
        quality_label_counts[quality_label_key] = quality_label_counts.get(quality_label_key, 0) + 1
        instruction_scene_sets.setdefault(episode.instruction, set()).add(scene_key)
        derived_target_key = (
            f"{episode.derived_target_side or 'unknown'}:{episode.derived_target_distance or 'unknown'}"
            if (episode.derived_target_side or episode.derived_target_distance)
            else "-"
        )
        instruction_target_sets.setdefault(episode.instruction, set()).add(
            episode.target_label or episode.target_type or derived_target_key
        )
        frame_lengths.append(len(episode.frames))
        duration_values.append(float(episode.trajectory_metrics.get("duration_seconds", 0.0)))
        action_change_values.append(float(episode.trajectory_metrics.get("action_change_count", 0.0)))
        stop_ratio_values.append(float(episode.trajectory_metrics.get("stop_ratio", 0.0)))
        turn_ratio_values.append(float(episode.trajectory_metrics.get("turn_ratio", 0.0)))
        if episode.capture_mode == "trajectory":
            trajectory_episodes.append(episode)
        for frame in episode.frames:
            magnitude = sum(abs(frame.control_action[axis]) for axis in ACTION_FIELDS)
            if magnitude <= 1e-6:
                zero_action_frame_count += 1

    empty_session_count, session_episode_counts = collect_session_stats(data_root)
    total_frames = sum(len(episode.frames) for episode in episodes)
    length_min = min(frame_lengths) if frame_lengths else 0
    length_max = max(frame_lengths) if frame_lengths else 0
    length_median = sorted(frame_lengths)[len(frame_lengths) // 2] if frame_lengths else 0

    summary = {
        "episode_count": len(episodes),
        "frame_count": total_frames,
        "warning_episode_count": sum(1 for episode in episodes if episode.warnings),
        "info_episode_count": sum(1 for episode in episodes if episode.info),
        "empty_session_count": empty_session_count,
        "session_episode_counts": session_episode_counts,
        "instructions": sorted({episode.instruction for episode in episodes}),
        "instruction_episode_counts": dict(sorted(instruction_episode_counts.items())),
        "instruction_frame_counts": dict(sorted(instruction_frame_counts.items())),
        "scene_episode_counts": dict(sorted(scene_counts.items())),
        "operator_episode_counts": dict(sorted(operator_counts.items())),
        "capture_mode_episode_counts": dict(sorted(capture_mode_counts.items())),
        "task_family_episode_counts": dict(sorted(task_family_counts.items())),
        "target_type_episode_counts": dict(sorted(target_type_counts.items())),
        "target_label_episode_counts": dict(sorted(target_label_counts.items(), key=lambda item: (-item[1], item[0]))),
        "target_description_episode_counts": dict(sorted(target_description_counts.items(), key=lambda item: (-item[1], item[0]))),
        "derived_target_side_episode_counts": dict(sorted(derived_target_side_counts.items(), key=lambda item: (-item[1], item[0]))),
        "derived_target_distance_episode_counts": dict(sorted(derived_target_distance_counts.items(), key=lambda item: (-item[1], item[0]))),
        "quality_label_episode_counts": dict(sorted(quality_label_counts.items(), key=lambda item: (-item[1], item[0]))),
        "quality_filtered_episode_count": sum(1 for episode in episodes if episode.exclude_from_processing),
        "instructions_with_multiple_scenes": {
            instruction: sorted(scene_set)
            for instruction, scene_set in sorted(instruction_scene_sets.items())
            if len(scene_set - {"-"}) >= 2
        },
        "instructions_with_multiple_targets": {
            instruction: sorted(target_set)
            for instruction, target_set in sorted(instruction_target_sets.items())
            if len(target_set - {"-"}) >= 2
        },
        "zero_action_frame_count": zero_action_frame_count,
        "zero_action_frame_ratio": 0.0 if total_frames == 0 else zero_action_frame_count / total_frames,
        "episode_length_min": length_min,
        "episode_length_median": length_median,
        "episode_length_max": length_max,
        "trajectory_metrics": {
            "duration_seconds": summarize_trajectory_metric_series(duration_values),
            "action_change_count": summarize_trajectory_metric_series(action_change_values),
            "stop_ratio": summarize_trajectory_metric_series(stop_ratio_values),
            "turn_ratio": summarize_trajectory_metric_series(turn_ratio_values),
        },
        "trajectory_episode_summary": {
            "episode_count": len(trajectory_episodes),
            "duration_seconds": summarize_trajectory_metric_series(
                [float(episode.trajectory_metrics.get("duration_seconds", 0.0)) for episode in trajectory_episodes]
            ),
            "action_change_count": summarize_trajectory_metric_series(
                [float(episode.trajectory_metrics.get("action_change_count", 0.0)) for episode in trajectory_episodes]
            ),
            "stop_ratio": summarize_trajectory_metric_series(
                [float(episode.trajectory_metrics.get("stop_ratio", 0.0)) for episode in trajectory_episodes]
            ),
            "turn_ratio": summarize_trajectory_metric_series(
                [float(episode.trajectory_metrics.get("turn_ratio", 0.0)) for episode in trajectory_episodes]
            ),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    instruction_rows = []
    total_episodes = max(1, summary["episode_count"])
    total_frames = max(1, summary["frame_count"])
    for instruction in summary["instructions"]:
        episode_count = summary["instruction_episode_counts"].get(instruction, 0)
        frame_count = summary["instruction_frame_counts"].get(instruction, 0)
        instruction_rows.append(
            f"<tr><td><code>{escape(instruction)}</code></td>"
            f"<td>{episode_count}</td>"
            f"<td>{episode_count / total_episodes:.1%}</td>"
            f"<td>{frame_count}</td>"
            f"<td>{frame_count / total_frames:.1%}</td></tr>"
        )

    scene_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{count}</td></tr>"
        for name, count in summary["scene_episode_counts"].items()
    )
    operator_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{count}</td></tr>"
        for name, count in summary["operator_episode_counts"].items()
    )
    capture_mode_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{count}</td></tr>"
        for name, count in summary["capture_mode_episode_counts"].items()
    )
    task_family_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{count}</td></tr>"
        for name, count in summary["task_family_episode_counts"].items()
    )
    target_type_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{count}</td></tr>"
        for name, count in summary["target_type_episode_counts"].items()
    )
    target_label_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{count}</td></tr>"
        for name, count in summary["target_label_episode_counts"].items()
    )
    derived_target_side_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{count}</td></tr>"
        for name, count in summary["derived_target_side_episode_counts"].items()
    )
    derived_target_distance_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{count}</td></tr>"
        for name, count in summary["derived_target_distance_episode_counts"].items()
    )
    quality_label_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{count}</td></tr>"
        for name, count in summary["quality_label_episode_counts"].items()
    )
    session_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{count}</td></tr>"
        for name, count in summary["session_episode_counts"].items()
    )
    multi_scene_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{escape(', '.join(values))}</td></tr>"
        for name, values in summary["instructions_with_multiple_scenes"].items()
    )
    multi_target_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{escape(', '.join(values))}</td></tr>"
        for name, values in summary["instructions_with_multiple_targets"].items()
    )
    review_script = _review_shared_script(data_root, episode_refs_payload, initial_reviews)

    index_html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Go2 数据集体检报告</title>
  <style>{CSS}</style>
</head>
<body>
  <div class=\"card\">
    <h1>Go2 数据集体检报告</h1>
    <p class=\"meta\">episodes={summary['episode_count']} frames={summary['frame_count']} warning_episodes={summary['warning_episode_count']} info_episodes={summary['info_episode_count']}</p>
    <p>指令s: <code>{escape(', '.join(summary['instructions']) or '无')}</code></p>
  </div>
  <div class=\"card\">
    <h2>录制分数修改</h2>
    <p class=\"meta\">这里显示的是源文件里的当前分数 `1/2/3`。如果要把某些 `3` 改回 `2`，可先用“指定分数”筛出，再直接修改并保存。</p>
    <div class=\"review-toolbar\">
      <label>筛选
        <select id=\"qualityFilter\" class=\"quality-filter\">
          <option value=\"all\">全部样本</option>
          <option value=\"excluded\">仅质量差（3）</option>
          <option value=\"overridden\">仅未保存修改</option>
          <option value=\"original\">仅无页面修改</option>
        </select>
      </label>
      <label>指定分数
        <select id=\"qualityScoreFilter\" class=\"quality-score-filter\">
          <option value=\"all\">全部分数</option>
          <option value=\"1\">仅 1 好样本</option>
          <option value=\"2\">仅 2 可用但不完美</option>
          <option value=\"3\">仅 3 失败但保留 / 质量差</option>
        </select>
      </label>
      <button id=\"saveAllReviews\" disabled>保存到源文件</button>
      <button id=\"bindReviewFile\">绑定记录文件</button>
      <button id=\"exportReview\">导出修改 JSON</button>
      <button id=\"resetAllReviews\">清空本地修改缓存</button>
      <label>导入修改 JSON <input id=\"importReviewFile\" type=\"file\" accept=\".json,application/json\" /></label>
      <span id=\"reviewFileStatus\" class=\"review-file-status\"></span>
    </div>
    <div class=\"review-summary\">
      <span class=\"review-pill\">1 好样本 <strong id=\"summaryLabel1\">0</strong></span>
      <span class=\"review-pill\">2 可用但不完美 <strong id=\"summaryLabel2\">0</strong></span>
      <span class=\"review-pill\">3 失败但保留 / 质量差 <strong id=\"summaryLabel3\">0</strong></span>
      <span class=\"review-pill\">未保存修改 <strong id=\"summaryOverridden\">0</strong></span>
      <span class=\"review-pill\">后续处理排除 <strong id=\"summaryExcluded\">0</strong></span>
    </div>
    <p id=\"reviewStatus\" class=\"meta\">修改后先保存在当前页面；点击“保存到源文件”后才会真正写回。未绑定时也会保存在浏览器缓存中。</p>
  </div>
  <div class=\"card\">
    <h2>Episode Filters</h2>
    <div class=\"controls\">
      <label>Session <input id=\"sessionFilter\" type=\"text\" placeholder=\"20260418_202212\" /></label>
      <label>Episode <input id=\"episodeFilter\" type=\"text\" placeholder=\"ep_000001\" /></label>
      <label>Instruction <input id=\"instructionFilter\" type=\"text\" placeholder=\"move to the door\" /></label>
      <label>Scene <input id=\"sceneFilter\" type=\"text\" placeholder=\"earth_left_door\" /></label>
      <label>Target <input id=\"targetFilter\" type=\"text\" placeholder=\"door / box\" /></label>
      <span class=\"playback-stats\" id=\"visibleEpisodeCount\"></span>
    </div>
  </div>
  <div class=\"grid\">
    <div class=\"card\">
      <h2>Dataset Overview</h2>
      <table>
        <tbody>
          <tr><th>Total episodes</th><td>{summary['episode_count']}</td></tr>
          <tr><th>Total frames</th><td>{summary['frame_count']}</td></tr>
          <tr><th>Empty sessions</th><td>{summary['empty_session_count']}</td></tr>
          <tr><th>Zero-action frames</th><td>{summary['zero_action_frame_count']}</td></tr>
          <tr><th>Zero-action ratio</th><td>{summary['zero_action_frame_ratio']:.1%}</td></tr>
          <tr><th>Quality-filtered episodes</th><td>{summary['quality_filtered_episode_count']}</td></tr>
          <tr><th>Episode length min / median / max</th><td>{summary['episode_length_min']} / {summary['episode_length_median']} / {summary['episode_length_max']}</td></tr>
          <tr><th>平均时长（秒）</th><td>{summary['trajectory_metrics']['duration_seconds']['mean']:.2f}</td></tr>
          <tr><th>平均动作切换次数</th><td>{summary['trajectory_metrics']['action_change_count']['mean']:.2f}</td></tr>
          <tr><th>平均停止占比</th><td>{summary['trajectory_metrics']['stop_ratio']['mean']:.1%}</td></tr>
          <tr><th>平均转向占比</th><td>{summary['trajectory_metrics']['turn_ratio']['mean']:.1%}</td></tr>
        </tbody>
      </table>
    </div>
    <div class=\"card\">
      <h2>Session 覆盖情况</h2>
      <table>
        <thead><tr><th>Session</th><th>Episode 数</th></tr></thead>
        <tbody>{session_rows or '<tr><td colspan="2">没有可用的 session 元数据。</td></tr>'}</tbody>
      </table>
    </div>
  </div>
  <div class=\"card\">
    <h2>指令统计</h2>
    <table>
      <thead><tr><th>指令</th><th>Episode 数</th><th>Episode 占比</th><th>帧数</th><th>帧占比</th></tr></thead>
      <tbody>{''.join(instruction_rows) or '<tr><td colspan="5">没有可用的指令统计。</td></tr>'}</tbody>
    </table>
  </div>
  <div class=\"grid\">
    <div class=\"card\">
      <h2>场景覆盖情况</h2>
      <table>
        <thead><tr><th>场景</th><th>Episode 数</th></tr></thead>
        <tbody>{scene_rows or '<tr><td colspan="2">没有可用的场景元数据。</td></tr>'}</tbody>
      </table>
    </div>
    <div class=\"card\">
      <h2>操作员覆盖情况</h2>
      <table>
        <thead><tr><th>操作员</th><th>Episode 数</th></tr></thead>
        <tbody>{operator_rows or '<tr><td colspan="2">没有可用的操作员元数据。</td></tr>'}</tbody>
      </table>
    </div>
    <div class=\"card\">
      <h2>采集模式统计</h2>
      <table>
        <thead><tr><th>采集模式</th><th>Episode 数</th></tr></thead>
        <tbody>{capture_mode_rows or '<tr><td colspan="2">没有可用的采集模式元数据。</td></tr>'}</tbody>
      </table>
    </div>
    <div class=\"card\">
      <h2>任务族统计</h2>
      <table>
        <thead><tr><th>任务族</th><th>Episode 数</th></tr></thead>
        <tbody>{task_family_rows or '<tr><td colspan="2">没有可用的任务族元数据。</td></tr>'}</tbody>
      </table>
    </div>
    <div class=\"card\">
      <h2>目标类型统计</h2>
      <table>
        <thead><tr><th>目标类型</th><th>Episode 数</th></tr></thead>
        <tbody>{target_type_rows or '<tr><td colspan="2">没有可用的目标元数据。</td></tr>'}</tbody>
      </table>
    </div>
    <div class=\"card\">
      <h2>语义目标统计</h2>
      <table>
        <thead><tr><th>target_label</th><th>Episode 数</th></tr></thead>
        <tbody>{target_label_rows or '<tr><td colspan="2">没有可用的语义目标标签。</td></tr>'}</tbody>
      </table>
    </div>
    <div class=\"card\">
      <h2>派生左右标签统计</h2>
      <table>
        <thead><tr><th>target_side_band</th><th>Episode 数</th></tr></thead>
        <tbody>{derived_target_side_rows or '<tr><td colspan="2">还没有派生左右标签。</td></tr>'}</tbody>
      </table>
    </div>
    <div class=\"card\">
      <h2>派生远近标签统计</h2>
      <table>
        <thead><tr><th>target_distance_band</th><th>Episode 数</th></tr></thead>
        <tbody>{derived_target_distance_rows or '<tr><td colspan="2">还没有派生远近标签。</td></tr>'}</tbody>
      </table>
    </div>
    <div class=\"card\">
      <h2>分数统计</h2>
      <table>
        <thead><tr><th>当前分数</th><th>Episode 数</th></tr></thead>
        <tbody>{quality_label_rows or '<tr><td colspan="2">当前还没有分数统计。</td></tr>'}</tbody>
      </table>
    </div>
  </div>
  <div class=\"grid\">
    <div class=\"card\">
      <h2>轨迹统计指标</h2>
      <table>
        <thead><tr><th>指标</th><th>均值</th><th>中位数</th><th>最小值</th><th>最大值</th></tr></thead>
        <tbody>
          <tr><td>时长（秒）</td><td>{summary['trajectory_metrics']['duration_seconds']['mean']:.2f}</td><td>{summary['trajectory_metrics']['duration_seconds']['median']:.2f}</td><td>{summary['trajectory_metrics']['duration_seconds']['min']:.2f}</td><td>{summary['trajectory_metrics']['duration_seconds']['max']:.2f}</td></tr>
          <tr><td>动作切换次数</td><td>{summary['trajectory_metrics']['action_change_count']['mean']:.2f}</td><td>{summary['trajectory_metrics']['action_change_count']['median']:.2f}</td><td>{summary['trajectory_metrics']['action_change_count']['min']:.0f}</td><td>{summary['trajectory_metrics']['action_change_count']['max']:.0f}</td></tr>
          <tr><td>停止占比</td><td>{summary['trajectory_metrics']['stop_ratio']['mean']:.1%}</td><td>{summary['trajectory_metrics']['stop_ratio']['median']:.1%}</td><td>{summary['trajectory_metrics']['stop_ratio']['min']:.1%}</td><td>{summary['trajectory_metrics']['stop_ratio']['max']:.1%}</td></tr>
          <tr><td>转向占比</td><td>{summary['trajectory_metrics']['turn_ratio']['mean']:.1%}</td><td>{summary['trajectory_metrics']['turn_ratio']['median']:.1%}</td><td>{summary['trajectory_metrics']['turn_ratio']['min']:.1%}</td><td>{summary['trajectory_metrics']['turn_ratio']['max']:.1%}</td></tr>
        </tbody>
      </table>
    </div>
    <div class=\"card\">
      <h2>仅 trajectory 模式</h2>
      <table>
        <tbody>
          <tr><th>trajectory episode 数</th><td>{summary['trajectory_episode_summary']['episode_count']}</td></tr>
          <tr><th>平均时长（秒）</th><td>{summary['trajectory_episode_summary']['duration_seconds']['mean']:.2f}</td></tr>
          <tr><th>平均动作切换次数</th><td>{summary['trajectory_episode_summary']['action_change_count']['mean']:.2f}</td></tr>
          <tr><th>平均停止占比</th><td>{summary['trajectory_episode_summary']['stop_ratio']['mean']:.1%}</td></tr>
          <tr><th>平均转向占比</th><td>{summary['trajectory_episode_summary']['turn_ratio']['mean']:.1%}</td></tr>
        </tbody>
      </table>
    </div>
    <div class=\"card\">
      <h2>跨场景重复指令</h2>
      <table>
        <thead><tr><th>指令</th><th>场景s</th></tr></thead>
        <tbody>{multi_scene_rows or '<tr><td colspan="2">当前还没有跨多个场景重复出现的指令。</td></tr>'}</tbody>
      </table>
    </div>
    <div class=\"card\">
      <h2>跨目标重复指令</h2>
      <table>
        <thead><tr><th>指令</th><th>目标s</th></tr></thead>
        <tbody>{multi_target_rows or '<tr><td colspan="2">当前还没有跨多个目标重复出现的指令。</td></tr>'}</tbody>
      </table>
    </div>
  </div>
  <div class=\"card\">
    <table>
      <thead><tr><th>Episode</th><th>指令</th><th>采集模式</th><th>source_type</th><th>任务族</th><th>目标</th><th>左右</th><th>远近</th><th>帧数</th><th>Duration(s)</th><th>Action Changes</th><th>Stop Ratio</th><th>Turn Ratio</th><th>场景</th><th>操作员</th><th>D435i</th><th>警告</th><th>信息</th><th>当前分数</th><th>设置分数</th><th>备注</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
  <script>
    {review_script}
    const qualityFilter = document.getElementById('qualityFilter');
    const qualityScoreFilter = document.getElementById('qualityScoreFilter');
    const saveAllReviewsButton = document.getElementById('saveAllReviews');
    const reviewRows = Array.from(document.querySelectorAll('[data-review-row]'));
    const sessionFilter = document.getElementById('sessionFilter');
    const episodeFilter = document.getElementById('episodeFilter');
    const instructionFilter = document.getElementById('instructionFilter');
    const sceneFilter = document.getElementById('sceneFilter');
    const targetFilter = document.getElementById('targetFilter');
    const visibleEpisodeCount = document.getElementById('visibleEpisodeCount');

    function includesValue(haystack, needle) {{
      return String(haystack || '').toLowerCase().includes(String(needle || '').trim().toLowerCase());
    }}

    function renderRow(row) {{
      const sessionId = row.dataset.sessionId || '';
      const episodeId = row.dataset.episodeId || '';
      const review = getEpisodeReview(sessionId, episodeId);
      const badge = row.querySelector('[data-review-badge]');
      const select = row.querySelector('[data-review-select]');
      const note = row.querySelector('[data-review-note]');
      if (badge) {{
        badge.textContent = labelText(review.quality_label);
        badge.className = `quality-badge ${{labelClass(review.quality_label)}}`;
      }}
      if (select) {{
        select.value = String(review.quality_label);
      }}
      if (note) {{
        note.value = review.quality_notes;
      }}
      row.classList.toggle('quality-excluded', review.exclude_from_processing);
      row.classList.toggle('quality-review', !review.exclude_from_processing && review.quality_overridden);
    }}

    function updateReviewSummary() {{
      const counts = {{ 1: 0, 2: 0, 3: 0 }};
      let excluded = 0;
      let overridden = 0;
      episodeRefs.forEach((item) => {{
        const review = getEpisodeReview(item.session_id, item.episode_id);
        counts[review.quality_label] = (counts[review.quality_label] || 0) + 1;
        if (review.exclude_from_processing) {{
          excluded += 1;
        }}
        if (review.quality_overridden) {{
          overridden += 1;
        }}
      }});
      document.getElementById('summaryLabel1').textContent = String(counts[1] || 0);
      document.getElementById('summaryLabel2').textContent = String(counts[2] || 0);
      document.getElementById('summaryLabel3').textContent = String(counts[3] || 0);
      document.getElementById('summaryOverridden').textContent = String(overridden);
      document.getElementById('summaryExcluded').textContent = String(excluded);
    }}

    function applyRowFilter() {{
      const mode = qualityFilter.value;
      const scoreMode = qualityScoreFilter.value;
      const sessionNeedle = sessionFilter.value;
      const episodeNeedle = episodeFilter.value;
      const instructionNeedle = instructionFilter.value;
      const sceneNeedle = sceneFilter.value;
      const targetNeedle = targetFilter.value;
      let visibleCount = 0;
      reviewRows.forEach((row) => {{
        const sessionId = row.dataset.sessionId || '';
        const episodeId = row.dataset.episodeId || '';
        const review = getEpisodeReview(sessionId, episodeId);
        let visible = true;
        if (mode === 'excluded') {{
          visible = review.exclude_from_processing;
        }} else if (mode === 'overridden') {{
          visible = review.quality_overridden || Boolean(review.quality_notes);
        }} else if (mode === 'original') {{
          visible = !review.quality_overridden && !review.quality_notes;
        }}
        if (visible && scoreMode !== 'all') {{
          visible = String(review.quality_label) === scoreMode;
        }}
        if (visible) {{
          visible = includesValue(row.dataset.sessionId || '', sessionNeedle)
            && includesValue(row.dataset.episodeId || '', episodeNeedle)
            && includesValue(row.dataset.instruction || '', instructionNeedle)
            && includesValue(row.dataset.scene || '', sceneNeedle)
            && includesValue(row.dataset.target || '', targetNeedle);
        }}
        row.hidden = !visible;
        if (visible) {{
          visibleCount += 1;
        }}
      }});
      if (visibleEpisodeCount) {{
        visibleEpisodeCount.textContent = `${{visibleCount}} / ${{reviewRows.length}} episodes`;
      }}
    }}

    function saveRowReview(row) {{
      const sessionId = row.dataset.sessionId || '';
      const episodeId = row.dataset.episodeId || '';
      const select = row.querySelector('[data-review-select]');
      const note = row.querySelector('[data-review-note]');
      upsertEpisodeReview(sessionId, episodeId, Number(select && select.value) || 0, note ? note.value : '');
      renderRow(row);
      updateReviewSummary();
      applyRowFilter();
      announceStatus(`已暂存 ${{sessionId}}/${{episodeId}} 的修改，点击“保存到源文件”完成写回。`);
    }}

    function saveAllReviews() {{
      persistReviewState('已保存当前页面修改到源文件')
        .then(() => {{
          reviewRows.forEach(renderRow);
          updateReviewSummary();
          applyRowFilter();
        }})
        .catch((error) => announceStatus(`写回失败：${{error}}`, true));
    }}

    reviewRows.forEach((row) => {{
      renderRow(row);
      const select = row.querySelector('[data-review-select]');
      const note = row.querySelector('[data-review-note]');
      if (select) {{
        select.addEventListener('change', () => saveRowReview(row));
      }}
      if (note) {{
        note.addEventListener('change', () => saveRowReview(row));
      }}
    }});

    qualityFilter.addEventListener('change', applyRowFilter);
    qualityScoreFilter.addEventListener('change', applyRowFilter);
    [sessionFilter, episodeFilter, instructionFilter, sceneFilter, targetFilter].forEach((node) => {{
      node.addEventListener('input', applyRowFilter);
    }});
    saveAllReviewsButton.addEventListener('click', saveAllReviews);
    document.getElementById('bindReviewFile').addEventListener('click', () => {{
      bindReviewFile().catch((error) => announceStatus(`绑定文件失败：${{error}}`, true));
    }});
    document.getElementById('exportReview').addEventListener('click', () => {{
      exportReviewFile().catch((error) => announceStatus(`导出失败：${{error}}`, true));
    }});
    document.getElementById('resetAllReviews').addEventListener('click', () => {{
      clearReviewCache();
      reviewRows.forEach(renderRow);
      updateReviewSummary();
      applyRowFilter();
      announceStatus('已清空当前数据集的本地修改缓存。');
    }});
    document.getElementById('importReviewFile').addEventListener('change', async (event) => {{
      const [file] = Array.from(event.target.files || []);
      if (!file) {{
        return;
      }}
      try {{
        await importReviewFile(file);
        reviewRows.forEach(renderRow);
        updateReviewSummary();
        applyRowFilter();
        announceStatus('已导入修改记录，点击“保存到源文件”完成写回。');
      }} catch (error) {{
        announceStatus(`导入失败：${{error}}`, true);
      }} finally {{
        event.target.value = '';
      }}
    }});
    window.addEventListener('go2-review-dirty-changed', (event) => {{
      const detail = event.detail || {{}};
      const dirty = Boolean(detail.dirty);
      const saving = Boolean(detail.saving);
      saveAllReviewsButton.disabled = saving || !dirty;
      saveAllReviewsButton.textContent = saving ? '正在保存...' : '保存到源文件';
    }});
    setReviewDirty(reviewDirty);
    window.addEventListener('go2-review-state-updated', () => {{
      reviewRows.forEach(renderRow);
      updateReviewSummary();
      applyRowFilter();
    }});

    const query = new URLSearchParams(window.location.search);
    if (query.has('session_id')) sessionFilter.value = query.get('session_id') || '';
    if (query.has('episode_id')) episodeFilter.value = query.get('episode_id') || '';
    if (query.has('instruction')) instructionFilter.value = query.get('instruction') || '';
    if (query.has('scene_id')) sceneFilter.value = query.get('scene_id') || '';
    if (query.has('target')) targetFilter.value = query.get('target') || '';
    if (query.has('quality_mode')) qualityFilter.value = query.get('quality_mode') || 'all';
    if (query.has('quality_score')) qualityScoreFilter.value = query.get('quality_score') || 'all';

    updateReviewSummary();
    applyRowFilter();
  </script>
</body>
</html>
"""
    index_path.write_text(index_html, encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir.resolve()
    if args.mode == "pipeline_hub":
        index_path = write_pipeline_hub(args, output_dir)
        print(f"已生成 pipeline hub：{index_path}")
        return

    if args.mode == "raw":
        if args.data_root is None:
            raise ValueError("--mode raw 时必须提供 --data-root")
        data_root = args.data_root.resolve()
        episodes = load_episodes(data_root, args.max_episodes, None)
        write_reports(episodes, output_dir, data_root, args.seed, args.num_samples)
        print(f"已为 {len(episodes)} 条 episode 生成体检报告，输出目录：{output_dir}")
        if args.serve:
            serve_report(output_dir, data_root, args.host, args.port)
        return

    if args.windows_root is None:
        raise ValueError("--mode processed_windows 时必须提供 --windows-root")
    windows_root = args.windows_root.resolve()
    filtered_root = args.filtered_root.resolve() if args.filtered_root is not None else None
    canonical_phase_root = args.canonical_phase_root.resolve() if args.canonical_phase_root is not None else None
    samples = load_processed_windows(windows_root, args.max_samples, canonical_phase_root)
    write_processed_reports(samples, output_dir, windows_root, filtered_root, canonical_phase_root)
    print(f"已为 {len(samples)} 条 processed windows 样本生成体检报告，输出目录：{output_dir}")
    if args.serve:
        serve_report(
            output_dir,
            windows_root,
            args.host,
            args.port,
            extra_file_roots=processed_file_roots(samples, windows_root),
        )


if __name__ == "__main__":
    main()
