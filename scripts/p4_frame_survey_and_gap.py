#!/mnt/wxh/go2_short_vln/envs/conda/smolvla/bin/python
"""P4-T4 NaVILA frame survey and paired habitat/Isaac image gap analysis.

Interpreter: /mnt/wxh/go2_short_vln/envs/conda/smolvla/bin/python
CPU-only, deterministic selection and measurements.  The script takes no
arguments and writes the three reports required by P4-T4 plus PNG evidence on
the /mnt data disk.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/home/wxh/go2_short_vln")
NAVILA_TRAIN = Path("/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/train")
GO2_ROOT = ROOT / "outputs/r2r_ablation"
REPORT_DIR = ROOT / "reports/p4"
PAIR_DIR = Path("/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/index/t4_pairs")
TASK_ID = "P4-T4"
FRAME_RE = re.compile(r"^frame_(\d+)\.jpg$", re.IGNORECASE)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dir_fingerprint(path: Path, suffix: str | None = None) -> dict:
    """Hash sorted relative path + byte size, without reading image payloads."""
    h = hashlib.sha256()
    count = 0
    total = 0
    pattern = "*.jpg" if suffix == ".jpg" else "*"
    files = [p for p in path.rglob(pattern) if p.is_file()]
    for p in sorted(files, key=lambda x: x.relative_to(path).as_posix()):
        rel = p.relative_to(path).as_posix()
        size = p.stat().st_size
        h.update(f"{rel}\0{size}\n".encode())
        count += 1
        total += size
    return {"path": str(path), "file_count": count, "total_bytes": total, "sha256_listing": h.hexdigest()}


def quantiles(values: list[int | float]) -> dict:
    if not values:
        return {k: None for k in ("min", "p25", "p50", "p75", "p90", "max")}
    a = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(a)),
        "p25": float(np.quantile(a, 0.25, method="linear")),
        "p50": float(np.quantile(a, 0.50, method="linear")),
        "p75": float(np.quantile(a, 0.75, method="linear")),
        "p90": float(np.quantile(a, 0.90, method="linear")),
        "max": float(np.max(a)),
    }


def survey_navila() -> tuple[dict, dict[str, list[Path]]]:
    modes = Counter()
    resolutions = Counter()
    sizes: list[int] = []
    damaged: list[dict] = []
    frames_by_ep: dict[str, list[Path]] = {}
    ep_dirs = sorted((p for p in NAVILA_TRAIN.iterdir() if p.is_dir()), key=lambda p: p.name)
    for ep_dir in ep_dirs:
        fs = []
        for p in ep_dir.iterdir():
            m = FRAME_RE.match(p.name)
            if m and p.is_file():
                fs.append((int(m.group(1)), p))
        fs.sort(key=lambda x: x[0])
        frames_by_ep[ep_dir.name] = [p for _, p in fs]
        for _, p in fs:
            try:
                sizes.append(p.stat().st_size)
                with Image.open(p) as im:
                    resolutions[f"{im.size[0]}x{im.size[1]}"] += 1
                    modes[im.mode] += 1
                    # verify reads all JPEG structure without retaining pixels.
                    im.verify()
            except Exception as exc:  # noqa: BLE001 - report every bad asset
                damaged.append({"path": str(p), "error": f"{type(exc).__name__}: {exc}"})
    result = {
        "task_id": TASK_ID,
        "generated_at": now_iso(),
        "generator_command": f"{sys.executable} {Path(__file__).resolve()}",
        "inputs": {
            "navila_train": dir_fingerprint(NAVILA_TRAIN, ".jpg"),
        },
        "frame_count": len(sizes),
        "episode_directory_count": len(ep_dirs),
        "resolution_distribution": dict(sorted(resolutions.items())),
        "resolution_exceptions": [
            {"resolution": k, "count": v} for k, v in sorted(resolutions.items()) if k != "512x512"
        ],
        "color_mode_distribution": dict(sorted(modes.items())),
        "file_size_bytes_quantiles": quantiles(sizes),
        "damaged_or_unreadable": damaged,
        "damage_check": {
            "method": "PIL.Image.open followed by Image.verify for every JPG; no pixel array retained",
            "coverage_fraction": 1.0,
        },
        "missing": [] if not damaged else ["damaged_or_unreadable_files_present"],
    }
    return result, frames_by_ep


def read_manifest(d: Path) -> tuple[str | None, str | None]:
    for name in ("manifest.json", "preflight.json", "metadata.json"):
        p = d / name
        if p.exists():
            try:
                obj = json.loads(p.read_text())
                ep = obj.get("episode_id")
                scene = obj.get("scene_id")
                if ep is not None:
                    return str(ep), scene
            except Exception:
                pass
    m = re.search(r"ep(\d+)(?:_|$)", d.name)
    return (m.group(1) if m else None), None


def discover_go2() -> tuple[dict[str, dict], list[dict]]:
    candidates: dict[str, list[dict]] = defaultdict(list)
    all_rgb_dirs = []
    for d in sorted(GO2_ROOT.iterdir(), key=lambda p: p.name):
        rgb = d / "rgb"
        if not rgb.is_dir():
            continue
        files = sorted((p for p in rgb.glob("*.jpg") if p.is_file()), key=lambda p: p.name)
        ep, scene = read_manifest(d)
        rec = {"episode_id": ep, "scene_id": scene, "run_dir": d, "rgb_dir": rgb, "frame_count": len(files)}
        all_rgb_dirs.append(rec)
        if ep is not None and files:
            candidates[ep].append(rec)
    chosen = {}
    for ep, rows in candidates.items():
        # Explicit reproducible rule required by T4: max frame count, then name.
        chosen[ep] = sorted(rows, key=lambda r: (-r["frame_count"], r["run_dir"].name))[0]
    return chosen, all_rgb_dirs


def numeric_go2_frames(rgb_dir: Path) -> list[Path]:
    def key(p: Path):
        m = re.search(r"(\d+)", p.stem)
        return (int(m.group(1)) if m else 10**12, p.name)
    return sorted((p for p in rgb_dir.glob("*.jpg") if p.is_file()), key=key)


def sample_positions(n: int, k: int = 8) -> list[int]:
    if n <= 0:
        return []
    if n == 1:
        return [0]
    return sorted({int(math.floor(i * (n - 1) / (k - 1))) for i in range(k)})


def image_metrics(path: Path) -> dict:
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Physical FOV cannot be recovered from an RGB frame.  This is explicitly
    # an image-space texture/scale proxy, not a camera-angle estimate.
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    edges = cv2.Canny(gray, 50, 150)
    horizon = estimate_horizon_row(gray)
    return {
        "brightness_mean": float(np.mean(gray)),
        "contrast_std": float(np.std(gray)),
        "saturation_mean": float(np.mean(hsv[:, :, 1]) / 255.0),
        "edge_density": float(np.mean(edges > 0)),
        "image_scale_proxy_laplacian": float(np.mean(np.abs(lap)) / 255.0),
        "horizon_row": horizon,
        "horizon_row_normalized": (float(horizon / gray.shape[0]) if horizon is not None else None),
    }


def estimate_horizon_row(gray: np.ndarray) -> int | None:
    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=max(25, int(w * 0.08)),
                            minLineLength=max(60, int(w * 0.20)), maxLineGap=12)
    candidates = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            dx, dy = int(x2 - x1), int(y2 - y1)
            length = float(math.hypot(dx, dy))
            if length < w * 0.20 or abs(math.degrees(math.atan2(dy, dx))) > 8.0:
                continue
            y = (int(y1) + int(y2)) / 2.0
            if 0.05 * h <= y <= 0.95 * h:
                candidates.append((y, length))
    if not candidates:
        return None
    candidates.sort()
    total = sum(weight for _, weight in candidates)
    acc = 0.0
    for y, weight in candidates:
        acc += weight
        if acc >= total / 2.0:
            return int(round(y))
    return int(round(candidates[-1][0]))


def aggregate(rows: list[dict]) -> dict:
    out = {}
    for key in ("brightness_mean", "contrast_std", "saturation_mean", "edge_density",
                "image_scale_proxy_laplacian", "horizon_row_normalized"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        out[key] = {
            "mean": float(np.mean(vals)) if vals else None,
            "median": float(np.median(vals)) if vals else None,
            "min": float(np.min(vals)) if vals else None,
            "max": float(np.max(vals)) if vals else None,
            "sample_count": len(vals),
        }
    return out


def diff_aggregate(a: dict, b: dict) -> dict:
    out = {}
    for key, av in a.items():
        bv = b.get(key, {})
        am, bm = av.get("mean"), bv.get("mean")
        out[key] = (float(am - bm) if am is not None and bm is not None else None)
    return out


def render_pair(nav_path: Path, go_path: Path, ep: str, scene: str | None, nav_idx: int, go_idx: int, out: Path) -> None:
    with Image.open(nav_path) as n:
        n = n.convert("RGB")
    with Image.open(go_path) as g:
        g = g.convert("RGB")
    target_h = 512
    n = n.resize((512, target_h), Image.Resampling.LANCZOS)
    g = g.resize((512, target_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (1024, 570), "white")
    canvas.paste(n, (0, 38)); canvas.paste(g, (512, 38))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        small = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        font = small = ImageFont.load_default()
    draw.text((8, 8), f"episode_id={ep}  scene={scene or 'UNKNOWN'}", fill="black", font=font)
    draw.text((8, 23), f"NaVILA | frame_{nav_idx}.jpg", fill="black", font=small)
    draw.text((520, 23), f"Go2 Isaac | frame_{go_idx:06d}.jpg", fill="black", font=small)
    draw.line((511, 38, 511, 550), fill=(220, 0, 0), width=2)
    canvas.save(out, "PNG", optimize=False)


def paired_analysis(frames_by_ep: dict[str, list[Path]], chosen: dict[str, dict], all_rgb_dirs: list[dict]) -> dict:
    paired = []
    for ep in sorted(set(frames_by_ep) & set(chosen), key=lambda x: int(x) if x.isdigit() else x):
        nav = frames_by_ep[ep]
        go = numeric_go2_frames(chosen[ep]["rgb_dir"])
        if not nav or not go:
            continue
        nav_pos = sample_positions(len(nav), 8)
        go_pos = sample_positions(len(go), 8)
        nrows, grows, samples = [], [], []
        for np_i, gp_i in zip(nav_pos, go_pos):
            nr = image_metrics(nav[np_i]); gr = image_metrics(go[gp_i])
            nrows.append(nr); grows.append(gr)
            samples.append({"navila_frame_index": int(re.search(r"(\d+)", nav[np_i].stem).group(1)),
                            "go2_frame_index": int(re.search(r"(\d+)", go[gp_i].stem).group(1)),
                            "navila": nr, "go2": gr})
        na = aggregate(nrows); ga = aggregate(grows)
        paired.append({
            "episode_id": ep,
            "scene_id": chosen[ep].get("scene_id"),
            "navila_frame_count": len(nav),
            "go2_frame_count": len(go),
            "go2_run_dir": str(chosen[ep]["run_dir"]),
            "sample_rule": "8 normalized trajectory positions floor(i*(N-1)/7), i=0..7; same rank on both sides",
            "samples": samples,
            "navila_aggregate": na,
            "go2_aggregate": ga,
            "paired_mean_difference_navila_minus_go2": diff_aggregate(na, ga),
        })

    # At least 8 fixed visual evidence images: first 8 numeric episode IDs.
    PAIR_DIR.mkdir(parents=True, exist_ok=True)
    for old in sorted(PAIR_DIR.glob("p4_t4_*.png")):
        old.unlink()
    for item in paired[:8]:
        ep = item["episode_id"]
        nav = frames_by_ep[ep]
        go = numeric_go2_frames(chosen[ep]["rgb_dir"])
        ni = int(re.search(r"(\d+)", nav[0].stem).group(1))
        gi = int(re.search(r"(\d+)", go[0].stem).group(1))
        render_pair(nav[0], go[0], ep, item.get("scene_id"), ni, gi, PAIR_DIR / f"p4_t4_ep{ep}.png")

    navila_count = sum(1 for r in all_rgb_dirs if r["episode_id"] is not None and r["frame_count"] > 0)
    duplicate_eps = {ep: sum(1 for r in all_rgb_dirs if r["episode_id"] == ep and r["frame_count"] > 0)
                     for ep in sorted({r["episode_id"] for r in all_rgb_dirs if r["episode_id"] and r["frame_count"] > 0})}
    missing = []
    if len(paired) != 62:
        missing.append(f"expected_62_pairs_but_found_{len(paired)}")
    return {
        "task_id": TASK_ID,
        "generated_at": now_iso(),
        "generator_command": f"{sys.executable} {Path(__file__).resolve()}",
        "inputs": {
            "navila_train": dir_fingerprint(NAVILA_TRAIN, ".jpg"),
            "go2_root": {"path": str(GO2_ROOT), "rgb_directory_count": len(all_rgb_dirs),
                         "nonempty_rgb_directory_count": navila_count,
                         "directory_fingerprint": dir_fingerprint(GO2_ROOT, ".jpg")},
        },
        "go2_directory_selection_rule": "per episode_id, choose non-empty rgb directory with maximum JPG count; ties by lexicographically smallest run directory name",
        "go2_nonempty_directory_count": navila_count,
        "go2_unique_episode_count": len(chosen),
        "duplicate_episode_directory_counts": {k: v for k, v in duplicate_eps.items() if v > 1},
        "pair_count": len(paired),
        "paired_episode_ids": [x["episode_id"] for x in paired],
        "paired_episodes": paired,
        "known_go2_camera": {"resolution": [512, 512], "hfov_deg": 96.7329,
                             "world_height_m_mean": 0.815, "optical_axis_pitch_deg_range": [-2.15, 4.22],
                             "source": "go2_matterport_base_cfg.py:301-307 (dispatch-provided, directly cited)"},
        "methods": {
            "fov_difference": {"status": "UNVERIFIABLE", "reason": "RGB pairs have no calibration, depth, known geometric landmarks, or NaVILA intrinsics; a physical degree FOV cannot be recovered from the images alone.",
                               "image_scale_proxy": "mean(abs(Laplacian(gray)))/255 over each sampled image; image-space texture proxy only, not a FOV estimate."},
            "horizon_optical_axis": {"method": "Canny(50,150) then HoughLinesP; retain lines >=20% width and <=8° from horizontal; length-weighted median midpoint row, normalized by image height.",
                                     "status": "PROXY_ONLY", "reason": "Detected horizontal structures are not guaranteed to be the geometric horizon; without scene geometry/intrinsics, optical-axis pitch in degrees remains UNVERIFIABLE."},
            "low_order_statistics": "Brightness=mean grayscale/255 scale restored to 0-255; contrast=grayscale std; saturation=mean HSV S/255; edge density=Canny pixel fraction; all aggregated over 8 normalized positions per episode.",
            "camera_height": {"status": "UNVERIFIABLE", "reason": "No NaVILA camera metadata, depth, metric scene-plane correspondences, or calibrated horizon/vanishing-point geometry is available. Same episode/start does not identify camera height from RGB alone; Go2 0.815 m and R2R-CE ~1.25 m are references only and are not used to assert a NaVILA value."},
            "visual_evidence": "p4_t4_ep<episode_id>.png for the first 8 paired episode IDs sorted numerically; frame index 0 on each side; fixed left=NaVILA, right=Go2 Isaac."
        },
        "missing": missing,
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    survey, frames_by_ep = survey_navila()
    chosen, all_rgb_dirs = discover_go2()
    gap = paired_analysis(frames_by_ep, chosen, all_rgb_dirs)
    (REPORT_DIR / "t4_frame_survey.json").write_text(json.dumps(survey, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    (REPORT_DIR / "t4_render_gap.json").write_text(json.dumps(gap, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"survey_frames": survey["frame_count"], "pairs": gap["pair_count"], "damaged": len(survey["damaged_or_unreadable"]), "missing": gap["missing"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
