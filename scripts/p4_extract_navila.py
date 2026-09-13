#!/usr/bin/env python3
"""Reproducible P4-T1 extractor/auditor (no command-line arguments)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path("/home/wxh/go2_short_vln")
MNT = Path("/mnt/wxh/go2_short_vln")
SRC = MNT / "downloads/navila_probe/R2R_train.tar.gz"
R2R = MNT / "data/navila_dataset/R2R"
TRAIN = R2R / "train"
INDEX = R2R / "index/t1_per_video_frames.json"
MANIFEST = PROJECT / "reports/p4/t1_extract_manifest.json"
REPORT = PROJECT / "reports/p4/T1_EXTRACT.md"
EXPECTED_SHA = "49ffc4f88e15ae4bdd2a8f01b16a8685deee0bd4d05711998d9ae88a99d4d228"
EXPECTED_BYTES = 19703787520
EXPECTED_JPG = 601125
EXPECTED_DIRS = 10819
FRAME_RE = re.compile(r"^frame_(\d+)\.jpg$")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def probe() -> dict:
    p = MNT / f".probe_t1_{os.getpid()}"
    payload = b"p4-t1-write-probe\n"
    try:
        with p.open("wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        n = p.stat().st_size
        p.unlink()
        return {"status": "success", "path": str(p), "bytes_written": n, "removed": True}
    except OSError as e:
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass
        return {"status": "failure", "path": str(p), "errno": e.errno, "error": str(e), "removed": not p.exists()}


def quarantine() -> dict:
    src = R2R / "annotations.json"
    dst = R2R / "annotations.INCOMPLETE-184991744B.json.quarantine"
    if src.exists() and dst.exists():
        raise RuntimeError("both annotations.json and quarantine file exist")
    if src.exists():
        if src.stat().st_size != 184_991_744:
            raise RuntimeError("annotations.json is not the expected truncated copy")
        src.rename(dst)
        action = "renamed"
    elif dst.exists():
        action = "already_quarantined"
    else:
        raise FileNotFoundError(src)
    size = dst.stat().st_size
    if size != 184_991_744:
        raise RuntimeError(f"quarantine size changed: {size}")
    return {"action": action, "path": str(dst), "size_bytes": size}


def quick_counts() -> tuple[int, int]:
    if not TRAIN.is_dir():
        return 0, 0
    dirs = 0
    jpg = 0
    for d in TRAIN.iterdir():
        if d.is_dir():
            dirs += 1
            jpg += sum(1 for p in d.iterdir() if p.is_file() and p.name.endswith(".jpg"))
    return jpg, dirs


def extract() -> tuple[str, float]:
    if quick_counts() == (EXPECTED_JPG, EXPECTED_DIRS):
        return "skipped_existing_complete", 0.0
    R2R.mkdir(parents=True, exist_ok=True)
    t = time.monotonic()
    # Plain GNU/POSIX tar; deliberately no -z.
    subprocess.run(["tar", "-xf", str(SRC), "-C", str(R2R)], check=True)
    return "extracted", time.monotonic() - t


def inventory() -> tuple[list[dict], list[str], int]:
    videos: list[dict] = []
    files: list[str] = []
    total_bytes = 0
    for d in sorted((p for p in TRAIN.iterdir() if p.is_dir()), key=lambda p: p.name):
        indices: list[int] = []
        vbytes = 0
        for p in d.iterdir():
            if p.is_file() and p.name.endswith(".jpg"):
                files.append(p.relative_to(MNT).as_posix())
                size = p.stat().st_size
                total_bytes += size
                vbytes += size
                m = FRAME_RE.match(p.name)
                if m:
                    indices.append(int(m.group(1)))
        indices.sort()
        videos.append({"video": d.name, "frame_count": len(indices), "first_index": indices[0] if indices else None,
                       "last_index": indices[-1] if indices else None,
                       "contiguous": bool(indices) and indices == list(range(indices[0], indices[-1] + 1)),
                       "bytes": vbytes})
    files.sort()
    return videos, files, total_bytes


def set_check(files: list[str], videos: list[dict]) -> dict:
    proc = subprocess.Popen(["tar", "-tf", str(SRC)], stdout=subprocess.PIPE, text=True)
    tar_files: set[str] = set()
    tar_dirs: set[str] = set()
    assert proc.stdout is not None
    for line in proc.stdout:
        name = line.rstrip("\n")
        if name == "train/":
            continue
        if name.endswith("/"):
            tar_dirs.add(f"data/navila_dataset/R2R/{name.rstrip('/')}")
        elif name.endswith(".jpg"):
            tar_files.add(f"data/navila_dataset/R2R/{name}")
    rc = proc.wait()
    if rc:
        raise RuntimeError(f"tar -tf failed ({rc})")
    disk_files = set(files)
    disk_dirs = {f"data/navila_dataset/R2R/train/{v['video']}" for v in videos}
    return {"tar_file_count": len(tar_files), "disk_file_count": len(disk_files),
            "missing_files": sorted(tar_files - disk_files), "extra_files": sorted(disk_files - tar_files),
            "tar_dir_count": len(tar_dirs), "disk_dir_count": len(disk_dirs),
            "missing_dirs": sorted(tar_dirs - disk_dirs), "extra_dirs": sorted(disk_dirs - tar_dirs)}


def samples(files: list[str]) -> tuple[list[dict], dict]:
    n = len(files)
    rule = {"description": "Sort all MNT-relative .jpg paths bytewise in POSIX lexical order; choose floor(i*(N-1)/63), i=0..63.",
            "ordering": "Python str.sort() over POSIX paths", "formula": "floor(i*(N-1)/63)", "count": 64}
    out = []
    for i in range(64):
        j = (i * (n - 1)) // 63
        p = MNT / files[j]
        out.append({"ordinal": i, "sorted_index": j, "path": files[j], "sha256": sha256(p), "bytes": p.stat().st_size})
    return out, rule


def main() -> None:
    started = time.monotonic()
    pr = probe()
    if pr["status"] != "success":
        raise OSError(pr.get("errno"), pr.get("error"), pr.get("path"))
    if SRC.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError("input byte count mismatch")
    actual = sha256(SRC)
    if actual != EXPECTED_SHA:
        raise RuntimeError("input SHA-256 mismatch")
    q = quarantine()
    mode, extract_seconds = extract()
    videos, files, total_bytes = inventory()
    if len(files) != EXPECTED_JPG or len(videos) != EXPECTED_DIRS:
        raise RuntimeError(f"count mismatch: jpg={len(files)}, dirs={len(videos)}")
    check = set_check(files, videos)
    if any(check[k] for k in ("missing_files", "extra_files", "missing_dirs", "extra_dirs")):
        raise RuntimeError("tar/disk set difference is non-empty")
    smp, rule = samples(files)
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps({"task_id": "P4-T1", "generated_at": now(), "videos": videos}, separators=(",", ":")) + "\n")
    du = shutil.disk_usage(TRAIN)
    train_disk_bytes = int(subprocess.check_output(["du", "-sb", str(TRAIN)], text=True).split()[0])
    manifest = {"task_id": "P4-T1", "generated_at": now(), "generator_command": "python3 scripts/p4_extract_navila.py",
                "inputs": {"path": str(SRC), "sha256": actual, "bytes": SRC.stat().st_size}, "probe": pr, "quarantine": q,
                "extraction": {"mode": mode, "extract_seconds": round(extract_seconds, 3), "elapsed_seconds": round(time.monotonic() - started, 3)},
                "counts": {"jpg": len(files), "video_directories": len(videos), "jpg_total_bytes": total_bytes},
                "disk_usage": {"train_logical_bytes": total_bytes, "train_disk_bytes": train_disk_bytes, "filesystem_free_bytes": du.free},
                "tar_disk_set_check": check, "sampling": rule, "samples": smp, "per_video_index": str(INDEX)}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    anomalies = sum(1 for v in videos if not v["contiguous"])
    sample_rows = "\n".join(f"| {s['ordinal']} | {s['sorted_index']} | `{s['path']}` | {s['bytes']} | `{s['sha256']}` |" for s in smp)
    REPORT.write_text(
        f"# P4-T1 解包摘要\n\n"
        f"- JPEG：{len(files):,}\n- video 目录：{len(videos):,}\n- JPEG 字节合计：{total_bytes:,}\n"
        f"- `train/` 磁盘占用（du -sb）：{train_disk_bytes:,} 字节；文件系统剩余：{du.free:,} 字节\n"
        f"- 解包模式：`{mode}`；耗时 {extract_seconds:.3f} 秒；脚本总耗时 {manifest['extraction']['elapsed_seconds']:.3f} 秒\n"
        f"- tar↔磁盘双向差集：文件/目录均为空\n- 抽样规则：{rule['description']}\n"
        f"- 隔离文件：`{q['path']}`，{q['size_bytes']:,} 字节（{q['action']}）\n"
        f"- 帧连续性异常：{anomalies}\n- 写权限探针：success，写入 {pr['bytes_written']} 字节后删除\n\n"
        "## 64 个确定性抽样哈希\n\n| ordinal | sorted_index | path | bytes | sha256 |\n|---:|---:|---|---:|---|\n"
        + sample_rows + "\n\n## 未完成项\n\n- 无（MISSING：无）\n"
    )


if __name__ == "__main__":
    main()
