"""CPU-only lazy R2R index; every JSONL row (including oversampling) is kept."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.smolvla.phase1_prompt import navila_history_layout


class NavilaR2RDataset(Dataset):
    """Eight history frames chosen by NaVILA's own rule (navila_history_layout).

    Short histories are front-padded with black frames, exactly as NaVILA does in
    training and evaluation; the last frame is always the current one. Byte-offset
    access avoids materializing all records or image paths. File handles are
    reopened per process, including after fork, and omitted when pickling.
    ``load_frames=False`` supports precomputed features without JPEG decoding.
    """

    def __init__(self, index_path: str | Path, frames_root: str | Path,
                 history_frames: int = 8, *, load_frames: bool = True,
                 exclude_videos: set[str] | None = None) -> None:
        if history_frames != 8:
            raise ValueError("P4-T6v2 freezes history_frames=8")
        self.index_path = Path(index_path)
        self.frames_root = Path(frames_root)
        self.history_frames = history_frames
        self.load_frames = load_frames
        self._offsets: list[int] = []
        with self.index_path.open("rb") as fh:
            while True:
                pos = fh.tell()
                line = fh.readline()
                if not line:
                    break
                if line.strip():
                    if exclude_videos and str(json.loads(line)["video"]) in exclude_videos:
                        continue
                    self._offsets.append(pos)
        self._fh = None
        self._pid = None

    def __len__(self) -> int:
        return len(self._offsets)

    def close(self) -> None:
        if getattr(self, "_fh", None) is not None:
            self._fh.close()
        self._fh = None

    def __getstate__(self):
        return {**self.__dict__, "_fh": None, "_pid": None}

    def __del__(self):
        self.close()

    def _read_record(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self._fh is None or self._fh.closed or self._pid != os.getpid():
            self.close()
            self._fh = self.index_path.open("rb")
            self._pid = os.getpid()
        self._fh.seek(self._offsets[index])
        return json.loads(self._fh.readline())

    @staticmethod
    def _history_indices(n_frames: int, count: int = 8) -> list[int | None]:
        return navila_history_layout(n_frames, count)

    def frame_paths(self, record: dict[str, Any]) -> list[Path | None]:
        """Paths of the eight history frames; None where NaVILA pads with black."""
        video = str(record["video"])
        if not video or Path(video).name != video or video in {".", ".."}:
            raise ValueError("video must be a single directory name")
        n_frames = int(record["n_frames"])
        indices = self._history_indices(n_frames, self.history_frames)
        for key, expected in (("frames_first", f"{video}/frame_0.jpg"),
                              ("frames_last", f"{video}/frame_{n_frames - 1}.jpg")):
            if key in record and record[key] != expected:
                raise ValueError(f"{key} disagrees with index reconstruction")
        return [None if i is None else self.frames_root / video / f"frame_{i}.jpg" for i in indices]

    @staticmethod
    def read_frame(path: str | Path) -> torch.Tensor:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            frame = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
            return frame.view(rgb.height, rgb.width, 3).permute(2, 0, 1).contiguous()

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._read_record(index)
        paths = self.frame_paths(record)
        item = {
            "instruction_raw": record["instruction_raw"],
            "instruction": record["instruction_raw"],
            "action_text": record["action_text"],
            "video_id": record["video_id"], "video": record["video"],
            "step": int(record["step"]),
            # "" rather than None marks padding so torch's default_collate still works
            "frame_paths": ["" if p is None else str(p) for p in paths],
            "history_padding_mask": [p is None for p in paths],
            "record": record,
        }
        if self.load_frames:
            current = self.read_frame(paths[-1])          # last is never padding
            item["frames"] = torch.stack([current if i == len(paths) - 1
                                          else torch.zeros_like(current) if p is None
                                          else self.read_frame(p)
                                          for i, p in enumerate(paths)])
        return item


NavilaR2R = NavilaR2RDataset
