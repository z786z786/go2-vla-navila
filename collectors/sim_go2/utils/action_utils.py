from __future__ import annotations

from typing import Sequence


def clip_train_action(action: Sequence[float], vx_min: float, vx_max: float, wz_min: float, wz_max: float) -> list[float]:
    return [
        max(vx_min, min(vx_max, float(action[0]))),
        max(wz_min, min(wz_max, float(action[1]))),
    ]


def raw_full_from_train_action(train_action: Sequence[float], default_vy: float = 0.0) -> list[float]:
    return [float(train_action[0]), float(default_vy), float(train_action[1])]


def build_action_chunk(actions: Sequence[Sequence[float]], start_index: int, chunk_len: int, pad_action: Sequence[float] | None = None) -> tuple[list[list[float]], list[int]]:
    if pad_action is None:
        pad_action = (0.0, 0.0)
    chunk: list[list[float]] = []
    mask: list[int] = []
    for index in range(start_index, start_index + int(chunk_len)):
        if index < len(actions):
            chunk.append([float(actions[index][0]), float(actions[index][1])])
            mask.append(1)
        else:
            chunk.append([float(pad_action[0]), float(pad_action[1])])
            mask.append(0)
    return chunk, mask


def is_valid_action_mask(mask: Sequence[int]) -> bool:
    seen_zero = False
    for value in mask:
        if value not in (0, 1):
            return False
        if value == 0:
            seen_zero = True
        elif seen_zero:
            return False
    return True
