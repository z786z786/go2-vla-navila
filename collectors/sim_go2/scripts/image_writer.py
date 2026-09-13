from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def to_uint8_rgb(image: Any) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected HWC image tensor, got shape {array.shape!r}")
    if array.shape[-1] > 3:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        max_value = float(array.max()) if array.size else 0.0
        if max_value <= 1.0 + 1.0e-6:
            array = np.clip(array, 0.0, 1.0) * 255.0
        else:
            array = np.clip(array, 0.0, 255.0)
    return np.ascontiguousarray(array.astype(np.uint8))


def save_rgb_image(image: Any, path: Path | str, image_format: str | None = None, jpeg_quality: int = 95) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    pil_image = Image.fromarray(to_uint8_rgb(image), mode="RGB")
    fmt = (image_format or path_obj.suffix.lstrip(".") or "png").upper()
    save_kwargs: dict[str, Any] = {}
    if fmt in {"JPG", "JPEG"}:
        fmt = "JPEG"
        save_kwargs["quality"] = int(jpeg_quality)
        save_kwargs["subsampling"] = 0
    pil_image.save(path_obj, format=fmt, **save_kwargs)


def to_float32_depth(depth: Any) -> np.ndarray:
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()
    array = np.asarray(depth)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected HW depth tensor, got shape {array.shape!r}")
    return np.ascontiguousarray(array.astype(np.float32))


def save_depth_image(
    depth: Any,
    path: Path | str,
    *,
    image_format: str | None = None,
    depth_scale: float = 1000.0,
    invalid_fill_value: float = 0.0,
    depth_min: float | None = None,
    depth_max: float | None = None,
) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    depth_array = to_float32_depth(depth)
    finite_mask = np.isfinite(depth_array)
    if depth_min is not None:
        finite_mask &= depth_array >= float(depth_min)
    if depth_max is not None:
        finite_mask &= depth_array <= float(depth_max)
    sanitized = np.where(finite_mask, depth_array, float(invalid_fill_value)).astype(np.float32, copy=False)

    fmt = (image_format or path_obj.suffix.lstrip(".") or "npy").lower()
    if fmt == "npy":
        np.save(path_obj, sanitized)
        return
    if fmt not in {"png"}:
        raise ValueError(f"Unsupported depth image format: {fmt}")

    scaled = np.clip(np.round(sanitized * float(depth_scale)), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    Image.fromarray(scaled).save(path_obj, format="PNG")
