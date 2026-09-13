"""Small dependency-free state serialization helpers for the M4/D5 collector."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def serialize_batched_euler_xyz(components: Sequence[Any]) -> list[float]:
    """Return a finite XYZ Euler vector from a batch-one Isaac tuple."""
    if len(components) != 3:
        raise ValueError("Euler output must contain three components")
    result: list[float] = []
    for component in components:
        if not hasattr(component, "__len__") or len(component) != 1:
            raise ValueError("Euler components must contain one single batch value")
        try:
            value = component[0]
            scalar = float(value.item()) if hasattr(value, "item") else float(value)
        except (TypeError, ValueError, IndexError) as error:
            raise ValueError("Euler components must contain one numeric batch value") from error
        if not math.isfinite(scalar):
            raise ValueError("Euler components must be finite")
        result.append(scalar)
    return result
