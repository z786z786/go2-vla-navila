from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class JsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def write(self, row: Mapping[str, Any]) -> None:
        self._handle.write(json.dumps(dict(row), ensure_ascii=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()
