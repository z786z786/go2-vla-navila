from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    yaml = None
    YAML_IMPORT_ERROR = exc
else:
    YAML_IMPORT_ERROR = None


CONFIG_REF_KEYS = (
    "scene_config",
    "robot_config",
    "domain_randomization_config",
    "packing_config",
)


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def ensure_symlink_or_copytree(link_path: Path, source_dir: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        return
    try:
        link_path.symlink_to(source_dir.resolve(), target_is_directory=True)
    except OSError:
        import shutil

        shutil.copytree(source_dir, link_path)


def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required for sim_data_collection configs.") from YAML_IMPORT_ERROR


def load_yaml(path: str | Path) -> dict[str, Any]:
    _require_yaml()
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping YAML at {path_obj}, got {type(payload)!r}")
    return payload


def dump_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    _require_yaml()
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def dump_json(path: str | Path, payload: Any) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def set_nested_value(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = mapping
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def load_resolved_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    collector_cfg = load_yaml(config_path)
    resolved: dict[str, Any] = {}
    sources: dict[str, str] = {"collector": str(config_path)}
    for ref_key in CONFIG_REF_KEYS:
        ref_value = collector_cfg.pop(ref_key, None)
        if not ref_value:
            continue
        referenced = (config_path.parent / str(ref_value)).resolve()
        deep_update(resolved, load_yaml(referenced))
        sources[ref_key.removesuffix("_config")] = str(referenced)
    deep_update(resolved, collector_cfg)
    resolved.setdefault("config_sources", {}).update(sources)
    return resolved


def apply_cli_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    key_map = {
        "runtime.output_dir": overrides.get("output_dir"),
        "runtime.num_episodes": overrides.get("num_episodes"),
        "runtime.seed": overrides.get("seed"),
        "runtime.max_steps_per_episode": overrides.get("max_steps_per_episode"),
        "runtime.headless": overrides.get("headless"),
        "runtime.auto_pack": overrides.get("auto_pack"),
    }
    for dotted_key, value in key_map.items():
        if value is not None:
            set_nested_value(config, dotted_key, value)
    return config


def resolve_output_dir(raw_path: str | Path, package_root: Path) -> Path:
    path_obj = Path(raw_path)
    if path_obj.is_absolute():
        return path_obj
    return (package_root / path_obj).resolve()


def stable_hash(text: str, digits: int = 12) -> str:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
    return digest[:digits]


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, content: str) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    path_obj.write_text(content, encoding="utf-8")
