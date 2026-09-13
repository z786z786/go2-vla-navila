"""One CUDA inference-only client for DT1 Isaac + base-SmolVLA coexistence.

It consumes the strict 3D sidecar from :mod:`smolvla_probe_contract`, loads
only the locally cached base model, performs one 50×3 action-chunk inference,
and writes diagnostics.  The output is deliberately never sent to Isaac: no
model action, task success, trainer, optimizer, or old normalizer is present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from .contracts import ACTION_CHUNK_SIZE, IMAGE_KEY, STATE_KEY, TASK_KEY
from .smolvla_probe_contract import (
    IDENTITY_NORMALIZATION,
    SmolVlaProbeContractError,
    fresh_identity_dataset_stats,
    probe_input_from_dict,
)


DEFAULT_BASE_MODEL = Path("/mnt/wxh/go2_short_vln/models/smolvla_base_c83c316")
DEFAULT_BASE_CONFIG_SHA256 = "650584b56c104720f7a3c91d1ec6bec9e8de8ac11e60c92ba2fa82d93eda147d"
DEFAULT_BASE_MODEL_SHA256 = "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb"


class SmolVlaProbeClientError(RuntimeError):
    """The isolated resource/interface probe cannot make a safe claim."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_rgb(input_path: Path, relative: str) -> Path:
    candidate = (input_path.parent / relative).resolve()
    if not candidate.is_relative_to(input_path.parent.resolve()) or not candidate.is_file():
        raise SmolVlaProbeClientError("probe RGB must be an existing file inside its input sidecar directory")
    return candidate


def _load_probe_input(path: Path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return probe_input_from_dict(payload)
    except (OSError, json.JSONDecodeError, SmolVlaProbeContractError) as exc:
        raise SmolVlaProbeClientError(f"invalid SmolVLA probe input: {exc}") from exc


def _build_new_3d_config(checkpoint: Path) -> Any:
    """Load base weights with a fresh 1-RGB/3-state/3-action interface."""
    try:
        from lerobot.configs import PreTrainedConfig
        from lerobot.configs.policies import FeatureType, PolicyFeature
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    except ImportError as exc:  # pragma: no cover - smolvla environment only
        raise SmolVlaProbeClientError("SmolVLA/LeRobot imports are unavailable in this process") from exc
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    if not isinstance(config, SmolVLAConfig):
        raise SmolVlaProbeClientError(f"cached base model is not SmolVLA: {type(config).__name__}")
    # Preserve the pretrained internal padded capacity (32/32), but make the
    # public model interface exactly the new DT1 3D contract.
    config.input_features = {
        IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
        STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(3,))}
    config.chunk_size = ACTION_CHUNK_SIZE
    config.n_action_steps = 10
    # This is a fresh diagnostic interface.  It must not retrieve a training
    # dataset normalizer from the base checkpoint or a legacy 30-D server.
    from lerobot.configs import NormalizationMode
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.IDENTITY,
        "ACTION": NormalizationMode.IDENTITY,
    }
    config.device = "cuda"
    return config


def run_probe(input_path: Path, checkpoint: Path, output_path: Path) -> dict[str, object]:
    """Run exactly one local base-model inference and record non-control facts."""
    request = _load_probe_input(input_path.resolve())
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir():
        raise SmolVlaProbeClientError(f"cached base model directory is absent: {checkpoint}")
    config_path, model_path = checkpoint / "config.json", checkpoint / "model.safetensors"
    if not config_path.is_file() or not model_path.is_file():
        raise SmolVlaProbeClientError("cached base model lacks config.json or model.safetensors")
    if checkpoint == DEFAULT_BASE_MODEL and (_sha256(config_path) != DEFAULT_BASE_CONFIG_SHA256 or _sha256(model_path) != DEFAULT_BASE_MODEL_SHA256):
        raise SmolVlaProbeClientError("cached base model hash differs from the DT0-reviewed artifact")
    rgb_path = _input_rgb(input_path.resolve(), request.rgb_path)
    if _sha256(rgb_path) != request.rgb_sha256:
        raise SmolVlaProbeClientError("probe RGB bytes differ from the sidecar digest")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite SmolVLA probe output: {output_path}")
    # No Hub fallback is acceptable for this resource measurement.
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        import numpy as np
        from PIL import Image
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    except ImportError as exc:  # pragma: no cover - smolvla environment only
        raise SmolVlaProbeClientError("SmolVLA probe dependencies are unavailable") from exc
    if not torch.cuda.is_available():
        raise SmolVlaProbeClientError("CUDA is required for the coexistence resource probe")
    image = np.asarray(Image.open(rgb_path).convert("RGB"))
    if image.shape != (512, 512, 3):
        raise SmolVlaProbeClientError(f"probe front RGB must be exactly 512×512×3, got {image.shape}")
    config = _build_new_3d_config(checkpoint)
    if config.input_features[STATE_KEY].shape != (3,) or config.output_features["action"].shape != (3,):
        raise SmolVlaProbeClientError("fresh SmolVLA configuration did not retain exact 3D state/action shapes")
    if config.max_state_dim != 32 or config.max_action_dim != 32:
        raise SmolVlaProbeClientError("base model's internal 32D padded capacities changed unexpectedly")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config, local_files_only=True, strict=True).to("cuda").eval()
    # Identity mappings require no old statistic values.  Pass fresh 3D stats
    # only as an inspectable diagnostic object, never as checkpoint state.
    stats = {
        key: {name: torch.tensor(values, dtype=torch.float32) for name, values in feature.items()}
        for key, feature in fresh_identity_dataset_stats().items()
    }
    preprocessor, postprocessor = make_smolvla_pre_post_processors(config, dataset_stats=stats)
    raw = {
        IMAGE_KEY: torch.from_numpy(image.copy()).permute(2, 0, 1).to(torch.float32) / 255.0,
        STATE_KEY: torch.tensor(request.state, dtype=torch.float32),
        TASK_KEY: request.task,
    }
    processed = preprocessor(raw)
    torch.cuda.synchronize()
    prepared_at = time.perf_counter()
    generator = torch.Generator(device="cuda").manual_seed(20260905)
    with torch.inference_mode():
        policy.reset()
        prediction = policy.predict_action_chunk(
            processed,
            noise=torch.randn((1, ACTION_CHUNK_SIZE, config.max_action_dim), generator=generator, device="cuda"),
        )
    torch.cuda.synchronize()
    inferred_at = time.perf_counter()
    action = postprocessor(prediction).to(torch.float32)
    if tuple(action.shape) != (1, ACTION_CHUNK_SIZE, 3) or not bool(torch.isfinite(action).all().item()):
        raise SmolVlaProbeClientError("SmolVLA did not produce a finite 1×50×3 diagnostic action chunk")
    # Deliberately do not return, clamp, or execute this chunk.
    result = {
        "format": "go2-dual-target-dt1-smolvla-probe-result-v1", "stage": "DT1",
        "status": "SMOLVLA_3D_INFERENCE_COMPLETED_NOT_TASK_APPROVED",
        "input_path": str(input_path.resolve()), "input_rgb_sha256": request.rgb_sha256,
        "checkpoint": str(checkpoint), "config_sha256": _sha256(config_path), "model_sha256": _sha256(model_path),
        "interface": {
            "image_key": IMAGE_KEY, "state_key": STATE_KEY, "state_dim": 3,
            "action_dim": 3, "chunk_size": ACTION_CHUNK_SIZE, "execute_action_steps": 10,
            "max_state_dim": config.max_state_dim, "max_action_dim": config.max_action_dim,
            "normalization": IDENTITY_NORMALIZATION, "dataset_stats": fresh_identity_dataset_stats(),
        },
        "output_shape": list(action.shape), "output_finite": True,
        "execute_model_actions": False, "training": False, "navigation_success_approved": False, "dt1_approved": False,
        "timing_ms": {"load_and_preprocess": (prepared_at - started) * 1000.0, "inference": (inferred_at - prepared_at) * 1000.0},
        "cuda_peak_allocated_mib": int(torch.cuda.max_memory_allocated() // (1024 * 1024)),
        "cuda_peak_reserved_mib": int(torch.cuda.max_memory_reserved() // (1024 * 1024)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_probe(args.input, args.checkpoint, args.output), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":  # pragma: no cover
    main()
