#!/usr/bin/env python3
"""3-D full-episode SmolVLA server with a separate protocol/version boundary."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Any

from src.inference import server as socket_server
from src.inference.full_protocol import PROTOCOL_VERSION, recv_message, send_message, validate_request_header, validate_response_header
from src.inference.full_state import validate_action_chunk
from src.navila_full.contracts import ACTION_KEY, IMAGE_KEY, STATE_KEY


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FullEpisodePolicyRuntime:
    """Fresh full-episode checkpoint runtime; intentionally has no action codec."""

    def __init__(self, checkpoint: Path):
        import cv2
        import numpy as np
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for full-episode inference")
        config = PreTrainedConfig.from_pretrained(checkpoint)
        if not isinstance(config, SmolVLAConfig):
            raise TypeError(f"expected SmolVLAConfig, got {type(config).__name__}")
        if set(config.input_features) != {IMAGE_KEY, STATE_KEY}:
            raise ValueError(f"unexpected checkpoint inputs: {sorted(config.input_features)}")
        if config.input_features[STATE_KEY].shape != (3,):
            raise ValueError(f"expected 3-D checkpoint state, got {config.input_features[STATE_KEY].shape}")
        if config.output_features[ACTION_KEY].shape != (3,) or config.chunk_size != 50 or config.n_obs_steps != 1 or config.n_action_steps != 10:
            raise ValueError("checkpoint does not satisfy the full 1-RGB/3-D/50x3/10-step contract")
        config.device = "cuda"
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config).to("cuda").eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(policy_cfg=config, pretrained_path=checkpoint, preprocessor_overrides={"device_processor": {"device": "cuda"}})
        self.config, self.checkpoint, self.checkpoint_sha256, self.codec_sha256 = config, checkpoint, sha256(checkpoint / "model.safetensors"), None
        self.cv2, self.np, self.torch = cv2, np, torch

    def infer(self, header: dict[str, Any], jpeg: bytes, seed: int) -> tuple[list[list[float]], dict[str, float], None]:
        started = time.perf_counter()
        image_bgr = self.cv2.imdecode(self.np.frombuffer(jpeg, dtype=self.np.uint8), self.cv2.IMREAD_COLOR)
        if image_bgr is None or image_bgr.shape != (512, 512, 3):
            raise ValueError("JPEG must decode to a 512x512 RGB image")
        image_rgb = self.cv2.cvtColor(image_bgr, self.cv2.COLOR_BGR2RGB)
        raw = {
            IMAGE_KEY: self.torch.from_numpy(image_rgb.copy()).permute(2, 0, 1).to(self.torch.float32) / 255.0,
            STATE_KEY: self.torch.tensor(header["state"], dtype=self.torch.float32),
            "task": header["instruction"],
        }
        processed = self.preprocessor(raw)
        self.torch.cuda.synchronize()
        preprocessed = time.perf_counter()
        noise = self.torch.randn((1, self.config.chunk_size, self.config.max_action_dim), generator=self.torch.Generator(device="cuda").manual_seed(seed), device="cuda")
        self.policy.reset()
        predicted = self.policy.predict_action_chunk(processed, noise=noise)
        self.torch.cuda.synchronize()
        inferred = time.perf_counter()
        actions = self.postprocessor(predicted).to(self.torch.float32).squeeze(0).tolist()
        complete = time.perf_counter()
        return validate_action_chunk(actions), {
            "preprocess": (preprocessed - started) * 1000.0,
            "inference": (inferred - preprocessed) * 1000.0,
            "postprocess": (complete - inferred) * 1000.0,
            "total": (complete - started) * 1000.0,
        }, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--request-log", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


DEFAULT_SEED = 20260905


def main() -> None:
    args = parse_args()
    if args.base_seed < 0:
        raise ValueError("--base-seed must be nonnegative")
    # Reuse hardened Unix-socket ownership/cleanup code, but bind every runtime
    # dependency process-locally to the new protocol and 3-D checkpoint class.
    socket_server.PROTOCOL_VERSION = PROTOCOL_VERSION
    socket_server.recv_message = recv_message
    socket_server.send_message = send_message
    socket_server.validate_request_header = validate_request_header
    socket_server.validate_response_header = validate_response_header
    socket_server.validate_action_chunk = validate_action_chunk
    socket_server.PolicyRuntime = FullEpisodePolicyRuntime
    socket_server.serve(args)


if __name__ == "__main__":
    main()
