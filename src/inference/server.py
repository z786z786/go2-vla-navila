#!/usr/bin/env python3
"""Offline SmolVLA Unix-socket inference server for M7."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import signal
import socket
import stat
import time
import traceback
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.inference.protocol import (
    PROTOCOL_VERSION,
    recv_message,
    send_message,
    validate_request_header,
    validate_response_header,
)
from src.inference.state import validate_action_chunk


IMAGE_KEY = "observation.images.front"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
DEFAULT_SOCKET = Path("/tmp/go2_smolvla_m7.sock")
DEFAULT_CHECKPOINT = Path(
    "/mnt/wxh/go2_short_vln/outputs/m6_1/smoke_20260901T104846+0800/checkpoints/step_002000"
)
DEFAULT_SEED = 20260831


def is_m61_checkpoint(checkpoint: Path) -> bool:
    return "m6_1" in checkpoint.parts


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def derive_request_seed(base_seed: int, episode_id: str, replan_index: int) -> int:
    checksum = zlib.crc32(f"{episode_id}:{replan_index}".encode("utf-8"))
    return (base_seed + checksum) % (2**31)


def build_success_response(
    *,
    request_id: str,
    actions: list[list[float]],
    checkpoint_sha256: str,
    timings_ms: dict[str, float],
    seed: int,
) -> dict[str, Any]:
    response = {
        "version": PROTOCOL_VERSION,
        "request_id": request_id,
        "status": "ok",
        "actions": validate_action_chunk(actions),
        "checkpoint_sha256": checkpoint_sha256,
        "timings_ms": timings_ms,
        "seed": seed,
    }
    validate_response_header(response)
    return response


def build_error_response(request_id: str, error: Exception) -> dict[str, Any]:
    response = {
        "version": PROTOCOL_VERSION,
        "request_id": request_id or "unknown",
        "status": "error",
        "error_type": type(error).__name__,
        "error": str(error)[:4096] or type(error).__name__,
    }
    validate_response_header(response)
    return response


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PolicyRuntime:
    def __init__(self, checkpoint: Path):
        import cv2
        import numpy as np
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from src.smolvla.bounded_actions import (
            CODEC_FILENAME,
            decode_latent_actions,
            load_codec_metadata,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for M7 inference")
        config = PreTrainedConfig.from_pretrained(checkpoint)
        if not isinstance(config, SmolVLAConfig):
            raise TypeError(f"expected SmolVLAConfig, got {type(config).__name__}")
        if set(config.input_features) != {IMAGE_KEY, STATE_KEY}:
            raise ValueError(f"unexpected checkpoint inputs: {sorted(config.input_features)}")
        if config.input_features[STATE_KEY].shape != (30,):
            raise ValueError(f"expected checkpoint state shape (30,), got {config.input_features[STATE_KEY].shape}")
        if config.output_features[ACTION_KEY].shape != (3,):
            raise ValueError(f"expected checkpoint action shape (3,), got {config.output_features[ACTION_KEY].shape}")
        if config.chunk_size != 50 or config.n_obs_steps != 1:
            raise ValueError(
                f"expected current-RGB 50-step policy, got n_obs_steps={config.n_obs_steps}, chunk={config.chunk_size}"
            )
        config.device = "cuda"
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config).to("cuda").eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
        self.config = config
        self.checkpoint = checkpoint
        self.checkpoint_sha256 = sha256(checkpoint / "model.safetensors")
        self.decode_latent_actions = decode_latent_actions
        if is_m61_checkpoint(checkpoint):
            self.codec = load_codec_metadata(checkpoint)
            self.codec_sha256 = sha256(checkpoint / CODEC_FILENAME)
        else:
            self.codec = None
            self.codec_sha256 = None
        self.cv2 = cv2
        self.np = np
        self.torch = torch

    def infer(
        self, header: dict[str, Any], jpeg: bytes, seed: int
    ) -> tuple[list[list[float]], dict[str, float], list[list[float]] | None]:
        started = time.perf_counter()
        encoded = self.np.frombuffer(jpeg, dtype=self.np.uint8)
        image_bgr = self.cv2.imdecode(encoded, self.cv2.IMREAD_COLOR)
        if image_bgr is None or image_bgr.shape != (512, 512, 3):
            raise ValueError("JPEG must decode to a 512x512 RGB image")
        image_rgb = self.cv2.cvtColor(image_bgr, self.cv2.COLOR_BGR2RGB)
        raw_batch = {
            IMAGE_KEY: self.torch.from_numpy(image_rgb.copy()).permute(2, 0, 1).to(self.torch.float32) / 255.0,
            STATE_KEY: self.torch.tensor(header["state"], dtype=self.torch.float32),
            "task": header["instruction"],
        }
        processed = self.preprocessor(raw_batch)
        self.torch.cuda.synchronize()
        preprocess_done = time.perf_counter()

        generator = self.torch.Generator(device="cuda").manual_seed(seed)
        noise = self.torch.randn(
            (1, self.config.chunk_size, self.config.max_action_dim),
            generator=generator,
            device="cuda",
        )
        self.policy.reset()
        predicted = self.policy.predict_action_chunk(processed, noise=noise)
        self.torch.cuda.synchronize()
        inference_done = time.perf_counter()

        latent_tensor = self.postprocessor(predicted).to(self.torch.float32)
        actions_tensor = (
            self.decode_latent_actions(latent_tensor) if self.codec is not None else latent_tensor
        )
        actions = actions_tensor.squeeze(0).tolist()
        latent_actions = latent_tensor.squeeze(0).tolist() if self.codec is not None else None
        postprocess_done = time.perf_counter()
        timings = {
            "preprocess": (preprocess_done - started) * 1000.0,
            "inference": (inference_done - preprocess_done) * 1000.0,
            "postprocess": (postprocess_done - inference_done) * 1000.0,
            "total": (postprocess_done - started) * 1000.0,
        }
        return validate_action_chunk(actions), timings, latent_actions


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--socket", type=Path, default=DEFAULT_SOCKET)
    parser.add_argument("--request-log", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def serve(args: argparse.Namespace) -> None:
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    runtime = PolicyRuntime(checkpoint)
    socket_path = args.socket.expanduser()
    if socket_path.exists() or socket_path.is_symlink():
        mode = os.lstat(socket_path).st_mode
        if not stat.S_ISSOCK(mode):
            raise FileExistsError(f"refusing to replace non-socket path: {socket_path}")
        socket_path.unlink()
    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    os.chmod(socket_path, 0o600)
    listener.listen(1)
    stopping = False

    def cleanup() -> None:
        try:
            listener.close()
        except OSError:
            pass
        try:
            if socket_path.exists() and stat.S_ISSOCK(os.lstat(socket_path).st_mode):
                socket_path.unlink()
        except OSError:
            pass

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True
        cleanup()

    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    ready = {
        "format": "go2-short-vln-m7-server-ready-v1",
        "ready_at_utc": now_utc(),
        "socket": str(socket_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": runtime.checkpoint_sha256,
        "bounded_action_codec_sha256": runtime.codec_sha256,
        "pid": os.getpid(),
    }
    args.ready_file.write_text(json.dumps(ready, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(ready), flush=True)

    while not stopping:
        try:
            connection, _ = listener.accept()
        except OSError:
            if stopping:
                break
            raise
        with connection:
            request_id = "unknown"
            received_at = now_utc()
            try:
                header, jpeg = recv_message(connection)
                request_id = str(header.get("request_id") or "unknown")
                validate_request_header(header, payload_size=len(jpeg))
                seed = derive_request_seed(
                    args.base_seed, str(header["episode_id"]), int(header["replan_index"])
                )
                actions, timings, latent_actions = runtime.infer(header, jpeg, seed)
                response = build_success_response(
                    request_id=request_id,
                    actions=actions,
                    checkpoint_sha256=runtime.checkpoint_sha256,
                    timings_ms=timings,
                    seed=seed,
                )
                log_row = {
                    "status": "ok",
                    "received_at_utc": received_at,
                    "responded_at_utc": now_utc(),
                    "request": header,
                    "response": response,
                    "latent_actions": latent_actions,
                }
            except Exception as error:
                response = build_error_response(request_id, error)
                log_row = {
                    "status": "error",
                    "received_at_utc": received_at,
                    "responded_at_utc": now_utc(),
                    "request_id": request_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            append_jsonl(args.request_log, log_row)
            try:
                send_message(connection, response)
            except (BrokenPipeError, ConnectionResetError):
                pass


def main() -> None:
    args = parse_args()
    if args.base_seed < 0:
        raise ValueError("--base-seed must be nonnegative")
    serve(args)


if __name__ == "__main__":
    main()
