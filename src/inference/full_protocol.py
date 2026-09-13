"""Versioned socket protocol for the full-episode 3-D policy (not M7-v1)."""

from __future__ import annotations

import math
import re
import socket
from collections.abc import Mapping
from typing import Any

from src.inference.protocol import FramingError, MAX_PAYLOAD_BYTES, encode_message, recv_message, send_message
from src.navila_full.contracts import POLICY_PROTOCOL_VERSION


PROTOCOL_VERSION = POLICY_PROTOCOL_VERSION
ACTION_STEPS, ACTION_DIM, STATE_DIM = 50, 3, 3
REQUEST_FIELDS = frozenset({"version", "request_id", "episode_id", "replan_index", "instruction", "state", "image_encoding", "image_height", "image_width", "image_channels", "jpeg_size"})
SUCCESS_RESPONSE_FIELDS = frozenset({"version", "request_id", "status", "actions", "checkpoint_sha256", "timings_ms", "seed"})
ERROR_RESPONSE_FIELDS = frozenset({"version", "request_id", "status", "error_type", "error"})
TIMING_FIELDS = frozenset({"preprocess", "inference", "postprocess", "total"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _exact(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    unknown, missing = sorted(set(value) - expected), sorted(expected - set(value))
    if unknown:
        raise ValueError(f"unknown {label} fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"missing {label} fields: {', '.join(missing)}")


def _string(value: Any, label: str, maximum: int) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} must be a non-empty string of at most {maximum} characters")


def _number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if nonnegative and result < 0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def validate_request_header(header: Mapping[str, Any], *, payload_size: int) -> None:
    if not isinstance(header, Mapping):
        raise ValueError("request header must be an object")
    _exact(header, REQUEST_FIELDS, "request")
    if header["version"] != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version: {header['version']!r}")
    _string(header["request_id"], "request_id", 128)
    _string(header["episode_id"], "episode_id", 128)
    _string(header["instruction"], "instruction", 4096)
    if isinstance(header["replan_index"], bool) or not isinstance(header["replan_index"], int) or header["replan_index"] < 0:
        raise ValueError("replan_index must be a nonnegative integer")
    state = header["state"]
    if not isinstance(state, list) or len(state) != STATE_DIM:
        raise ValueError(f"request must contain a {STATE_DIM}-D state")
    for index, value in enumerate(state):
        _number(value, f"state[{index}]")
    if header["image_encoding"] != "jpeg":
        raise ValueError("image_encoding must be 'jpeg'")
    if (header["image_height"], header["image_width"], header["image_channels"]) != (512, 512, 3):
        raise ValueError("image shape must be (512, 512, 3)")
    if isinstance(header["jpeg_size"], bool) or not isinstance(header["jpeg_size"], int) or header["jpeg_size"] != payload_size or not 0 < payload_size <= MAX_PAYLOAD_BYTES:
        raise ValueError("jpeg_size does not match payload size")


def validate_response_header(header: Mapping[str, Any]) -> None:
    if not isinstance(header, Mapping):
        raise ValueError("response header must be an object")
    if header.get("status") == "error":
        _exact(header, ERROR_RESPONSE_FIELDS, "error response")
        if header["version"] != PROTOCOL_VERSION:
            raise ValueError("error response protocol version mismatch")
        _string(header["request_id"], "request_id", 128)
        _string(header["error_type"], "error_type", 128)
        _string(header["error"], "error", 4096)
        return
    if header.get("status") != "ok":
        raise ValueError("response status must be 'ok' or 'error'")
    _exact(header, SUCCESS_RESPONSE_FIELDS, "response")
    if header["version"] != PROTOCOL_VERSION:
        raise ValueError("response protocol version mismatch")
    _string(header["request_id"], "request_id", 128)
    actions = header["actions"]
    if not isinstance(actions, list) or len(actions) != ACTION_STEPS:
        raise ValueError(f"response must contain {ACTION_STEPS} actions")
    for row_index, row in enumerate(actions):
        if not isinstance(row, list) or len(row) != ACTION_DIM:
            raise ValueError(f"action[{row_index}] must contain {ACTION_DIM} values")
        for column_index, value in enumerate(row):
            _number(value, f"action[{row_index}][{column_index}]")
    if not isinstance(header["checkpoint_sha256"], str) or not SHA256_RE.fullmatch(header["checkpoint_sha256"]):
        raise ValueError("checkpoint_sha256 must be 64 lowercase hex characters")
    if not isinstance(header["timings_ms"], Mapping):
        raise ValueError("timings_ms must be an object")
    _exact(header["timings_ms"], TIMING_FIELDS, "timing")
    for name, value in header["timings_ms"].items():
        _number(value, f"timings_ms.{name}", nonnegative=True)
    if isinstance(header["seed"], bool) or not isinstance(header["seed"], int) or header["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")


def unix_request(socket_path: str, header: Mapping[str, Any], payload: bytes, *, timeout_s: float) -> dict[str, Any]:
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be positive and finite")
    validate_request_header(header, payload_size=len(payload))
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout_s)
        connection.connect(socket_path)
        send_message(connection, header, payload)
        response, extra = recv_message(connection, max_payload_bytes=0)
    if extra:
        raise FramingError("response must not include a binary payload")
    validate_response_header(response)
    if response["request_id"] != header["request_id"]:
        raise ValueError("response request_id does not match request")
    if response["status"] == "error":
        raise RuntimeError(f"{response['error_type']}: {response['error']}")
    return response
