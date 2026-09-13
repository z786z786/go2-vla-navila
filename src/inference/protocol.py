#!/usr/bin/env python3
"""Length-prefixed protocol shared by the M7 Isaac client and policy server."""

from __future__ import annotations

import json
import math
import re
import socket
import struct
from collections.abc import Mapping, Sequence
from typing import Any


PROTOCOL_VERSION = "go2-smolvla-m7-v1"
MAX_HEADER_BYTES = 1024 * 1024
MAX_PAYLOAD_BYTES = 4 * 1024 * 1024
ACTION_STEPS = 50
ACTION_DIM = 3
STATE_DIM = 30

REQUEST_FIELDS = frozenset(
    {
        "version",
        "request_id",
        "episode_id",
        "replan_index",
        "instruction",
        "state",
        "image_encoding",
        "image_height",
        "image_width",
        "image_channels",
        "jpeg_size",
    }
)
SUCCESS_RESPONSE_FIELDS = frozenset(
    {
        "version",
        "request_id",
        "status",
        "actions",
        "checkpoint_sha256",
        "timings_ms",
        "seed",
    }
)
ERROR_RESPONSE_FIELDS = frozenset(
    {"version", "request_id", "status", "error_type", "error"}
)
TIMING_FIELDS = frozenset({"preprocess", "inference", "postprocess", "total"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FramingError(ValueError):
    """Raised when a peer sends an incomplete or oversized frame."""


def _recv_exact(sock: Any, size: int, *, allow_clean_eof: bool = False) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            if allow_clean_eof and remaining == size:
                raise EOFError("peer closed connection")
            raise FramingError(f"truncated frame: expected {size} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def encode_message(header: Mapping[str, Any], payload: bytes = b"") -> bytes:
    if not isinstance(header, Mapping):
        raise TypeError("header must be a mapping")
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    try:
        header_bytes = json.dumps(
            dict(header), separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"header is not strict JSON: {exc}") from exc
    if not 0 < len(header_bytes) <= MAX_HEADER_BYTES:
        raise FramingError(f"invalid header length: {len(header_bytes)}")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise FramingError(f"invalid payload length: {len(payload)}")
    return (
        struct.pack("!Q", len(header_bytes))
        + header_bytes
        + struct.pack("!Q", len(payload))
        + payload
    )


def recv_message(
    sock: Any,
    *,
    max_header_bytes: int = MAX_HEADER_BYTES,
    max_payload_bytes: int = MAX_PAYLOAD_BYTES,
) -> tuple[dict[str, Any], bytes]:
    header_size = struct.unpack("!Q", _recv_exact(sock, 8, allow_clean_eof=True))[0]
    if not 0 < header_size <= max_header_bytes:
        raise FramingError(f"invalid header length: {header_size}")
    header_bytes = _recv_exact(sock, header_size)
    payload_size = struct.unpack("!Q", _recv_exact(sock, 8))[0]
    if payload_size > max_payload_bytes:
        raise FramingError(f"invalid payload length: {payload_size}")
    payload = _recv_exact(sock, payload_size) if payload_size else b""
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FramingError(f"invalid JSON header: {exc}") from exc
    if not isinstance(header, dict):
        raise FramingError("JSON header must be an object")
    return header, payload


def send_message(sock: Any, header: Mapping[str, Any], payload: bytes = b"") -> None:
    sock.sendall(encode_message(header, payload))


def unix_request(
    socket_path: str,
    header: Mapping[str, Any],
    payload: bytes,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be positive and finite")
    validate_request_header(header, payload_size=len(payload))
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout_s)
        connection.connect(socket_path)
        send_message(connection, header, payload)
        response, response_payload = recv_message(connection, max_payload_bytes=0)
    if response_payload:
        raise FramingError("response must not include a binary payload")
    validate_response_header(response)
    if response["request_id"] != header["request_id"]:
        raise ValueError(
            f"response request_id {response['request_id']!r} does not match {header['request_id']!r}"
        )
    if response["status"] == "error":
        raise RuntimeError(f"{response['error_type']}: {response['error']}")
    return response


def _check_exact_fields(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = set(value)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        raise ValueError(f"unknown {label} fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"missing {label} fields: {', '.join(missing)}")


def _finite_number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite")
    if nonnegative and converted < 0:
        raise ValueError(f"{label} must be nonnegative")
    return converted


def _nonempty_string(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} must be a non-empty string of at most {maximum} characters")
    return value


def validate_request_header(header: Mapping[str, Any], *, payload_size: int) -> None:
    if not isinstance(header, Mapping):
        raise ValueError("request header must be an object")
    _check_exact_fields(header, REQUEST_FIELDS, "request")
    if header["version"] != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version: {header['version']!r}")
    _nonempty_string(header["request_id"], "request_id", 128)
    _nonempty_string(header["episode_id"], "episode_id", 128)
    _nonempty_string(header["instruction"], "instruction", 4096)
    replan = header["replan_index"]
    if isinstance(replan, bool) or not isinstance(replan, int) or replan < 0:
        raise ValueError("replan_index must be a nonnegative integer")
    state = header["state"]
    if not isinstance(state, list) or len(state) != STATE_DIM:
        raise ValueError(f"request must contain a {STATE_DIM}-D state")
    for index, value in enumerate(state):
        _finite_number(value, f"state[{index}]")
    if header["image_encoding"] != "jpeg":
        raise ValueError("image_encoding must be 'jpeg'")
    expected_shape = (512, 512, 3)
    actual_shape = (header["image_height"], header["image_width"], header["image_channels"])
    if actual_shape != expected_shape:
        raise ValueError(f"image shape must be {expected_shape}, got {actual_shape}")
    jpeg_size = header["jpeg_size"]
    if isinstance(jpeg_size, bool) or not isinstance(jpeg_size, int):
        raise ValueError("jpeg_size must be an integer")
    if jpeg_size != payload_size or not 0 < jpeg_size <= MAX_PAYLOAD_BYTES:
        raise ValueError(f"jpeg_size {jpeg_size} does not match payload size {payload_size}")


def _validate_actions(actions: Any) -> None:
    if not isinstance(actions, list) or len(actions) != ACTION_STEPS:
        raise ValueError(f"response must contain {ACTION_STEPS} actions")
    for row_index, row in enumerate(actions):
        if not isinstance(row, list) or len(row) != ACTION_DIM:
            raise ValueError(f"action[{row_index}] must contain {ACTION_DIM} values")
        for column_index, value in enumerate(row):
            _finite_number(value, f"action[{row_index}][{column_index}]")


def validate_response_header(header: Mapping[str, Any]) -> None:
    if not isinstance(header, Mapping):
        raise ValueError("response header must be an object")
    status = header.get("status")
    if status == "error":
        _check_exact_fields(header, ERROR_RESPONSE_FIELDS, "error response")
        if header["version"] != PROTOCOL_VERSION:
            raise ValueError("error response protocol version mismatch")
        _nonempty_string(header["request_id"], "request_id", 128)
        _nonempty_string(header["error_type"], "error_type", 128)
        _nonempty_string(header["error"], "error", 4096)
        return
    if status != "ok":
        raise ValueError(f"response status must be 'ok' or 'error', got {status!r}")
    _check_exact_fields(header, SUCCESS_RESPONSE_FIELDS, "response")
    if header["version"] != PROTOCOL_VERSION:
        raise ValueError("response protocol version mismatch")
    _nonempty_string(header["request_id"], "request_id", 128)
    _validate_actions(header["actions"])
    if not isinstance(header["checkpoint_sha256"], str) or not SHA256_RE.fullmatch(
        header["checkpoint_sha256"]
    ):
        raise ValueError("checkpoint_sha256 must be 64 lowercase hex characters")
    timings = header["timings_ms"]
    if not isinstance(timings, Mapping):
        raise ValueError("timings_ms must be an object")
    _check_exact_fields(timings, TIMING_FIELDS, "timing")
    for name, value in timings.items():
        _finite_number(value, f"timings_ms.{name}", nonnegative=True)
    seed = header["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
