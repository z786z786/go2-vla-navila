import io
import errno
import json
import math
import os
import socket
import struct
import tempfile
import threading
import unittest

from src.inference.protocol import (
    FramingError,
    MAX_HEADER_BYTES,
    PROTOCOL_VERSION,
    encode_message,
    recv_message,
    send_message,
    unix_request,
    validate_request_header,
    validate_response_header,
)


class FragmentedSocket:
    def __init__(self, payload: bytes, chunk_size: int = 3):
        self._payload = io.BytesIO(payload)
        self._chunk_size = chunk_size

    def recv(self, size: int) -> bytes:
        return self._payload.read(min(size, self._chunk_size))


def valid_request() -> dict:
    return {
        "version": PROTOCOL_VERSION,
        "request_id": "short_vln_v1_0004:0",
        "episode_id": "short_vln_v1_0004",
        "replan_index": 0,
        "instruction": "Move forward and stop.",
        "state": [0.0] * 30,
        "image_encoding": "jpeg",
        "image_height": 512,
        "image_width": 512,
        "image_channels": 3,
        "jpeg_size": 4,
    }


class ProtocolTests(unittest.TestCase):
    def test_fragmented_reads_reconstruct_header_and_payload(self):
        header = valid_request()
        encoded = encode_message(header, b"jpeg")

        actual_header, actual_payload = recv_message(FragmentedSocket(encoded))

        self.assertEqual(actual_header, header)
        self.assertEqual(actual_payload, b"jpeg")

    def test_truncated_payload_is_rejected(self):
        encoded = encode_message(valid_request(), b"jpeg")[:-1]

        with self.assertRaisesRegex(FramingError, "truncated"):
            recv_message(FragmentedSocket(encoded))

    def test_oversized_header_is_rejected_before_allocation(self):
        encoded = struct.pack("!Q", MAX_HEADER_BYTES + 1)

        with self.assertRaisesRegex(FramingError, "header length"):
            recv_message(FragmentedSocket(encoded))

    def test_request_rejects_unknown_oracle_field(self):
        header = valid_request()
        header["goal_direction"] = [1.0, 0.0]

        with self.assertRaisesRegex(ValueError, "unknown request fields.*goal_direction"):
            validate_request_header(header, payload_size=4)

    def test_request_rejects_wrong_state_shape_and_nonfinite_state(self):
        short = valid_request()
        short["state"] = [0.0] * 29
        with self.assertRaisesRegex(ValueError, "30-D state"):
            validate_request_header(short, payload_size=4)

        nonfinite = valid_request()
        nonfinite["state"][5] = math.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_request_header(nonfinite, payload_size=4)

    def test_request_rejects_declared_jpeg_size_mismatch(self):
        with self.assertRaisesRegex(ValueError, "jpeg_size"):
            validate_request_header(valid_request(), payload_size=3)

    def test_response_accepts_only_finite_50_by_3_actions(self):
        response = {
            "version": PROTOCOL_VERSION,
            "request_id": "short_vln_v1_0004:0",
            "status": "ok",
            "actions": [[0.1, 0.0, -0.2] for _ in range(50)],
            "checkpoint_sha256": "a" * 64,
            "timings_ms": {
                "preprocess": 1.0,
                "inference": 2.0,
                "postprocess": 1.0,
                "total": 4.0,
            },
            "seed": 20260831,
        }
        validate_response_header(response)

        response["actions"][49][2] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_response_header(response)

    def test_send_message_round_trips_over_real_socket_pair(self):
        left, right = socket.socketpair()
        try:
            send_message(left, valid_request(), b"jpeg")
            header, payload = recv_message(right)
        finally:
            left.close()
            right.close()

        self.assertEqual(header, valid_request())
        self.assertEqual(payload, b"jpeg")

    def test_unix_request_returns_validated_response(self):
        response = {
            "version": PROTOCOL_VERSION,
            "request_id": "short_vln_v1_0004:0",
            "status": "ok",
            "actions": [[0.1, 0.0, -0.2] for _ in range(50)],
            "checkpoint_sha256": "a" * 64,
            "timings_ms": {
                "preprocess": 1.0,
                "inference": 2.0,
                "postprocess": 1.0,
                "total": 4.0,
            },
            "seed": 20260831,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "policy.sock")
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                listener.bind(path)
            except PermissionError as error:
                listener.close()
                if error.errno == errno.EPERM:
                    self.skipTest("the local desktop sandbox forbids AF_UNIX bind; run on Isaac host")
                raise
            listener.listen(1)

            def serve_once():
                connection, _ = listener.accept()
                with connection:
                    request_header, request_payload = recv_message(connection)
                    validate_request_header(request_header, payload_size=len(request_payload))
                    send_message(connection, response)
                listener.close()

            thread = threading.Thread(target=serve_once)
            thread.start()
            actual = unix_request(path, valid_request(), b"jpeg", timeout_s=1.0)
            thread.join(timeout=1.0)

        self.assertEqual(actual, response)


if __name__ == "__main__":
    unittest.main()
