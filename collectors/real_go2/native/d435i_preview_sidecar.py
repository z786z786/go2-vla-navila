#!/usr/bin/env python3
import argparse
import json
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image


class FrameStore:
    def __init__(self, color_topic: str, depth_topic: str, min_depth_m: float, max_depth_m: float,
                 preview_width: int = 640, alive_threshold_s: float = 1.0, fresh_threshold_s: float = 0.3,
                 color_preview_fps: float = 12.0, depth_preview_fps: float = 8.0,
                 enable_color_preview: bool = True, enable_depth_preview: bool = True):
        self.color_topic = color_topic
        self.depth_topic = depth_topic
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.preview_width = preview_width
        self.alive_threshold_s = alive_threshold_s
        self.fresh_threshold_s = fresh_threshold_s
        self.color_preview_interval_s = 0.0 if color_preview_fps <= 0 else 1.0 / color_preview_fps
        self.depth_preview_interval_s = 0.0 if depth_preview_fps <= 0 else 1.0 / depth_preview_fps
        self.enable_color_preview = enable_color_preview
        self.enable_depth_preview = enable_depth_preview
        self.lock = threading.Lock()
        self.color_jpeg: bytes = b""
        self.depth_jpeg: bytes = b""
        self.color_stamp = 0.0
        self.depth_stamp = 0.0
        self.color_seq = 0
        self.depth_seq = 0
        self.color_encoding = ""
        self.depth_encoding = ""
        self.last_color_encode_wall_time = 0.0
        self.last_depth_encode_wall_time = 0.0
        self.color_received = threading.Event()
        self.depth_received = threading.Event()
        self.shutdown = False

    def _resize(self, image: np.ndarray) -> np.ndarray:
        if self.preview_width <= 0:
            return image
        h, w = image.shape[:2]
        if w <= self.preview_width:
            return image
        scale = self.preview_width / float(w)
        return cv2.resize(image, (self.preview_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)

    def _jpeg(self, image: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return encoded.tobytes() if ok else b""

    def on_color(self, msg: Image) -> None:
        try:
            now = time.time()
            if not self.enable_color_preview:
                with self.lock:
                    self.color_stamp = msg.header.stamp.to_sec() if msg.header.stamp else now
                    self.color_seq += 1
                    self.color_encoding = msg.encoding
                    self.last_color_encode_wall_time = now
                self.color_received.set()
                return
            with self.lock:
                if self.color_preview_interval_s > 0.0 and now - self.last_color_encode_wall_time < self.color_preview_interval_s:
                    self.color_stamp = msg.header.stamp.to_sec() if msg.header.stamp else now
                    self.color_seq += 1
                    self.color_encoding = msg.encoding
                    return
            array = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding in ('rgb8', 'bgr8'):
                image = array.reshape((msg.height, msg.width, 3))
                if msg.encoding == 'rgb8':
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif msg.encoding in ('rgba8', 'bgra8'):
                image = array.reshape((msg.height, msg.width, 4))
                code = cv2.COLOR_RGBA2BGR if msg.encoding == 'rgba8' else cv2.COLOR_BGRA2BGR
                image = cv2.cvtColor(image, code)
            elif msg.encoding == 'mono8':
                image = array.reshape((msg.height, msg.width))
            else:
                return
            image = self._resize(image)
            jpeg = self._jpeg(image)
            if not jpeg:
                return
            with self.lock:
                self.color_jpeg = jpeg
                self.color_stamp = msg.header.stamp.to_sec() if msg.header.stamp else time.time()
                self.color_seq += 1
                self.color_encoding = msg.encoding
                self.last_color_encode_wall_time = now
            self.color_received.set()
        except Exception:
            return

    def on_depth(self, msg: Image) -> None:
        try:
            now = time.time()
            if not self.enable_depth_preview:
                with self.lock:
                    self.depth_stamp = msg.header.stamp.to_sec() if msg.header.stamp else now
                    self.depth_seq += 1
                    self.depth_encoding = msg.encoding
                    self.last_depth_encode_wall_time = now
                self.depth_received.set()
                return
            with self.lock:
                if self.depth_preview_interval_s > 0.0 and now - self.last_depth_encode_wall_time < self.depth_preview_interval_s:
                    self.depth_stamp = msg.header.stamp.to_sec() if msg.header.stamp else now
                    self.depth_seq += 1
                    self.depth_encoding = msg.encoding
                    return
            if msg.encoding == '16UC1':
                depth = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width)).astype(np.float32) / 1000.0
            elif msg.encoding == '32FC1':
                depth = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
            else:
                return
            valid = np.isfinite(depth) & (depth > 0.0)
            clipped = np.clip(depth, self.min_depth_m, self.max_depth_m)
            norm = ((clipped - self.min_depth_m) / max(1e-6, self.max_depth_m - self.min_depth_m) * 255.0).astype(np.uint8)
            norm[~valid] = 0
            colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
            colored[~valid] = (0, 0, 0)
            colored = self._resize(colored)
            jpeg = self._jpeg(colored)
            if not jpeg:
                return
            with self.lock:
                self.depth_jpeg = jpeg
                self.depth_stamp = msg.header.stamp.to_sec() if msg.header.stamp else time.time()
                self.depth_seq += 1
                self.depth_encoding = msg.encoding
                self.last_depth_encode_wall_time = now
            self.depth_received.set()
        except Exception:
            return

    def wait_ready(self, timeout_s: float) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline and not self.shutdown:
            if self.color_received.is_set() and self.depth_received.is_set():
                return True
            time.sleep(0.05)
        return self.color_received.is_set() and self.depth_received.is_set()

    def status(self) -> Dict[str, Any]:
        now = time.time()
        with self.lock:
            color_age = -1.0 if self.color_stamp <= 0 else max(0.0, now - self.color_stamp)
            depth_age = -1.0 if self.depth_stamp <= 0 else max(0.0, now - self.depth_stamp)
            return {
                'running': True,
                'camera_node_running': (0 <= color_age < self.alive_threshold_s) or (0 <= depth_age < self.alive_threshold_s),
                'color_valid': 0 <= color_age < self.alive_threshold_s,
                'color_fresh': 0 <= color_age < self.fresh_threshold_s,
                'depth_valid': 0 <= depth_age < self.alive_threshold_s,
                'depth_fresh': 0 <= depth_age < self.fresh_threshold_s,
                'color_age_s': color_age,
                'depth_age_s': depth_age,
                'color_seq': self.color_seq,
                'depth_seq': self.depth_seq,
                'color_encoding': self.color_encoding,
                'depth_encoding': self.depth_encoding,
            }


def discover_topics(color_topic: str, depth_topic: str, color_camera_info_topic: str, depth_camera_info_topic: str, prefix: str = '/camera/') -> Dict[str, Any]:
    published = rospy.get_published_topics()
    names = {name for name, _ in published}
    imu_topics = sorted([name for name in names if name.startswith(prefix) and ('accel' in name or 'gyro' in name or 'imu' in name)])
    resolved = [
        color_topic,
        depth_topic,
        color_camera_info_topic,
        depth_camera_info_topic,
    ]
    if '/tf' in names:
        resolved.append('/tf')
    if '/tf_static' in names:
        resolved.append('/tf_static')
    resolved.extend(imu_topics)
    return {
        'published_topics': sorted(names),
        'imu_topics': imu_topics,
        'resolved_topics': resolved,
        'missing_topics': [name for name in [color_topic, depth_topic] if name not in names],
    }


def run_probe(args: argparse.Namespace) -> int:
    rospy.init_node('d435i_probe', anonymous=True, disable_signals=True)
    store = FrameStore(args.color_topic, args.depth_topic, args.min_depth_m, args.max_depth_m, args.preview_width,
                       args.alive_threshold_s, args.fresh_threshold_s, args.color_preview_fps, args.depth_preview_fps,
                       not args.disable_color_preview, not args.disable_depth_preview)
    color_sub = rospy.Subscriber(args.color_topic, Image, store.on_color, queue_size=1)
    depth_sub = rospy.Subscriber(args.depth_topic, Image, store.on_depth, queue_size=1)
    try:
        ready = store.wait_ready(args.timeout_s)
        discovery = discover_topics(
            args.color_topic,
            args.depth_topic,
            args.color_camera_info_topic,
            args.depth_camera_info_topic,
            '/camera/',
        )
        payload = {
            'ready': ready,
            'requested_topics': [args.color_topic, args.depth_topic, args.color_camera_info_topic, args.depth_camera_info_topic, '/tf', '/tf_static'],
            'resolved_topics': discovery['resolved_topics'],
            'missing_topics': discovery['missing_topics'],
            'imu_topics': discovery['imu_topics'],
        }
        print(json.dumps(payload, ensure_ascii=False))
        return 0 if ready else 3
    finally:
        color_sub.unregister()
        depth_sub.unregister()


def run_server(args: argparse.Namespace) -> int:
    rospy.init_node('d435i_preview_sidecar', anonymous=True, disable_signals=True)
    store = FrameStore(args.color_topic, args.depth_topic, args.min_depth_m, args.max_depth_m, args.preview_width,
                       args.alive_threshold_s, args.fresh_threshold_s, args.color_preview_fps, args.depth_preview_fps,
                       not args.disable_color_preview, not args.disable_depth_preview)
    rospy.Subscriber(args.color_topic, Image, store.on_color, queue_size=1)
    rospy.Subscriber(args.depth_topic, Image, store.on_depth, queue_size=1)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == '/healthz' or self.path == '/status.json':
                body = json.dumps(store.status(), ensure_ascii=False).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == '/latest/color.jpg':
                data = store.color_jpeg
                if not data:
                    self.send_error(404, 'color unavailable')
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if self.path == '/latest/depth.jpg':
                data = store.depth_jpeg
                if not data:
                    self.send_error(404, 'depth unavailable')
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(404, 'not found')

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)

    def shutdown(*_: Any) -> None:
        store.shutdown = True
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    server.serve_forever(poll_interval=0.2)
    server.server_close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument('--color-topic', default='/camera/color/image_raw')
        p.add_argument('--depth-topic', default='/camera/depth/image_rect_raw')
        p.add_argument('--color-camera-info-topic', default='/camera/color/camera_info')
        p.add_argument('--depth-camera-info-topic', default='/camera/depth/camera_info')
        p.add_argument('--min-depth-m', type=float, default=0.2)
        p.add_argument('--max-depth-m', type=float, default=3.0)
        p.add_argument('--preview-width', type=int, default=480)
        p.add_argument('--alive-threshold-s', type=float, default=1.0)
        p.add_argument('--fresh-threshold-s', type=float, default=0.3)
        p.add_argument('--color-preview-fps', type=float, default=12.0)
        p.add_argument('--depth-preview-fps', type=float, default=8.0)
        p.add_argument('--disable-color-preview', action='store_true')
        p.add_argument('--disable-depth-preview', action='store_true')

    probe = sub.add_parser('probe')
    add_common(probe)
    probe.add_argument('--timeout-s', type=float, default=6.0)

    serve = sub.add_parser('serve')
    add_common(serve)
    serve.add_argument('--port', type=int, default=18091)

    args = parser.parse_args()
    if args.mode == 'probe':
        return run_probe(args)
    return run_server(args)


if __name__ == '__main__':
    raise SystemExit(main())
