"""Opt-in bounded lossless image writer; frozen dataset code stays untouched."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import types


class AsyncRgbWriter:
    def __init__(self, workers=2, pending_limit=4):
        if workers < 1 or pending_limit < 1:
            raise ValueError('positive worker and queue limits required')
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.pending = deque()
        self.limit = pending_limit
        self.paths = set()
        self.closed = False
        self.submit_seconds = 0.

    def __call__(self, path, rgb):
        import numpy as np
        import cv2
        started = time.monotonic()
        path = Path(path)
        if self.closed or path in self.paths or path.exists():
            raise ValueError('closed writer or duplicate image target')
        if hasattr(rgb, 'detach'):
            rgb = rgb.detach().cpu()
        if hasattr(rgb, 'numpy'):
            rgb = rgb.numpy()
        if rgb.ndim == 4:
            rgb = rgb[0]
        if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4) or rgb.dtype != np.uint8:
            raise ValueError('expected uint8 RGB or RGBA')
        # Copy now: a sensor may overwrite its output immediately after return.
        bgr = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR).copy()
        def save():
            if not cv2.imwrite(str(path), bgr):
                raise OSError('image write failed: '+str(path))
        while len(self.pending) >= self.limit:
            self.pending.popleft().result()
        self.paths.add(path)
        # Reset/first RGB is hashed immediately by the frozen paired audit.
        if path.parent.name in ('rgb_reset', 'rgb_pre'):
            save()
        else:
            self.pending.append(self.pool.submit(save))
        self.submit_seconds += time.monotonic()-started

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            while self.pending:
                self.pending.popleft().result()
        finally:
            self.pool.shutdown(wait=True)


def run_live_dataset_fast(*args, **kwargs):
    """Expert-only I/O candidate: identical physics/captures, bounded async writes."""
    from .zoh_dataset_loop import run_live_dataset
    if kwargs.get('mode', 'expert') != 'expert':
        raise ValueError('async writer is not approved for policy image reads')
    writer = AsyncRgbWriter()
    namespace = dict(run_live_dataset.__globals__, _write_rgb=writer)
    isolated = types.FunctionType(run_live_dataset.__code__, namespace,
                                  run_live_dataset.__name__, run_live_dataset.__defaults__)
    isolated.__kwdefaults__ = run_live_dataset.__kwdefaults__
    started = time.monotonic()
    try:
        result = isolated(*args, **kwargs)
    finally:
        writer.close()
    result['io_candidate'] = dict(name='bounded_async_png_v1', images=len(writer.paths),
        submit_seconds=writer.submit_seconds, loop_and_drain_seconds=time.monotonic()-started,
        pending_limit=writer.limit, physics_or_camera_schedule_changed=False)
    return result
