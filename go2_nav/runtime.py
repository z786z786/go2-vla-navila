"""Observation-driven two-box mock environment and explicit runtime adapters.

The mock policy detects colored rectangles in rendered RGB; it is a deterministic
integration controller, not a trained VLA and not evidence of navigation quality.
"""
from collections import deque
import importlib
import math
from pathlib import Path
from typing import Protocol
from .contracts import Action, Observation


class Environment(Protocol):
    def reset(self, instruction: str) -> Observation: ...
    def step(self, action: Action) -> Observation: ...
    def close(self) -> None: ...


class Policy(Protocol):
    def predict(self, observation: Observation) -> Action: ...
    def reset(self) -> None: ...


def load_factory(spec, **kwargs):
    """Load an explicitly supplied runtime/model factory; never fall back to mock."""
    module, separator, name = spec.partition(":")
    if not separator or not module or not name:
        raise ValueError("factory must be module:callable")
    factory = getattr(importlib.import_module(module), name)
    return factory(**kwargs)


class ExternalEnvironment:
    """Bridge reset/step/close implementations for Isaac Sim or Unitree runtimes."""
    def __init__(self, backend, factory, enable_robot=False):
        if backend not in {"isaacsim", "unitree"}:
            raise ValueError("unsupported runtime")
        if backend == "unitree" and not enable_robot:
            raise ValueError("Unitree execution requires --enable-robot")
        self.runtime = load_factory(factory)
        for name in ("reset", "step", "close"):
            if not callable(getattr(self.runtime, name, None)):
                raise TypeError(f"runtime lacks {name}")

    def reset(self, instruction):
        result = self.runtime.reset(instruction)
        return result if isinstance(result, Observation) else Observation(**result)

    def step(self, action):
        result = self.runtime.step(action)
        return result if isinstance(result, Observation) else Observation(**result)

    def close(self):
        self.runtime.close()


class ModelPolicy:
    """Checkpoint injection and context packing for a SmolVLA/LLaDA-V predictor.

    The factory returns a callable accepting the context dictionary and returning
    Action or its dictionary representation. Model-specific processors belong to
    the factory, so dimensions and normalizers are never guessed here.
    """
    def __init__(self, factory, checkpoint, history=1, state=False, denoise_steps=10):
        if not Path(checkpoint).exists():
            raise FileNotFoundError(checkpoint)
        if history < 1 or denoise_steps < 1:
            raise ValueError("history and denoise_steps must be positive")
        self.frames = deque(maxlen=history)
        self.include_state = state
        self.denoise_steps = denoise_steps
        self.predictor = load_factory(factory, checkpoint=checkpoint)
        if not callable(self.predictor):
            raise TypeError("model factory must return a callable")

    def reset(self):
        self.frames.clear()
        reset = getattr(self.predictor, "reset", None)
        if reset:
            reset()

    def predict(self, observation):
        self.frames.append(observation.rgb_path)
        context = {"instruction": observation.instruction, "frames": list(self.frames),
                   "denoise_steps": self.denoise_steps}
        if self.include_state:
            context["state"] = observation.state
        result = self.predictor(context)
        return result if isinstance(result, Action) else Action(**result)


class TwoBoxMock:
    dt = 0.2

    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.targets = {"red": (1.4, 0.55), "blue": (1.4, -0.55)}
        self.reset("go to the red box")

    def reset(self, instruction):
        self.x = self.y = self.yaw = 0.0
        self.tick = 0
        self.instruction = instruction
        self.last = Action()
        return self.observe()

    def observe(self):
        w, h = 96, 64
        pixels = bytearray([32, 36, 42] * (w*h))
        # A simple forward pinhole rendering. The policy reads pixels only.
        visible = []
        for color, (tx, ty) in self.targets.items():
            dx, dy = tx-self.x, ty-self.y
            forward = math.cos(self.yaw)*dx + math.sin(self.yaw)*dy
            lateral = -math.sin(self.yaw)*dx + math.cos(self.yaw)*dy
            if forward <= 0.1:
                continue
            cx = int(w/2 - 48*lateral/forward)
            size = min(62, max(2, int(12/forward)))
            visible.append((forward, color, cx, size))
        for _, color, cx, size in sorted(visible, reverse=True):
            rgb = bytes((230, 40, 40) if color == "red" else (40, 80, 230))
            for y in range(max(0,h//2-size//2), min(h,h//2+size//2)):
                for x in range(max(0,cx-size//2), min(w,cx+size//2)):
                    i = (y*w+x)*3
                    pixels[i:i+3] = rgb
        path = self.output/f"frame_{self.tick:04d}.ppm"
        path.write_bytes(f"P6\n{w} {h}\n255\n".encode()+pixels)
        return Observation(self.tick*self.dt, self.instruction, str(path), (self.last.vx, 0., self.last.wz))

    def step(self, action):
        if not action.stop:
            self.x += action.vx*math.cos(self.yaw)*self.dt
            self.y += action.vx*math.sin(self.yaw)*self.dt
            self.yaw += action.wz*self.dt
        self.last = action
        self.tick += 1
        return self.observe()

    def close(self):
        pass


class ColorBoxPolicy:
    def __init__(self, stop_mode="hysteresis"):
        if stop_mode not in {"explicit", "hysteresis"}:
            raise ValueError("invalid stop_mode")
        self.required = 2 if stop_mode == "hysteresis" else 1
        self.reset()

    def reset(self):
        self.near_count = 0

    def predict(self, observation):
        target = "red" if "red" in observation.instruction else "blue" if "blue" in observation.instruction else None
        if target is None:
            raise ValueError("mock instruction must name a red or blue box")
        header, dimensions, maximum, rgb = Path(observation.rgb_path).read_bytes().split(b"\n", 3)
        w, h = map(int, dimensions.split())
        if header != b"P6" or maximum != b"255" or len(rgb) != w*h*3:
            raise ValueError("mock policy expects generated P6 RGB frames")
        points = []
        for i in range(w*h):
            r, g, b = rgb[i*3:i*3+3]
            if (r > 180 and b < 100) if target == "red" else (b > 180 and r < 100):
                points.append((i % w, i // w))
        if not points:
            self.near_count = 0
            return Action(0., 0.3)
        width = max(x for x,y in points)-min(x for x,y in points)+1
        self.near_count = self.near_count+1 if width >= 38 else 0
        if self.near_count >= self.required:
            return Action(stop=True)
        center = sum(x for x,y in points)/len(points)
        error = (w/2-center)/(w/2)
        return Action(max(0.04, 0.3*(1-abs(error))), 0.8*error)
