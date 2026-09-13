"""Capture camera evidence around the unmodified official NaVILA PD planner."""

import atexit
import hashlib
import os
from pathlib import Path
import runpy
import signal
import sys

import cv2
import numpy as np


SOURCE = Path("/mnt/wxh/go2_short_vln/third_party/NaVILA-Bench/scripts/demo_planner.py")
VIDEO_DIR = Path(os.environ["M1_VIDEO_DIR"])
SCREENSHOT_DIR = Path(os.environ["M1_SCREENSHOT_DIR"])
ASSET_ROOT = os.environ.get("M1_ISAAC_ASSET_ROOT")
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

writer = None
frames = 0
last_frame = None
timeline_extension_applied = False
timeline_stop_subscription = None
source_sha256 = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
original_rotate = cv2.rotate


def configure_project_asset_root():
    """Point only this M1 process at the verified local official Go2 assets."""
    if not ASSET_ROOT:
        return

    # AppLauncher forwards unknown Kit arguments to SimulationApp.  Supplying
    # these settings before the application starts prevents startup extensions
    # from first attempting the unavailable cloud asset root.  This is a
    # process-local storage location override, not a NaVILA planner change.
    kit_asset_args = (
        f"--/persistent/isaac/asset_root/default={ASSET_ROOT}",
        f"--/persistent/isaac/asset_root/cloud={ASSET_ROOT}",
        f"--/persistent/isaac/asset_root/nvidia={ASSET_ROOT}",
        "--/persistent/isaac/asset_root/timeout=1.0",
    )
    from omni.isaac.lab.app import AppLauncher

    original_init = AppLauncher.__init__

    def init_with_project_asset_root(self, *args, **kwargs):
        # The official script parses its own arguments before it constructs
        # AppLauncher, so add Kit-only arguments at this lifecycle point.
        # AppLauncher forwards them to SimulationApp without exposing them to
        # the NaVILA argparse parser.
        for kit_arg in kit_asset_args:
            if not any(arg.startswith(kit_arg.split("=", 1)[0] + "=") for arg in sys.argv):
                sys.argv.append(kit_arg)
        print("M1 pre-init local Isaac asset roots added to Kit arguments (process-local)", flush=True)
        original_init(self, *args, **kwargs)
        import carb
        import omni.timeline
        import omni.usd

        settings = carb.settings.get_settings()
        settings.set_string("/persistent/isaac/asset_root/default", ASSET_ROOT)
        settings.set_string("/persistent/isaac/asset_root/cloud", ASSET_ROOT)
        settings.set_float("/persistent/isaac/asset_root/timeout", 30.0)
        hang_detector_before = settings.get("/app/hangDetector/enabled")
        settings.set_bool("/app/hangDetector/enabled", False)
        print(f"M1 local Isaac asset root: {ASSET_ROOT}")
        print(
            "M1 Kit hang detector: "
            f"{hang_detector_before} -> {settings.get('/app/hangDetector/enabled')} (process-local)",
            flush=True,
        )
        # Importing Isaac Lab can resolve its asset-root constants, so this
        # must happen only after the process-local root is installed above.
        from omni.isaac.lab.sim import SimulationContext

        # The Isaac 4.1 camera-enabled experience carries a non-looping,
        # two-second timeline.  The camera's Replicator orchestrator stops
        # that timeline when it reaches the endpoint, before the official
        # single episode finishes.  ManagerBasedEnv calls sim.reset() only
        # after the stage is built and before it starts playback, which is the
        # lifecycle point at which Kit accepts the new endpoint.
        original_reset = SimulationContext.reset

        def reset_with_extended_timeline(sim_context, *reset_args, **reset_kwargs):
            global timeline_extension_applied
            if not timeline_extension_applied:
                timeline = omni.timeline.get_timeline_interface()
                before_end_time = timeline.get_end_time()
                stage = omni.usd.get_context().get_stage()
                time_codes_per_second = stage.GetTimeCodesPerSecond()
                before_end_time_code = stage.GetEndTimeCode()
                stage.SetEndTimeCode(10000.0 * time_codes_per_second)
                timeline.set_end_time(10000.0)
                timeline.commit()
                timeline_extension_applied = True
                print(
                    "M1 timeline end time before sim.reset: "
                    f"timeline {before_end_time} -> {timeline.get_end_time()} seconds; "
                    f"stage {before_end_time_code} -> {stage.GetEndTimeCode()} time codes "
                    "(process-local)",
                    flush=True,
                )
            return original_reset(sim_context, *reset_args, **reset_kwargs)

        SimulationContext.reset = reset_with_extended_timeline

        # Keep a diagnostic subscription alive for the event that causes
        # SimulationContext's standalone callback to close Kit.  It also
        # reports the live endpoint, which can be changed by a late-loading
        # camera extension after reset.
        global timeline_stop_subscription

        def log_timeline_stop(event):
            timeline = omni.timeline.get_timeline_interface()
            stage = omni.usd.get_context().get_stage()
            stage_end = stage.GetEndTimeCode() if stage is not None else "none"
            print(
                "M1 timeline STOP observed: "
                f"current={timeline.get_current_time()} end={timeline.get_end_time()} "
                f"stage_end={stage_end}",
                flush=True,
            )

        timeline_stop_subscription = (
            omni.timeline.get_timeline_interface()
            .get_timeline_event_stream()
            .create_subscription_to_pop_by_type(
                int(omni.timeline.TimelineEventType.STOP), log_timeline_stop, order=0
            )
        )

        # Record, then preserve, AppLauncher's existing shutdown behavior for
        # every signal it installs a handler for.  This is diagnostic only.
        for signal_number, signal_name in (
            (signal.SIGINT, "SIGINT"),
            (signal.SIGTERM, "SIGTERM"),
            (signal.SIGABRT, "SIGABRT"),
            (signal.SIGSEGV, "SIGSEGV"),
        ):
            previous_handler = signal.getsignal(signal_number)

            def log_then_delegate(signum, frame, previous_handler=previous_handler, signal_name=signal_name):
                print(f"M1 signal observed: {signal_name}", flush=True)
                previous_handler(signum, frame)

            signal.signal(signal_number, log_then_delegate)

    AppLauncher.__init__ = init_with_project_asset_root


def capture_rotate(image, rotate_code):
    """Return the official rotation result unchanged, with a side-channel capture."""
    global frames, last_frame, writer
    frame = original_rotate(image, rotate_code)
    if frame is None:
        return frame
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim != 3 or frame.shape[2] < 3:
        return frame
    rgb = frame[:, :, :3]
    # A late-loading camera extension can restore a finite stage endpoint.
    # Before that endpoint is reached, extend only the running Kit timeline.
    # The official planner action, observation, and termination code remain
    # untouched.
    if frames and frames % 100 == 0:
        try:
            import omni.timeline
            import omni.usd

            timeline = omni.timeline.get_timeline_interface()
            current_time = timeline.get_current_time()
            end_time = timeline.get_end_time()
            stage = omni.usd.get_context().get_stage()
            stage_end = stage.GetEndTimeCode() if stage is not None else "none"
            print(
                "M1 timeline snapshot: "
                f"frame={frames} current={current_time} end={end_time} stage_end={stage_end}",
                flush=True,
            )
            if end_time <= current_time + 2.0:
                new_end = current_time + 10000.0
                timeline.set_end_time(new_end)
                timeline.commit()
                if stage is not None:
                    stage.SetEndTimeCode(new_end * stage.GetTimeCodesPerSecond())
                print(f"M1 timeline endpoint extended at frame={frames} to {new_end}", flush=True)
        except Exception as exc:
            print(f"M1 timeline monitor warning: {exc}", flush=True)
    if writer is None:
        height, width = rgb.shape[:2]
        writer = cv2.VideoWriter(
            str(VIDEO_DIR / "pd_planner_camera.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (width, height),
        )
        cv2.imwrite(str(SCREENSHOT_DIR / "camera_first.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if writer is not None and writer.isOpened():
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if frames % 50 == 0:
        cv2.imwrite(
            str(SCREENSHOT_DIR / f"camera_{frames:05d}.png"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        )
    last_frame = rgb.copy()
    frames += 1
    return frame


def finish_capture():
    if writer is not None:
        writer.release()
    if last_frame is not None:
        cv2.imwrite(str(SCREENSHOT_DIR / "camera_last.png"), cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR))
    manifest = VIDEO_DIR / "pd_planner_camera_manifest.txt"
    manifest.write_text(
        "official_source=" + str(SOURCE) + "\n"
        "official_source_sha256=" + source_sha256 + "\n"
        "capture_method=cv2.rotate side-channel; planner source and logic unmodified\n"
        "asset_root_override=" + (ASSET_ROOT if ASSET_ROOT else "none") + "\n"
        "asset_root_override_method=Kit pre-init plus AppLauncher post-init before NaVILA config import\n"
        "timeline_end_time=10000.0 seconds (process-local on first PLAY event; planner unchanged)\n"
        "captured_frames=" + str(frames) + "\n"
        "last_frame_shape=" + (str(last_frame.shape) if last_frame is not None else "none") + "\n"
        "last_frame_minmax="
        + (f"{int(last_frame.min())},{int(last_frame.max())}" if last_frame is not None else "none")
        + "\n"
    )


atexit.register(finish_capture)
cv2.rotate = capture_rotate
configure_project_asset_root()
sys.path.insert(0, str(SOURCE.parent))
runpy.run_path(str(SOURCE), run_name="__main__")
