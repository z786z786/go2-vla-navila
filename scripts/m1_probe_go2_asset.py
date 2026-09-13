"""Minimal Isaac 4.1 asset-access probe for M1 diagnostics."""

import os
import time

from omni.isaac.lab.app import AppLauncher


DEFAULT_URL = (
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/4.1/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd"
)
URL = os.environ.get("GO2_ASSET_URL", DEFAULT_URL)

app_launcher = AppLauncher({"headless": True, "enable_cameras": False})
simulation_app = app_launcher.app

import carb
import omni.client
import omni.usd


def emit(*values):
    message = " ".join(str(value) for value in values)
    print(message, flush=True)
    carb.log_info(message)


settings = carb.settings.get_settings()
emit("asset_root_default=", settings.get("/persistent/isaac/asset_root/default"))
emit("asset_root_cloud=", settings.get("/persistent/isaac/asset_root/cloud"))
emit("asset_root_timeout=", settings.get("/persistent/isaac/asset_root/timeout"))

started = time.monotonic()
stat_result, stat_entry = omni.client.stat(URL)
emit("stat_result=", stat_result, "seconds=", round(time.monotonic() - started, 3))
emit("stat_entry=", stat_entry)

context = omni.usd.get_context()
started = time.monotonic()
opened = context.open_stage(URL)
for _ in range(8):
    simulation_app.update()
emit("open_stage_return=", opened, "seconds=", round(time.monotonic() - started, 3))
stage = context.get_stage()
emit("stage_valid=", stage is not None)
emit("root_identifier=", stage.GetRootLayer().identifier if stage is not None else None)
emit("default_prim=", stage.GetDefaultPrim().GetPath() if stage is not None else None)

simulation_app.close()
