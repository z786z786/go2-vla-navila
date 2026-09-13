#!/usr/bin/env python3
"""One shared-GPU DT1 preview run; never used for formal DT1 evidence."""
from __future__ import annotations
import sys
from dataclasses import replace

from src.dual_target import gpu_wait

SHARED_PREVIEW_POLICY = replace(
    gpu_wait.SHARED_COEXISTENCE_DT1_POLICY,
    policy_id="dt1_shared_preview_12g_v1",
    required_free_mib=12 * 1024,
)

def main() -> None:
    # Read the run identity/project root without importing Isaac modules.
    argv = list(sys.argv[1:])
    if "--run-id" not in argv or "--gpu-project-root" not in argv:
        raise SystemExit("shared preview requires --run-id and --gpu-project-root")
    run_id = argv[argv.index("--run-id") + 1]
    project_root = __import__("pathlib").Path(argv[argv.index("--gpu-project-root") + 1])
    waiter = gpu_wait.Dt1GpuWaiter(
        project_root=project_root, run_id=run_id,
        argv=[sys.executable, __file__, *argv], policy=SHARED_PREVIEW_POLICY,
        required_free_mib=SHARED_PREVIEW_POLICY.required_free_mib,
        required_samples=SHARED_PREVIEW_POLICY.required_samples,
        interval_s=SHARED_PREVIEW_POLICY.sample_interval_s,
    )
    result = waiter.wait()
    if result.get("status") != "GPU_READY_LOCKED_RECHECKED":
        raise SystemExit(f"shared preview GPU admission failed: {result}")
    # Import the live runner only after the waiter has completed.  Importing
    # it before AppLauncher can load omni.kit_app too early and segfault Kit.
    from src.dual_target import runner
    original = runner.acquire_live_admission
    def shared_admission(*args, **kwargs):
        kwargs["policy"] = SHARED_PREVIEW_POLICY
        return original(*args, **kwargs)
    runner.acquire_live_admission = shared_admission
    runner.main()

if __name__ == "__main__":
    main()
