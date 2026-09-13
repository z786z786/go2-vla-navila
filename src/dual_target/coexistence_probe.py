"""DT1-only Isaac + SmolVLA resource probe. Never executes model actions."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .gpu_wait import (acquire_live_admission, probe_gpu, EXCLUSIVE_DT1_POLICY,
                       SHARED_COEXISTENCE_DT1_POLICY)
from .smolvla_probe_contract import (
    SmolVlaProbeInput, SMOLVLA_PROBE_SCHEMA, IDENTITY_NORMALIZATION,
    fresh_identity_dataset_stats,
)


def wait_for_model(child, *, sample, record, timeout_s=300.0, allowed_foreign_pids=()):
    """Bounded wait with live resource evidence; only terminate our child."""
    deadline = time.monotonic() + timeout_s
    concurrent_seen = False
    try:
        while True:
            snapshot = sample()
            record(snapshot)
            pids = set(snapshot['compute_pids'])
            if pids - {os.getpid(), child.pid} - set(allowed_foreign_pids):
                raise RuntimeError('foreign GPU compute appeared during coexistence probe')
            if snapshot['free_mib'] < 2048:
                raise RuntimeError('coexistence free memory dropped below 2 GiB')
            concurrent_seen |= {os.getpid(), child.pid} <= pids
            code = child.poll()
            if code is not None:
                if code != 0:
                    raise RuntimeError(f'model probe exited {code}')
                if not concurrent_seen:
                    raise RuntimeError('no sample proves both model and Isaac GPU residency')
                return
            if time.monotonic() >= deadline:
                raise TimeoutError('model coexistence probe exceeded bounded wall time')
            time.sleep(.5)
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)


def main():
    # Same reviewed app setup and admission as the expert path; no model or
    # Isaac CUDA initialization before the project lease has been acquired.
    from . import runner as r
    from .runtime import PreResetEvidenceCapture, SubstepContactLatch
    from omni.isaac.lab.app import AppLauncher

    parser = r._live_parser()
    parser.add_argument('--shared-coexistence', action='store_true')
    parser.add_argument('--model-python', type=Path, default=Path(
        '<external-data-root>'))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if (not args.live or args.paired_reset or args.shared_smoke or args.contact_precheck
            or args.motion_precheck or not args.go2_usd):
        parser.error('coexistence requires exclusive --live with no expert/diagnostic mode flags')
    if not args.model_python.is_file():
        parser.error('model interpreter missing')
    args.seed = 3401 if args.seed is None else args.seed
    asset_root = r._asset_root_from_go2_usd(args.go2_usd)
    r._configure_dt1_kit_args_before_app(asset_root)
    root = r.initialize_dt1_run(args.output_dir, run_id=r._safe_run_id(args.run_id),
                                actual_argv=[sys.executable, '-m', 'src.dual_target.coexistence_probe', *sys.argv[1:]])
    lease = app = runtime = capture = None
    try:
        lease = acquire_live_admission(args.gpu_project_root, run_id=args.run_id,
                                       actual_argv=sys.argv, state_path=args.gpu_admission_state,
                                       policy=SHARED_COEXISTENCE_DT1_POLICY if args.shared_coexistence else EXCLUSIVE_DT1_POLICY)
        allowed_foreign = lease.snapshot.compute_pids if args.shared_coexistence else ()
        (root/'resource_policy.json').write_text(json.dumps({
            'admission_policy': SHARED_COEXISTENCE_DT1_POLICY.policy_id if args.shared_coexistence else EXCLUSIVE_DT1_POLICY.policy_id,
            'allowed_foreign_pids': list(allowed_foreign), 'minimum_free_mib_during_run': 2048,
            'training': False,
        }, indent=2))
        app = AppLauncher(args)
        kit = r._configure_dt1_runtime_after_app(asset_root)
        (root/'kit_runtime.json').write_text(json.dumps(kit, indent=2))
        scene, task = r._scene_for_task(args.group_id, args.color_configuration, args.target_color)
        runtime = r.build_live_low_level_runtime(args, scene)
        adapter, base = runtime.adapter, runtime.adapter.base_env
        adapter.reset()
        contact = r.validate_live_target_contact_sensors(base)
        (root/'target_contact_view_contract.json').write_text(json.dumps(contact, indent=2))
        latch = SubstepContactLatch(threshold_n=1.0)
        capture = PreResetEvidenceCapture(
            base, snapshot=lambda env: r._default_runtime_snapshot(env),
            on_substep=lambda: latch.capture({
                color: r._target_contact_force(base.scene[f'{color}_target_contacts'], color=color)
                for color in ('red', 'blue')
            }),
        )
        capture.install()
        for _ in range(r.GO2_WARMUP_STEPS):
            _, _, done, _, _, hits, _ = adapter.step((0., 0., 0.), latch)
            if r._bool_at_zero(done, 'warmup_done') or capture.records or any(hits):
                raise RuntimeError('auto-reset or collision during coexistence warmup')
        r._render_without_physics(base)
        rgb_path = root/'front.png'
        r._write_rgb(rgb_path, r._camera_rgb(base.scene['rgbd_camera']))
        before = r._live_clock(base)
        pose_before = r._robot_pose_w(base)
        state = r._default_runtime_snapshot(base)['body_velocity_body']
        request = SmolVlaProbeInput(
            schema_version=SMOLVLA_PROBE_SCHEMA, rgb_path='front.png', rgb_sha256=r.sha256(rgb_path),
            state=tuple(state), task=task.instruction, action_dim=3, action_chunk_size=50,
            execute_action_steps=10, normalization=IDENTITY_NORMALIZATION,
            dataset_stats=fresh_identity_dataset_stats(), inference_only=True,
            execute_model_actions=False, training=False, navigation_success_approved=False, dt1_approved=False,
        )
        input_path, output_path = root/'model_input.json', root/'model_result.json'
        input_path.write_text(json.dumps(request.as_dict(), indent=2, allow_nan=False))
        env = dict(os.environ)
        env.pop('PYTHONHOME', None)
        env['PYTHONPATH'] = '<external-workspace>:<external-data-root>'
        env['HF_HOME'] = '<external-data-root>'
        env['HF_HUB_OFFLINE'] = env['TRANSFORMERS_OFFLINE'] = '1'
        command = [str(args.model_python), '-m', 'src.dual_target.smolvla_probe_client',
                   '--input', str(input_path), '--output', str(output_path)]
        samples = []
        with (root/'model.log').open('w') as log, (root/'gpu_samples.jsonl').open('w') as gpu_log:
            def record(snapshot):
                row = {'wall_time_s': time.time(), **snapshot}
                samples.append(row)
                gpu_log.write(json.dumps(row)+'\n')
                gpu_log.flush()
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            (root/'model_process.json').write_text(json.dumps({
                'isaac_pid': os.getpid(), 'model_pid': child.pid, 'command': command,
            }, indent=2))
            started = time.monotonic()
            wait_for_model(child, sample=lambda: probe_gpu().as_dict(), record=record,
                           allowed_foreign_pids=allowed_foreign)
        after = r._live_clock(base)
        pose_after = r._robot_pose_w(base)
        if before != after or not r._same_pose(pose_before, pose_after):
            raise RuntimeError('physics moved while waiting for model inference')
        model = json.loads(output_path.read_text())
        if (model['output_shape'] != [1, 50, 3] or model['output_finite'] is not True
                or model['execute_model_actions'] is not False or model['training'] is not False):
            raise RuntimeError('model result violates inference-only 3D contract')
        result = {
            'status': 'COEXISTENCE_PASSED_NOT_DT1_APPROVED', 'stage': 'DT1',
            'model_result_sha256': r.sha256(output_path), 'input_sha256': r.sha256(input_path),
            'sampled_total_peak_mib': max(s['used_mib'] for s in samples),
            'sampled_min_free_mib': min(s['free_mib'] for s in samples),
            'sampling_interval_s': .5, 'wall_wait_s': time.monotonic()-started,
            'clock_before': list(before), 'clock_after': list(after),
            'pose_before': pose_before, 'pose_after': pose_after, 'physics_advanced': False,
            'execute_model_actions': False, 'training': False, 'dt1_approved': False,
        }
        (root/'coexistence_result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        r.write_dt1_run_status(root, 'RUNNING', run_id=args.run_id,
                              detail='coexistence only; full DT1 review remains', extra={'coexistence': result})
    except BaseException as exc:
        evidence = r.write_dt1_failure_traceback(root, exc=exc)
        r.write_dt1_run_status(root, 'FAILED', run_id=args.run_id, detail=str(exc), extra=evidence)
        raise
    finally:
        try:
            if capture is not None:
                capture.restore()
            if runtime is not None:
                runtime.history_env.close()
        finally:
            try:
                if app is not None:
                    app.app.close()
            finally:
                if lease is not None:
                    lease.close()


if __name__ == '__main__':
    main()
