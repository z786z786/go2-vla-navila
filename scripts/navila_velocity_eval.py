"""Run the official setup/scorers with SmolVLA replacing the text-action loop.

Official source remains untouched. The replaced block and its source hash are
saved for audit. This evaluates SmolVLA; it does not reproduce NaVILA-8B.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rollout(env, infos, episode, output, checkpoint, seed):
    import numpy as np
    import torch
    from PIL import Image
    import imageio.v2 as imageio
    from scripts.navila_velocity_rpc import PolicyClient
    from src.inference.navila_velocity import velocity_stop, hold_steps, STOP_THRESHOLDS
    from src.dual_target.zoh_control import CommandBoundary

    dt = float(env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation)
    hold = hold_steps(dt)
    # Same outer timeout as official navila_eval.py (50 simulation seconds).
    max_steps = 100 * 0.5 / dt
    instruction = episode['instruction']['instruction_text']
    (output / 'episode.json').write_text(json.dumps(episode, indent=2))
    (output / 'rgb').mkdir()
    client = None
    frames = 0
    reason = None
    boundary = CommandBoundary()
    with (output / 'commands.jsonl').open('x') as commands, \
         imageio.get_writer(str(output / 'rollout.mp4'), fps=round(1 / dt)) as video:
        try:
            client = PolicyClient(checkpoint, seed, output)
            (output / 'adapter.json').write_text(json.dumps({
                **client.metadata, 'stop_thresholds_vx_vy_wz': STOP_THRESHOLDS,
                'stop_rule': 'single raw prediction; strict per-axis absolute thresholds',
                'official_goal_used_by_policy_or_stop': False,
                'command_boundary': boundary.spec.metadata(),
                'official_control_dt_s': dt, 'hold_steps': hold,
                'model_rgb_resize': '512x512 PIL bilinear; no rotation',
            }, indent=2))
            # Official frame orientation is retained; only model-required resize.
            step = 0
            applied = [0., 0., 0.]
            while True:
                rgb = infos['observations']['camera_obs'][0, :, :, :3].detach().cpu().numpy()
                if step % hold == 0:
                    image_path = output / 'rgb' / f'{step:06d}.png'
                    Image.fromarray(rgb).resize((512,512), Image.Resampling.BILINEAR).save(image_path)
                    raw = client.action(image_path, None, instruction)
                    stop = velocity_stop(raw)  # Before clipping/slew; no GT signal.
                    if stop:
                        env.set_stop_called(True)
                        # Update official metrics at the STOP pose, without moving.
                        env.measure_manager.update_measures()
                        infos['measurements'] = env.measure_manager.get_measurements()
                        applied = [0.,0.,0.]
                    else:
                        applied = list(boundary.update(raw))
                    commands.write(json.dumps({'control_step':step, 'simulation_time_s':step*dt,
                        'raw_velocity':raw, 'applied_velocity':applied, 'stop':stop})+'\n')
                    commands.flush()
                    if stop:
                        video.append_data(rgb)
                        frames += 1
                        reason = 'model_low_velocity_stop'
                        break
                video.append_data(rgb)
                frames += 1
                with torch.inference_mode():
                    _, _, done, infos = env.step(torch.tensor(applied, device=env.unwrapped.device))
                # Retain wrapper terminal conditions and official outer time limit.
                if bool(done) or env.is_stop_called or step > max_steps:
                    reason = 'official_environment_done' if bool(done) else 'official_time_limit'
                    break
                step += 1
            metrics = {k:float(v) for k,v in infos['measurements'].items()}
            if not all(np.isfinite(v) for v in metrics.values()):
                raise ValueError('nonfinite official metric')
            (output / 'measurements.json').write_text(json.dumps(metrics, indent=2, allow_nan=False))
            (output / 'result.json').write_text(json.dumps({
                'status':'COMPLETE', 'episode_id':episode['episode_id'], 'metrics':metrics,
                'termination':reason, 'control_steps':step, 'video_frames':frames,
                'checkpoint':str(checkpoint), 'policy_seed':seed,
            }, indent=2))
        finally:
            if client is not None:
                client.close()


def main():
    # Pin the complete Pillow stack before Kit adds its bundled Python paths.
    import PIL.Image, PIL.ImageFont, PIL.ImageDraw, PIL.ImageColor
    import torchvision
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--policy-seed', type=int, default=20260906)
    p.add_argument('--navila-root', type=Path, default=Path('/mnt/wxh/go2_short_vln/third_party/NaVILA-Bench'))
    p.add_argument('--asset-root', type=Path, default=Path('/mnt/wxh/go2_short_vln/assets/isaac_sim_4_1'))
    args, official_args = p.parse_known_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    source_path = args.navila_root / 'scripts/navila_eval.py'
    source = source_path.read_text()
    begin = '    # NaViLA training gets image observations each 0.5s, visualize every 0.1s'
    end = '    # close the simulator'
    if source.count(begin) != 1 or source.count(end) != 1:
        raise ValueError('official source layout changed; review required')
    lo, hi = source.index(begin), source.index(end)
    replacement = '    _velocity_rollout(env, infos, episode)\n\n'
    adapted = source[:lo] + replacement + source[hi:]
    (args.output / 'official_navila_eval.py').write_text(source)
    (args.output / 'adapted_navila_eval.py').write_text(adapted)
    utils = args.navila_root / 'isaaclab_exts/omni.isaac.vlnce/omni/isaac/vlnce/utils'
    (args.output / 'provenance.json').write_text(json.dumps({
        'official_source_sha256': sha(source_path),
        'official_measures_sha256': sha(utils / 'measures.py'),
        'official_wrappers_sha256': sha(utils / 'wrappers.py'),
        'checkpoint_manifest_sha256': sha(args.checkpoint / 'checkpoint_manifest.json'),
        'argv':sys.argv, 'adapter_source_sha256':sha(__file__),
    }, indent=2))
    sys.path.insert(0, str(args.navila_root / 'scripts'))
    os.chdir(args.navila_root)
    sys.argv = [str(source_path)] + official_args
    from src.inference.isaac_client import configure_app_launcher
    configure_app_launcher(args.asset_root)
    namespace = {'__name__':'__main__', '__file__':str(source_path),
        '_velocity_rollout':lambda env, infos, episode: rollout(
            env, infos, episode, args.output, args.checkpoint, args.policy_seed)}
    exec(compile(adapted, str(source_path), 'exec'), namespace)


if __name__ == '__main__':
    main()
