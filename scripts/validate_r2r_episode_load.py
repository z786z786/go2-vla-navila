#!/usr/bin/env python3
"""Bounded R2R -> Isaac loading diagnostic; never runs a navigation policy."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path('/mnt/wxh/go2_short_vln')
DATA = ROOT / 'data/r2r_vlnce'


def load(path):
    with gzip.open(path, 'rt') as f:
        return json.load(f)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def point(p):
    x, y, z = p
    return [x, -z, y]


def quat_mul(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return [w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
            w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v]


def rotate(q, v):
    return quat_mul(quat_mul(q, [0, *v]), [q[0], -q[1], -q[2], -q[3]])[1:]


def rotation(q_xyzw):
    x, y, z, w = q_xyzw
    norm = math.sqrt(w*w+x*x+y*y+z*z)
    # A maps Habitat world (X,Y,Z) to Isaac world (X,-Z,Y).
    # Habitat forward -Z becomes +Y, whereas Go2 forward is +X.
    # Hence R_go2 = A R_habitat A^-1 Rz(+90 degrees).
    q = quat_mul([w/norm, x/norm, -z/norm, y/norm],
                 [math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
    return q


def distance(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))


def angle(a, b):
    dot = abs(sum(x*y for x,y in zip(a,b)))
    dot /= math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
    return math.degrees(2*math.acos(min(1, dot)))


def convert(e, gt):
    out = copy.deepcopy(e)
    out['start_position'] = point(e['start_position'])
    out['start_rotation'] = rotation(e['start_rotation'])
    out['reference_path'] = [point(p) for p in e['reference_path']]
    for g in out['goals']:
        g['position'] = point(g['position'])
    out['gt_locations'] = [point(p) for p in gt['locations']]
    out['gt_actions'] = gt['actions']
    out['gt_forward_steps'] = gt.get('forward_steps')
    return out


def preflight(args):
    original_path = DATA / 'R2R_VLNCE_v1-3/train/train.json.gz'
    gt_path = DATA / 'R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz'
    train = load(original_path)['episodes']
    gt = load(gt_path)
    selected = [e for e in train if str(e['episode_id']) == args.episode_id]
    assert len(selected) == 1
    original = selected[0]
    converted = convert(original, gt[str(original['episode_id'])])
    forward_errors = []
    for e in train:
        q = e['start_rotation']
        qh = [q[3], *q[:3]]
        norm = math.sqrt(sum(x*x for x in qh))
        forward_h = rotate([v/norm for v in qh], [0,0,-1])
        forward_i = rotate(rotation(q), [1,0,0])
        forward_errors.append(distance(point(forward_h), forward_i))
    assert max(forward_errors) < 1e-10
    navila = load(ROOT / 'assets/vln_ce_isaac/vln_ce_isaac_v1.json.gz')['episodes']
    val = load(DATA / 'R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz')['episodes']
    by_key = {(e['scene_id'], e['instruction']['instruction_text']):e for e in val}
    comparisons = []
    for e in navila:
        h = by_key[(e['scene_id'], e['instruction']['instruction_text'])]
        rp, hp = e['reference_path'], [point(p) for p in h['reference_path']]
        comparisons.append({'episode_id': e['episode_id'],
            'start_offset_m': distance(e['start_position'], point(h['start_position'])),
            'rotation_difference_deg': angle(e['start_rotation'], rotation(h['start_rotation'])),
            'path_same_length': len(rp)==len(hp),
            'path_max_offset_m': max(distance(p,q) for p,q in zip(rp,hp)) if len(rp)==len(hp) else None})
    save(args.output / 'original_episode.json', original)
    save(args.output / 'isaac_episode.json', converted)
    receipt = {'source': str(original_path), 'source_sha256': hashlib.sha256(original_path.read_bytes()).hexdigest(),
        'gt_source': str(gt_path), 'episode_id': original['episode_id'], 'scene_id': original['scene_id'],
        'instruction': original['instruction']['instruction_text'],
        'position_transform': '[x_h, -z_h, y_h]',
        'rotation_transform': 'Habitat XYZW -> A R_h A^-1 Rz(+90deg) -> Isaac WXYZ; preserve original heading',
        'train_forward_axis_checks': len(train), 'max_forward_axis_error': max(forward_errors),
        'official_navila_comparison': {'matched':len(comparisons),
            'unmodified_starts':sum(r['start_offset_m']<1e-5 for r in comparisons),
            'unmodified_rotations':sum(r['rotation_difference_deg']<1e-3 for r in comparisons),
            'unmodified_paths':sum(r['path_same_length'] and r['path_max_offset_m']<1e-5 for r in comparisons),
            'note':'NaVILA episodes contain pose/path differences; not an exact rotation oracle for original R2R.'},
        'sources':['https://aihabitat.org/docs/habitat-sim/coordinate-frame-tutorial.html',
            'https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/utils/geometry_utils.py'],
        'passed':True}
    save(args.output / 'official_navila_comparison.json', comparisons)
    save(args.output / 'preflight.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return converted


def render(args, episode):
    from src.inference.isaac_client import configure_app_launcher, configure_runtime_assets, configure_env_episode
    configure_app_launcher(args.asset_root)
    from omni.isaac.lab.app import AppLauncher
    launcher = AppLauncher(args)
    app = launcher.app
    env = None
    try:
        import cv2
        import numpy as np
        import torch
        import gymnasium as gym
        import omni.usd
        import omni.physx
        import omni.isaac.vlnce.config
        from omni.isaac.lab_tasks.utils import parse_env_cfg
        from pxr import Usd, UsdGeom

        local_go2 = configure_runtime_assets(args.asset_root)
        cfg = parse_env_cfg('go2_matterport_vision', num_envs=1)
        configure_env_episode(cfg, episode, local_go2)
        scene_id = Path(episode['scene_id']).stem
        usd = ROOT / f'assets/vln_ce_isaac/matterport_usd/{scene_id}/{scene_id}.usd'
        cfg.scene.terrain.obj_filepath = str(usd)
        cfg.seed = 20260910
        # Headless IsaacLab defaults this to False; PhysX queries otherwise
        # return no hits even when the rendered/collidable terrain is present.
        cfg.sim.enable_scene_query_support = True
        # Keep navigation truth out of rendered images.
        for key in ('disk_1','disk_2'):
            getattr(cfg.scene,key).init_state.pos = (0,0,-100)
        cfg.scene.height_scanner.debug_vis = False
        cfg.scene.lidar_sensor.debug_vis = False
        print('R2R_PHASE constructing environment', flush=True)
        env = gym.make('go2_matterport_vision', cfg=cfg, render_mode=None)
        env.reset()
        base = env.unwrapped
        scene = base.scene
        robot = scene['robot']
        camera = scene['rgbd_camera']
        stage = omni.usd.get_context().get_stage()
        terrain = stage.GetPrimAtPath('/World/matterport')
        assert terrain.IsValid()
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default','render']).ComputeWorldBound(terrain).ComputeAlignedRange()
        lower, upper = list(bbox.GetMin()), list(bbox.GetMax())
        meshes = sum(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(terrain))
        assert meshes > 0
        query = omni.physx.get_physx_scene_query_interface()

        def support(p):
            hits = []
            def on_hit(hit):
                name = str(hit.collision)
                if '/World/matterport' in name:
                    pos = [float(x) for x in hit.position]
                    hits.append({'position':pos, 'normal':[float(x) for x in hit.normal], 'collision':name})
                return True
            query.raycast_all((p[0],p[1],p[2]+0.5), (0.,0.,-1.), 2.0, on_hit)
            floor = [h for h in hits if abs(h['normal'][2])>0.5]
            nearest = min(floor, key=lambda h:abs(h['position'][2]-p[2])) if floor else None
            return {'point':p, 'nearest_surface':nearest,
                    'vertical_error_m':abs(nearest['position'][2]-p[2]) if nearest else None,
                    'supported_within_025m':nearest is not None and abs(nearest['position'][2]-p[2])<=0.25}

        def place(p, q):
            pose = [p[0],p[1],p[2]+0.4,*q]
            robot.write_root_pose_to_sim(torch.tensor([pose],dtype=torch.float32,device=base.device))
            robot.write_root_velocity_to_sim(torch.zeros((1,6),device=base.device))
            robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_pos))
            robot.set_joint_position_target(robot.data.default_joint_pos)
            scene.write_data_to_sim()
            for _ in range(2):
                base.sim.step(render=True)
                scene.update(cfg.sim.dt)
            camera.update(cfg.sim.dt, force_recompute=True)

        # Warm up rendering while repeatedly holding the requested spawn pose.
        for _ in range(8):
            place(episode['start_position'],episode['start_rotation'])
        images = []
        indices = sorted(set([0,len(episode['reference_path'])//2,len(episode['reference_path'])-1]))
        initial_state = None
        for ordinal, idx in enumerate(indices):
            p = episode['reference_path'][idx]
            if idx==0:
                p = episode['start_position']
                q = episode['start_rotation']
            else:
                neighbor = episode['reference_path'][min(idx+1,len(episode['reference_path'])-1)]
                if idx==len(episode['reference_path'])-1:
                    previous = episode['reference_path'][idx-1]
                    yaw = math.atan2(p[1]-previous[1],p[0]-previous[0])
                else:
                    yaw = math.atan2(neighbor[1]-p[1],neighbor[0]-p[0])
                q = [math.cos(yaw/2),0,0,math.sin(yaw/2)]
            for _ in range(3):
                place(p,q)
            rgb = camera.data.output['rgb'][0].detach().cpu().numpy()[...,:3].astype(np.uint8)
            depth = camera.data.output['distance_to_image_plane'][0].detach().cpu().numpy().squeeze()
            # Save unrotated sensor bytes; visual QA decides presentation orientation.
            name = ['start','midpoint','goal'][ordinal]
            path = args.output / f'{name}_rgb.png'
            assert cv2.imwrite(str(path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            finite = np.isfinite(depth) & (depth>0) & (depth<100)
            display = np.clip(np.nan_to_num(depth,nan=10,posinf=10,neginf=0)/10,0,1)
            cv2.imwrite(str(args.output/f'{name}_depth.png'),cv2.applyColorMap((display*255).astype(np.uint8),cv2.COLORMAP_TURBO))
            images.append({'name':name,'path':str(path),'reference_index':idx,'rgb_shape':list(rgb.shape),
                'rgb_std':float(rgb.std()),'valid_depth_fraction':float(finite.mean()),
                'depth_median_m':float(np.median(depth[finite])) if finite.any() else None,
                'camera_position':camera.data.pos_w[0].detach().cpu().tolist(),
                'camera_quaternion_ros_wxyz':camera.data.quat_w_ros[0].detach().cpu().tolist(),
                'requested_base_position':[p[0],p[1],p[2]+0.4], 'requested_base_quaternion_wxyz':q})
            if idx==0:
                pos = robot.data.root_pos_w[0].detach().cpu().tolist()
                actual_q = robot.data.root_quat_w[0].detach().cpu().tolist()
                initial_state = {'actual_base_position':pos,'actual_base_quaternion_wxyz':actual_q,
                    'position_error_m':distance(pos,[p[0],p[1],p[2]+0.4]),'rotation_error_deg':angle(actual_q,q)}
            print('R2R_PHASE captured '+name,flush=True)
        # At final waypoint the start is unobstructed by the robot during floor queries.
        support_points = episode['reference_path'] + episode['gt_locations']
        unique_points = list({tuple(p):p for p in support_points}.values())
        support_rows = [support(p) for p in unique_points]
        start_support = support(episode['start_position'])
        checks = {'usd_loaded':meshes>0, 'spawn_position':initial_state['position_error_m']<0.02,
            'spawn_rotation':initial_state['rotation_error_deg']<1,
            'rgb':all(r['rgb_std']>5 for r in images),
            'depth':all(r['valid_depth_fraction']>0.5 for r in images),
            'start_floor':start_support['supported_within_025m'],
            'path_floor':all(r['supported_within_025m'] for r in support_rows)}
        receipt = {'episode_id':episode['episode_id'],'scene_id':scene_id,'usd_path':str(usd),
            'scene_query_support_enabled':cfg.sim.enable_scene_query_support,
            'mesh_count':meshes,'world_bounds':[lower,upper], 'initial_state':initial_state,
            'images':images,'start_support':start_support,'path_support':support_rows,
            'checks':checks,'passed':all(checks.values()),
            'method':'Go2 task loaded, reset, pose-held RGB/depth snapshots; waypoint teleport previews, not navigation rollout',
            'not_tested':['closed-loop locomotion','full-route collision clearance','all TRAIN scenes'],
            'completed_unix':time.time()}
        save(args.output/'render_report.json',receipt)
        print('R2R_RESULT '+json.dumps({'passed':receipt['passed'],'checks':checks,'initial_state':initial_state}),flush=True)
        if not receipt['passed']:
            raise RuntimeError('One or more loading checks failed; see render_report.json')
    except Exception as e:
        save(args.output/'render_failure.json',{'error':repr(e)})
        raise
    finally:
        # Reports and PNGs are persisted before Kit shutdown.
        app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episode-id',default='1')
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--asset-root',type=Path,default=ROOT/'assets/isaac_sim_4_1')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--enable_cameras',action='store_true')
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'preflight.json').exists():
        raise FileExistsError('Use a fresh diagnostic output directory')
    episode = preflight(args)
    if args.render:
        render(args,episode)


if __name__=='__main__':
    main()
