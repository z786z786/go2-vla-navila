#!/usr/bin/env python3
"""Rich, causal single-episode Go2 expert recording for later ablations."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_r2r_episode_load import ROOT, DATA, load, save, convert, rotate
from src.dual_target.zoh_control import CommandBoundary, ZohSpec
from src.dual_target.r2r_path_expert import ContinuousRouteExpert


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def history8(ids):
    """NaVILA sampling rule, with explicit nulls for black-frame padding."""
    padded=[None]*max(0,8-len(ids))+list(ids)
    n=len(padded)
    result=[padded[int(i*(n-1)/7)] for i in range(7)]+[padded[-1]]
    return {'frame_ids':result,'padding_mask':[i is None for i in result]}


class RouteExpert:
    def __init__(self, path, goal):
        self.path=[]
        for p in path:
            if not self.path or math.dist(p,self.path[-1])>1e-4:self.path.append(p)
        if math.dist(self.path[-1],goal)>.05:self.path.append(goal)
        self.index=0
        self.goal=goal

    def command(self,s):
        p=s['position_w'];q=s['quaternion_wxyz']
        forward=rotate(q,[1,0,0]);yaw=math.atan2(forward[1],forward[0])
        while self.index<len(self.path)-1 and math.dist(p[:2],self.path[self.index][:2])<.25:
            self.index+=1
        target=self.path[self.index]
        dx,dy=target[0]-p[0],target[1]-p[1]
        error=math.atan2(math.sin(math.atan2(dy,dx)-yaw),math.cos(math.atan2(dy,dx)-yaw))
        goal_distance=math.dist(p[:2],self.goal[:2])
        stop=self.index==len(self.path)-1 and goal_distance<=.30
        raw=[0.,0.,0.] if stop else [min(.35, .8*math.hypot(dx,dy))*max(0.,math.cos(error)) if abs(error)<.75 else 0.,0.,1.5*error]
        return raw, {'waypoint_index':self.index,'waypoint_count':len(self.path),'waypoint':target,
            'waypoint_distance_xy_m':math.hypot(dx,dy),'heading_error_rad':error,
            'goal_distance_xy_m':goal_distance,'stop_intent':stop,
            'phase':'stop' if stop else ('turn' if abs(error)>=.75 else 'track')}


def still(s):
    v=s['linear_body'];w=s['angular_body']
    return math.hypot(v[0],v[1])<.05 and abs(w[2])<.10


def run(args):
    import numpy as np
    import torch
    import cv2
    from src.inference.isaac_client import configure_app_launcher,configure_runtime_assets,configure_env_episode
    from omni.isaac.lab.app import AppLauncher
    output=args.output
    output.mkdir(parents=True,exist_ok=False)
    (output/'rgb').mkdir()
    source=DATA/'R2R_VLNCE_v1-3/train/train.json.gz'
    gt_source=DATA/'R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz'
    original=next(e for e in load(source)['episodes'] if str(e['episode_id'])==args.episode_id)
    gt=load(gt_source)[str(original['episode_id'])]
    episode=convert(original,gt)
    save(output/'original_episode.json',original);save(output/'isaac_episode.json',episode)
    spec=ZohSpec();boundary=CommandBoundary(spec)
    expert_class=ContinuousRouteExpert if args.expert=='continuous' else RouteExpert
    expert=expert_class(episode['gt_locations'],episode['goals'][0]['position'])
    configure_app_launcher(args.asset_root)
    launcher=AppLauncher(args);app=launcher.app
    streams=[];summary={'status':'starting','success':False,'training_eligible':False}
    started=time.monotonic()
    try:
        import gymnasium as gym
        import omni.isaac.vlnce.config
        from omni.isaac.lab_tasks.utils import parse_env_cfg,get_checkpoint_path
        from omni.isaac.vlnce.utils import RslRlVecEnvHistoryWrapper
        from rsl_rl.runners import OnPolicyRunner
        import cli_args
        local_go2=configure_runtime_assets(args.asset_root)
        cfg=parse_env_cfg(args.task,num_envs=1)
        configure_env_episode(cfg,episode,local_go2)
        cfg.seed=args.seed
        # This IsaacLab version requires True for TensorAPI contact reporting.
        # Do not confuse disabling global reports with disabling tensor sensors.
        import carb
        cfg.sim.disable_contact_processing=True
        carb.settings.get_settings().set_bool('/physics/disableContactProcessing',True)
        cfg.episode_length_s=args.max_seconds+20
        cfg.scene.terrain.obj_filepath=str(ROOT/f"assets/vln_ce_isaac/matterport_usd/{Path(episode['scene_id']).stem}/{Path(episode['scene_id']).stem}.usd")
        for key in ('disk_1','disk_2'):getattr(cfg.scene,key).init_state.pos=(0,0,-100)
        cfg.scene.height_scanner.debug_vis=False;cfg.scene.lidar_sensor.debug_vis=False
        env=gym.make(args.task,cfg=cfg,render_mode=None)
        env=RslRlVecEnvHistoryWrapper(env,history_length=args.history_length)
        agent=cli_args.parse_rsl_rl_cfg(args.task,args)
        checkpoint=Path(get_checkpoint_path(str(args.navila_root/'logs/rsl_rl'/agent.experiment_name),args.load_run,agent.load_checkpoint))
        runner=OnPolicyRunner(env,agent.to_dict(),log_dir=None,device=agent.device)
        runner.load(str(checkpoint));policy=runner.get_inference_policy(device=env.unwrapped.device)
        base=env.unwrapped;robot=base.scene['robot'];camera=base.scene['rgbd_camera'];contacts=base.scene['contact_forces']
        physics_contacts=[]
        original_sim_step=base.sim.step
        def record_physics_contacts(*a,**kw):
            value=original_sim_step(*a,**kw)
            if contacts.is_initialized:
                forces=contacts.contact_physx_view.get_net_contact_forces(dt=cfg.sim.dt)
                physics_contacts.append(forces.detach().cpu().numpy().copy())
            return value
        base.sim.step=record_physics_contacts
        def arr(value):return value[0].detach().cpu().tolist()
        def snapshot():
            d=robot.data
            return {'physics_step':int(base._sim_step_counter),'sim_time_s':float(base._sim_step_counter*cfg.sim.dt),
                'position_w':arr(d.root_pos_w),'quaternion_wxyz':arr(d.root_quat_w),
                'linear_world':arr(d.root_lin_vel_w),'angular_world':arr(d.root_ang_vel_w),
                'linear_body':arr(d.root_lin_vel_b),'angular_body':arr(d.root_ang_vel_b),
                'joint_position':arr(d.joint_pos),'joint_velocity':arr(d.joint_vel),
                'joint_position_target':arr(d.joint_pos_target),'applied_torque':arr(d.applied_torque),
                'contact_force_world':arr(contacts.data.net_forces_w),
                'contact_substep_peak_force_n':float(max((np.linalg.norm(f,axis=-1).max() for f in physics_contacts),default=0.))}
        obs,infos=env.reset()
        reset_evidence=[]
        orig_reset=base._reset_idx
        def capture_reset(ids):
            if len(ids):
                reset_evidence.append({'state':snapshot(),'terminated':bool(base.reset_terminated[0]),
                    'truncated':bool(base.reset_time_outs[0]),'active_terms':base.termination_manager.active_terms})
            return orig_reset(ids)
        base._reset_idx=capture_reset
        def step(cmd):
            nonlocal obs
            physics_contacts.clear()
            c=torch.tensor(cmd,device=base.device,dtype=torch.float32)
            obs[:,6:9]=c;env.proprio_obs_buf[:,-1,6:9]=c
            with torch.no_grad():
                low_action=policy(obs)
                obs,reward,done,info=env.step(low_action)
            return low_action,reward,done,info
        def stream(name):
            f=(output/name).open('x',buffering=1);streams.append(f);return f
        lowlog=stream('low_level.jsonl');highlog=stream('actions.jsonl');framelog=stream('frames.jsonl');warm=stream('warmup.jsonl')
        def emit(f,x):f.write(json.dumps(x,allow_nan=False,separators=(',',':'))+'\n')
        metadata={'schema':'r2r-go2-ablation-v1','episode_id':args.episode_id,'split':'train',
            'scene_id':episode['scene_id'],'trajectory_id':episode['trajectory_id'],'instruction':episode['instruction']['instruction_text'],
            'source':str(source),'source_sha256':sha(source),'gt_source':str(gt_source),'gt_sha256':sha(gt_source),
            'collector_sha256':sha(__file__),'coordinate_adapter_sha256':sha(Path(__file__).with_name('validate_r2r_episode_load.py')),
            'command_boundary_sha256':sha(Path(__file__).resolve().parents[1]/'src/dual_target/zoh_control.py'),
            'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'seed':args.seed,
            'zoh':spec.metadata(),'numeric_hz':50,'primary_rgb_hz':5,'navila_keyframe_interval_s':.5,
            'rgb_schedule':'union of action boundaries (10 steps) and history keyframes (25 steps), plus final frame',
            'history':'NaVILA 8-frame prefix-uniform sampling, null/black padding; most recent 0.5s keyframe, current RGB separate',
            'history_reference_sha256':sha(args.navila_root/'scripts/navila_eval.py'),
            'low_level_proprio_history_length':args.history_length,
            'rgb_encoding':{'format':'JPEG','quality':90,'shape':[512,512,3],'orientation':'raw camera RGB, no rotation or truth overlay'},
            'camera_intrinsics':arr(camera.data.intrinsic_matrices),'camera_offset':str(cfg.scene.rgbd_camera.offset),
            'joint_names':robot.joint_names,'contact_body_names':contacts.body_names,
            'contact_tensor_api_configured':cfg.sim.disable_contact_processing,
            'stop_rules':{'goal_distance_xy_m':.30,'body_planar_speed_mps':.05,'body_yaw_rate_radps':.10,'continuous_hold_s':2.0},
            'expert':{'type':'sequential_gt_waypoints','waypoint_acceptance_xy_m':.25,'max_forward_mps':.35,'heading_gain':1.5,'forward_heading_gate_rad':.75},
            'truth_fields':'original/isaac episode, expert diagnostics and world pose are audit-only; learner exports must use allowlists',
            'depth_saved':False,'physics_substep_recording':False,'actual_speed_average':'mean of 50Hz post-step body velocities within command interval'}
        if args.expert=='continuous':metadata['expert']=expert.metadata()
        metadata['path_expert_sha256']=sha(Path(__file__).resolve().parents[1]/'src/dual_target/r2r_path_expert.py')
        save(output/'manifest.json',metadata)
        # A freshly placed Go2 can briefly trigger base_contact/bad_orientation
        # while its joints settle. Retry the same episode a bounded number of
        # times; never silently continue after an actual reset.
        warmup_attempts=0
        while True:
            warmup_attempts += 1
            warmup_reset_start=len(reset_evidence)
            for i in range(100):
                pre=snapshot();a,_,done,_=step([0.,0.,0.]);post=reset_evidence[-1]['state'] if reset_evidence else snapshot()
                emit(warm,{'attempt':warmup_attempts,'index':i,'pre':pre,'post':post,'command':[0.,0.,0.]})
                if bool(done[0]) or len(reset_evidence)>warmup_reset_start:break
            if not bool(done[0]) and len(reset_evidence)==warmup_reset_start:break
            if warmup_attempts>=3:
                raise RuntimeError('warmup terminated after 3 bounded retries; retained pre-reset evidence')
            reset_evidence.clear()
            obs,infos=env.reset()
            base=env.unwrapped;robot=base.scene['robot'];camera=base.scene['rgbd_camera'];contacts=base.scene['contact_forces']
            physics_contacts.clear()
            time.sleep(.2)
        contacts.update(cfg.sim.dt,force_recompute=True)
        save(output/'contact_diagnostics.json',{
            'configured_disable_contact_processing':cfg.sim.disable_contact_processing,
            'runtime_disable_contact_processing':carb.settings.get_settings().get('/physics/disableContactProcessing'),
            'sensor_buffer':arr(contacts.data.net_forces_w),
            'direct_physx_forces':contacts.contact_physx_view.get_net_contact_forces(dt=cfg.sim.dt).detach().cpu().tolist(),
            'substep_peak_force_n':float(max((np.linalg.norm(f,axis=-1).max() for f in physics_contacts),default=0.)),
            'body_names':contacts.body_names})
        start_physics=int(base._sim_step_counter);capture_start=time.monotonic()
        frame_count=0;navila_ids=[];current_rgb=None;interval=None;rows=0;hold=0;reason='timeout';path_length=0.
        stop_positive=0;failed_interval=False;max_force=0.
        def frame(index,s,keyframe=False):
            nonlocal frame_count,current_rgb
            img=camera.data.output['rgb'][0].detach().cpu().numpy()[...,:3]
            assert img.shape==(512,512,3) and img.dtype==np.uint8
            name=f'rgb/{frame_count:06d}.jpg'
            assert cv2.imwrite(str(output/name),cv2.cvtColor(img,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,90])
            depth=camera.data.output['distance_to_image_plane'][0].detach().cpu().numpy()
            finite=np.isfinite(depth)&(depth>0)&(depth<100)
            emit(framelog,{'frame_id':frame_count,'path':name,'control_index':index,'physics_step':s['physics_step'],
                'sim_time_s':s['sim_time_s'],'episode_time_s':(s['physics_step']-start_physics)*cfg.sim.dt,
                'camera_frame_counter':int(camera.frame[0]),'rgb_std':float(img.std()),
                'camera_position_w':arr(camera.data.pos_w),'camera_quaternion_ros_wxyz':arr(camera.data.quat_w_ros),
                'depth_valid_fraction':float(finite.mean()),'navila_keyframe':keyframe})
            current_rgb=frame_count
            if keyframe:navila_ids.append(frame_count)
            frame_count+=1
            return current_rgb
        def flush_interval():
            if interval is None:return
            samples=interval.pop('_actual_samples')
            interval['mean_actual_velocity_body']=np.mean(samples,axis=0).tolist()
            interval['duration_s']=interval['executed_low_steps']*.02
            interval['complete_command_interval']=interval['executed_low_steps']==10
            interval['post_state']=post
            interval['displacement_world']=(np.asarray(post['position_w'])-np.asarray(interval['pre_state']['position_w'])).tolist()
            emit(highlog,interval)
        print('COLLECTION_START '+args.episode_id,flush=True)
        for i in range(int(args.max_seconds*50)):
            pre=snapshot()
            if i%10==0 or i%25==0:frame(i,pre,keyframe=i%25==0)
            if i%10==0:
                raw,diagnostics=expert.command(pre);applied=list(boundary.update(raw))
                stop_positive+=int(diagnostics['stop_intent'])
                interval={'action_index':i//10,'low_start_index':i,'rgb_frame_id':current_rgb,
                    'history':history8(navila_ids),'history_action_indices':list(range(max(0,i//10-7),i//10)),
                    'pre_state':pre,'raw_command':raw,'applied_command':applied,'expert':diagnostics,
                    'stop_intent':diagnostics['stop_intent'],'command_zero':max(abs(v) for v in applied)<1e-8,
                    'executed_low_steps':0,'_actual_samples':[]}
            a,reward,done,info=step(applied)
            reset=bool(reset_evidence)
            post=reset_evidence[-1]['state'] if reset else snapshot()
            body=[post['linear_body'][0],post['linear_body'][1],post['angular_body'][2]]
            goal_distance=math.dist(post['position_w'][:2],expert.goal[:2])
            zero=max(abs(v) for v in applied)<1e-8
            stopped=still(post)
            task_region=goal_distance<=.30 and expert.index==len(expert.path)-1
            hold=hold+1 if diagnostics['stop_intent'] and zero and stopped and task_region and not reset else 0
            forces=np.linalg.norm(np.asarray(post['contact_force_world']),axis=-1)
            max_force=max(max_force,float(forces.max()))
            path_length+=math.dist(pre['position_w'][:2],post['position_w'][:2])
            record={'low_index':i,'action_index':i//10,'rgb_frame_id':current_rgb,'pre':pre,'post':post,
                'raw_command':raw,'applied_command':applied,'low_level_action':arr(a),
                'reward':float(reward[0]),'stop_intent':diagnostics['stop_intent'],'command_zero':zero,
                'actual_stationary':stopped,'inside_stop_region':task_region,'hold_steps':hold,
                'environment_done':bool(done[0]),'pre_reset_state_used':reset,
                'terminated':bool(reset_evidence[-1]['terminated']) if reset else False,
                'truncated':bool(reset_evidence[-1]['truncated']) if reset else False}
            emit(lowlog,record);rows+=1
            interval['_actual_samples'].append(body);interval['executed_low_steps']+=1
            interval['actual_stationary_post']=stopped;interval['inside_stop_region_post']=task_region
            interval['hold_steps_post']=hold;interval['environment_done']=bool(done[0])
            success=hold>=100
            ending=success or bool(done[0]) or reset or i==int(args.max_seconds*50)-1
            if i%10==9 or ending:flush_interval();interval=None
            if i%250==0:print(f'PROGRESS steps={i} waypoint={expert.index}/{len(expert.path)} goal={goal_distance:.3f} hold={hold}',flush=True)
            if ending:
                reason='success' if success else ('environment_termination' if reset or bool(done[0]) else 'timeout')
                if not reset:frame(i+1,post)
                break
        wall=time.monotonic()-capture_start
        save(output/'reset_events.json',reset_evidence)
        for f in streams:f.flush()
        sizes={name:sum(p.stat().st_size for p in (output/name).glob('*')) if (output/name).is_dir() else (output/name).stat().st_size for name in ['rgb','low_level.jsonl','actions.jsonl','frames.jsonl','warmup.jsonl']}
        sim_seconds=rows*.02
        summary={'status':'complete','success':reason=='success','training_eligible':False,'termination_reason':reason,
            'low_level_records':rows,'rgb_frames':frame_count,'action_records':(rows+9)//10,'stop_positive_actions':stop_positive,
            'final_hold_steps':hold,'final_goal_distance_xy_m':goal_distance,'path_length_xy_m':path_length,
            'sim_seconds':sim_seconds,'collection_wall_seconds':wall,'realtime_factor':sim_seconds/wall,
            'bytes_by_category':sizes,'total_recorded_bytes':sum(sizes.values()),
            'projected_bytes_per_sim_minute':sum(sizes.values())/sim_seconds*60,'maximum_contact_force_n':max_force,
            'reset_count':len(reset_evidence),'note':'training eligibility requires separate audit; contact includes normal foot support'}
        save(output/'summary.json',summary)
        print('COLLECTION_RESULT '+json.dumps(summary),flush=True)
    except BaseException as e:
        summary.update(status='failed',success=False,training_eligible=False,error=repr(e))
        save(output/'summary.json',summary)
        if 'reset_evidence' in locals():save(output/'reset_events.json',reset_evidence)
        raise
    finally:
        for f in streams:f.close()
        app.close()


def main():
    sys.path.insert(0,str(ROOT/'third_party/NaVILA-Bench/scripts'))
    import cli_args
    from omni.isaac.lab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--episode-id',default='1')
    p.add_argument('--max-seconds',type=float,default=120.)
    p.add_argument('--expert',choices=['legacy','continuous'],default='legacy',
                   help='continuous is an opt-in candidate until live comparison passes')
    p.add_argument('--asset-root',type=Path,default=ROOT/'assets/isaac_sim_4_1')
    p.add_argument('--navila-root',type=Path,default=ROOT/'third_party/NaVILA-Bench')
    p.add_argument('--task',default='go2_matterport_vision');p.add_argument('--history_length',type=int,default=9)
    p.add_argument('--seed',type=int,default=20260910);p.add_argument('--use_cnn',action='store_true',default=None)
    p.add_argument('--use_rnn',action='store_true',default=False)
    cli_args.add_rsl_rl_args(p);AppLauncher.add_app_launcher_args(p)
    args=p.parse_args();run(args)


if __name__=='__main__':main()
