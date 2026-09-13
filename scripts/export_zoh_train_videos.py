"""Export only the current 16 training expert trajectories using CPU ffmpeg."""
import hashlib
import json
from pathlib import Path
import subprocess

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def rows(p):
    return [json.loads(x) for x in p.read_text().splitlines()]

def main():
    root=Path('/mnt/wxh/go2_short_vln/outputs/dual_target_v2/zoh_dataset_0906_v2')
    approval=json.loads(Path('/home/wxh/go2_short_vln/reports/dual_target_v1/zoh_v2_dataset_approval.json').read_text())
    assert approval['status']=='APPROVED'
    assert sha(root/'split_manifest.json')==approval['split_manifest_sha256']
    assert sha(root/'dataset/dataset_manifest.json')==approval['dataset_manifest_sha256']
    slots=[s for s in json.loads((root/'split_manifest.json').read_text())['slots'] if s['split']=='train']
    assert len(slots)==16 and [s['index'] for s in slots]==list(range(16))
    out=root.parent/'zoh_train_expert_videos_0906_v1';out.mkdir(exist_ok=False)
    exports=[];previews=[]
    for slot in slots:
        ep=Path(slot['episode_path']);pre=rows(ep/'pre_action.jsonl');post=rows(ep/'post_step_events.jsonl')
        high=rows(ep/'high_level_pre_action.jsonl')
        assert len(pre)==len(post)==10*len(high)
        assert all(a['task']==slot['instruction'] for a in pre)
        assert all(abs(b['sim_time_after_s']-a['sim_time_s']-.02)<1e-5 for a,b in zip(pre,post))
        assert all(not any(b['collision_latched_substeps']) and not b['fallen'] for b in post)
        frames=[ep/pre[0]['rgb_path']]+[ep/'rgb_post'/f"{b['observation_seq_after']:06d}.png" for b in post]
        assert all(p.is_file() for p in frames)
        duration=post[-1]['sim_time_after_s']-pre[0]['sim_time_s']
        name=f"{slot['index']+1:02d}_{slot['trajectory_id']}"
        video=out/(name+'.mp4');preview=out/(name+'.png')
        vf=("pad=512:608:0:64:color=white,"
            f"drawtext=text='Go to {slot['target_color'].upper()} box and stop in front.':x=12:y=9:fontsize=21:fontcolor=black,"
            f"drawtext=text='TRAIN {slot['index']+1:02d}/16 | Expert | 1x simulation':x=12:y=38:fontsize=18:fontcolor=black,"
            f"drawtext=text='{slot['geometry_group_id']} | {slot['color_configuration']}':x=12:y=584:fontsize=16:fontcolor=black")
        with (out/(name+'.encode.log')).open('x') as log:
            cmd=['ffmpeg','-hide_banner','-loglevel','error','-n','-threads','2','-f','image2pipe','-framerate','50',
                '-vcodec','png','-i','pipe:0','-filter_threads','1','-vf',vf,'-an','-c:v','libx264','-threads','2',
                '-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
            child=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
            try:
                for frame in frames:child.stdin.write(frame.read_bytes())
                child.stdin.close()
                assert child.wait()==0,'encoder failed'
            finally:
                if child.poll() is None:child.terminate();child.wait(timeout=10)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=nb_frames,duration,width,height,r_frame_rate','-of','json',str(video)],text=True))['streams'][0]
        assert int(probe['nb_frames'])==len(frames) and abs(float(probe['duration'])-len(frames)/50)<.001
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-ss',str(duration/2),'-i',str(video),
            '-frames:v','1','-threads','1',str(preview)],check=True)
        previews.append(preview)
        record=dict(number=slot['index']+1,trajectory_id=slot['trajectory_id'],instruction=slot['instruction'],
            source_episode=str(ep),video=video.name,video_sha256=sha(video),preview=preview.name,
            training_frames=len(high),video_frames=len(frames),fps=50,simulation_duration_s=duration,
            video_duration_s=float(probe['duration']),initial_frame_included=True,
            evidence_sha256={n:sha(ep/n) for n in ('pre_action.jsonl','post_step_events.jsonl','high_level_pre_action.jsonl')})
        exports.append(record)
        (out/'export_manifest.json').write_text(json.dumps(dict(status='EXPORTING',exports=exports),indent=2))
        print(json.dumps(dict(number=record['number'],video=video.name,seconds=duration)),flush=True)
    assert sum(r['training_frames'] for r in exports)==770
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','image2pipe','-vcodec','png','-i','pipe:0',
        '-filter_threads','1','-vf','scale=256:304,tile=4x4','-frames:v','1','-threads','1',str(out/'overview.png')],
        input=b''.join(p.read_bytes() for p in previews),check=True)
    (out/'export_manifest.json').write_text(json.dumps(dict(status='COMPLETE',source_dataset=str(root),
        split='train',episodes=16,training_fps=5,training_frames=770,video_fps=50,
        note='Original full-rate expert RGB; no warmup or inference waits. One initial frame included; not model rollouts.',
        split_manifest_sha256=sha(root/'split_manifest.json'),exports=exports),indent=2))
    lines=['# 当前训练集：16 条专家轨迹视频','',
        '来自 zoh_dataset_0906_v2 的 train split，供当前 ZOH 5k 模型训练。不是旧版数据，也不是模型 rollout。',
        '视频使用原始 50 Hz 图像，仿真时间 1×，含起始画面和末尾实际停稳段；训练实际使用其中 5 Hz 观测，共 770 帧。',
        '不含 warmup；每条因加入起始帧比动作时长多 0.02 秒。','',
        '|编号|轨迹|指令|时长|视频|','|---|---|---|---|---|']
    for r in exports:
        target='红箱子' if 'red box' in r['instruction'] else '蓝箱子'
        lines.append(f"|{r['number']:02d}|{r['trajectory_id']}|走到{target}前停稳|{r['video_duration_s']:.2f}s|[播放]({r['video']})|")
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='COMPLETE',episodes=16,output=str(out))),flush=True)

if __name__=='__main__':main()
