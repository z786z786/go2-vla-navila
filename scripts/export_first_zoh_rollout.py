"""Export the completed first validation rollout without changing its evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();ep=a.episode.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    pre=[json.loads(l) for l in (ep/'pre_action.jsonl').read_text().splitlines()]
    post=[json.loads(l) for l in (ep/'post_step_events.jsonl').read_text().splitlines()]
    assert len(pre)==len(post)==1500
    assert all(abs(b['sim_time_after_s']-a['sim_time_s']-.02)<1e-5 for a,b in zip(pre,post))
    result=json.loads((ep.parents[1]/'stage_status.json').read_text())['episode_result']
    assert result['status']=='FAILED_TIMEOUT'
    frames=[ep/pre[0]['rgb_path']]+[ep/'rgb_post'/f"{b['observation_seq_after']:06d}.png" for b in post]
    assert all(p.is_file() for p in frames)
    video=out/'first_validation_rollout.mp4'
    vf=("pad=512:608:0:64:color=white,"
        "drawtext=text='Go to RED box and stop in front.':x=12:y=9:fontsize=21:fontcolor=black,"
        "drawtext=text='Checkpoint 1000 | Validation 01 | 1x simulation':x=12:y=38:fontsize=16:fontcolor=black,"
        "drawtext=text='Outcome - TIMEOUT at 30s | no collision':x=12:y=584:fontsize=17:fontcolor=black")
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-n','-threads','2',
         '-f','image2pipe','-framerate','50','-vcodec','png','-i','pipe:0',
         '-filter_threads','1','-vf',vf,'-an','-c:v','libx264','-threads','2',
         '-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    with (out/'encode.log').open('x') as log:
        child=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        try:
            for path in frames:child.stdin.write(path.read_bytes())
            child.stdin.close()
            if child.wait()!=0:raise RuntimeError('ffmpeg failed; see encode.log')
        finally:
            if child.poll() is None:child.terminate();child.wait(timeout=10)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,nb_frames,r_frame_rate,duration','-of','json',str(video)],text=True))
    stream=probe['streams'][0];assert int(stream['nb_frames'])==len(frames)
    assert abs(float(stream['duration'])-len(frames)/50)<.001
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-ss','15','-i',str(video),
        '-frames:v','1','-threads','1',str(out/'preview.png')],check=True)
    meta=dict(source_episode=str(ep),source_frames=len(frames),fps=50,simulation_duration_s=30,
        video_duration_s=len(frames)/50,initial_frame_included=True,inference_wall_waits_excluded=True,
        result=result['status'],probe=probe,sha256=hashlib.sha256(video.read_bytes()).hexdigest())
    (out/'export.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))


if __name__=='__main__':main()
