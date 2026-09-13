"""CPU video export of every actual executed frame, including partial final holds."""
import json
from pathlib import Path
import subprocess
from scripts.zoh_compare_common import sha

def export_video(ep,task,result,directory):
    ep=Path(ep);directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    pre=[json.loads(x) for x in (ep/'pre_action.jsonl').read_text().splitlines()]
    post=[json.loads(x) for x in (ep/'post_step_events.jsonl').read_text().splitlines()]
    if len(pre)!=len(post) or not pre or len(pre)!=result['steps']:raise ValueError('video evidence length mismatch')
    if any(abs(b['sim_time_after_s']-a['sim_time_s']-.02)>1e-5 for a,b in zip(pre,post)):
        raise ValueError('video time grid mismatch')
    frames=[ep/pre[0]['rgb_path']]+[ep/'rgb_post'/f"{r['observation_seq_after']:06d}.png" for r in post]
    if not all(p.is_file() for p in frames):raise ValueError('video source image missing')
    name=task['run_id'];video=directory/(name+'.mp4');preview=directory/(name+'.png')
    color='RED' if 'red box' in pre[0]['task'] else 'BLUE'
    label=result['status'].replace('_',' ').upper()
    vf=("pad=512:608:0:64:color=white,"
        f"drawtext=text='Go to {color} box and stop in front.':x=10:y=9:fontsize=21:fontcolor=black,"
        f"drawtext=text='{task['condition']} | {task['evaluation_split']} {task['slot']:02d}':x=10:y=38:fontsize=16:fontcolor=black,"
        f"drawtext=text='{label} | 1x simulation':x=10:y=584:fontsize=16:fontcolor=black")
    with (directory/(name+'.encode.log')).open('x') as log:
        child=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n','-threads','2','-f','image2pipe',
            '-framerate','50','-vcodec','png','-i','pipe:0','-filter_threads','1','-vf',vf,'-an','-c:v','libx264',
            '-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],
            stdin=subprocess.PIPE,stderr=log)
        try:
            for frame in frames:child.stdin.write(frame.read_bytes())
            child.stdin.close()
            if child.wait()!=0:raise RuntimeError('ffmpeg encoding failed')
        finally:
            if child.poll() is None:child.terminate();child.wait(timeout=10)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=nb_frames,duration','-of','json',str(video)],text=True))['streams'][0]
    if int(probe['nb_frames'])!=len(frames) or abs(float(probe['duration'])-len(frames)/50)>.001:
        raise ValueError('encoded frame count/duration mismatch')
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-ss',str(len(pre)*.02*.8),'-i',str(video),
        '-frames:v','1','-threads','1',str(preview)],check=True)
    return dict(video='videos/'+video.name,video_sha256=sha(video),video_frames=len(frames),
        video_duration_s=float(probe['duration']),terminal_partial_hold_steps=len(pre)%10,
        video_note='50 Hz actual executed frames plus initial frame; partial final hold retained at actual duration')
