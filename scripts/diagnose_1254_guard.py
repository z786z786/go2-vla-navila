import os,sys,time,subprocess,signal,json
from pathlib import Path
sys.path.insert(0,'/home/wxh/go2_short_vln')
from src.dual_target.coexistence_watch import sample_group
root=Path('/mnt/wxh/go2_short_vln/outputs/r2r_ablation/diag1254_static_v1')
root.mkdir(exist_ok=False)
for i in range(3):
    s=sample_group(-1)
    if s['free_mib']<10240:raise RuntimeError('admission below 10GiB')
    if i<2:time.sleep(30)
env=os.environ.copy();env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None);env['OMNI_KIT_ACCEPT_EULA']='YES'
cmd=['/mnt/wxh/go2_short_vln/envs/conda/navila-isaac/bin/python','/home/wxh/go2_short_vln/scripts/validate_r2r_episode_load.py','--episode-id','1254','--output',str(root/'data'),'--render','--headless','--enable_cameras']
with (root/'run.log').open('x') as log,(root/'gpu.jsonl').open('x',buffering=1) as gpu:
    p=subprocess.Popen(cmd,cwd='/home/wxh/go2_short_vln',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    start=time.monotonic();result={}
    try:
        while p.poll() is None:
            s=sample_group(p.pid);gpu.write(json.dumps(s)+'\n')
            if s['free_mib']<2048:raise RuntimeError('memory protection')
            if time.monotonic()-start>2400:raise RuntimeError('wall timeout')
            time.sleep(1)
        result['exit_code']=p.returncode
    except BaseException as e:result['error']=repr(e)
    finally:
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        (root/'guard.json').write_text(json.dumps(result))
print(result)
