"""Wait for the one existing collection, then run CPU-only conversion/reviews.

This controller never collects, trains or issues milestone approval.
"""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time

base = Path('/mnt/wxh/go2_short_vln')
batch = base/'outputs/dual_target_v1/dt2_tiny_0906_v2'
dataset = base/'outputs/dual_target_v1/dt2_tiny_dataset_0906_v1'
visual = base/'outputs/dual_target_v1/dt2_tiny_0906_v2_visual'
source = Path('/home/wxh/go2_short_vln')
result_path = batch/'collection_result.json'
started = time.monotonic()
while not result_path.exists():
    if time.monotonic()-started > 7200:
        raise TimeoutError('existing collection did not finish within 2h; no replacement launched')
    time.sleep(10)
result = json.loads(result_path.read_text())
assert result['status'] == 'COLLECTION_COMPLETE_NOT_DT2_APPROVED', result
plan = json.loads((batch/'tiny_plan.json').read_text())
env = dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
           PYTHONPATH=f'{source}:{base}/third_party/lerobot/src',PYTHONUNBUFFERED='1')
python = str(base/'envs/conda/smolvla/bin/python')
def run(name, command):
    print(json.dumps({'event':'cpu_check_started','check':name}),flush=True)
    with (batch/f'{name}.log').open('x') as log:
        subprocess.run(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
    print(json.dumps({'event':'cpu_check_completed','check':name}),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    conversion = pool.submit(run,'full_conversion',[python,'-m','src.dual_target.tiny_convert',
        '--tiny-plan',str(batch/'tiny_plan.json'),'--output',str(dataset)])
    images = pool.submit(run,'full_visual',[str(base/'envs/conda/navila-isaac/bin/python'),
        '/tmp/dt1_matrix_visual_review_0906.py','--output-dir',str(visual),
        *[s['episode_path'] for s in plan['slots']]])
    conversion.result()
    images.result()
run('root_checks',[python,'/tmp/dt2_root_checks_0906.py','--tiny-plan',str(batch/'tiny_plan.json'),
    '--dataset',str(dataset),'--visual',str(visual/'visual_review.json'),
    '--output',str(batch/'root_dataset_review.json')])
print('DT2 CPU checks complete; root manual review still required; no training started',flush=True)
