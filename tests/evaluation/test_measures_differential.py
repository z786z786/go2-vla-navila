import importlib.util, json, random, unittest
from pathlib import Path
import numpy as np
from evaluation.episodes import load_dev100
from evaluation.measures import evaluate_trajectory

OFFICIAL = Path('/mnt/wxh/go2_short_vln/third_party/NaVILA-Bench/isaaclab_exts/omni.isaac.vlnce/omni/isaac/vlnce/utils/measures.py')
spec = importlib.util.spec_from_file_location('official_measures', OFFICIAL)
official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)

class Tensor:
    def __init__(self, value): self.value = np.asarray(value)
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value
class Robot:
    def __init__(self, p): self.data = type('D', (), {'root_pos_w':[Tensor(p)]})()
class Env:
    def __init__(self, p): self.is_stop_called=False; self.unwrapped=self; self.scene={'robot':Robot(p)}
    def move(self,p): self.scene['robot'].data.root_pos_w[0]=Tensor(p)

def run_official(positions, stop, ep):
    env=Env(positions[0]); manager=official.add_measurement(env, ep); manager.reset_measures(); out=[manager.get_measurements()]
    for i,p in enumerate(positions[1:],1):
        env.move(p); env.is_stop_called=stop is not None and i>=stop; manager.update_measures(); out.append(manager.get_measurements())
    return out

class DifferentialTests(unittest.TestCase):
    def test_official_direct_stepwise_random_and_dev100(self):
        rng=random.Random(20260913); trajectories=[]
        for _ in range(200):
            n=rng.randint(2,12); base=np.array([rng.uniform(-4,4) for _ in range(3)])
            pts=[base.copy()]
            for i in range(n-1):
                if i%4==0: pts.append(pts[-1].copy())
                else: pts.append(pts[-1]+np.array([rng.uniform(-2,2) for _ in range(3)]))
            stop=rng.choice([None,1,n-1])
            trajectories.append((pts,stop,{'gt_locations':[[0.,0.,0.],[3.,0.,0.],[3.,3.,0.]],'goals':[{'radius':3.0}]}))
        for ep in load_dev100():
            pts=np.asarray(ep['gt_locations'],dtype=float); trajectories.append((pts, len(pts)//2, ep))
        counters={'trajectories':0,'step_comparisons':0,'metric_comparisons':0,'mismatches':0}
        for pts,stop,ep in trajectories:
            replica=evaluate_trajectory(pts,stop,ep)['steps']; truth=run_official(pts,stop,ep); counters['trajectories']+=1
            keys=('path_length','distance_to_goal','success','spl','oracle_navigation_error','oracle_success')
            for a,b in zip(replica,truth):
                counters['step_comparisons']+=1
                for k in keys:
                    counters['metric_comparisons']+=1
                    equal=(a[k]==b[k]) or (isinstance(a[k],float) and isinstance(b[k],float) and np.isnan(a[k]) and np.isnan(b[k]))
                    if not equal: counters['mismatches']+=1
        self.assertGreater(counters['step_comparisons'],0); self.assertEqual(counters['mismatches'],0)
        Path('reports/p1').mkdir(parents=True,exist_ok=True); Path('reports/p1/p1e_cpu_evidence.json').write_text(json.dumps(counters,indent=2)+'\n')
