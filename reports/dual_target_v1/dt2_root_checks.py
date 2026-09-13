"""Root's separate raw/LeRobot numerical recomputation, never auto-approves DT2."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from src.dual_target.tiny_plan import read,sha,verify_binding,SOURCE_ROOT,build_plan
from src.dual_target.contracts import IMAGE_KEY,STATE_KEY
from src.dual_target.reset_audit import reset_audit_from_dict,validate_paired_resets
from reports.dual_target_v1.dt1_independent_checks import review_trace


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def review(plan_path,dataset_root,visual_path):
    plan = read(plan_path)
    assert plan == build_plan(plan_path.parent)
    contract = verify_binding(SOURCE_ROOT)
    manifest = read(dataset_root/'dataset_manifest.json')
    assert manifest['status'] == 'CONVERTED_NOT_DT2_APPROVED'
    assert manifest['plan_sha256'] == sha(plan_path)
    assert manifest['task_contract_sha256'] == plan['task_contract_sha256']
    for name,digest in manifest['files_sha256'].items():
        assert sha(dataset_root/name) == digest, name
    visual = read(visual_path)
    assert len(visual) == 16 and all(v['target_visible_throughout'] for v in visual)
    dataset = LeRobotDataset('local/dual_target_v1_tiny',root=dataset_root/'lerobot',
                            delta_timestamps={'action':[i*.02 for i in range(50)]})
    assert set(dataset.features) == {IMAGE_KEY,STATE_KEY,'action','timestamp','frame_index','episode_index','index','task_index'}
    weights = np.load(dataset_root/'episode_balanced_sampler_weights.npy')
    normalizer = read(dataset_root/'normalizer.json')
    groups, states, actions, reviews, resets = set(),[],[],[],{}
    offset = checked = 0
    actual_all_images = 0
    for index,slot in enumerate(plan['slots']):
        p = Path(slot['episode_path'])
        m = read(p/'episode_manifest.json')
        pre,post = rows(p/'pre_action.jsonl'),rows(p/'post_step_events.jsonl')
        assert len(pre) == len(post) and len(pre) > 51
        assert visual[index]['episode'] == str(p)
        assert m['slot_a_center_xy'] == slot['scene']['slot_a_center_xy']
        assert m['slot_b_center_xy'] == slot['scene']['slot_b_center_xy']
        assert m['start_xy'] == slot['scene']['start_xy']
        assert m['thresholds'] == contract['parking_thresholds']
        for field in ('geometry_group_id','color_configuration','target_color','instruction','target_slot'):
            assert m[field] == slot[field]
        groups.add(slot['geometry_group_id'])
        assert m['seed'] == slot['pair_seed'] and m['repeat'] == 0
        trace = review_trace(pre,post,{'correct_parking_center_xy':m['parking_centers_xy'][m['target_slot']],
            'other_parking_center_xy':m['parking_centers_xy']['B' if m['target_slot']=='A' else 'A'],
            'thresholds':m['thresholds'],'physics_dt_s':.005,'decimation':4,'contact_threshold_n':1.})
        assert post[-1]['scorer_status'] == 'success'
        raw_state = np.asarray([r['body_velocity_body'] for r in pre],dtype=np.float64)
        raw_action = np.asarray([r['applied_action'] for r in pre],dtype=np.float64)
        assert np.isfinite(raw_state).all() and np.isfinite(raw_action).all()
        assert np.array_equal(raw_action[:,1],np.zeros(len(pre)))
        assert all(np.array_equal(r['raw_action'],r['applied_action']) for r in pre)
        assert np.equal(raw_action[-51:],0).all()
        assert (np.linalg.norm(raw_state[-51:,:2],axis=1) < .03).all()
        assert (np.abs(raw_state[-51:,2]) < .05).all()
        assert pre[-1]['sim_time_s']-pre[-51]['sim_time_s'] >= 1-1e-6
        assert all(r['in_correct_parking_region'] for r in post[-51:])
        pair = reset_audit_from_dict(read(p/'paired_reset_audit.json'))
        resets.setdefault((slot['geometry_group_id'],slot['color_configuration']),[]).append(pair)
        n = len(pre)
        # All image artifacts must exist and match converter provenance; loader
        # pixels/chunks are separately recomputed at deterministic diverse anchors.
        audit = read(dataset_root/'raw_audit.json')[index]
        assert audit['source_manifest_sha256'] == sha(p.parent.parent/'source_manifest.json')
        for i,r in enumerate(pre):
            assert r['task'] == slot['instruction'] and sha(p/r['rgb_path']) == audit['rgb_sha256'][i]
        assert np.allclose(weights[offset:offset+n],1/(16*n),rtol=0,atol=1e-15)
        anchors = sorted({0,2,n//5,2*n//5,3*n//5,4*n//5,n-51,n-50,n-1})
        for anchor in anchors:
            item = dataset[offset+anchor]
            assert item['episode_index'].item() == index and item['frame_index'].item() == anchor
            assert item['task'] == slot['instruction']
            assert np.allclose(item[STATE_KEY].numpy(),raw_state[anchor],rtol=1e-6,atol=1e-7)
            query = np.arange(anchor,anchor+50)
            expected = raw_action[np.minimum(query,n-1)].astype(np.float32)
            assert np.array_equal(item['action'].numpy(),expected)
            assert np.array_equal(item['action_is_pad'].numpy(),query>=n)
            raw_rgb = np.asarray(Image.open(p/pre[anchor]['rgb_path']).convert('RGB'))
            assert np.array_equal((item[IMAGE_KEY].permute(1,2,0).numpy()*255).round().astype(np.uint8),raw_rgb)
            checked += 1
        states.append(raw_state)
        actions.append(raw_action)
        offset += n
        actual_all_images += visual[index]['real_image_count']
        reviews.append({'slot':index,'frames':n,'trace':trace,'real_terminal_pre_frames_checked':51,
                        'target_minimum_color_pixels':visual[index]['minimum_color_pixels'][slot['target_color']]})
    assert len(groups) == 4 and len(resets) == 8
    for pair in resets.values():
        assert len(pair) == 2
        validate_paired_resets(*pair)
    for name,parts in (('state',states),('action',actions)):
        array = np.concatenate(parts)
        stat = normalizer[name]
        assert stat['count'] == len(array) == manifest['frames']
        std = array.std(axis=0)
        for key,expected in (('mean',array.mean(axis=0)),('raw_std',std),
                             ('std',np.where(std<1e-6,1,std)),('min',array.min(axis=0)),('max',array.max(axis=0))):
            assert np.allclose(stat[key],expected,rtol=1e-9,atol=1e-12), (name,key)
    assert offset == len(dataset) == len(weights)
    assert np.isclose(weights.sum(),1) and normalizer['action_loss_active_channels'] == [True,False,True]
    events = rows(plan_path.parent/'collection_events.jsonl')
    launches = [e['slot'] for e in events if e['event']=='launching']
    exits = [e for e in events if e['event']=='child_exited']
    assert launches == list(range(8,16)) and len(exits)==8 and all(e['exit_code']==0 for e in exits)
    samples = [r for p in plan_path.parent.glob('*_gpu_samples.jsonl') for r in rows(p)]
    assert samples and min(r['free_mib'] for r in samples)>=2048
    return {'stage':'DT2','status':'ROOT_NUMERICAL_CHECKS_PASS_NOT_APPROVAL',
        'review_mode':'single_agent_separate_raw_and_real_loader_recomputation',
        'episodes':16,'geometry_groups':4,'paired_resets':8,'frames':offset,
        'real_images_audited':actual_all_images,'root_loader_anchors':checked,
        'fresh_normalizer_numpy_recomputed':True,'sampler_weights_recomputed':True,
        'train_only':True,'truth_excluded_from_dataset':True,'per_slot_retries':0,
        'gpu_samples':len(samples),'owned_peak_mib':max(r['owned_used_mib'] for r in samples),
        'minimum_actual_free_mib':min(r['free_mib'] for r in samples),
        'plan_sha256':sha(plan_path),'dataset_manifest_sha256':sha(dataset_root/'dataset_manifest.json'),
        'visual_review_sha256':sha(visual_path),'episodes_review':reviews,
        'training_started':False,'navigation_success_approved':False}


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--tiny-plan',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--visual',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=review(a.tiny_plan,a.dataset,a.visual)
    with a.output.open('x') as f:
        f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='episodes_review'},indent=2))
