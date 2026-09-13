"""Root's offline resource evidence checks; never approves navigation or DT1."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review(root):
    def read(name):
        return json.loads((root/name).read_text())
    result, model, request, process = map(read, (
        'coexistence_result.json', 'model_result.json', 'model_input.json', 'model_process.json'))
    samples = [json.loads(line) for line in (root/'gpu_samples.jsonl').read_text().splitlines()]
    assert result['status'] == 'COEXISTENCE_PASSED_NOT_DT1_APPROVED'
    assert result['model_result_sha256'] == sha(root/'model_result.json')
    assert result['input_sha256'] == sha(root/'model_input.json')
    assert request['rgb_path'] == 'front.png'
    assert request['rgb_sha256'] == model['input_rgb_sha256'] == sha(root/'front.png')
    assert len(request['state']) == 3 and all(math.isfinite(v) for v in request['state'])
    assert set(request) == {
        'schema_version', 'rgb_path', 'rgb_sha256', 'state', 'task', 'action_dim',
        'action_chunk_size', 'execute_action_steps', 'normalization', 'dataset_stats',
        'inference_only', 'execute_model_actions', 'training', 'navigation_success_approved', 'dt1_approved'}
    assert request['task'] in ('Go to the red box and stop in front of it.',
                               'Go to the blue box and stop in front of it.')
    assert request['inference_only'] is True
    for item in (result, model, request):
        assert all(item[k] is False for k in ('execute_model_actions', 'training', 'dt1_approved'))
    assert model['navigation_success_approved'] is False
    assert model['output_shape'] == [1, 50, 3] and model['output_finite'] is True
    assert model['model_sha256'] == '7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb'
    assert model['config_sha256'] == '650584b56c104720f7a3c91d1ec6bec9e8de8ac11e60c92ba2fa82d93eda147d'
    assert model['interface']['state_dim'] == model['interface']['action_dim'] == 3
    assert model['interface']['chunk_size'] == 50
    assert model['interface']['execute_action_steps'] == 10
    assert result['clock_before'] == result['clock_after']
    assert result['pose_before'] == result['pose_after']
    assert result['physics_advanced'] is False
    owned = {process['isaac_pid'], process['model_pid']}
    policy = read('resource_policy.json')
    allowed = set(policy['allowed_foreign_pids'])
    assert len(owned) == 2 and all(isinstance(p, int) and p > 1 for p in owned)
    assert samples and any(owned <= set(s['compute_pids']) for s in samples)
    assert all(not (set(s['compute_pids']) - owned - allowed) for s in samples)
    peak, free = max(s['used_mib'] for s in samples), min(s['free_mib'] for s in samples)
    assert free >= 2048
    assert peak == result['sampled_total_peak_mib'] and free == result['sampled_min_free_mib']
    assert all(a['wall_time_s'] < b['wall_time_s'] for a, b in zip(samples, samples[1:]))
    launcher = root.parent/(root.name+'_launcher_receipt.json')
    whole_run = root.parent/(root.name+'_startup_gpu_samples.jsonl')
    launch = json.loads(launcher.read_text())
    startup = [json.loads(line) for line in whole_run.read_text().splitlines()]
    assert startup and all(s['owned_pgid'] == launch['pgid'] for s in startup)
    assert all(s['free_mib'] >= 2048 for s in startup)
    assert all(s['owned_used_mib'] == sum(p['used_mib'] for p in s['processes'] if p['owned']) for s in startup)
    owned_peak = max(s['owned_used_mib'] for s in startup)
    return {'coexistence_evidence_passed': True, 'reviewer': 'root_same_agent_recomputation',
            'sample_count': len(samples), 'sampled_total_peak_mib': peak,
            'sampled_min_free_mib': free, 'model_timing_ms': model['timing_ms'],
            'admission_policy': policy['admission_policy'], 'allowed_foreign_pids': sorted(allowed),
            'whole_run_sampled_owned_peak_mib': owned_peak,
            'whole_run_sampled_total_peak_mib': max(s['used_mib'] for s in startup),
            'whole_run_min_free_mib': min(s['free_mib'] for s in startup),
            'whole_run_sample_count': len(startup),
            'launcher_sha256': sha(launcher), 'whole_run_samples_sha256': sha(whole_run),
            'sampling_scope': 'Isaac-resident model loading and inference; not continuous global peak',
            'model_actions_executed': False, 'navigation_success_approved': False,
            'dt1_approved': False,
            'evidence_sha256': {str(p): sha(p) for p in sorted(root.iterdir())
                                if p.is_file() and p.name != 'root_coexistence_review.json'}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = review(args.run_dir)
    encoded = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as output:
            output.write(encoded)
    print(encoded)
