"""One DT2 eight-episode collection queue; fail stops without automatic retry."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .tiny_plan import SOURCE_ROOT, DATA_ROOT, build_plan, read, sha
from .tiny_runtime import SHARED_DT2_POLICY
from .gpu_wait import Dt1GpuWaiter, EXCLUSIVE_DT1_POLICY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiny-plan', type=Path, required=True)
    parser.add_argument('--shared-dt2', action='store_true')
    args = parser.parse_args()
    plan_path = args.tiny_plan.resolve()
    plan = read(plan_path)
    if plan != build_plan(plan_path.parent):
        raise ValueError('noncanonical DT2 plan')
    root = plan_path.parent
    receipt = {'stage': 'DT2', 'pid': os.getpid(), 'actual_argv': sys.argv,
               'plan_sha256': sha(plan_path), 'shared_dt2': args.shared_dt2,
               'policy': SHARED_DT2_POLICY.policy_id if args.shared_dt2 else EXCLUSIVE_DT1_POLICY.policy_id,
               'training': False, 'dt3_started': False}
    with (root/'collection_receipt.json').open('x') as f:
        json.dump(receipt, f, indent=2)
    policy = SHARED_DT2_POLICY if args.shared_dt2 else EXCLUSIVE_DT1_POLICY
    child = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt('DT2 collection interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with (root/'collection_events.jsonl').open('x', buffering=1) as events:
        def record(event, **fields):
            events.write(json.dumps({'stage': 'DT2', 'wall_time_s': time.time(),
                                     'event': event, **fields})+'\n')
        try:
            for slot in plan['slots'][8:]:
                record('waiting_gpu', slot=slot['slot_index'], run_id=slot['run_id'])
                waiter = Dt1GpuWaiter(project_root=DATA_ROOT, run_id=slot['run_id'],
                    argv=sys.argv, required_free_mib=policy.required_free_mib,
                    required_samples=policy.required_samples, interval_s=policy.sample_interval_s,
                    policy=policy)
                state = waiter.wait()
                if state['status'] != 'GPU_READY_LOCKED_RECHECKED':
                    raise KeyboardInterrupt('resource queue cancelled; do not launch a child')
                command = ['bash', str(SOURCE_ROOT/'scripts/dual_target_dt2_episode.sh'),
                           str(plan_path), str(slot['slot_index'])]
                if args.shared_dt2:
                    command.append('--shared-dt2')
                record('launching', slot=slot['slot_index'], command=command)
                child = subprocess.Popen(command)
                code = child.wait()
                record('child_exited', slot=slot['slot_index'], child_pid=child.pid, exit_code=code)
                child = None
                if code:
                    raise RuntimeError(f'DT2 slot {slot["slot_index"]} failed with exit {code}')
            record('all_new_episodes_completed', count=8)
            (root/'collection_result.json').write_text(json.dumps({
                'stage': 'DT2', 'status': 'COLLECTION_COMPLETE_NOT_DT2_APPROVED',
                'new_successes': 8, 'training': False, 'dt3_started': False}, indent=2))
        except BaseException as exc:
            record('stopped', reason=str(exc), exception=type(exc).__name__)
            (root/'collection_result.json').write_text(json.dumps({
                'stage': 'DT2', 'status': 'INTERRUPTED_RESUMABLE' if isinstance(exc, KeyboardInterrupt) else 'FAILED',
                'reason': str(exc), 'training': False, 'dt3_started': False}, indent=2))
            raise
        finally:
            if child is not None and child.poll() is None:
                child.terminate()  # owned shell trap cleans only its isolated runtime group
                child.wait(timeout=45)


if __name__ == '__main__':
    main()
