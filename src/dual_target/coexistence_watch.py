"""One bounded, CPU-only sample for the DT1 launcher's memory watchdog."""
import argparse
import json
import os
import subprocess
import time

from .gpu_wait import probe_gpu


def sample_group(pgid):
    snapshot = probe_gpu().as_dict()
    query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                            '--format=csv,noheader,nounits'], check=True,
                           capture_output=True, text=True, timeout=10)
    processes = []
    for line in query.stdout.splitlines():
        pid, memory = (int(v.strip()) for v in line.split(','))
        try:
            owned = os.getpgid(pid) == pgid
        except ProcessLookupError:
            owned = False
        processes.append({'pid': pid, 'used_mib': memory, 'owned': owned})
    return {'wall_time_s': time.time(), **snapshot, 'owned_pgid': pgid,
            'owned_used_mib': sum(p['used_mib'] for p in processes if p['owned']),
            'processes': processes}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pgid', type=int, required=True)
    args = parser.parse_args()
    if args.pgid <= 1 or args.pgid == os.getpgrp():
        parser.error('require distinct owned child process group')
    row = sample_group(args.pgid)
    print(json.dumps(row), flush=True)
    if row['free_mib'] < 2048:
        raise SystemExit(75)
