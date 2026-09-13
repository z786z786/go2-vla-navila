"""Cancellation must stop a shell's waiter && launcher chain before launch."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CancelExitTest(unittest.TestCase):
    def test_cancelled_cli_exits_nonzero_and_records_resumable_state(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            out = root / 'outputs/dual_target_v1'
            out.mkdir(parents=True)
            (out/'dt1_gpu_wait_cancel.json').write_text(json.dumps({
                'project_root': str(root.resolve()), 'run_id': 'cancel_regression',
            }))
            snapshots = root/'snapshots.json'
            snapshots.write_text('[]')
            result = subprocess.run([
                sys.executable, '-m', 'src.dual_target.gpu_wait',
                '--project-root', str(root), '--run-id', 'cancel_regression',
                '--test-snapshots', str(snapshots),
            ], capture_output=True, text=True, timeout=10)
            state = json.loads(result.stdout)
            self.assertEqual(state['status'], 'INTERRUPTED_RESUMABLE')
            self.assertEqual(result.returncode, 130)


if __name__ == '__main__':
    unittest.main()
