import os
import subprocess
import sys
import unittest

from src.dual_target.coexistence_probe import wait_for_model
from src.dual_target.gpu_wait import (
    SHARED_COEXISTENCE_DT1_POLICY, EXCLUSIVE_DT1_POLICY,
    parse_nvidia_smi_snapshot, eligibility, GpuWaitError,
)


class ModelWaitTests(unittest.TestCase):
    def test_shared_probe_accepts_only_admitted_foreign_processes(self):
        for allowed in ((777,), ()):
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(.1)'])
            sample = lambda: {'compute_pids': [os.getpid(), child.pid, 777], 'free_mib': 8000}
            if allowed:
                wait_for_model(child, sample=sample, record=lambda x: None, allowed_foreign_pids=allowed)
                self.assertEqual(child.returncode, 0)
            else:
                with self.assertRaises(RuntimeError):
                    wait_for_model(child, sample=sample, record=lambda x: None)
                self.assertIsNotNone(child.poll())

    def test_shared_policy_and_driver_reserved_memory(self):
        snapshot = parse_nvidia_smi_snapshot('24576,6603,80,17513', '777,6550')
        self.assertEqual(snapshot.free_mib, 17513)
        self.assertTrue(eligibility(snapshot, required_free_mib=SHARED_COEXISTENCE_DT1_POLICY.required_free_mib,
                                    allow_existing_compute=True)[0])
        self.assertFalse(eligibility(snapshot)[0])
        self.assertEqual(EXCLUSIVE_DT1_POLICY.required_free_mib, 20480)
        low = parse_nvidia_smi_snapshot('24576,6603,80,16000', '777,6550')
        self.assertFalse(eligibility(low, required_free_mib=16384, allow_existing_compute=True)[0])
        with self.assertRaises(GpuWaitError):
            parse_nvidia_smi_snapshot('24576,6603,80,19000', '')

    def test_records_concurrent_residency_and_propagates_model_failure(self):
        for exit_code in (0, 7):
            child = subprocess.Popen([sys.executable, '-c', f'import time; time.sleep(.1); raise SystemExit({exit_code})'])
            rows = []
            sample = lambda: {'compute_pids': [os.getpid(), child.pid], 'free_mib': 8000}
            if exit_code:
                with self.assertRaises(RuntimeError):
                    wait_for_model(child, sample=sample, record=rows.append)
            else:
                wait_for_model(child, sample=sample, record=rows.append)
            self.assertGreaterEqual(len(rows), 1)
            self.assertEqual(child.returncode, exit_code)

    def test_low_memory_terminates_only_created_child(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        with self.assertRaises(RuntimeError):
            wait_for_model(child, sample=lambda: {'compute_pids': [], 'free_mib': 100}, record=lambda x: None)
        self.assertIsNotNone(child.poll())
