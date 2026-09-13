import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.dual_target.tiny_plan import groups, task_at, build_plan, verify_binding
from src.dual_target.runner import run_live_expert
from src.dual_target.tiny_runtime import SHARED_DT2_POLICY
from src.dual_target.gpu_wait import LiveGpuSnapshot, eligibility


class TinyPlanTests(unittest.TestCase):
    def test_user_authorized_eight_gib_shared_collection(self):
        self.assertTrue(SHARED_DT2_POLICY.allow_existing_compute)
        self.assertEqual(SHARED_DT2_POLICY.required_free_mib, 8192)
        self.assertTrue(eligibility(LiveGpuSnapshot(24576,16000,(777,)),
                                   required_free_mib=8192, allow_existing_compute=True)[0])
        self.assertFalse(eligibility(LiveGpuSnapshot(24576,17000,(777,)),
                                    required_free_mib=8192, allow_existing_compute=True)[0])

    def test_four_train_groups_and_all_balanced_combinations(self):
        root = Path(__file__).resolve().parents[2]
        plan = build_plan(Path('/tmp/dt2_tiny_test'), source_root=root, data_root=Path('/tmp/data'))
        self.assertEqual(len(groups()), 4)
        self.assertEqual(len(plan['slots']), 16)
        self.assertEqual(sum(s['reuse_approved_dt1'] for s in plan['slots']), 8)
        for start in range(0, 16, 4):
            rows = plan['slots'][start:start+4]
            self.assertEqual({(r['color_configuration'], r['target_color']) for r in rows},
                             {(cfg, color) for cfg in ('A_red_B_blue', 'A_blue_B_red') for color in ('red', 'blue')})
            self.assertEqual(rows[0]['pair_seed'], rows[1]['pair_seed'])
            self.assertEqual(rows[2]['pair_seed'], rows[3]['pair_seed'])
            self.assertTrue(all(r['split'] == 'train' and r['repeat'] == 0 for r in rows))

    def test_scene_extension_is_explicit_and_dt1_default_preserved(self):
        args = SimpleNamespace(group_id='dt1_dev_000', color_configuration='A_red_B_blue', target_color='red')
        scene, task = task_at(8)
        with patch('src.dual_target.runner.build_live_low_level_runtime', side_effect=RuntimeError('sentinel')) as build:
            with self.assertRaisesRegex(RuntimeError, 'sentinel'):
                run_live_expert(args, None, scene_task=(scene, task))
            self.assertIs(build.call_args.args[1], scene)
            with self.assertRaisesRegex(RuntimeError, 'sentinel'):
                run_live_expert(args, None)
            self.assertEqual(build.call_args.args[1].group.geometry_group_id, 'dt1_dev_000')

    def test_locked_contract_unchanged_and_rejected_task_index(self):
        root = Path(__file__).resolve().parents[2]
        lock = verify_binding(root)
        self.assertEqual(lock['parking_thresholds']['required_duration_s'], 1.)
        for bad in (-1, 16, True):
            with self.assertRaises(ValueError):
                task_at(bad)


if __name__ == '__main__':
    unittest.main()
