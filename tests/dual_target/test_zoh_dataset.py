import ast
import json
from pathlib import Path
import unittest

from src.dual_target.zoh_dataset import grouped_tasks, TerminalHolds, complete_rows, low_stop
from src.dual_target.zoh_dataset_convert import require_train_split
from src.dual_target.tiny_data import chunk_indices


class DatasetTests(unittest.TestCase):
    def test_fixed_4_2_2_and_no_test_training_geometry(self):
        tasks=grouped_tasks()
        self.assertEqual(len(tasks),32)
        self.assertEqual([sum(s==split for s,_,_ in tasks) for split in ('train','validation','test')],[16,8,8])
        ownership={}
        for split,_,t in tasks:
            for key in (t.geometry_group_id,t.lineage_root_id,t.geometry_signature):
                self.assertIn(ownership.setdefault(key,split),(split,))

    def test_six_full_holds_not_six_frames(self):
        c=TerminalHolds()
        for i in range(59):self.assertFalse(c.observe(i,True))
        self.assertTrue(c.observe(59,True))

    def test_partly_stopped_hold_not_counted(self):
        c=TerminalHolds()
        for i in range(69):self.assertFalse(c.observe(i,i>=5))
        self.assertTrue(c.observe(69,True))
        self.assertFalse(c.observe(70,False))
        self.assertEqual(c.count,0)

    def test_stop_checks_both_raw_and_applied(self):
        limits=dict(body_linear_speed_mps=.03,body_yaw_rate_radps=.05)
        self.assertTrue(low_stop([0,0,0],[0,0,0],[0,0,0],True,limits))
        for raw,applied,body,correct in [([.1,0,0],[0,0,0],[0,0,0],True),
            ([0,0,0],[.1,0,0],[0,0,0],True),([0,0,0],[0,0,0],[.04,0,0],True),
            ([0,0,0],[0,0,0],[0,0,0],False)]:
            self.assertFalse(low_stop(raw,applied,body,correct,limits))

    def test_partial_final_hold_excluded_with_actual_duration(self):
        high=[dict(low_level_start_index=i*10) for i in range(3)]
        pre=[dict(sim_time_s=i*.02) for i in range(23)]
        post=[dict(sim_time_after_s=(i+1)*.02) for i in range(23)]
        rows,segments=complete_rows(high,pre,post)
        self.assertEqual(len(rows),2)
        self.assertEqual(segments[-1]['low_level_steps'],3)
        self.assertAlmostEqual(segments[-1]['actual_duration_s'],.06)
        self.assertFalse(segments[-1]['complete'])

    def test_padding_only_within_episode(self):
        indices,pad=chunk_indices(6,8,chunk_size=5)
        self.assertEqual(indices,[6,7,7,7,7])
        self.assertEqual(pad,[False,False,True,True,True])

    def test_heldout_rejected_for_normalizer(self):
        require_train_split({'split':'train'})
        for split in ('validation','test',None):
            with self.assertRaises(ValueError):require_train_split({'split':split})

    def test_four_approved_gate_partial_tails_are_not_exported(self):
        root=Path('reports/dual_target_v1/remote_runs/zoh_gate_0906_v1')
        if not root.exists():self.skipTest('local gate mirror unavailable')
        seen=[]
        for ep in root.glob('runs/*expert/episodes/*'):
            read=lambda n:[json.loads(l) for l in (ep/n).read_text().splitlines()]
            high,pre,post=read('high_level_pre_action.jsonl'),read('pre_action.jsonl'),read('post_step_events.jsonl')
            rows,segments=complete_rows(high,pre,post)
            self.assertEqual(len(rows),len(high)-1)
            self.assertFalse(segments[-1]['complete'])
            seen.append(segments[-1]['low_level_steps'])
        self.assertEqual(sorted(seen),[1,1,3,3])

    def test_extension_explicitly_overrides_latched_success_on_failure(self):
        source=Path('src/dual_target/zoh_dataset_loop.py').read_text()
        ast.parse(source)
        for token in ('ScoreStatus.FAILED_COLLISION','ScoreStatus.FAILED_FALLEN',
                      'extension_lost','terminal_holds.observe','segment_log.write'):
            self.assertIn(token,source)
        self.assertIn('decision.status is ScoreStatus.SUCCESS and tail_ready',source)


if __name__=='__main__':unittest.main()
