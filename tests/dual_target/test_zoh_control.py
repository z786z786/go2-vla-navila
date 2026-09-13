import unittest
import ast
from pathlib import Path
from src.dual_target.zoh_control import CommandBoundary, ZohClock, ZohSpec, probe_command, validate_trajectory_splits


class ZohTests(unittest.TestCase):
    def test_time_and_configurable_chunk(self):
        self.assertEqual(ZohSpec().action_dt_s, .2)
        self.assertEqual(ZohSpec(chunk_size=10).chunk_size, 10)
        with self.assertRaises(ValueError):
            ZohSpec(physics_dt_s=.02)

    def test_bounds_rates_reset_and_zero_vy(self):
        b = CommandBoundary()
        self.assertEqual(b.update([5,8,-5]), (.1,0.,-.2))
        for _ in range(20):
            previous = b.previous
            v = b.update([5,8,-5])
            for old,new,rate in zip(previous,v,b.spec.rate_limits):
                self.assertLessEqual(abs(new-old),rate*.2+1e-12)
        self.assertEqual(v,(.5,0.,-.5))
        b.reset()
        self.assertEqual(b.previous,(0.,0.,0.))

    def test_reject_nonfinite_without_state_change(self):
        b=CommandBoundary()
        for raw in ([float('nan'),0,0],[0,0,float('inf')],[True,0,0],[0,0]):
            with self.assertRaises(ValueError): b.update(raw)
            self.assertEqual(b.previous,(0.,0.,0.))

    def test_exact_hold_and_identical_expert_policy_boundary(self):
        expert, policy = ZohClock(), ZohClock()
        for i in range(700):
            raw = probe_command(i//10) if i%10 == 0 else None
            a,b=expert.advance(raw),policy.advance(raw)
            self.assertEqual(a,b)
            if i%10: self.assertEqual(a,previous)
            previous=a
        with self.assertRaises(ValueError): probe_command(70)

    def test_midhold_update_rejected(self):
        c=ZohClock();c.advance([0,0,0])
        with self.assertRaises(ValueError):c.advance([0,0,0])
        self.assertEqual(c.step,1)

    def test_split_lineage_not_only_trajectory_id(self):
        a=dict(trajectory_id='t1',geometry_group_id='g1',lineage_root_id='g1',split='train')
        b=dict(trajectory_id='t2',geometry_group_id='g2',lineage_root_id='g1',split='test')
        with self.assertRaises(ValueError):validate_trajectory_splits([a,b])
        b['lineage_root_id']='g2'
        self.assertTrue(validate_trajectory_splits([a,b]))

    def test_model_branch_has_only_three_learner_inputs(self):
        tree=ast.parse(Path('src/dual_target/zoh_loop.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Attribute) and n.func.attr=='action']
        self.assertEqual(len(calls),1)
        self.assertEqual(len(calls[0].args),3)
        self.assertFalse(calls[0].keywords)
        self.assertNotIn('pose',ast.unparse(calls[0]))

    def test_independent_checker_preserves_raw_stop_requirement(self):
        from src.dual_target.zoh_parking_review import review_trace
        from reports.dual_target_v1 import dt1_independent_checks as old
        # Exercise the old checker's full synthetic positive/negative suite
        # against the new checker without editing the original implementation.
        original=old.review_trace
        try:
            old.review_trace=review_trace
            old.self_test()
        finally:
            old.review_trace=original


if __name__=='__main__': unittest.main()
