import unittest
from dataclasses import replace
from src.dual_target.zoh_scoring import ZohActionAudit, ZohScoreFrame
from src.dual_target.zoh_control import ZohClock
from src.dual_target.scoring import ScoreFrame, AutonomousParkingScorer, ScoreStatus


def frame(i,raw,applied,expected,**extra):
    return ZohScoreFrame(sim_time_s=(i+1)*.02,physics_step=(i+1)*4,
        observation_seq=i+1,raw_action=raw,applied_action=applied,expected_applied=expected,
        body_vx_mps=0.,body_vy_mps=0.,body_yaw_rate_radps=0.,in_correct_parking_region=True,**extra)


class ZohScoringTests(unittest.TestCase):
    def test_slew_no_longer_invalid_and_raw_not_replaced(self):
        clock=ZohClock();audit=ZohActionAudit();scorer=AutonomousParkingScorer()
        for i in range(60):
            raw,applied=clock.advance([.5,.8,.5] if i%10==0 else None)
            f=frame(i,raw,applied,audit.observe(raw,applied,i))
            decision=scorer.observe(f)
            self.assertEqual(decision.status,ScoreStatus.IN_PROGRESS)
            self.assertEqual(f.raw_action,(.5,.8,.5))

    def test_full_real_stop_duration_unchanged(self):
        audit=ZohActionAudit();scorer=AutonomousParkingScorer()
        for i in range(51):
            f=frame(i,[0,0,0],[0,0,0],audit.observe([0,0,0],[0,0,0],i))
            d=scorer.observe(f)
            self.assertEqual(d.status,ScoreStatus.SUCCESS if i==50 else ScoreStatus.IN_PROGRESS)

    def test_raw_nonzero_cannot_be_hidden_by_clip(self):
        audit=ZohActionAudit();scorer=AutonomousParkingScorer()
        for i in range(70):
            d=scorer.observe(frame(i,[-.2,0,0],[0,0,0],audit.observe([-.2,0,0],[0,0,0],i)))
            self.assertEqual(d.status,ScoreStatus.IN_PROGRESS)

    def test_wrong_slew_mid_hold_and_duplicate_rejected(self):
        a=ZohActionAudit()
        with self.assertRaises(ValueError):a.observe([.5,0,0],[.5,0,0],0)
        a=ZohActionAudit();a.observe([.5,0,0],[.1,0,0],0)
        with self.assertRaises(ValueError):a.observe([.5,0,0],[.2,0,0],1)
        with self.assertRaises(ValueError):a.observe([.4,0,0],[.1,0,0],1)
        with self.assertRaises(ValueError):a.observe([.5,0,0],[.1,0,0],0)

    def test_collision_priority_and_nan(self):
        f=frame(0,[.5,0,0],[.1,0,0],[.1,0,0],collision=True)
        self.assertEqual(AutonomousParkingScorer().observe(f).status,ScoreStatus.FAILED_COLLISION)
        self.assertEqual(AutonomousParkingScorer().observe(replace(f,collision=False,raw_action=[float('nan'),0,0])).status,ScoreStatus.INVALID_SAMPLE)

    def test_actual_motion_prevents_success(self):
        scorer=AutonomousParkingScorer()
        for i in range(70):
            d=scorer.observe(replace(frame(i,[0,0,0],[0,0,0],[0,0,0]),body_vx_mps=.2))
            self.assertEqual(d.status,ScoreStatus.IN_PROGRESS)


if __name__=='__main__':unittest.main()
