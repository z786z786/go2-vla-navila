from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import zoh_train_bc_common as common
from scripts.zoh_train_bc_common import CONDITIONS,SEED,tasks_for,metric_rows


class TwoCheckpointTrainTests(unittest.TestCase):
    def test_exact_paired_train_matrix(self):
        values={name:dict(checkpoint=name,checkpoint_manifest_sha256='x') for name in CONDITIONS}
        tasks=tasks_for(Path('evaluation'),values)
        self.assertEqual(len(tasks),32)
        self.assertEqual(len({x['run_id'] for x in tasks}),32)
        self.assertEqual({x['slot'] for x in tasks},set(range(16)))
        self.assertEqual({x['condition'] for x in tasks},set(CONDITIONS))
        self.assertTrue(all(x['evaluation_split']=='train' and x['policy_seed']==SEED for x in tasks))
        for slot in range(16):self.assertEqual([x['condition'] for x in tasks if x['slot']==slot],list(CONDITIONS))

    def test_unfinished_is_not_failure(self):
        result=dict(condition=CONDITIONS[0],status='FAILED_TIMEOUT',
            review=dict(collision=False,reached_correct_region=True,reached_other_region=False,final_distance_m=.4))
        rows=metric_rows([result])
        self.assertEqual(rows[0]['completed'],1);self.assertEqual(rows[0]['timeout'],1)
        self.assertEqual(rows[1]['completed'],0);self.assertIsNone(rows[1]['mean_final_parking_distance_m'])

    def test_training_delivery_parent_is_created(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(common,'ROOT',Path(directory)/'new_root'), \
                patch.object(common,'RUNS',{},create=True):
            # No conditions means this specifically tests nested parent creation.
            with patch.object(common,'CONDITIONS',()):common.publish_training_evidence()
            self.assertTrue((Path(directory)/'new_root/delivery/training').is_dir())


if __name__=='__main__':unittest.main()
