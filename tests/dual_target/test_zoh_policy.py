import io
import json
from collections import deque
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from src.dual_target.zoh_policy_contract import validate_action_chunk
from src.dual_target.zoh_policy_rpc import PolicyClient
from src.dual_target.contracts import RED_TASK, BLUE_TASK


class ZohPolicyTests(unittest.TestCase):
    def test_new_chunk_contract_preserves_unclamped_raw(self):
        self.assertEqual(validate_action_chunk([[2,0,-3]]*5)[0],(2.,0.,-3.))
        for bad in ([[0,0,0]]*50,[[0,0]]*5,[[float('nan'),0,0]]*5,[[True,0,0]]*5):
            with self.assertRaises(ValueError):validate_action_chunk(bad)

    def test_every_high_level_call_requests_new_chunk_and_executes_only_first(self):
        with tempfile.TemporaryDirectory() as d:
            c=PolicyClient.__new__(PolicyClient);c.queue=deque();c.index=0;c.output=Path(d)
            c.child=SimpleNamespace(stdin=io.BytesIO())
            chunks=iter([{'actions':[[.2,0,.3]]+[[.9,0,.9]]*4},
                         {'actions':[[.1,0,-.2]]+[[.9,0,.9]]*4}])
            c.receive=lambda timeout:next(chunks)
            self.assertEqual(c.action('/tmp/front.png',[0,0,0],RED_TASK),[.2,0,.3])
            self.assertEqual(c.action('/tmp/front2.png',[.1,0,0],BLUE_TASK),[.1,0,-.2])
            requests=[json.loads(l) for l in c.child.stdin.getvalue().splitlines()]
            self.assertEqual(len(requests),2)
            self.assertEqual(set(requests[0]),{'observation.images.front','observation.state','task'})
            logs=[json.loads(l) for l in (Path(d)/'model_chunks.jsonl').read_text().splitlines()]
            self.assertTrue(all(x['execute_steps']==1 and x['action_dt_s']==.2 for x in logs))


if __name__=='__main__':unittest.main()
