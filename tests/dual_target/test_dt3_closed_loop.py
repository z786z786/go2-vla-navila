import ast
from collections import deque
import json
from pathlib import Path
import tempfile
import unittest

from src.dual_target.tiny_policy_rpc import PolicyClient


class ClosedLoopTests(unittest.TestCase):
    def test_model_loop_only_supplies_rgb_velocity_instruction(self):
        tree=ast.parse(Path('src/dual_target/tiny_policy_loop.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Attribute) and n.func.attr=='action']
        self.assertEqual(len(calls),1)
        self.assertEqual(len(calls[0].args),3)
        self.assertFalse(calls[0].keywords)
        for prohibited in ('ParkingExpert','parking_center','terminal_heading','robot_yaw_rad'):
            self.assertNotIn(prohibited,ast.unparse(calls[0]))

    def test_chunk_execution_replans_at_ten_and_preserves_raw(self):
        class Pipe:
            def __init__(self): self.sent=[]
            def write(self,data): self.sent.append(json.loads(data))
            def flush(self): pass
        class Child:
            stdin=Pipe()
        with tempfile.TemporaryDirectory() as folder:
            client=PolicyClient.__new__(PolicyClient)
            client.queue=deque()
            client.index=0
            client.output=Path(folder)
            client.child=Child()
            chunk=[[i*.1,9.,-2.] for i in range(50)]
            client.receive=lambda timeout:{'actions':chunk}
            for i in range(21):
                action=client.action(Path(folder)/f'{i}.png',[0,0,0],
                    'Go to the red box and stop in front of it.')
                self.assertEqual(action,chunk[i%10])
            self.assertEqual(len(client.child.stdin.sent),3)
            for request in client.child.stdin.sent:
                self.assertEqual(set(request),{'observation.images.front','observation.state','task'})
            self.assertTrue(client.child.stdin.sent[1]['observation.images.front'].endswith('/10.png'))


if __name__=='__main__':
    unittest.main()
