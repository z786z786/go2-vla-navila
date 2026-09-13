"""Bounded local pipe transport; only RGB, velocity and instruction reach policy."""
import json
import os
from pathlib import Path
import select
import subprocess
import time
from collections import deque

from .contracts import IMAGE_KEY,STATE_KEY,validate_policy_input,validate_action_chunk


class PolicyClient:
    def __init__(self,checkpoint,seed,output):
        self.queue=deque()
        self.index=0
        self.metadata={'checkpoint':str(checkpoint),'policy_seed':seed,'chunk_size':50,
                       'execute_steps':10,'policy_inputs':[IMAGE_KEY,STATE_KEY,'task']}
        self.output=Path(output)
        self.log=(self.output/'model.log').open('x')
        env=dict(os.environ,PYTHONPATH='<external-workspace>:<external-data-root>',
            HF_HOME='<external-data-root>',HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
        env.pop('PYTHONHOME',None)
        self.child=subprocess.Popen([
            '<external-data-root>','-m','src.dual_target.tiny_policy_server',
            '--checkpoint',str(checkpoint),'--seed',str(seed)],stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=self.log,env=env,bufsize=0)
        self.buffer=b''
        try:
            ready=self.receive(180)
            if ready.get('status')!='ready':
                raise ValueError('model startup failed')
            self.metadata.update(ready)
        except BaseException:
            self.close()
            raise

    def receive(self,timeout):
        deadline=time.monotonic()+timeout
        while b'\n' not in self.buffer:
            remaining=deadline-time.monotonic()
            if remaining<=0:
                raise TimeoutError('model response deadline exceeded')
            readable,_,_=select.select([self.child.stdout],[],[],min(remaining,1))
            if readable:
                part=os.read(self.child.stdout.fileno(),65536)
                if not part:
                    raise RuntimeError('model pipe closed; inspect model.log')
                self.buffer+=part
                if len(self.buffer)>1024*1024:
                    raise ValueError('model response too large')
        line,self.buffer=self.buffer.split(b'\n',1)
        return json.loads(line)

    def action(self,rgb_path,state,task):
        if not self.queue:
            payload=validate_policy_input({IMAGE_KEY:str(rgb_path),STATE_KEY:state,'task':task})
            start=time.monotonic()
            self.child.stdin.write((json.dumps(payload)+'\n').encode())
            self.child.stdin.flush()
            response=self.receive(60)
            chunk=validate_action_chunk(response['actions'])
            record={'chunk_index':self.index,'input':payload,'raw_chunk':chunk,
                    'execute_steps':10,'wall_wait_s':time.monotonic()-start}
            with (self.output/'model_chunks.jsonl').open('a') as log:
                log.write(json.dumps(record,allow_nan=False)+'\n')
            self.index+=1
            self.queue.extend(chunk[:10])
        return list(self.queue.popleft())

    def close(self):
        if self.child.poll() is None:
            self.child.terminate()
            try:
                self.child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.child.kill()
                self.child.wait(timeout=10)
        self.log.close()
