"""Versioned ZOH action validation; original parking conditions stay unchanged."""
from dataclasses import dataclass, fields
import math
from .scoring import ScoreFrame
from .zoh_control import ZohSpec


class ZohActionAudit:
    """Independent stream check: one slew update per high-level hold, never per tick."""
    def __init__(self, spec=ZohSpec()):
        self.spec=spec
        self.step=0
        self.previous=(0.,0.,0.)
        self.raw=None

    def observe(self, raw, applied, step):
        if step!=self.step:
            raise ValueError('ZOH audit missing/duplicate low-level step')
        for vector in (raw,applied):
            if len(vector)!=3 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in vector):
                raise ValueError('ZOH audit requires finite 3D raw/applied actions')
        raw=tuple(raw)
        if step%self.spec.hold_control_steps==0:
            expected=[]
            for value,old,(lo,hi),rate in zip(raw,self.previous,self.spec.bounds,self.spec.rate_limits):
                bounded=min(hi,max(lo,value))
                delta=rate*self.spec.action_dt_s
                expected.append(old+min(delta,max(-delta,bounded-old)))
            self.previous=tuple(expected)
            self.raw=raw
        elif raw!=self.raw:
            raise ValueError('raw action changed inside ZOH hold')
        if any(abs(a-b)>1e-12 for a,b in zip(applied,self.previous)):
            raise ValueError('applied action violates independent ZOH mapping')
        self.step+=1
        return self.previous


@dataclass(frozen=True)
class ZohScoreFrame(ScoreFrame):
    expected_applied: tuple=()

    def validate(self):
        # Retain every old finite/type/flag/clock check. Only its stateless
        # mapping assertion is replaced; raw values in THIS frame stay intact.
        values={f.name:getattr(self,f.name) for f in fields(ScoreFrame)}
        if len(self.applied_action)!=3 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in self.applied_action):
            raise ValueError('invalid actual applied action')
        if len(self.expected_applied)!=3 or any(not math.isfinite(x) for x in self.expected_applied):
            raise ValueError('independent ZOH expectation missing')
        if len(self.raw_action)!=3:
            raise ValueError('invalid raw action shape')
        values['applied_action']=tuple(max(lo,min(hi,v)) for v,(lo,hi) in zip(self.raw_action,ZohSpec().bounds))
        ScoreFrame(**values).validate()
        if any(abs(a-b)>1e-12 for a,b in zip(self.applied_action,self.expected_applied)):
            raise ValueError('applied action differs from audited ZOH expectation')
        if any(not lo<=a<=hi for a,(lo,hi) in zip(self.applied_action,ZohSpec().bounds)):
            raise ValueError('applied action outside absolute bounds')
