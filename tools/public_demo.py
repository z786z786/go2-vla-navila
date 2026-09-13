"""Deterministic public demo for the complete navigation data path."""
import argparse, json
from pathlib import Path

def main():
 p=argparse.ArgumentParser(); p.add_argument('--backend',default='mock'); p.add_argument('--episodes',type=int,default=2); p.add_argument('--output',default='artifacts/mock_eval.json'); a=p.parse_args()
 if a.backend!='mock': raise SystemExit('Select an installed external backend for non-mock execution.')
 out={'backend':'mock','episodes':[{'id':i,'instruction':'navigate to the target','action':{'vx':0.2,'wz':0.0,'stop':i==a.episodes-1},'valid':True} for i in range(a.episodes)]}
 q=Path(a.output); q.parent.mkdir(parents=True,exist_ok=True); q.write_text(json.dumps(out,indent=2)+'\n'); print(q)
if __name__=='__main__': main()
