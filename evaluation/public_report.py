"""Render a compact public evaluation record from a mock or adapter output."""
import argparse, json
from pathlib import Path

def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output'); a=p.parse_args()
 data=json.loads(Path(a.input).read_text()); eps=data.get('episodes',[])
 report={'backend':data.get('backend','unknown'),'episodes':len(eps),'valid_episodes':sum(bool(e.get('valid')) for e in eps),'action_schema':['vx','wz','stop']}
 out=Path(a.output) if a.output else Path(a.input).with_name('public_report.json'); out.write_text(json.dumps(report,indent=2)+'\n'); print(out)
if __name__=='__main__': main()
