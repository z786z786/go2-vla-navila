"""Copy unmodified real RGB examples for root visual inspection (no synthesis)."""
import json
from pathlib import Path
import shutil

base=Path('/mnt/wxh/go2_short_vln/outputs/dual_target_v1')
visual=base/'dt2_tiny_0906_v2_visual'
out=visual/'root_examples'
out.mkdir(exist_ok=False)
review=json.loads((visual/'visual_review.json').read_text())
manifest=[]
for index,item in enumerate(review):
    if index<8:
        continue
    p=Path(item['episode'])
    pre=[json.loads(s) for s in (p/'pre_action.jsonl').read_text().splitlines()]
    for label,relative in (('first',pre[0]['rgb_path']),
                           ('minimum_target',item['minimum_color_pixels'][item['target']][1]),
                           ('last_pre',pre[-1]['rgb_path'])):
        source=p/relative
        target=out/f's{index:02d}_{label}.png'
        shutil.copyfile(source,target)
        manifest.append({'slot':index,'kind':label,'target':item['target'],'source':str(source),'copy':target.name})
(out/'sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'unmodified_real_rgb_copies':len(manifest),'output':str(out)}))
