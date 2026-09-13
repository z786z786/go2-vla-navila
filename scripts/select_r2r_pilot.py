"""Deterministic six-scene pilot; planar routes only for current XY expert."""
import gzip,json,math
from pathlib import Path

root=Path('/mnt/wxh/go2_short_vln/data/r2r_vlnce')
episodes=json.load(gzip.open(root/'R2R_VLNCE_v1-3/train/train.json.gz'))['episodes']
gt=json.load(gzip.open(root/'R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz'))
candidates=[]
for e in episodes:
    scene=Path(e['scene_id']).stem
    if scene=='7y3sRwLe3Va':continue
    pts=gt[str(e['episode_id'])]['locations']
    heights=[p[1] for p in pts]
    if max(heights)-min(heights)>.15:continue
    length=sum(math.hypot(b[0]-a[0],b[2]-a[2]) for a,b in zip(pts,pts[1:]))
    headings=[]
    for a,b in zip(e['reference_path'],e['reference_path'][1:]):
        if math.hypot(b[0]-a[0],b[2]-a[2])>.2:headings.append(math.atan2(-(b[2]-a[2]),b[0]-a[0]))
    turns=[math.degrees(math.atan2(math.sin(b-a),math.cos(b-a))) for a,b in zip(headings,headings[1:])]
    candidates.append(dict(episode_id=str(e['episode_id']),scene=scene,length_m=length,turns_deg=turns,total_turn_deg=sum(map(abs,turns)),height_span_m=max(heights)-min(heights)))
selected=[];seen=set()
for label,lo,hi,turning in [('short_gentle',3,6,False),('short_turn',3,6,True),('medium_gentle',6,10,False),('medium_turn',6,10,True),('long_gentle',10,16,False),('long_turn',10,16,True)]:
    options=[c for c in candidates if lo<=c['length_m']<hi and c['scene'] not in seen and ((c['total_turn_deg']>=90) if turning else (c['total_turn_deg']<90))]
    if not options:raise RuntimeError('no candidate '+label)
    c=min(options,key=lambda c:(abs(c['length_m']-(lo+hi)/2),int(c['episode_id'])))
    c['category']=label;selected.append(c);seen.add(c['scene'])
print(json.dumps(selected,indent=2))
