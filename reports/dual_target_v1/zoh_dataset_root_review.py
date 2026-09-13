"""Independent final dataset checks; derived outputs separate from frozen data."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from src.dual_target.tiny_plan import sha
from src.dual_target.zoh_dataset import make_plan, complete_rows, TerminalHolds, low_stop
from src.dual_target.zoh_review import review_episode, read_lines
from src.dual_target.expert import ParkingExpert
from src.dual_target.layouts import Vec2
from src.dual_target.reset_audit import validate_paired_resets


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    plan=json.loads((a.root/'split_manifest.json').read_text());assert plan==make_plan(a.root)
    queue=json.loads((a.root/'queue_status.json').read_text());assert queue['status']=='DATASET_READY_AWAITING_ROOT_REVIEW'
    assert [x['index'] for x in queue['completed']]==list(range(32))
    dm=json.loads((a.root/'dataset/dataset_manifest.json').read_text())
    assert not dm['test_exported'] and not dm['training']
    assert dm['plan_sha256']==sha(a.root/'split_manifest.json')
    for name,digest in dm['files_sha256'].items():assert sha(a.root/'dataset'/name)==digest,name
    results=[];pairs={};hashes={};train=[];panels=[]
    for slot in plan['slots']:
        ep=Path(slot['episode_path']);review_episode(ep,'expert')
        pre=read_lines(ep/'pre_action.jsonl');post=read_lines(ep/'post_step_events.jsonl');high=read_lines(ep/'high_level_pre_action.jsonl')
        meta=json.loads((ep/'episode_manifest.json').read_text());pause=json.loads((ep/'pause_no_physics.json').read_text())
        reset=json.loads((ep/'paired_reset_audit.json').read_text());pairs.setdefault((slot['geometry_group_id'],slot['color_configuration']),[]).append(reset)
        rows,segments=complete_rows(high,pre,post);assert len(rows)==len(high) and segments==read_lines(ep/'high_level_segments.jsonl')
        assert not pause['physics_advanced'] and pause['pose_before']==pause['pose_after']
        assert all(pre[i]['body_velocity_body']==post[i-1]['body_velocity_body'] for i in range(1,len(pre)))
        assert all(h['task']==slot['instruction'] for h in high)
        center=Vec2(*meta['parking_centers_xy'][slot['target_slot']]);expert=ParkingExpert();errors=[]
        for h in high:
            i=h['low_level_start_index'];pose=post[i-1]['robot_pose_w'] if i else pause['pose_before'];w,x,y,z=pose[3:]
            command=expert.command(robot_xy=Vec2(*pose[:2]),robot_yaw_rad=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)),parking_center=center,terminal_heading_rad=0.).as_list()
            errors.append(max(abs(x-y) for x,y in zip(command,h['raw_action'])))
        assert max(errors)<1e-9
        tail=TerminalHolds()
        for i,(b,c) in enumerate(zip(pre,post)):
            valid=all(low_stop(b['raw_action'],b['applied_action'],v,c['in_correct_parking_region'],meta['thresholds']) for v in [b['body_velocity_body'],c['body_velocity_body']])
            ready=tail.observe(i,valid)
        assert ready
        for path in ep.glob('rgb_*/*.png'):hashes[str(path.relative_to(a.root))]=sha(path)
        counts=[]
        for h in high:
            with Image.open(ep/h['rgb_path']) as im:
                assert im.size==(512,512);v=np.asarray(im.convert('RGB')).astype('float32');r,g,b=v[:,:,0],v[:,:,1],v[:,:,2]
                counts.append({'red':int(((r>80)&(r>1.15*g)&(r>1.15*b)).sum()),'blue':int(((b>80)&(b>1.15*g)&(b>1.15*r)).sum())})
        worst=min(range(len(high)),key=lambda i:counts[i][slot['target_color']]);assert counts[worst][slot['target_color']]>=100 and min(counts[0].values())>=100
        audit=json.loads((ep/'dataset_audit.json').read_text());assert [v['counts'] for v in audit['visibility']]==counts
        selected=[0,len(high)//3,2*len(high)//3,len(high)-1,worst]
        panel=Image.new('RGB',(1280,285),'white');draw=ImageDraw.Draw(panel)
        for col,i in enumerate(selected):
            with Image.open(ep/high[i]['rgb_path']) as im:panel.paste(im.convert('RGB').resize((256,256)),(col*256,29))
            draw.text((col*256+2,2),f"s{slot['index']:02} {slot['target_color']} h{i} "+('MIN' if col==4 else ''),fill='black')
        panels.append(panel)
        if len(panels)==4:
            sheet=Image.new('RGB',(1280,1140),'white')
            for j,panel in enumerate(panels):sheet.paste(panel,(0,285*j))
            sheet.save(a.output/f"sheet_{slot['index']//4:02}.png");panels=[]
        results.append(dict(index=slot['index'],split=slot['split'],frames=len(high),replay_max_error=max(errors),terminal_full_holds=tail.count,
            minimum_target_pixels=counts[worst][slot['target_color']],edge_warning_frames=audit['edge_warning_frames'],final_distance_m=math.dist(post[-1]['robot_pose_w'][:2],(center.x,center.y))))
        if slot['split']=='train':train.extend(high)
    pair_results=[validate_paired_resets(*pair) for pair in pairs.values()]
    normalizer=json.loads((a.root/'dataset/normalizer.json').read_text());assert normalizer['source_split']=='train'
    for key,field in [('state','body_velocity_body'),('action','applied_action')]:
        arr=np.array([h[field] for h in train]);stats=normalizer[key]
        assert np.allclose(arr.mean(0),stats['mean'],atol=1e-10)
        # Compare the explicitly recorded raw and safe scales according to exporter format.
        raw=arr.std(0);safe=np.where(raw<1e-6,1.,raw)
        assert np.allclose(safe,stats['std'],atol=1e-10)
    (a.output/'image_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    summary=dict(status='NUMERICAL_PASSED_VISUAL_REVIEW_PENDING',episodes=results,pairs=pair_results,total_images_hashed=len(hashes),
        total_high_images=sum(x['frames'] for x in results),dataset_files_verified=len(dm['files_sha256']),train_normalizer_independently_verified=True)
    (a.output/'review.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
