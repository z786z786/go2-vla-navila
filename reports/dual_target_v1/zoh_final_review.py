"""Read frozen v2 evidence; write a separate root-review artifact directory."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from src.dual_target.zoh_review import review_episode, read_lines
from src.dual_target.expert import ParkingExpert
from src.dual_target.layouts import Vec2
from src.dual_target.reset_audit import validate_paired_resets


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--images',action='store_true')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    queue=json.loads((a.root/'queue_status.json').read_text())
    assert queue['status']=='GATE_TRACES_PASSED_AWAITING_ROOT_REVIEW' and len(queue['completed'])==5
    summary=dict(status='NUMERICAL_REVIEW_COMPLETE_AWAITING_VISUAL_REVIEW',episodes=[],pairs=[],image_audit=[],source_hashes={})
    manifest=json.loads((a.root/'gate_manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():
        actual=sha(Path(name)); assert actual==digest, name
        summary['source_hashes'][name]=actual
    pair_groups={}
    image_rows=[]
    image_hashes={}
    for item in queue['completed']:
        ep=a.root/'runs'/item['run_id']/'episodes'/item['episode_result']['episode_id']
        result=review_episode(ep,item['mode'])
        pre=read_lines(ep/'pre_action.jsonl');post=read_lines(ep/'post_step_events.jsonl');high=read_lines(ep/'high_level_pre_action.jsonl')
        m=json.loads((ep/'episode_manifest.json').read_text());reset=json.loads((ep/'paired_reset_audit.json').read_text());pause=json.loads((ep/'pause_no_physics.json').read_text())
        assert not pause['physics_advanced'] and pause['pose_before']==pause['pose_after']
        assert pause['physics_step_before']==pause['physics_step_after'] and pause['sim_time_before_s']==pause['sim_time_after_s']
        assert m['slot_'+m['target_slot'].lower()+'_color']==m['target_color']
        assert all(r['task']==m['instruction'] for r in high)
        for i,r in enumerate(pre[1:],1):
            assert r['body_velocity_body']==post[i-1]['body_velocity_body']
        if item['mode']=='expert':
            pair_groups.setdefault(m['color_configuration'],[]).append(reset)
            expert=ParkingExpert();errors=[]
            center=Vec2(*m['parking_centers_xy'][m['target_slot']])
            for h in high:
                i=h['low_level_start_index'];pose=post[i-1]['robot_pose_w'] if i else pause['pose_before']
                w,x,y,z=pose[3:];yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
                predicted=expert.command(robot_xy=Vec2(*pose[:2]),robot_yaw_rad=yaw,parking_center=center,terminal_heading_rad=0.).as_list()
                errors.append(max(abs(x-y) for x,y in zip(predicted,h['raw_action'])))
            assert max(errors)<1e-9, 'expert not replayable from 5 Hz snapshots'
            tail=[]
            for before,after in reversed(list(zip(pre,post))):
                if (any(abs(x)>1e-12 for x in before['raw_action']+before['applied_action'])
                    or math.hypot(*after['body_velocity_body'][:2])>=m['thresholds']['body_linear_speed_mps']
                    or abs(after['body_velocity_body'][2])>=m['thresholds']['body_yaw_rate_radps']
                    or not after['in_correct_parking_region']): break
                tail.append(after)
            duration=tail[0]['sim_time_after_s']-pre[len(pre)-len(tail)]['sim_time_s']
            assert duration>=1.-1e-6
            result.update(expert_replay_max_error=max(errors),actual_stop_duration_s=duration,
                final_distance_to_parking_m=math.dist(post[-1]['robot_pose_w'][:2],(center.x,center.y)),
                terminal_partial_hold_low_steps=len(pre)%10,
                final_planar_speed=math.hypot(*post[-1]['body_velocity_body'][:2]),
                final_abs_yaw_speed=abs(post[-1]['body_velocity_body'][2]))
        result.update(run_id=item['run_id'],instruction=m['instruction'],color_configuration=m['color_configuration'],
            scored_duration_s=post[-1]['sim_time_after_s']-pre[0]['sim_time_s'],
            raw_evidence_sha256={n:sha(ep/n) for n in ('pre_action.jsonl','post_step_events.jsonl','high_level_pre_action.jsonl','episode_manifest.json','paired_reset_audit.json')})
        summary['episodes'].append(result)
        if a.images:
            from PIL import Image, ImageDraw
            import numpy as np
            visibility=[]
            for path in sorted(ep.glob('rgb_*/*.png')):
                image_hashes[str(path.relative_to(a.root))]=sha(path)
            for field,hashfield in [('first_rgb_path','first_rgb_sha256'),('learner_first_rgb_path','learner_first_rgb_sha256')]:
                assert sha(ep/reset[field])==reset[hashfield]
            for h in high:
                with Image.open(ep/h['rgb_path']) as im:
                    assert im.size==(512,512)
                    v=np.asarray(im.convert('RGB')).astype('float32');r,g,b=v[:,:,0],v[:,:,1],v[:,:,2]
                    counts={'red':int(((r>80)&(r>1.15*g)&(r>1.15*b)).sum()),'blue':int(((b>80)&(b>1.15*g)&(b>1.15*r)).sum())}
                    visibility.append(counts)
            worst=min(range(len(high)),key=lambda i:visibility[i][m['target_color']])
            summary['image_audit'].append(dict(run_id=item['run_id'],high_images=len(high),
                initial_counts=visibility[0],minimum_target_pixels=visibility[worst][m['target_color']],worst_index=worst,
                worst_path=str((ep/high[worst]['rgb_path']).relative_to(a.root))))
            if item['mode']=='expert':
                selected=[0,len(high)//3,2*len(high)//3,len(high)-1,worst]
                row=Image.new('RGB',(5*256,290),'white');draw=ImageDraw.Draw(row)
                for col,i in enumerate(selected):
                    with Image.open(ep/high[i]['rgb_path']) as im:
                        row.paste(im.convert('RGB').resize((256,256)),(col*256,34))
                    draw.text((col*256+3,2),f'{m["color_configuration"]} -> {m["target_color"]}',fill='black')
                    draw.text((col*256+3,17),f'high={i} '+('worst visibility' if col==4 else ''),fill='black')
                image_rows.append(row)
    for key,pair in pair_groups.items():
        assert len(pair)==2
        summary['pairs'].append(validate_paired_resets(*pair))
    if a.images:
        sheet=Image.new('RGB',(1280,290*len(image_rows)),'white')
        for i,row in enumerate(image_rows):sheet.paste(row,(0,i*290))
        sheet.save(a.output/'expert_contact_sheet.png')
        (a.output/'image_hashes.json').write_text(json.dumps(image_hashes,indent=2)+'\n')
        summary['total_image_files_hashed']=len(image_hashes)
    (a.output/'numerical_review.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
