"""Pinned B-10k validation-layout rollout contract and evidence report."""
import csv
import io
import shutil

from scripts.zoh_compare_common import BASE,SEED,write_json,sha
from scripts.zoh_b10k_train_common import TRAINING,CHECKPOINT,checkpoint

ROOT=BASE/'b10k_eval_rollout_0907_v1'
CONDITION='B_scheduler_expert_10k'


def tasks_for(root,value):
    return [dict(condition=CONDITION,slot=slot,step=10000,policy_seed=SEED,
        evaluation_split='validation',run_id=f'{root.name}_{CONDITION}_s{slot:02d}',**value)
        for slot in range(16,24)]


def metrics(results):
    n=len(results)
    return dict(condition=CONDITION,split='validation',completed=n,planned=8,
        success=sum(r['status']=='success' for r in results),
        wrong_target_stop=sum(r['status']=='failed_wrong_target_stop' for r in results),
        timeout=sum(r['status']=='FAILED_TIMEOUT' for r in results),
        collision=sum(r['review']['collision'] for r in results),
        entered_correct=sum(r['review']['reached_correct_region'] for r in results),
        entered_other=sum(r['review']['reached_other_region'] for r in results),
        mean_final_parking_distance_m=(sum(r['review']['final_distance_m'] for r in results)/n if n else None))


def publish_training_evidence():
    destination=ROOT/'delivery/training';destination.mkdir(parents=True,exist_ok=False)
    for name in ('training_config.json','parameter_scope.json','result.json','single_update_gate.json',
                 'final_reload_gate.json','checkpoint_verification.json','resource_profiles.json'):
        shutil.copy2(TRAINING/name,destination/name)
    shutil.copy2(CHECKPOINT/'checkpoint_manifest.json',destination/'checkpoint_010000_manifest.json')


def publish(results,complete=False):
    delivery=ROOT/'delivery';delivery.mkdir(exist_ok=True);summary=metrics(results)
    write_json(delivery/'results.json',{'complete':complete,'scope':'B-10k validation layouts only; test forbidden',
        'training':False,'validation_used':True,'test_used':False,'results':results,'metrics':summary})
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(summary));writer.writeheader();writer.writerow(summary)
    (delivery/'summary.csv').write_text(buffer.getvalue())
    distance='—' if summary['mean_final_parking_distance_m'] is None else f"{summary['mean_final_parking_distance_m']:.3f}"
    lines=['# B scheduler expert-only 10k：validation rollout','',
        '**状态：'+('8/8完成；已停止，等待用户。' if complete else f"进行中：{len(results)}/8；未完成不计作失败。")+'**','',
        '固定使用B fresh-10k checkpoint_010000、policy seed 20260906，以及全部8个冻结validation任务。',
        '本轮不训练、不使用独立test。validation结果不回流训练或选择checkpoint。',
        '每条保留50 Hz真实执行轨迹和视频；末尾不足0.2秒的partial hold按实际时长保留。','',
        '|完成|成功|停错目标|超时|碰撞|进入正确区|进入错误区|末距均值(m)|',
        '|---|---|---|---|---|---|---|---|',
        f"|{summary['completed']}/8|{summary['success']}|{summary['wrong_target_stop']}|{summary['timeout']}|{summary['collision']}|{summary['entered_correct']}|{summary['entered_other']}|{distance}|",'',
        '## 逐条视频','',
        '|slot|指令/episode|结果|视频|','|---|---|---|---|']
    for r in results:lines.append(f"|{r['slot']:02d}|{r['episode_id']}|{r['status']}|[播放]({r['video']})|")
    (delivery/'README.md').write_text('\n'.join(lines)+'\n')
