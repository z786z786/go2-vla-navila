"""Pinned B/C 5000-step train-layout comparison and report."""
import csv
import io
import json
from pathlib import Path
import shutil

from scripts.zoh_compare_common import BASE,SOURCE,SEED,sha,write_json,verify_training

ROOT=BASE/'two_ckpt_train_0907_v2'
CONDITIONS=('B_scheduler_expert','C_scheduler_full')


def checkpoints():
    return {name:verify_training(name) for name in CONDITIONS}


def tasks_for(root,values):
    return [dict(condition=name,slot=slot,step=5000,policy_seed=SEED,
        evaluation_split='train',run_id=f'{root.name}_{name}_s{slot:02d}',**values[name])
        for slot in range(16) for name in CONDITIONS]


def metric_rows(results):
    rows=[]
    for name in CONDITIONS:
        selected=[r for r in results if r['condition']==name]
        n=len(selected)
        rows.append(dict(condition=name,split='train',completed=n,planned=16,
            success=sum(r['status']=='success' for r in selected),
            wrong_target_stop=sum(r['status']=='failed_wrong_target_stop' for r in selected),
            timeout=sum(r['status']=='FAILED_TIMEOUT' for r in selected),
            collision=sum(r['review']['collision'] for r in selected),
            entered_correct=sum(r['review']['reached_correct_region'] for r in selected),
            entered_other=sum(r['review']['reached_other_region'] for r in selected),
            mean_final_parking_distance_m=(sum(r['review']['final_distance_m'] for r in selected)/n if n else None)))
    return rows


def publish_training_evidence():
    destination=ROOT/'delivery/training';destination.mkdir(parents=True,exist_ok=False)
    from scripts.zoh_compare_common import RUNS
    for condition in CONDITIONS:
        run=RUNS[condition];target=destination/condition;target.mkdir()
        for name in ('training_config.json','parameter_scope.json','result.json','single_update_gate.json',
                     'full_gradient_gate.json','full_weight_update_gate.json','resource_profiles.json'):
            if (run/name).is_file():shutil.copy2(run/name,target/name)
        shutil.copy2(run/'checkpoint_005000/checkpoint_manifest.json',target/'checkpoint_005000_manifest.json')


def publish(results,complete=False):
    delivery=ROOT/'delivery';delivery.mkdir(exist_ok=True)
    metrics=metric_rows(results)
    write_json(delivery/'results.json',dict(complete=complete,scope='B/C train layouts only; eval not started',
        results=results,metrics=metrics))
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    (delivery/'summary.csv').write_text(buffer.getvalue())
    lines=['# B/C 两个 5k checkpoint：train rollout','',
        '**状态：'+('32/32完成；已停止，等待用户决定是否运行eval。' if complete else f'进行中：{len(results)}/32；未完成不计作失败。')+'**','',
        'B：warmup/cosine＋expert-only；C：相同warmup/cosine＋全量微调（含视觉/VLM）。',
        '两组同base、数据、80000采样索引、micro4/accum4、loss、5Hz/chunk5/execute1、policy seed及成对reset。',
        '这里只运行全部16个train任务；train是已见布局拟合检查，不代表泛化。未启动validation或独立test。','',
        '|条件|完成|成功|停错目标|超时|碰撞|进入正确区|进入错误区|末距均值(m)|','|---|---|---|---|---|---|---|---|---|']
    for r in metrics:
        distance='—' if r['mean_final_parking_distance_m'] is None else f"{r['mean_final_parking_distance_m']:.3f}"
        lines.append(f"|{r['condition']}|{r['completed']}/{r['planned']}|{r['success']}|{r['wrong_target_stop']}|{r['timeout']}|{r['collision']}|{r['entered_correct']}|{r['entered_other']}|{distance}|")
    if complete:
        b,c=metrics
        lines+=['',f"C−B成功数差：{c['success']-b['success']:+d}/16。该差异仅描述单训练seed/推理seed结果，不作统计显著性声明。",'',
            '## 训练证据','',
            '- [B训练配置](training/B_scheduler_expert/training_config.json)；[B 5k hash](training/B_scheduler_expert/checkpoint_005000_manifest.json)。',
            '- [C训练配置](training/C_scheduler_full/training_config.json)；[C 5k hash](training/C_scheduler_full/checkpoint_005000_manifest.json)。']
    lines+=['','## 逐条视频','','|slot|条件|指令/episode|结果|视频|','|---|---|---|---|---|']
    for r in results:
        lines.append(f"|{r['slot']:02d}|{r['condition']}|{r['episode_id']}|{r['status']}|[播放]({r['video']})|")
    (delivery/'README.md').write_text('\n'.join(lines)+'\n')
