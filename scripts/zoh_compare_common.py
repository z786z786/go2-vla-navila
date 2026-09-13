"""Pinned three-checkpoint contract and CPU-only comparison reporting."""
import csv
import hashlib
import io
import json
import shutil
from pathlib import Path

BASE=Path('/mnt/wxh/go2_short_vln/outputs/dual_target_v2')
SOURCE=Path('/home/wxh/go2_short_vln')
ROOT=BASE/'three_ckpt_compare_0907_v1'
RUNS={
    'A_constant_expert':BASE/'zoh_train_5000_0906_v2',
    'B_scheduler_expert':BASE/'zoh_scheduler_5000_0907_v1',
    'C_scheduler_full':BASE/'zoh_full_5000_0907_v1',
}
SEED=20260906

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def write_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def verify_training(name):
    run=RUNS[name];result=json.loads((run/'result.json').read_text())
    if result['optimizer_updates']!=5000 or not result['status'].endswith('COMPLETE_AWAITING_REVIEW'):
        raise ValueError('training did not finish '+name)
    path=run/'checkpoint_005000';manifest=json.loads((path/'checkpoint_manifest.json').read_text())
    if manifest['optimizer_updates']!=5000:raise ValueError('wrong checkpoint step')
    for f,d in manifest['files_sha256'].items():
        target=(path/f).resolve()
        if not target.is_relative_to(path.resolve()) or sha(target)!=d:raise ValueError('checkpoint drift '+f)
    cfg=json.loads((run/'training_config.json').read_text())
    for k,v in dict(effective_batch=16,micro_batch=4,gradient_accumulation=4,seed=SEED,fps=5,chunk_size=5,execute_steps=1).items():
        if cfg.get(k)!=v:raise ValueError('uncontrolled config difference '+k)
    reference=json.loads((RUNS['A_constant_expert']/'training_config.json').read_text())
    for k in ('dataset_manifest_sha256','normalizer_sha256','sampled_indices_sha256','betas','eps','weight_decay','gradient_clip','loss'):
        if cfg[k]!=reference[k]:raise ValueError('uncontrolled data/optimizer difference '+k)
    if sha(run/'sampled_indices.npy')!=cfg['sampled_indices_sha256']:raise ValueError('sampler drift')
    if name!='A_constant_expert':
        if cfg['scheduler']!='cosine_decay_with_warmup' or cfg['scheduler_warmup_steps_actual']!=166:
            raise ValueError('scheduler comparison mismatch')
        if not json.loads((run/'single_update_gate.json').read_text())['passed']:raise ValueError('training gate failed')
    if name=='C_scheduler_full':
        scope=json.loads((run/'parameter_scope.json').read_text())
        if not scope or not all(x['trainable'] for x in scope):raise ValueError('not full-policy training')
        if not json.loads((run/'full_gradient_gate.json').read_text())['groups']:raise ValueError('missing full gradients')
        if not json.loads((run/'full_weight_update_gate.json').read_text()):raise ValueError('missing full updates')
    return dict(checkpoint=str(path),checkpoint_manifest_sha256=sha(path/'checkpoint_manifest.json'))

def tasks_for(root,checkpoints):
    # Paired order limits temporal bias from shared GPU load. No independent test slots.
    return [dict(condition=name,slot=slot,step=5000,policy_seed=SEED,
        evaluation_split='train' if slot<16 else 'validation',
        run_id=f'{root.name}_{name}_s{slot:02d}',**checkpoints[name])
        for slot in range(24) for name in RUNS]

def publish_training_evidence():
    destination=ROOT/'delivery/training';destination.mkdir(exist_ok=False)
    for condition,run in RUNS.items():
        target=destination/condition;target.mkdir()
        for name in ('training_config.json','parameter_scope.json','result.json','single_update_gate.json',
                     'full_gradient_gate.json','full_weight_update_gate.json','resource_profiles.json'):
            if (run/name).is_file():shutil.copy2(run/name,target/name)
        shutil.copy2(run/'checkpoint_005000/checkpoint_manifest.json',target/'checkpoint_005000_manifest.json')

def metric_rows(results):
    output=[]
    for name in RUNS:
        for split,total in [('train',16),('validation',8)]:
            rows=[r for r in results if r['condition']==name and r['evaluation_split']==split]
            n=len(rows)
            output.append(dict(condition=name,split=split,completed=n,planned=total,
                success=sum(r['status']=='success' for r in rows),
                wrong_target_stop=sum(r['status']=='failed_wrong_target_stop' for r in rows),
                timeout=sum(r['status']=='FAILED_TIMEOUT' for r in rows),
                collision=sum(r['review']['collision'] for r in rows),
                entered_correct=sum(r['review']['reached_correct_region'] for r in rows),
                entered_other=sum(r['review']['reached_other_region'] for r in rows),
                mean_final_parking_distance_m=sum(r['review']['final_distance_m'] for r in rows)/n if n else None))
    return output

def publish(results,complete=False):
    delivery=ROOT/'delivery';delivery.mkdir(exist_ok=True)
    metrics=metric_rows(results)
    write_json(delivery/'results.json',dict(complete=complete,results=results,metrics=metrics))
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    (delivery/'summary.csv').write_text(buffer.getvalue())
    lines=['# 三个 5k checkpoint 对比','',
        '**状态：'+('72/72 已完成，自动轨迹复核及视频导出完成。' if complete else f'进行中，已完成 {len(results)}/72；未完成不能计作失败。')+'**','',
        'A：constant LR＋expert-only；B：warmup/cosine＋expert-only；C：相同warmup/cosine＋全量微调（含视觉/VLM）。',
        '三组同base、同16条训练轨迹、同80000采样索引、micro4/accum4、同目标步数5000。C启用视觉激活重计算以节省显存；全量参数使用FP32 master。',
        '评估覆盖所有16个train任务及8个validation任务，单一固定policy seed=20260906；不使用独立test集，不依据离线loss挑checkpoint。train是已见场景拟合检查，不是独立泛化成绩。',
        '每条最长30秒仿真时间；控制、配对重置规则、停车/碰撞标准一致。停错目标与超时均为模型结果；基础设施中断不冒充任务失败。','',
        '|条件|场景|完成|成功|停错目标|超时|碰撞|进入正确停车区|','|---|---|---|---|---|---|---|---|']
    for r in metrics:
        lines.append(f"|{r['condition']}|{r['split']}|{r['completed']}/{r['planned']}|{r['success']}|{r['wrong_target_stop']}|{r['timeout']}|{r['collision']}|{r['entered_correct']}|")
    if complete:
        lines+=['','## 对照差异（描述性，不作统计显著性声明）','']
        for split in ('train','validation'):
            rows=[r for r in metrics if r['split']==split];a,b,c=rows
            lines.append(f"- {split}：B−A 成功数差 {b['success']-a['success']:+d}/{a['planned']}；C−B 成功数差 {c['success']-b['success']:+d}/{a['planned']}。")
        lines+=['','B−A用于观察scheduler作用；C−B用于观察在同scheduler下扩大微调范围的作用。只有一个训练seed/推理seed，不能将差异解释为稳定因果结论。',
            '这是自动证据报告；成功轨迹经独立停车重算，附全部视频供视觉复核，不宣称已人工逐条看完。']
        lines+=['','## 训练与 checkpoint 证据','']
        for name in RUNS:
            lines.append(f'- {name}：[训练配置](training/{name}/training_config.json)、[5k 文件哈希](training/{name}/checkpoint_005000_manifest.json)、[训练结果](training/{name}/result.json)。')
    lines+=['','## 逐条结果与视频','','|任务slot|条件|场景|指令目标|结果|视频|','|---|---|---|---|---|---|']
    for r in results:
        lines.append(f"|{r['slot']:02d}|{r['condition']}|{r['evaluation_split']}|{r['episode_id']}|{r['status']}|[播放]({r['video']})|")
    (delivery/'README.md').write_text('\n'.join(lines)+'\n')
