#!/bin/bash
# P5-phase1 formal training, parameters confirmed by the user 2026-09-13:
# 2 epochs, 32 layers, LoRA 1e-4 / lm_head 2e-5, checkpoint every 0.4 epoch (all kept),
# optimizer + scheduler state in latest/. Resume after any interruption: run this again.
cd /root/autodl-tmp/go2_short_vln_repo
export GO2_DATA_ROOT=/root/autodl-tmp/go2_short_vln
export HF_HOME=$GO2_DATA_ROOT/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export GO2_TB_ROOT=/root/tf-logs
PY=/root/miniconda3/envs/smolvla/bin/python
RUN=p5_phase1_lora_r2r_2ep_0913
LOG=/root/autodl-tmp/$RUN.log
# Guards: never two trainers on one run dir, never start on a GPU something else holds
# (a kill -9 can leave DataLoader workers alive with the CUDA context; clean those first).
if pgrep -f "p5_phase1_train.py --run-name $RUN" > /dev/null; then
  echo "$(date -Is) === refuse: a process for $RUN is still alive: $(pgrep -f "p5_phase1_train.py --run-name $RUN" | tr '\n' ' ') ===" >> $LOG; exit 2
fi
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$USED" -gt 1000 ]; then
  echo "$(date -Is) === refuse: GPU already has ${USED} MiB in use ===" >> $LOG; exit 3
fi
echo "$(date -Is) === start ===" >> $LOG
$PY scripts/p5_phase1_train.py --run-name $RUN --out-dir $GO2_DATA_ROOT/checkpoints \
  --epochs 2 --batch-size 16 --workers 2 --lr-lora 1e-4 --lr-head 2e-5 --weight-decay 0 \
  --warmup-ratio 0.03 --grad-clip 1.0 --seed 20260913 \
  --save-every-epochs 0.4 --eval-every-epochs 0.2 --eval-records 2048 --greedy-records 32 \
  >> $LOG 2>&1; RC=$?
echo "$(date -Is) === exit $RC ===" >> $LOG
