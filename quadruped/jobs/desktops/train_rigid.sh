#!/bin/bash
# Rigid-data control surrogate: the recipe exactly (train_v23.sbatch), corpus go2_rigid_v2.
#   train_rigid.sh SEED
cd ~/qrc/quadruped
export PYTHONPATH=$HOME/qrc:$HOME/qrc/src OMP_NUM_THREADS=4
P=~/miniconda3/envs/nedm/bin/python
C=~/qrc/data/go2_rigid_v2; L=nnrom_rig_s$1
$P -u train.py --corpus $C --out ~/qrc/$L --seed $1 --device cuda > ~/qrc/logs/train_$L.log 2>&1
echo "STAGE1 $L exit $? $(date +%H:%M)"
$P -u train.py --corpus $C --out ~/qrc/${L}_ms100 --seed $1 --device cuda \
  --init-from ~/qrc/$L/best.pt --rollout-loss-steps 100 --rollout-loss-batch 4 --epochs 8 \
  --steps-per-epoch 1000 --lr 1e-4 --min-lr 1e-5 --warmup-steps 200 --select-window 3 \
  > ~/qrc/logs/train_${L}_ms100.log 2>&1
echo "STAGE2 $L exit $? $(date +%H:%M)"
