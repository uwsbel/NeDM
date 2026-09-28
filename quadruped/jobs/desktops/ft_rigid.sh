#!/bin/bash
# Fine-tune the base policy in a rigid-data surrogate with the recipe exactly
# (ft_data.sbatch): 1024 branches x 100 steps, iteration 1000, snapshots every 500.
#   ft_rigid.sh SEED
cd ~/qrc/quadruped
export PYTHONPATH=$HOME/qrc:$HOME/qrc/src OMP_NUM_THREADS=4
P=~/miniconda3/envs/nedm/bin/python
L=ft_rig_s$1
$P -u finetune.py --model ~/qrc/nnrom_rig_s$1_ms100/best.pt --policy ~/qrun/policy/policy.pt \
  --corpus ~/qrc/data/go2_rigid_v2 --out ~/qrc/$L --method ppo --device cuda \
  --seed 0 --steps 100 --branches 1024 --iters 1000 --target-dw 1e9 \
  --snapshot-every 500 > ~/qrc/logs/$L.log 2>&1
echo "FT $L exit $? $(date +%H:%M)"
