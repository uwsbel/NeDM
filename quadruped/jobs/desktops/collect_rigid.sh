#!/bin/bash
# Rigid-ground control corpus: the SAME 24 shards x 50 episodes as go2_crm_v2 (same seeds,
# --pushes 2 --long-fraction 0.25), only --terrain rigid. Shards given as arguments.
cd ~/qrc/quadruped
export PYTHONPATH=$HOME/qrc:$HOME/qrc/src OMP_NUM_THREADS=1
P=~/miniconda3/envs/nedm/bin/python
U=~/.cache/nedm_cluster_stage/assets/data/robot/go2_irrvis/urdf/go2_description.urdf
for S in "$@"; do
  SEED=$(( 20260921 + S * 1000 ))
  ( nice -n 10 $P collect.py --corpus go2_rigid_v2_s${S} --out ~/qrc/data --policy ~/qrun/policy/policy.pt       --urdf $U --episodes 50 --terrain rigid --pushes 2 --long-fraction 0.25 --seed $SEED       > ~/qrc/logs/rigid_s${S}.log 2>&1; echo "SHARD_${S} exit $?" ) &
done
wait
echo COLLECT_DONE
