#!/bin/bash
# The PPO recipe policy (seed 8 surrogate, iteration 1000) on d33, so analytic and PPO
# compare on one build: the analytic grid used the same seed-8 surrogate.
cd ~/qrc/quadruped
export PYTHONPATH=$HOME/qrc:$HOME/qrc/src OMP_NUM_THREADS=8
~/miniconda3/envs/nedm/bin/python -u evaluate.py --policy ~/qrc/ppo_s8_it1000.pt --urdf ~/.cache/nedm_cluster_stage/assets/data/robot/go2_irrvis/urdf/go2_description.urdf --corpus ~/qrun/quadruped/data/go2_crm_v2 --out ~/qrc/eval_paths_d33b/ppo_s8_it1000 --label ppo_s8_it1000 --terrain crm --paths 4 --seconds 15 --spawn-spread 1.0 > ~/qrc/logs/evp2/ppo_s8_it1000.log 2>&1
echo "EVAL ppo_s8_it1000 rc=$? $(date +%H:%M)"
