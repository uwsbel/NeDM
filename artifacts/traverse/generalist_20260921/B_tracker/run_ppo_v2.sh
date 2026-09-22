#!/bin/bash
# PPO round 2 (PLAN v2 B5 improvement round): refit dynamics model (cache v3: hold-mode + harvested failures), stronger
# speed-tracking term (weight 1.5 vs 0.5 in round 1), otherwise the round-1 recipe. Run after nrd_tag_v3 is synced.
set -u
cd /home/harry/NeDM-traverse_mppi
PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts
K=artifacts/traverse/generalist_20260921/B_tracker
$PY scripts/gb_train_tracker.py --out $K/ppo_v2 --nrd $K/nrd_tag_v3/ckpt_best.pt --cache $K/cache_v3 \
   --grid artifacts/traverse/crm_f104_v1/grids/arena_f104_50h_v1/grid.npz --num-envs 2048 --max-iterations 1000 \
   --save-interval 100 --imitation-samples 131072 --seed 2 --speed-weight 1.5 > $K/ppo_v2.log 2>&1
echo "ppo_v2 rc=$? $(date)"
