#!/bin/bash
# Plan 28 step 2 training variants on the multi-arena cache (run on the cluster login node after syncing wp7_cache_v1).
#   A  fine-tune the deployed arena_v1 model on the new train arenas + the arena_v1 tracker episodes (anti-forgetting mix)
#   B  same data, trained from scratch (does arena_v1 pre-training matter?)
#   C  fine-tune on the new arenas only (how much arena_v1 knowledge is lost / needed)
cd /work1/dannegrut/harry/nedm
INIT=artifacts/traverse/wp2_mapv2_pt_dag_ro8_amd/ckpt_best.pt
MIX="--cache artifacts/traverse/wp2_z2_cache_v6 --extra-train-cache artifacts/traverse/wp2_z2_cache_dagger_v2 --z1-extra-cache artifacts/traverse/wp2_z2_cache_v6_pt"
sbatch -J wp7_ftA slurm/wp7_finetune.sbatch wp7_ft_mix_amd    --init-from $INIT --rollout-steps 8 --steps 12000 --lr 1e-4 --min-lr 1e-5 --warmup-steps 300 $MIX
sbatch -J wp7_ftB slurm/wp7_finetune.sbatch wp7_scratch_mix_amd --rollout-steps 8 --steps 20000 --lr 3e-4 --min-lr 3e-5 --warmup-steps 1000 $MIX
sbatch -J wp7_ftC slurm/wp7_finetune.sbatch wp7_ft_new_amd    --init-from $INIT --rollout-steps 8 --steps 12000 --lr 1e-4 --min-lr 1e-5 --warmup-steps 300
squeue -u $USER
