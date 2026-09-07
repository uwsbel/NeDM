#!/bin/bash
# Wave 2 of the stall ablation (notes §13.4). Wave 1: best family = event-frac .6, K 80, progress loss; stall metrics
# plateau by step 2k while val loss rises (overfit); closed loop from rest still completes ~57 % of stalled runs.
# Axes here: regularisation (dropout, weight decay, input-noise), more weight on matched successes, longer horizon,
# stronger progress term, combinations, seeds. 6k steps (plateau at 2k), lr 3e-4.
cd /work1/dannegrut/harry/nedm
INIT=artifacts/traverse/wp2_mapv2_pt_dag_ro8_amd/ckpt_best.pt
FT="--init-from $INIT --lr 3e-4 --min-lr 3e-5 --warmup-steps 500"
MIX="--cache artifacts/traverse/wp2_z2_cache_v6 --extra-train-cache artifacts/traverse/wp2_z2_cache_dagger_v2 --z1-extra-cache artifacts/traverse/wp2_z2_cache_v6_pt"
B="$FT --rollout-steps 80 --steps 6000 --event-frac 0.6 --progress-weight 1"
sub() { local name=$1; shift; sbatch -J $name slurm/wp8_stall.sbatch $name "$@"; }
# regularisation
sub wp8b_drop1   $B --dropout 0.1
sub wp8b_drop2   $B --dropout 0.2
sub wp8b_wd5     $B --weight-decay 0.5
sub wp8b_noise05 $B --context-noise 0.05
sub wp8b_noise10 $B --context-noise 0.10
sub wp8b_noise20 $B --context-noise 0.20
# kind weighting (duplicates = weight): matched successes x3, and approach x2 as well
sub wp8b_match3    $B --event-kinds approach stuck launch recovery matched matched matched
sub wp8b_match3ap2 $B --event-kinds approach approach stuck launch recovery matched matched matched
# horizon
sub wp8b_k160      $FT --rollout-steps 160 --steps 5000 --event-frac 0.6 --progress-weight 1
sub wp8b_k120_p9   $FT --rollout-steps 120 --steps 6000 --event-frac 0.9 --progress-weight 1
# loss strength
sub wp8b_prog3     $FT --rollout-steps 80 --steps 6000 --event-frac 0.6 --progress-weight 3
sub wp8b_vx5       $FT --rollout-steps 80 --steps 6000 --event-frac 0.6 --vx-weight 5
# combinations
sub wp8b_combo     $FT --rollout-steps 120 --steps 6000 --event-frac 0.6 --progress-weight 1 --dropout 0.1 --context-noise 0.05 --event-kinds approach stuck launch recovery matched matched matched
sub wp8b_combo_mix $FT --rollout-steps 120 --steps 6000 --event-frac 0.6 --progress-weight 1 --dropout 0.1 --context-noise 0.05 --event-kinds approach stuck launch recovery matched matched matched $MIX
sub wp8b_combo_s1  $FT --rollout-steps 120 --steps 6000 --event-frac 0.6 --progress-weight 1 --dropout 0.1 --context-noise 0.05 --event-kinds approach stuck launch recovery matched matched matched --seed 1
squeue -u $USER | wc -l
