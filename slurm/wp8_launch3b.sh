#!/bin/bash
# Wave 3b: tracker-matched AR(1) action noise (per-step |d| ~0.025, lag-1 +0.3), corrected events (recoveries only), stuck kind
# weighted by its episode count, approach validation scored as |pred-rec|. Same base as wave 3.
cd /work1/dannegrut/harry/nedm
INIT=artifacts/traverse/wp2_mapv2_pt_dag_ro8_amd/ckpt_best.pt
B="--init-from $INIT --lr 3e-4 --min-lr 3e-5 --warmup-steps 500 --rollout-steps 80 --steps 6000 --event-frac 0.6 --progress-weight 1 --stall-eval-jitter 0.03 --event-kind-weights episodes"
sub() { local name=$1; shift; sbatch -J $name slurm/wp8_stall.sbatch $name "$@"; }
sub wp8d_ar03        $B --action-noise 0.0265 --action-noise-rho 0.3
sub wp8d_ar03_sm     $B --action-noise 0.0265 --action-noise-rho 0.3 --action-smooth-p 0.3
sub wp8d_ar03_sm_s1  $B --action-noise 0.0265 --action-noise-rho 0.3 --action-smooth-p 0.3 --seed 1
sub wp8d_ar05_sm     $B --action-noise 0.05   --action-noise-rho 0.3 --action-smooth-p 0.3
squeue -u $USER | wc -l
