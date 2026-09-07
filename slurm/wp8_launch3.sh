#!/bin/bash
# Wave 3 (notes §13.9): control-input augmentation against the constant-throttle stall fingerprint (audit 2026-09-07).
# Base = the wave-1 primary (p .6, K 80, progress loss, lr 3e-4, 6k steps); selection by the jitter-robust stall score.
cd /work1/dannegrut/harry/nedm
INIT=artifacts/traverse/wp2_mapv2_pt_dag_ro8_amd/ckpt_best.pt
B="--init-from $INIT --lr 3e-4 --min-lr 3e-5 --warmup-steps 500 --rollout-steps 80 --steps 6000 --event-frac 0.6 --progress-weight 1 --stall-eval-jitter 0.03"
sub() { local name=$1; shift; sbatch -J $name slurm/wp8_stall.sbatch $name "$@"; }
sub wp8c_base          $B
sub wp8c_an02          $B --action-noise 0.02
sub wp8c_an03          $B --action-noise 0.03
sub wp8c_an05          $B --action-noise 0.05
sub wp8c_an10          $B --action-noise 0.10
sub wp8c_an03_sm       $B --action-noise 0.03 --action-smooth-p 0.3
sub wp8c_an05_sm       $B --action-noise 0.05 --action-smooth-p 0.3
sub wp8c_an03_sm_s1    $B --action-noise 0.03 --action-smooth-p 0.3 --seed 1
squeue -u $USER | wc -l
