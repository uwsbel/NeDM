#!/bin/bash
# Wave 5: the constant-throttle cue attacked directly -- random windows get a perfectly constant throttle input (hold), on top of
# the tracker-matched AR(1) noise; with and without the momentum-loss events.
cd /work1/dannegrut/harry/nedm
INIT=artifacts/traverse/wp2_mapv2_pt_dag_ro8_amd/ckpt_best.pt
B="--init-from $INIT --lr 3e-4 --min-lr 3e-5 --warmup-steps 500 --rollout-steps 80 --steps 6000 --event-frac 0.6 --progress-weight 1 --stall-eval-jitter 0.03 --event-kind-weights episodes --action-noise 0.0265 --action-noise-rho 0.3"
sub() { local name=$1; shift; sbatch -J $name slurm/wp8_stall.sbatch $name "$@"; }
sub wp8f_hold3        $B --action-hold-p 0.3
sub wp8f_hold5        $B --action-hold-p 0.5
sub wp8f_hold3_mom    $B --action-hold-p 0.3 --event-kinds approach launch matched momentum momentum momentum recovery stuck
sub wp8f_hold5_mom_s1 $B --action-hold-p 0.5 --event-kinds approach launch matched momentum momentum momentum recovery stuck --seed 1
squeue -u $USER | wc -l
