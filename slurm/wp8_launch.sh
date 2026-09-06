#!/bin/bash
# Stall-reproduction ablation grid (plan §30, notes §13). Run on the cluster login node after syncing code + events.json.
#   axes: event fraction p (0 / .3 / .6 / .9) x rollout horizon K (8 / 40 / 80 / 120 steps = 0.4 / 2 / 4 / 6 s)
#         x loss (plain / progress / delta-scale / progress+vx) x init (fine-tune / scratch) x data (new / mixed) x lr x seed
cd /work1/dannegrut/harry/nedm
INIT=artifacts/traverse/wp2_mapv2_pt_dag_ro8_amd/ckpt_best.pt
FT="--init-from $INIT --lr 3e-4 --min-lr 3e-5 --warmup-steps 500"
MIX="--cache artifacts/traverse/wp2_z2_cache_v6 --extra-train-cache artifacts/traverse/wp2_z2_cache_dagger_v2 --z1-extra-cache artifacts/traverse/wp2_z2_cache_v6_pt"
S12=12000; S80=10000
sub() { local name=$1; shift; sbatch -J $name slurm/wp8_stall.sbatch $name "$@"; }
# grid 1: sampling x horizon (fine-tune, new data)
for p in 0 3 6; do
  sub wp8_p${p}_k8   $FT --rollout-steps 8  --steps $S12 --event-frac 0.$p
  sub wp8_p${p}_k40  $FT --rollout-steps 40 --steps $S12 --event-frac 0.$p
  sub wp8_p${p}_k80  $FT --rollout-steps 80 --steps $S80 --event-frac 0.$p
done
# grid 2: loss, at (p3,k40) and (p6,k80)
sub wp8_p3_k40_prog   $FT --rollout-steps 40 --steps $S12 --event-frac 0.3 --progress-weight 1
sub wp8_p3_k40_ds     $FT --rollout-steps 40 --steps $S12 --event-frac 0.3 --delta-scale
sub wp8_p3_k40_progvx $FT --rollout-steps 40 --steps $S12 --event-frac 0.3 --progress-weight 1 --vx-weight 5
sub wp8_p6_k80_prog   $FT --rollout-steps 80 --steps $S80 --event-frac 0.6 --progress-weight 1
sub wp8_p6_k80_ds     $FT --rollout-steps 80 --steps $S80 --event-frac 0.6 --delta-scale
sub wp8_p6_k80_progvx $FT --rollout-steps 80 --steps $S80 --event-frac 0.6 --progress-weight 1 --vx-weight 5
# grid 3: init and data mix
sub wp8_p3_k40_prog_scratch --lr 3e-4 --min-lr 3e-5 --warmup-steps 1000 --rollout-steps 40 --steps 20000 --event-frac 0.3 --progress-weight 1
sub wp8_p6_k80_prog_scratch --lr 3e-4 --min-lr 3e-5 --warmup-steps 1000 --rollout-steps 80 --steps 14000 --event-frac 0.6 --progress-weight 1
sub wp8_p3_k40_prog_mix $FT --rollout-steps 40 --steps $S12 --event-frac 0.3 --progress-weight 1 $MIX
sub wp8_p6_k80_prog_mix $FT --rollout-steps 80 --steps $S80 --event-frac 0.6 --progress-weight 1 $MIX
# grid 4: learning rate
sub wp8_p3_k40_prog_lr1 --init-from $INIT --lr 1e-4 --min-lr 1e-5 --warmup-steps 300 --rollout-steps 40 --steps $S12 --event-frac 0.3 --progress-weight 1
sub wp8_p6_k80_prog_lr1 --init-from $INIT --lr 1e-4 --min-lr 1e-5 --warmup-steps 300 --rollout-steps 80 --steps $S80 --event-frac 0.6 --progress-weight 1
# grid 5: seeds
sub wp8_p6_k80_prog_s1 $FT --rollout-steps 80 --steps $S80 --event-frac 0.6 --progress-weight 1 --seed 1
sub wp8_p6_k80_prog_s2 $FT --rollout-steps 80 --steps $S80 --event-frac 0.6 --progress-weight 1 --seed 2
# grid 6: heavier event share, longer horizon
sub wp8_p9_k80_prog  $FT --rollout-steps 80  --steps $S80 --event-frac 0.9 --progress-weight 1
sub wp8_p6_k120_prog $FT --rollout-steps 120 --steps 8000 --event-frac 0.6 --progress-weight 1
squeue -u $USER
