#!/bin/bash
# WHEN did seed 6's run go wrong? The same probe along its own checkpoints, rolled in its
# own surrogate and in a healthy one, so the onset of the tumbling policy is dated.
#SBATCH -J kyle-probe2
#SBATCH -p mi2104x
#SBATCH -N 1
#SBATCH -t 00:40:00
#SBATCH -o /work1/dannegrut/kyle/logs/%x-%j.out
#SBATCH -e /work1/dannegrut/kyle/logs/%x-%j.err
set -uo pipefail
K=/work1/dannegrut/kyle
module purge
module load gnu12 rocm
source $K/nedm/env-scoring.sh
export PYTHONPATH=$K/qrun:$K/qrun/src:$PYTHONPATH
cd $K/qrun/quadruped
L=$K/qrun/ft_h/long_nnrom_v2h_s6_ms100_s0/snapshots
M="$K/qrun/nnrom_v2h_s6_ms100_s0/best.pt $K/qrun/nnrom_v2h_s8_ms100_s0/best.pt"
for C in dw2 dw4 dw5 it500 it1000 it1500; do
  [ -f $L/policy_$C.pt ] || continue
  echo "======== s6 $C"
  python -u diagnostics/probe_ood.py --policy $L/policy_$C.pt --models $M \
    --corpus $K/qrun/data/go2_crm_v2 --branches 256 --steps 100 --device cuda
done
echo PROBE2_DONE
