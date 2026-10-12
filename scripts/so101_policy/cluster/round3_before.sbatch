#!/bin/bash
#SBATCH -J sp-r3-before
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 01:30:00
#SBATCH -o /work1/dannegrut/harry/experiments/so101_policy_20261011/logs/%x_%j.out
# Round 3 'before' numbers on one MI350X: the round-2 grasp NRD (s61, s62; = ~/so101_policy/ckpt/nrd_r2_s6x.pt) on the
# policy_v3 evaluation folders, open loop (free rollout from the first record with the recorded MuJoCo commands):
# lift / drop flag agreement, end error, cube error while MuJoCo holds the cube (collect_round.py eval --carry), per
# policy and noise level. Run after round3_collect.sbatch (sbatch --dependency=afterok:<collect job>).
# Outputs: $B/round3/before/r2_policy_{pgate,val_full}.json
set -euo pipefail
set +u; source /etc/profile; module load pytorch/2.10.0; source /work1/dannegrut/harry/venvs/nedm/bin/activate; set -u
B=/work1/dannegrut/harry/experiments/so101_policy_20261011
C=${SP_CODE:-$B/round3/code}
P=$B/data/policy_v3
O=$B/round3/before
export OMP_NUM_THREADS=4 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 PYTHONPATH="$C/src:${PYTHONPATH:-}"
cd "$C"
mkdir -p "$O"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} start $(date -Is)"
sha256sum $B/runs/nrd_ft_r2_s61/best.pt $B/runs/nrd_ft_r2_s62/best.pt
for set in pgate val_full; do
  [ -f "$O/r2_policy_$set.json" ] && continue
  echo "eval $set $(date -Is)"
  python3.12 -P scripts/so101_policy/collect_round.py eval --data "$P/$set" --device cuda --carry \
    --model "s61=$B/runs/nrd_ft_r2_s61/best.pt" "s62=$B/runs/nrd_ft_r2_s62/best.pt" --out "$O/r2_policy_$set.json" \
    > "$O/r2_policy_$set.log" 2>&1
  grep -E " all | pol_..._[df] |noise_" "$O/r2_policy_$set.log" | head -60
done
echo "done $(date -Is)"
