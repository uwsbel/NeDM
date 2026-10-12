#!/bin/bash
# Submit the NRD data-round fine-tune of both seeds (nrd_round_ft.sbatch) on AMD with a resubmission chain for the 4 h
# cap. Run from the worktree root on luffy:
#   bash scripts/so101_policy/cluster/submit_nrd_round_ft.sh            (PART=mi3501x, ROUND=noisy_v1, CHAIN=1)
# Steps: (1) rsync the worktree's src/, scripts/, configs/ to $B/code_nrd_round (contact_nrd source sha256 printed);
# (2) per seed one job on PART, then CHAIN follow-up jobs with --dependency=afternotok:<previous> and
# --kill-on-invalid-dep=yes: a follow-up starts only if the previous job failed or hit the time limit (train.py resumes
# from contact_state.pt, saved at every check), and it is cancelled by SLURM when the previous job succeeds.
# PART=mi3508x SEEDS_PER_JOB=2 runs both seeds in one whole-node job on GPUs 0 and 1 (same ledger rate per GPU-hour).
set -euo pipefail
PART=${PART:-mi3501x}
ROUND=${ROUND:-noisy_v1}
CHAIN=${CHAIN:-1}
B=/work1/dannegrut/harry/experiments/so101_policy_20261011
C=$B/code_nrd_round
ssh amd "mkdir -p $C $B/logs"
rsync -a --delete --exclude __pycache__ src scripts configs "amd:$C/"
ssh amd "cd $C && sha256sum src/nedm/contact_nrd/*.py configs/so101_policy/nrd_round/ft_${ROUND}_s6*.json"
submit() {   # submit <export vars> [dependency] -> job id
  local dep=""
  [ -n "${2:-}" ] && dep="--dependency=afternotok:$2 --kill-on-invalid-dep=yes"
  ssh amd "cd $C && sbatch --parsable -p $PART $dep --export=ALL,$1,ROUND=$ROUND scripts/so101_policy/cluster/nrd_round_ft.sbatch"
}
if [ "${SEEDS_PER_JOB:-1}" = 2 ]; then
  groups=("SEEDS=61+62")
else
  groups=("SEED=61" "SEED=62")
fi
for g in "${groups[@]}"; do
  j=$(submit "$g")
  echo "$g: job $j"
  for k in $(seq 1 "$CHAIN"); do
    j=$(submit "$g" "$j")
    echo "$g: follow-up $k job $j (afternotok)"
  done
done
