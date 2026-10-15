#!/bin/bash
# build9: up to 8 commands in parallel on one 8 x MI350X node (mi3508x), one GPU each (HIP_VISIBLE_DEVICES only:
# setting ROCR_VISIBLE_DEVICES as well composes the filters). JOBS = a file with one shell command per line (run in
# $REPO with render_env.sh sourced); line i runs on GPU i; each line's output goes to $VIRAL/logs/<job>_<i>.out.
#
#   sbatch -J v9-multi -t 03:00:00 --export=ALL,JOBS=/path/jobs.txt scripts/so101_policy/cluster/viral_multi.sbatch
#
# mi3508x bills 8 x 0.175 = 1.4 ledger-h per hour (whole node).
#SBATCH -J v9-multi
#SBATCH -A dannegrut
#SBATCH -p mi3508x
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -o /work1/dannegrut/harry/experiments/so101_policy_20261011/viral/logs/%x_%j.out
set -uo pipefail
VIRAL=${VIRAL:-/work1/dannegrut/harry/experiments/so101_policy_20261011/viral}
export REPO=$VIRAL/repo
set +u; source "$REPO/scripts/so101_policy/cluster/render_env.sh"; set -u
export MUJOCO_GL=${MUJOCO_GL:-disable}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
cd "$REPO"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} start $(date -Is)"
i=0
pids=()
while IFS= read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue;; esac
  ( export HIP_VISIBLE_DEVICES=$i; export MIOPEN_USER_DB_PATH=${TMPDIR:-/tmp}/miopen-$USER-${SLURM_JOB_ID:-$$}-$i
    export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH; mkdir -p "$MIOPEN_USER_DB_PATH"
    echo "gpu $i: $line"; bash -c "$line" ) > "$VIRAL/logs/${SLURM_JOB_ID:-x}_$i.out" 2>&1 &
  pids+=($!)
  i=$((i + 1))
done < "$JOBS"
st=0
for p in "${pids[@]}"; do wait "$p" || st=1; done
echo "exit $st $(date -Is)"
exit $st
