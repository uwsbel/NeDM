#!/bin/bash
# build9 (viral student): run one command on one MI350X with torch + the Newton/Warp renderer (render_env.sh of
# build8_amd_render.md) and the build9 code in $VIRAL/repo.
#
#   sbatch -J v9-ppo -t 03:00:00 scripts/so101_policy/cluster/viral_run.sbatch python3.12 -P scripts/so101_policy/... 
#
# mi3501x bills 0.175 ledger-h per hour.
#SBATCH -J v9-run
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -o /work1/dannegrut/harry/experiments/so101_policy_20261011/viral/logs/%x_%j.out
set -uo pipefail
VIRAL=${VIRAL:-/work1/dannegrut/harry/experiments/so101_policy_20261011/viral}
export REPO=$VIRAL/repo
set +u; source "$REPO/scripts/so101_policy/cluster/render_env.sh"; set -u
export MUJOCO_GL=${MUJOCO_GL:-disable}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
cd "$REPO"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} start $(date -Is) gpu $(rocminfo | grep -m1 -o 'gfx9[0-9a-z]*')"
echo "cmd: $*"
"$@"
status=$?
echo "exit $status $(date -Is)"
exit $status
