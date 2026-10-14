#!/bin/bash
# Run any command with torch + the Newton/Warp batch renderer sharing one MI350X (gfx950) in one process.
#
#   sbatch scripts/so101_policy/cluster/render_run.sbatch python3.12 -P scripts/so101_policy/cluster/render_amd_smoke.py ...
#   sbatch -p mi2101x scripts/so101_policy/cluster/render_run.sbatch ...         # MI210 (gfx90a); the build has it
#
# Needs the one-time setup of build8_amd_render.md (venv + render_warp_build.sbatch). The command runs in $REPO
# (default $RENDER_AMD/repo: rsync src scripts configs tests contact_nrd there). mi3501x bills 0.175 ledger-h per hour.
#SBATCH -J r8-render
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -o /work1/dannegrut/harry/experiments/so101_policy_20261011/render_amd/logs/%x_%j.out
set -uo pipefail
RENDER_AMD=${RENDER_AMD:-/work1/dannegrut/harry/experiments/so101_policy_20261011/render_amd}
set +u; source "${REPO:-$RENDER_AMD/repo}/scripts/so101_policy/cluster/render_env.sh"; set -u
export MUJOCO_GL=${MUJOCO_GL:-disable}      # no EGL on the compute nodes; mujoco is only used to export meshes
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
cd "$REPO"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} start $(date -Is) gpu $(rocminfo | grep -m1 -o 'gfx9[0-9a-z]*')"
echo "cmd: $*"
"$@"
status=$?
echo "exit $status $(date -Is)"
exit $status
