#!/bin/bash
# The architecture ablation of bouncing_ball: one model per array task (0: 3M, 1: 1T, 2: 1M).
#
#   sbatch scripts/contact_nrd/arch_ablation/bouncing_ball/train.sbatch            # all three
#   sbatch --array=1 scripts/contact_nrd/arch_ablation/bouncing_ball/train.sbatch  # the 1T only
#
# A full run is longer than the 4 h limit: submit the same command again and each unfinished run continues from its
# last check (every 10 000 updates); a finished run is skipped. The data must be in place (reproduce.sh bouncing_ball
# download, on a node with network access). Once all three are complete, write the score table:
#   scripts/contact_nrd/arch_ablation/reproduce.sh bouncing_ball test
# Submit from the repository root with a Python environment that has torch active (or set PY); the settings of
# scripts/contact_nrd/arch_ablation/reproduce.sh (ARTIFACTS, DATA_ROOT, SEED, ONE_UPDATES, RUN_TAG, ...) pass through.
#
#SBATCH -J arch-bouncing_ball
#SBATCH -o arch-bouncing_ball.%A_%a.out
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 04:00:00
#SBATCH -p mi3001x
#SBATCH --array=0-2
set -euo pipefail
CASE=bouncing_ball
ARCHS=(3m 1t 1m)
ARCH=${ARCHS[${SLURM_ARRAY_TASK_ID:-0}]}
cd "${SLURM_SUBMIT_DIR:-.}"
export PYTHONUNBUFFERED=1
echo "node: $(hostname -s)  job: ${SLURM_JOB_ID:-}  case: $CASE  arch: $ARCH"
scripts/contact_nrd/arch_ablation/reproduce.sh "$CASE" "$ARCH"
echo "ARCH_ABLATION_DONE $CASE $ARCH"
