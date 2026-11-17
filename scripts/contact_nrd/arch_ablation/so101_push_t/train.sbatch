#!/bin/bash
# The architecture ablation of so101_push_t: one model per array task (0: 3M, 1: 1T, 2: 1M).
#
#   sbatch scripts/contact_nrd/arch_ablation/so101_push_t/train.sbatch            # all three
#   sbatch --array=1 scripts/contact_nrd/arch_ablation/so101_push_t/train.sbatch  # the 1T only
#
# A full run is longer than the 4 h limit: submit the same command again and each unfinished run continues from its
# last check (every 10 000 updates); a finished run is skipped. The data must be in place (reproduce.sh so101_push_t
# download, on a node with network access). Once all three are complete, write the score table:
#   scripts/contact_nrd/arch_ablation/reproduce.sh so101_push_t test
# Submit from the repository root with a Python environment that has torch active (or set PY); the settings of
# scripts/contact_nrd/arch_ablation/reproduce.sh (ARTIFACTS, DATA_ROOT, SEED, ONE_UPDATES, RUN_TAG, ...) pass through.
#
# The original SO101 runs trained on the larger push-T corpus, compared with its 3T, and gave the one-network models
# the 3T's total of 150000 updates (reproduce.sh's header):
#   TRAIN_DATA=<...>/so101_push_rl/nrd_data/train MODEL_3T=<...>/so101_push_rl/nrd/rot_sc8_s61/best.pt \
#   ONE_UPDATES=150000 sbatch scripts/contact_nrd/arch_ablation/so101_push_t/train.sbatch
#
#SBATCH -J arch-so101_push_t
#SBATCH -o arch-so101_push_t.%A_%a.out
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 04:00:00
#SBATCH -p mi3001x
#SBATCH --array=0-2
set -euo pipefail
CASE=so101_push_t
ARCHS=(3m 1t 1m)
ARCH=${ARCHS[${SLURM_ARRAY_TASK_ID:-0}]}
cd "${SLURM_SUBMIT_DIR:-.}"
export PYTHONUNBUFFERED=1
echo "node: $(hostname -s)  job: ${SLURM_JOB_ID:-}  case: $CASE  arch: $ARCH"
scripts/contact_nrd/arch_ablation/reproduce.sh "$CASE" "$ARCH"
echo "ARCH_ABLATION_DONE $CASE $ARCH"
