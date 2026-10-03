#!/bin/bash
# Run any render command on one MI210 with torch and Warp sharing the GPU.
#
#   sbatch scripts/render/hpcfund/run.sbatch python scripts/render/go2_nrd_rollout.py run \
#       --quadruped $WORK/qrun --model ... --policy ... --starts ... --urdf ... --out out/go2
#
# mi2101x bills 0.1 node-hours per hour. A 1024-world, 5 s Go2 rollout takes about 3 minutes.
#SBATCH -J nrd-render
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -uo pipefail
WORK=${WORK:-/work1/dannegrut/$USER}
REPO=${REPO:-$WORK/NeDM}
module purge
module load gnu12 rocm
source "$REPO/scripts/render/hpcfund/env.sh"
cd "$REPO"
echo "host $(hostname) job $SLURM_JOB_ID: $*"
"$@"
