#!/bin/bash
#SBATCH -J cn-convert
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 01:00:00
set -euo pipefail
set +u
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
set -u
export PYTHONPATH="$CN_CODE/src:${PYTHONPATH:-}"
cd "$CN_CODE"
python3.12 -m nedm.contact_nrd.systems "$@"
