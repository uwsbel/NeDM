#!/bin/bash
#SBATCH -J cn-train
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 04:00:00
set -euo pipefail
set +u
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONFAULTHANDLER=1
export PYTHONPATH="$CN_CODE/src:${PYTHONPATH:-}"
cd "$CN_CODE"
python3.12 -m nedm.contact_nrd.train --data "$CN_DATA" --config "$CN_CONFIG" --output-dir "$CN_RUN" "$@"
