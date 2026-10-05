#!/bin/bash
#SBATCH -J ball-horizon
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 00:15:00
set -euo pipefail
set +u
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
set -u
export PYTHONPATH="$BALL_CODE/src:${PYTHONPATH:-}"
cd "$BALL_CODE"
python3.12 -m nedm.bouncing_ball.horizon --data "$BALL_ROOT/data" --checkpoint "$BALL_ROOT/runs/nrd_v2/diagnostic_warmup.pt" --report "$BALL_ROOT/horizon_diagnosis.json" --output-data "$BALL_ROOT/data_margin20ms" --margin-s .02
