#!/bin/bash
#SBATCH -J ball-calibrate
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
python3.12 -m unittest discover -s tests/bouncing_ball -p test_model_validation.py
python3.12 -m nedm.bouncing_ball.calibrate --data "${BALL_DATA:-$BALL_ROOT/data}" --checkpoint "${BALL_CHECKPOINT:-$BALL_ROOT/runs/nrd_v2/diagnostic_warmup.pt}" --output-dir "$BALL_RUN" "$@"
