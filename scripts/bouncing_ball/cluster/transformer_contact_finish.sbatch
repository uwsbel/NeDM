#!/bin/bash
#SBATCH -J ball-tcontact-finish
#SBATCH -A dannegrut
#SBATCH -p devel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:30:00
set -euo pipefail
set +u
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$BALL_CODE/src:/work1/dannegrut/harry/experiments/ball_span_v1_20260930/plot_deps:${PYTHONPATH:-}"
export MPLCONFIGDIR="$BALL_ROOT/mpl_cache"
cd "$BALL_CODE"
python3.12 tests/bouncing_ball/test_transformer_contact.py
python3.12 -m nedm.bouncing_ball.transformer_contact_checks --root "$BALL_ROOT" --data /work1/dannegrut/harry/experiments/ball_span_v1_20260930/final_data
python3.12 -m nedm.bouncing_ball.transformer_contact_report --root "$BALL_ROOT"
