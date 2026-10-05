#!/bin/bash
#SBATCH -J ball-latent-report
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
export MPLCONFIGDIR="$BALL_ROOT/mpl_cache"
export PYTHONPATH="$BALL_CODE/src:/work1/dannegrut/harry/experiments/ball_span_v1_20260930/plot_deps:${PYTHONPATH:-}"
cd "$BALL_CODE"
python3.12 -m nedm.bouncing_ball.latent_report --root "$BALL_ROOT"
python3.12 -m nedm.bouncing_ball.latent_diagnostics --root "$BALL_ROOT" --data "$BALL_DATA" --output "$BALL_ROOT/diagnostics_phase_clips"
