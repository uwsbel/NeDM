#!/bin/bash
#SBATCH -J ball-tcontact-test
#SBATCH -A dannegrut
#SBATCH -p devel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:30:00
set -euo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
source "$BALL_CODE/scripts/bouncing_ball/cluster/collection_env.sh"
"$NRD_PYTHON" -m nedm.bouncing_ball.campaign --campaign configs/bouncing_ball/transformer_contact_certification.json --output-dir "$BALL_ROOT/fresh_test" --workers 16
"$NRD_PYTHON" -m nedm.bouncing_ball.merge --primary /work1/dannegrut/harry/experiments/ball_span_v1_20260930/data --heldout "$BALL_ROOT/fresh_test" --output-dir "$BALL_ROOT/certification_data" --margin-s .02
