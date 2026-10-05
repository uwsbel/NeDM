#!/bin/bash
#SBATCH -J ball-report
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
export MPLCONFIGDIR="$BALL_ROOT/mpl_cache"
if ! test -f "$BALL_ROOT/plot_deps/.ready"; then
    python3.12 -m pip install --no-cache-dir --target "$BALL_ROOT/plot_deps" --no-deps matplotlib contourpy cycler fonttools kiwisolver packaging pillow pyparsing python-dateutil six
    touch "$BALL_ROOT/plot_deps/.ready"
fi
export PYTHONPATH="$BALL_ROOT/plot_deps:$PYTHONPATH"
python3.12 -m pip list --format=freeze --path "$BALL_ROOT/plot_deps" > "$BALL_RUN/plot_runtime.txt"
python3.12 -m nedm.bouncing_ball.report --data "${BALL_DATA:-$BALL_ROOT/data}" --run "$BALL_RUN" --reference "$BALL_ROOT/reference"
