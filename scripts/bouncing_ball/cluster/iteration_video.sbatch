#!/bin/bash
#SBATCH -J ball-iter-video
#SBATCH -A dannegrut
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:15:00
set -euo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
source "$BALL_CODE/scripts/bouncing_ball/cluster/collection_env.sh"
if ! test -d "$BALL_ROOT/video_deps/imageio_ffmpeg"; then
    "$NRD_PYTHON" -m pip install --no-cache-dir --no-deps --target "$BALL_ROOT/video_deps" imageio-ffmpeg
fi
export PYTHONPATH="$BALL_ROOT/video_deps:$BALL_ROOT/plot_deps:$PYTHONPATH"
"$NRD_PYTHON" -m nedm.bouncing_ball.iteration_video --run "$BALL_RUN" --output-dir "$BALL_VIDEO" --workers 8 "$@"
