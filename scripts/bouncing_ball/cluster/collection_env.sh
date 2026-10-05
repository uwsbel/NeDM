#!/bin/bash
# The existing Chrono 10 binary bundle preserves the approved pilot runtime.
source /work1/dannegrut/harry/nrd/env.sh
BALL_CHRONO_ROOT=/work1/dannegrut/harry/nrd/chrono10_cpu
export LD_LIBRARY_PATH="$BALL_CHRONO_ROOT/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$BALL_CHRONO_ROOT/lib/python3.12/site-packages:$NRD_PYSTACK:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$BALL_CODE/src:$PYTHONPATH"
cd "$BALL_CODE"
