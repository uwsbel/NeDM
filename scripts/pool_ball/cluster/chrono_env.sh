#!/bin/bash
# Chrono 10 CPU bundle (same runtime as the bouncing-ball study) on AMD.
source /work1/dannegrut/harry/nrd/env.sh
POOL_CHRONO_ROOT=/work1/dannegrut/harry/nrd/chrono10_cpu
export LD_LIBRARY_PATH="$POOL_CHRONO_ROOT/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$POOL_CHRONO_ROOT/lib/python3.12/site-packages:$NRD_PYSTACK:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$POOL_CODE/src:$PYTHONPATH"
cd "$POOL_CODE"
