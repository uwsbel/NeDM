#!/bin/bash
# Rigid EVALUATION shard pool inside a running soil allocation (arena_gator_20260925, module E6b). Run from the login node:
#   setsid nohup srun --jobid=<soil job id> --overlap -N1 -n1 -c <workers> \
#     bash $G3/tools/e6b/ag_eval_pool.sh <tasklist.txt> <out dir> <workers> <deadline epoch> > <log> 2>&1 &
# Environment = gen_array_g.sbatch's rigid recipe (plain Chrono build, lavapipe, one thread per collector), exactly as
# ag_rigid_pool.sh; collector, runner, source root and fingerprint are fixed in ag_eval_pool.py (r2 tree).
set -eo pipefail
unset NEDM_VEHICLE
source /work1/dannegrut/harry/nrd/env.sh
nrd_pychrono
nrd_use_lavapipe
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
export AG_EVAL_TASKLIST=$1 AG_POOL_OUT=$2 AG_POOL_WORKERS=$3 AG_POOL_DEADLINE=$4 AG_POOL_CONCURRENT=${5:-1}
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
T=$G3/tools/e6b
echo "eval pool: host=$(hostname) job=${SLURM_JOB_ID}.${SLURM_STEP_ID:-?} workers=$3 tasklist=$1 out=$2"
cat "$1"
sha256sum $(grep -v '^#' "$1") $T/ag_eval_pool.py $G3/r2/source/scripts/ag_rigid_runner.py $G3/r2/source/scripts/ag_gen_collect_ext.py \
  $G3/r2/source/scripts/gen_collect_ext.py $G3/r2/source/scripts/ag_vehicle.py $G3/r2/source/scripts/gen_arenas.json \
  $G3/runtime/gator_runtime_fingerprint.json
exec "$NRD_PYTHON" -P -u $T/ag_eval_pool.py
