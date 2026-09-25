#!/bin/bash
# Rigid shard pool inside a running soil allocation (arena_gator_20260925, module E3b2). Run from the login node:
#   setsid nohup srun --jobid=<soil job id> --overlap -N1 -n1 -c <workers> \
#     bash $G3/source/scripts/ag_rigid_pool.sh <tasks.json> <out dir> <shards spec> <workers> <deadline epoch> \
#     > $G3/rigid_v1/pool/logs/step_<job>.out 2>&1 &
# Environment = gen_array_g.sbatch's rigid recipe (plain Chrono build, lavapipe, one thread per collector); the collector,
# runner, source root and runtime fingerprint of each shard are set by ag_rigid_pool.py from the shard number.
set -eo pipefail
unset NEDM_VEHICLE
source /work1/dannegrut/harry/nrd/env.sh
nrd_pychrono
nrd_use_lavapipe
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
export AG_POOL_TASKS=$1 AG_POOL_OUT=$2 AG_POOL_SHARDS=$3 AG_POOL_WORKERS=$4 AG_POOL_DEADLINE=$5
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
echo "rigid pool: host=$(hostname) job=${SLURM_JOB_ID}.${SLURM_STEP_ID:-?} workers=$4 shards=$3 tasks=$1 out=$2"
sha256sum "$1" $G3/source/scripts/ag_rigid_pool.py $G3/source/scripts/ag_rigid_runner.py $G3/source/scripts/ag_gen_collect_ext.py \
  $G3/source/scripts/gen_collect_ext.py $G3/r2/source/scripts/gen_collect_ext.py $G3/r2/source/scripts/gen_arenas.json \
  $G3/runtime/gator_runtime_fingerprint.json
exec "$NRD_PYTHON" -P -u $G3/source/scripts/ag_rigid_pool.py
