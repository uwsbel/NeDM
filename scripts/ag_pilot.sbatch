#!/bin/bash
#SBATCH -A dannegrut
#SBATCH -N 1
#SBATCH -n 1
# Gator cluster pilot (arena_gator_20260925, module E3b1): one node runs BOTH
#   soil : crm_worker.py (unchanged), one process per GPU, CRM_COLLECTOR = ag_crm_collect.py (the Gator dispatcher;
#          its name contains crm_collect, so --crm-config is forwarded), environment = ag_soil.sbatch's
#   rigid: ag_rigid_runner.py (gen_runner_g.py copy that keeps rich telemetry) on AG_RIGID_WORKERS CPU cores, rows of
#          shard = SLURM_ARRAY_TASK_ID, GEN_COLLECTOR = ag_gen_collect_ext.py, environment = gen_array_g.sbatch's
# Both parts get every input path explicitly; NEDM_VEHICLE is unset (the rows carry --vehicle).
#   sbatch -p mi3501x -c 24 -t 04:00:00 --array=0-1 -J ag_pilot_gator -o $G3/pilot_gator/logs/%x_%A_%a.out \
#     --export=ALL,CRM_TASKS=...,CRM_OUT=...,CRM_COLLECTOR=...,CRM_CONFIG=configs/crm_main.json,CRM_BUDGET_S=...,GEN_TASKS=...,GEN_OUT=...,GEN_COLLECTOR=...,AG_RIGID_WORKERS=16 \
#     $G3/source/scripts/ag_pilot.sbatch
set -uo pipefail
unset NEDM_VEHICLE
: "${CRM_COLLECTOR:?}"; : "${CRM_CONFIG:?}"; : "${CRM_TASKS:?}"; : "${CRM_OUT:?}"
: "${GEN_TASKS:?}"; : "${GEN_OUT:?}"; : "${GEN_COLLECTOR:?}"
case "$CRM_COLLECTOR" in *crm_collect*) ;; *) echo "CRM_COLLECTOR name must contain crm_collect"; exit 2;; esac
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
TAG=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$CRM_OUT/logs" "$GEN_OUT/logs"
echo "host=$(hostname) job=${SLURM_JOB_ID} array=$TAG part=${SLURM_JOB_PARTITION} cpus=${SLURM_CPUS_PER_TASK:-?}"
sha256sum "$CRM_COLLECTOR" "$GEN_COLLECTOR" "$G3/source/scripts/ag_vehicle.py" "$G3/source/scripts/ag_gator_belly.json" \
  "$G3/source/scripts/crm_worker.py" "$G3/source/scripts/ag_rigid_runner.py" "$G3/$CRM_CONFIG" "$CRM_TASKS" "$GEN_TASKS" \
  "$G3/runtime/gator_runtime_fingerprint.json"

(   # ---- soil (GPU)
  source /work1/dannegrut/harry/nrd/env.sh
  export CHRONO_BUILD=$NRD_ROOT/chrono-build-fsi
  nrd_pychrono
  export CRM_ROOT=$G3 CRM_CHRONO_DATA=$CHRONO_BUILD/data OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CRM_OMP=4
  export CRM_GPUS=${CRM_GPUS_OVERRIDE:-$(rocminfo | grep -c -E "^\s+Name:\s+gfx")}
  test -f "$CRM_ROOT/$CRM_CONFIG" || { echo "missing config $CRM_ROOT/$CRM_CONFIG"; exit 2; }
  echo "soil: host=$(hostname) gpus=$CRM_GPUS tasks=$CRM_TASKS out=$CRM_OUT collector=$CRM_COLLECTOR config=$CRM_CONFIG"
  exec "$NRD_PYTHON" -P -u "$CRM_ROOT/source/scripts/crm_worker.py"
) > "$CRM_OUT/logs/soil_$TAG.out" 2>&1 &
SOIL=$!

(   # ---- rigid (CPU)
  source /work1/dannegrut/harry/nrd/env.sh
  nrd_pychrono
  nrd_use_lavapipe
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
  export FDM_RUNTIME_FINGERPRINT=$G3/runtime/gator_runtime_fingerprint.json
  export GEN_ROOT=$G3 GEN_CHRONO_DATA=/work1/dannegrut/harry/nrd/chrono-build/data
  export GEN_RUNNER=$G3/source/scripts/ag_rigid_runner.py AG_KEEP_RICH=1
  export PYTHONPATH="$G3/source/scripts:$G3/source/src:${PYTHONPATH:-}"
  export GEN_WORKERS=${AG_RIGID_WORKERS:-16}
  echo "rigid: host=$(hostname) workers=$GEN_WORKERS tasks=$GEN_TASKS out=$GEN_OUT collector=$GEN_COLLECTOR"
  exec "$NRD_PYTHON" -P -u "$GEN_RUNNER"
) > "$GEN_OUT/logs/rigid_$TAG.out" 2>&1 &
RIGID=$!

wait $RIGID; echo "rigid part exit $? at $(date +%T)"
wait $SOIL; echo "soil part exit $? at $(date +%T)"
