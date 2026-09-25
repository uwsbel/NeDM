#!/bin/bash
#SBATCH -A dannegrut
#SBATCH -N 1
#SBATCH -n 1
# Soil (CRM) worker job for arena_gator_20260925: a copy of crm_collect.sbatch with CRM_ROOT = the G3 experiment root
# (source tree G3/source, cases G3/cases, config G3/configs). One worker process per GPU (crm_worker.py, unchanged).
# CRM_COLLECTOR and CRM_CONFIG must be passed explicitly (no defaults): the worker forwards --crm-config only when the
# collector's path contains "crm_collect", and a missing config would silently run the collector's 0.5 ms default.
# submit with (see scripts/ag_soil_launch.sh):
#   sbatch -p <partition> -c <all CPUs of that node type> -t <limit> --array=0-(n-1) \
#     --export=ALL,CRM_TASKS=...,CRM_OUT=...,CRM_COLLECTOR=$G3/source/scripts/crm_collect.py,CRM_CONFIG=configs/crm_main.json,CRM_BUDGET_S=... \
#     $G3/source/scripts/ag_soil.sbatch
set -uo pipefail
: "${CRM_COLLECTOR:?CRM_COLLECTOR must be exported explicitly}"
: "${CRM_CONFIG:?CRM_CONFIG must be exported explicitly}"
: "${CRM_TASKS:?}"; : "${CRM_OUT:?}"
case "$CRM_COLLECTOR" in *crm_collect*) ;; *) echo "CRM_COLLECTOR name must contain crm_collect (else no --crm-config)"; exit 2;; esac
source /work1/dannegrut/harry/nrd/env.sh
export CHRONO_BUILD=$NRD_ROOT/chrono-build-fsi   # FSI/HIP build (superset of the rigid build)
nrd_pychrono
export CRM_ROOT=/work1/dannegrut/harry/experiments/arena_gator_20260925
export CRM_CHRONO_DATA=$CHRONO_BUILD/data
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CRM_GPUS=${CRM_GPUS_OVERRIDE:-$(rocminfo | grep -c -E "^\s+Name:\s+gfx")}
export CRM_OMP=4
test -f "$CRM_ROOT/$CRM_CONFIG" || { echo "missing config $CRM_ROOT/$CRM_CONFIG"; exit 2; }
echo "host=$(hostname) job=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} part=${SLURM_JOB_PARTITION} gpus=$CRM_GPUS"
echo "tasks=$CRM_TASKS out=$CRM_OUT config=$CRM_CONFIG collector=$CRM_COLLECTOR"
sha256sum "$CRM_COLLECTOR" "$CRM_ROOT/source/scripts/crm_worker.py" "$CRM_ROOT/$CRM_CONFIG" "$CRM_TASKS"
exec "$NRD_PYTHON" -P -u "$CRM_ROOT/source/scripts/crm_worker.py"
