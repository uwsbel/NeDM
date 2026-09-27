#!/bin/bash
# arena_gator_20260925 soil track step 2: soil workers on the soil_v3 task file as an 'srun --overlap' step inside one
# of OUR running soil_v2 allocations (each new worker shares one GPU with the soil_v2 worker already on it). Same worker
# (crm_worker.py, unchanged), same frozen dispatcher, config and output folder as ag_soil.sbatch; the worker's JOB tag
# is '<job id>s2' so its status files and shuffle order differ from the soil_v2 worker of the same job.
#   (started by scripts/ag_s2_soil_overlap.sh) bash ag_s2_soil_step.sh <tasks.json> <out dir> <budget s> [<n gpus>]
set -uo pipefail
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
export CRM_TASKS=$1 CRM_OUT=$2 CRM_BUDGET_S=$3
export CRM_COLLECTOR=$G3/source/scripts/ag_crm_collect.py CRM_CONFIG=configs/crm_main.json
unset NEDM_VEHICLE
source /work1/dannegrut/harry/nrd/env.sh
export CHRONO_BUILD=$NRD_ROOT/chrono-build-fsi
nrd_pychrono
export CRM_ROOT=$G3
export CRM_CHRONO_DATA=$CHRONO_BUILD/data
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CRM_GPUS=${4:-$(rocminfo | grep -c -E "^\s+Name:\s+gfx")}
export CRM_OMP=4
test -f "$CRM_ROOT/$CRM_CONFIG" || { echo "missing config"; exit 2; }
echo "S2 overlap soil step host=$(hostname) job=${SLURM_JOB_ID} step=${SLURM_STEP_ID:-} gpus=$CRM_GPUS budget=$CRM_BUDGET_S $(date +%T)"
echo "tasks=$CRM_TASKS out=$CRM_OUT config=$CRM_CONFIG collector=$CRM_COLLECTOR"
sha256sum "$CRM_COLLECTOR" "$CRM_ROOT/source/scripts/crm_worker.py" "$CRM_ROOT/$CRM_CONFIG" "$CRM_TASKS"
export SLURM_JOB_ID=${SLURM_JOB_ID}s2
exec "$NRD_PYTHON" -P -u "$CRM_ROOT/source/scripts/crm_worker.py"
