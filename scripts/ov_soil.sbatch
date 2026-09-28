#!/bin/bash
#SBATCH -A dannegrut
#SBATCH -N 1
#SBATCH -n 1
# Soil (CRM) worker job for offroad_vehicles_20260927 (module M3): a copy of ag_soil.sbatch with CRM_ROOT = the G4
# experiment root (source tree G4/source, config G4/configs). One worker process per GPU (crm_worker.py, unchanged).
# CRM_COLLECTOR and CRM_CONFIG must be passed explicitly (no defaults): the worker forwards --crm-config only when the
# collector's path contains "crm_collect", and a missing config would silently run the collector's 0.5 ms default.
# Rows may name their own soil config ("config": "configs/crm_m113.json" for the 0.5 ms M113 rows; crm_worker.py:97).
# Extra guards against the Gator study's version:
#   - NEDM_VEHICLE must be unset (every row names its vehicle; submit with env -u NEDM_VEHICLE);
#   - the frozen dispatcher files must match G4/source/DISPATCHER.sha256 (written by ov_stage.sh), else the job stops;
#   - CRM_EPISODE_TIMEOUT_S (the worker's per-episode kill time) must be exported by the launcher.
# submit with scripts/ov_soil_launch.sh (it sets every variable below):
#   sbatch -p <partition> -c <all CPUs of that node type> -t <limit> --array=0-(n-1) \
#     --export=ALL,CRM_TASKS=...,CRM_OUT=...,CRM_COLLECTOR=$G4/source/scripts/ov_crm_collect.py,CRM_CONFIG=configs/crm_main.json,CRM_BUDGET_S=...,CRM_EPISODE_TIMEOUT_S=... \
#     $G4/source/scripts/ov_soil.sbatch
set -uo pipefail
G4=/work1/dannegrut/harry/experiments/offroad_vehicles_20260927
: "${CRM_COLLECTOR:?CRM_COLLECTOR must be exported explicitly}"
: "${CRM_CONFIG:?CRM_CONFIG must be exported explicitly}"
: "${CRM_TASKS:?}"; : "${CRM_OUT:?}"; : "${CRM_EPISODE_TIMEOUT_S:?CRM_EPISODE_TIMEOUT_S must be exported by ov_soil_launch.sh}"
case "$CRM_COLLECTOR" in *crm_collect*) ;; *) echo "CRM_COLLECTOR name must contain crm_collect (else no --crm-config)"; exit 2;; esac
if [[ -n "${NEDM_VEHICLE+x}" ]]; then echo "NEDM_VEHICLE is set ($NEDM_VEHICLE): refused (submit with env -u NEDM_VEHICLE)"; exit 2; fi
source /work1/dannegrut/harry/nrd/env.sh
export CHRONO_BUILD=$NRD_ROOT/chrono-build-fsi   # FSI/HIP build (superset of the rigid build)
nrd_pychrono
export CRM_ROOT=$G4
export CRM_CHRONO_DATA=$CHRONO_BUILD/data
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CRM_GPUS=${CRM_GPUS_OVERRIDE:-$(rocminfo | grep -c -E "^\s+Name:\s+gfx")}
export CRM_OMP=4
test -f "$CRM_ROOT/$CRM_CONFIG" || { echo "missing config $CRM_ROOT/$CRM_CONFIG"; exit 2; }
test -f "$CRM_ROOT/source/DISPATCHER.sha256" || { echo "missing $CRM_ROOT/source/DISPATCHER.sha256 (run ov_stage.sh)"; exit 2; }
( cd "$CRM_ROOT/source" && sha256sum --quiet -c DISPATCHER.sha256 ) || { echo "frozen dispatcher files differ from DISPATCHER.sha256: refused"; exit 2; }
echo "host=$(hostname) job=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} part=${SLURM_JOB_PARTITION} gpus=$CRM_GPUS"
echo "tasks=$CRM_TASKS out=$CRM_OUT config=$CRM_CONFIG collector=$CRM_COLLECTOR episode_timeout_s=$CRM_EPISODE_TIMEOUT_S budget_s=${CRM_BUDGET_S:-}"
sha256sum "$CRM_COLLECTOR" "$CRM_ROOT/source/scripts/crm_worker.py" "$CRM_ROOT/$CRM_CONFIG" "$CRM_TASKS" "$CRM_ROOT/source/DISPATCHER.sha256"
exec "$NRD_PYTHON" -P -u "$CRM_ROOT/source/scripts/crm_worker.py"
