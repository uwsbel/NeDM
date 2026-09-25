#!/bin/bash
# E3b1: HMMWV re-drive of the Gator pilot's rigid routes inside a running pilot allocation (srun --overlap), so the
# HMMWV and Gator rows of a group run on the same node (rigid Chrono is deterministic per node only) and the HMMWV
# chassis contact forces are recorded too.  Environment = gen_array_g.sbatch (HMMWV runtime fingerprint, unchanged
# gen_collect_ext.py), runner = ag_rigid_runner.py (keeps rich telemetry).
#   srun --jobid=<pilot job id of array task k> --overlap -N1 -n1 -c 16 bash ag_pilot_hmmwv_rigid.sh <shard k>
set -eo pipefail
unset NEDM_VEHICLE
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
source /work1/dannegrut/harry/nrd/env.sh
nrd_pychrono
nrd_use_lavapipe
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
export FDM_RUNTIME_FINGERPRINT=/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909/pilot_runtime_412394.json
export GEN_ROOT=$G3 GEN_CHRONO_DATA=/work1/dannegrut/harry/nrd/chrono-build/data AG_KEEP_RICH=1
export GEN_COLLECTOR=$G3/source/scripts/gen_collect_ext.py
export GEN_TASKS=$G3/tasks/pilot_hmmwv_rigid.json GEN_OUT=$G3/pilot_gator/rigid_hmmwv
export PYTHONPATH="$G3/source/scripts:$G3/source/src:${PYTHONPATH:-}"
export GEN_WORKERS=${AG_RIGID_WORKERS:-16}
export SLURM_ARRAY_TASK_ID=$1
echo "hmmwv rigid: host=$(hostname) shard=$1 workers=$GEN_WORKERS collector=$GEN_COLLECTOR $(sha256sum $GEN_COLLECTOR | cut -c1-16)"
exec "$NRD_PYTHON" -P -u $G3/source/scripts/ag_rigid_runner.py
