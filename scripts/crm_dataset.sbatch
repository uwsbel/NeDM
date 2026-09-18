#!/bin/bash
#SBATCH -A dannegrut
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
# submit with: -p mi2101x -c 16 --export=ALL,CRM_RUNS=<collect dir>/runs,CRM_DS=<out npz>
set -eo pipefail
source /work1/dannegrut/harry/nrd/env.sh
nrd_pychrono
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CRM=/work1/dannegrut/harry/experiments/crm_f104_20260916
cd $CRM/source/scripts
# corridors are sampled from the ONE overhead reference depth image rendered with OptiX on luffy ($CRM/static_map_v1)
"$NRD_PYTHON" -P -u f104_n2_dataset.py --root $CRM --runs "$CRM_RUNS/*_route_*:designed" "$CRM_RUNS/*_op_*:on_policy" \
  --out "$CRM_DS" --workers ${SLURM_CPUS_PER_TASK:-16}
