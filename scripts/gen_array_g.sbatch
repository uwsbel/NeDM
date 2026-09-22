#!/bin/bash
#SBATCH -A dannegrut
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
# Copy of gen_array.sbatch for the generalist experiment root: same env recipe, same rigid runtime fingerprint and
# chrono-build data path; GEN_ROOT / GEN_COLLECTOR / GEN_TASKS / GEN_OUT / GEN_WORKERS come from --export.
# Always pass GEN_ROOT explicitly (with --export=ALL a GEN_ROOT left in the submitting shell would be inherited):
#   sbatch --parsable -p mi2104x -c 128 -t 03:00:00 --array=0-5 -J rigid_g \
#     --export=ALL,GEN_ROOT=$G,GEN_TASKS=$G/tasks_rigid.json,GEN_OUT=$G/rigid_v1 -o $G/rigid_v1/logs/%x_%A_%a.out $G/source/scripts/gen_array_g.sbatch
set -eo pipefail
source /work1/dannegrut/harry/nrd/env.sh
nrd_pychrono
nrd_use_lavapipe
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
campaign=/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909
export FDM_RUNTIME_FINGERPRINT=$campaign/pilot_runtime_412394.json
export GEN_ROOT=${GEN_ROOT:-/work1/dannegrut/harry/experiments/generalist_20260921}
export GEN_COLLECTOR=${GEN_COLLECTOR:-$GEN_ROOT/source/scripts/gen_collect_ext.py}
export GEN_CHRONO_DATA=${GEN_CHRONO_DATA:-/work1/dannegrut/harry/nrd/chrono-build/data}
export GEN_RUNNER=${GEN_RUNNER:-$GEN_ROOT/source/scripts/gen_runner_g.py}
export PYTHONPATH="$GEN_ROOT/source/scripts:$GEN_ROOT/source/src:${PYTHONPATH:-}"
export GEN_WORKERS=${GEN_WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-16} - 2 ))}
"$NRD_PYTHON" -P -u "$GEN_RUNNER"
