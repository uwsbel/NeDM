#!/bin/bash
# Fixed initial validation references; CPU inference only, no training/test.
#SBATCH -J fdm_initial_diag
#SBATCH -A dannegrut
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:20:00
#SBATCH -o /work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909/logs/%x_%j.out
set -eo pipefail
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
: "${DIAG_SNAPSHOT:?Set frozen diagnostic script snapshot}"
: "${DIAG_OUT:?Set fresh output}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8
campaign=/work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909
cd "$DIAG_SNAPSHOT"
python3.12 -u scripts/diagnose_traverse_fdm_diverse_initial_routes.py \
    --campaign "$campaign" --code-root "$campaign/snapshots/online_v7" \
    --source-manifest-sha256 a6cd57f77fe981a17909cc5163c3ee72f12bba82f32739ad192f651151389d16 \
    --full-route-support "$DIAG_SNAPSHOT/full_route_support.json" \
    --out "$DIAG_OUT" --threads 8
