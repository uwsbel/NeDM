#!/bin/bash
# Chrono cost profile: where a CRM episode's wall time goes (SPH vs multibody vs coupling
# vs the Python loop), rigid ground for contrast, the surrogate on the same GPU, and the
# SPH kernel split under rocprofv3. One MI210, runs sequential so nothing contends.
# COST: ~20-30 min x 0.4 = ~0.15-0.2 node-hours.
#SBATCH -J kyle-costprof
#SBATCH -p mi2104x
#SBATCH -N 1
#SBATCH -t 00:40:00
#SBATCH -o /work1/dannegrut/kyle/logs/%x-%j.out
#SBATCH -e /work1/dannegrut/kyle/logs/%x-%j.err
set -uo pipefail
K=/work1/dannegrut/kyle
module purge
module load gnu12 rocm
source $K/nedm/env-scoring.sh
export PYTHONPATH=$K/qrun:$K/qrun/src:$PYTHONPATH
export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8
cd $K/qrun/quadruped
O=$K/qrun/costprof; mkdir -p $O
URDF=$K/qrun/assets/data/robot/go2_irrvis/urdf/go2_description.urdf
POL=$K/qrun/policy/policy.pt
MOD=$K/qrun/nnrom_v2h_s8_ms100_s0/best.pt
echo "host $(hostname) job $SLURM_JOB_ID code $(cat $K/qrun/.source_commit)+cost_profile"
python -u diagnostics/cost_profile.py --policy $POL --urdf $URDF --model $MOD \
  --kinds crm,rigid --cases straight,turn --reps 2 --out $O/profile.json
echo "PLAIN_DONE rc=$?"
# Kernel split inside the SPH step. Profiler overhead inflates wall time, so only the
# kernel fractions from this run are used, never its timings.
rocprofv3 --kernel-trace --stats --output-format csv -d $O/rocprof -o crm -- \
  python -u diagnostics/cost_profile.py --policy $POL --urdf $URDF \
  --kinds crm --cases straight --reps 1 --out $O/profile_rocprof.json
echo "ROCPROF_DONE rc=$?"
find $O/rocprof -name "*stats*" | head
