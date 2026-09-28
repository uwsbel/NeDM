#!/bin/bash
# v2 CORPUS: 1200 episodes = 24 shards x 50, packed 4 per mi2104x node (one per MI210).
#
# WHY v2 EXISTS. v1 captured every row one physics step AFTER the policy acted, so at a
# control row the recorded state already carried the PD kick from the action chosen there
# and the policy could not be rolled on it (obs_truth.py; commit 07f440a4). v2 captures
# before the step. Shards 0-15 reuse v1 seeds, so v1 and v2 share their episode draws
# and differ only in capture timing; shards 16-23 are new episodes.
#
# WHY mi2104x. mi2101x was fully allocated. 4 GPUs at 0.4x is the same 10 GPU-h per
# charged node-hour, provided all four GPUs are used for the whole job -- hence exactly
# 4 equal shards per node. v1 measured ~1.75 h per 50-episode shard.
# COST: 6 nodes x ~2 h x 0.4 = ~4.8 node-hours (1500 total, ~487 used). -t caps a hang
# at 6 x 3.5 x 0.4 = 8.4.
#SBATCH -J kyle-go2-v2
#SBATCH -p mi2104x
#SBATCH -N 1
#SBATCH -t 03:30:00
#SBATCH --array=0-5%6
#SBATCH -o /work1/dannegrut/kyle/logs/%x-%A_%a.out
#SBATCH -e /work1/dannegrut/kyle/logs/%x-%A_%a.err

set -uo pipefail
K=/work1/dannegrut/kyle
module purge
module load gnu12 rocm
source $K/nedm/env-scoring.sh
export PYTHONPATH=$K/qrun:$K/qrun/src:$PYTHONPATH
cd $K/qrun/quadruped
A=${SLURM_ARRAY_TASK_ID}
echo "task $A  host $(hostname)  job ${SLURM_JOB_ID:-?}  code $(cat $K/qrun/.source_commit)"
python doctor.py --action collect || exit 1

for G in 0 1 2 3; do
  S=$(( A * 4 + G ))
  SEED=$(( 20260921 + S * 1000 ))
  ( export ROCR_VISIBLE_DEVICES=$G HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8
    python collect.py --corpus go2_crm_v2_s${S} --out $K/qrun/data \
      --policy $K/qrun/policy/policy.pt \
      --urdf $K/qrun/assets/data/robot/go2_irrvis/urdf/go2_description.urdf \
      --episodes 50 --terrain crm --pushes 2 --long-fraction 0.25 --seed $SEED \
      > $K/logs/go2-v2-s${S}.log 2>&1
    rc=$?
    echo "SHARD_${S} exit $rc" ) &
done
wait
echo "TASK_${A}_DONE"
