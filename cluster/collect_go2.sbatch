#!/bin/bash
# Collect Go2 CRM episodes on the cluster, ONE SHARD PER ARRAY TASK.
#
#   sbatch -p mi2101x -t 03:00:00 --array=0-15%16 \
#          --export=ALL,NPF=50,NSHARD=16,SEED_OFFSET=10000000 collect_go2.sbatch
#
# WHY THE CLUSTER AND NOT THE DESKTOPS. This is the one job in the pipeline that is
# embarrassingly parallel and genuinely large: the existing corpus is 795 episodes /
# 1.55 h, and the author's question is whether 4 h or 10 h changes anything. A desktop
# collects one episode per slot at ~3 min for a 41 s WIDE episode; 24 nodes at
# concurrency 8 collect 192 at a time. The desktops are also the only boxes that can
# fine-tune, so spending them on collection stalls everything else.
#
# COST, measured from the scoring benchmark (515 episodes per charged node-hour at 16 s).
# WIDE episodes are 41.25 s, ~2.6x, so ~200 episodes per charged node-hour. 400 episodes
# is ~2 node-hours; 2,400 episodes (~10 h of data) is ~12. Against 1,138 remaining this
# is affordable, which is the answer to "why not just collect more".
#
#SBATCH -J kyle-crmcollect
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -o /work1/dannegrut/kyle/logs/%x-%A_%a.out
#SBATCH -e /work1/dannegrut/kyle/logs/%x-%A_%a.err

set -o pipefail
source /work1/dannegrut/kyle/env.sh
module purge
module load gnu12 rocm
source /work1/dannegrut/kyle/nedm/env-scoring.sh

NPF=${NPF:-50}                       # episodes per command family
NSHARD=${NSHARD:-16}
SEED_OFFSET=${SEED_OFFSET:-10000000} # new territory; existing corpora used 9.3-9.9M
CONC=${CONC:-8}                      # saturates one MI210, measured

export NEDM_WIDE=1                   # 41.25 s episodes, 200 m ground, stratified pushes
export NEDM_TILT_PITCH_DEG=${NEDM_TILT_PITCH_DEG:-1.5}
export NEDM_SHARD=$SLURM_ARRAY_TASK_ID
export NEDM_NSHARD=$NSHARD
export NEDM_REPO=$NEDM_REPO
export NEDM_PY=$(which python)
export NEDM_CHRONO_PYTHONPATH=$NEDM_CHRONO/bin
export NEDM_GO2_CKPT=$NEDM_CKPTS/go2_cts_150k.pt
export NEDM_DATASET_ROOT=$KWORK/nedm/collect/go2_crm_wide_${SEED_OFFSET}

echo "== host $(hostname)  shard $SLURM_ARRAY_TASK_ID of $NSHARD  npf $NPF  conc $CONC =="
echo "== root $NEDM_DATASET_ROOT =="
t0=$(date +%s)
cd "$NEDM_REPO"
python scripts/collection/drive_go2_collection.py crm "$NPF" "$CONC" "$SEED_OFFSET"
rc=$?
t1=$(date +%s); el=$((t1-t0))
n=$(ls -d $NEDM_DATASET_ROOT/crm_*_* 2>/dev/null | wc -l)
echo "== rc=$rc  elapsed ${el}s  episodes on disk ${n}  charged $(awk -v s=$el 'BEGIN{printf "%.4f", s/3600*0.1}') node-hours =="
