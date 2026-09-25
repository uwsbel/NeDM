#!/bin/bash
# arena_gator_20260925 E5: ci_train.py runs packed on one MI350X node (mi3501x, 4 h cap), all lines of one job list at
# once. Job list (AG_JOBS, tab-separated, '#' lines ignored): tag <TAB> output folder (relative to G3) <TAB> ci_train
# arguments (without --out/--tag). Each run writes <out>/<tag>*.pt|json|_logits.npz and <G3>/e5/logs/<tag>_<job>.log.
#   sbatch -J ag_train --export=ALL,AG_JOBS=$G3/e5/jobs/<list>.tsv $G3/source/scripts/ag_train.sbatch
#SBATCH -A dannegrut
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p mi3501x
#SBATCH -c 24
#SBATCH -t 03:50:00
#SBATCH -o /work1/dannegrut/harry/experiments/arena_gator_20260925/e5/logs/%x_%j.out
set -o pipefail
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
: "${AG_JOBS:?export AG_JOBS=<job list tsv>}"
unset NEDM_VEHICLE
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID} MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${SLURM_JOB_ID}
cd $G3/source
export PYTHONPATH=$G3/source/src:$G3/source/scripts:${PYTHONPATH:-}
mapfile -t LINES < <(grep -v '^#' "$AG_JOBS" | grep -v '^[[:space:]]*$')
N=${#LINES[@]}; THR=$(( 24 / (N > 0 ? N : 1) )); [ $THR -lt 1 ] && THR=1
echo "host=$(hostname) job=$SLURM_JOB_ID runs=$N omp=$THR jobs=$AG_JOBS sha256=$(sha256sum $AG_JOBS | cut -c1-16)"
echo "ci_train.py $(sha256sum scripts/ci_train.py | cut -c1-16)  ga_train.py $(sha256sum scripts/ga_train.py | cut -c1-16)  python $(which python3.12)"
python3.12 -c "import torch; print('torch', torch.__version__, 'device', torch.cuda.get_device_name(0))"
PIDS=(); TAGS=()
for L in "${LINES[@]}"; do
  IFS=$'\t' read -r TAG OUT ARGS <<< "$L"
  mkdir -p $G3/$OUT
  LOG=$G3/e5/logs/${TAG}_${SLURM_JOB_ID}.log
  echo "start $TAG -> $G3/$OUT : $ARGS"
  ( OMP_NUM_THREADS=$THR python3.12 -u scripts/ci_train.py $ARGS --out $G3/$OUT --tag $TAG > $LOG 2>&1; echo "exit: $?" >> $LOG ) &
  PIDS+=($!); TAGS+=($TAG)
done
RC=0
for i in "${!PIDS[@]}"; do
  wait ${PIDS[$i]}
  E=$(tail -1 $G3/e5/logs/${TAGS[$i]}_${SLURM_JOB_ID}.log)
  echo "${TAGS[$i]} $E"
  [ "$E" = "exit: 0" ] || RC=1
done
echo "all done rc=$RC"
exit $RC
