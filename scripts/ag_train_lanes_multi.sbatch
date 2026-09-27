#!/bin/bash
# arena_gator_20260925 E5a (resumed): scripts/ag_train_lanes.sbatch for nodes with several GPUs. Identical lane logic (lines of
# one lane run one after the other, lanes at the same time, each line waits for its --ds manifests, a failed run does not
# stop its lane), plus: lane k runs on GPU (k-1) mod NGPU (HIP_VISIBLE_DEVICES, NGPU = number of /dev/dri/renderD* devices, or AG_NGPU) and the node's
# cores are shared evenly between the lanes. Written because every MI350X/MI300X node was taken at 11:00 on 2026-09-25
# (another user's 7 + 11 pending 4 h jobs, our fairshare priority far below theirs). Same environment and trainer (G3/source).
#   sbatch -p <partition> -c <cores> -J ag_train_x --export=ALL,AG_JOBS=$G3/e5/jobs/<list>.tsv $G3/tools/e5a/scripts/ag_train_lanes_multi.sbatch
#SBATCH -A dannegrut
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p mi2508x
#SBATCH -c 128
#SBATCH -t 03:50:00
#SBATCH -o /work1/dannegrut/harry/experiments/arena_gator_20260925/e5/logs/%x_%j.out
set -o pipefail
G3=${AG_G3:-/work1/dannegrut/harry/experiments/arena_gator_20260925}
: "${AG_JOBS:?export AG_JOBS=<job list tsv>}"
unset NEDM_VEHICLE
if [ -z "${AG_NO_ENV:-}" ]; then
  source /etc/profile
  module load pytorch/2.10.0
  source /work1/dannegrut/harry/venvs/nedm/bin/activate
fi
export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID} MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${SLURM_JOB_ID}
cd $G3/source
export PYTHONPATH=$G3/source/src:$G3/source/scripts:${PYTHONPATH:-}
PY=${AG_PY:-python3.12}
mapfile -t LINES < <(grep -v '^#' "$AG_JOBS" | grep -v '^[[:space:]]*$')
LANES=$(printf '%s\n' "${LINES[@]}" | cut -f1 | sort -u)
NL=$(echo "$LANES" | wc -w); NC=${SLURM_CPUS_PER_TASK:-24}; THR=$(( NC / (NL > 0 ? NL : 1) )); [ $THR -gt 16 ] && THR=16; [ $THR -lt 1 ] && THR=1
NGPU=${AG_NGPU:-$(python3 -c "import os; print(len([d for d in os.listdir('/dev/dri') if d.startswith('renderD')]))" 2>/dev/null || echo 1)}; [ "$NGPU" -ge 1 ] 2>/dev/null || NGPU=1
echo "host=$(hostname) job=$SLURM_JOB_ID runs=${#LINES[@]} lanes=$NL ngpu=$NGPU omp=$THR jobs=$AG_JOBS sha256=$(sha256sum $AG_JOBS | cut -c1-16)"
echo "ci_train.py $(sha256sum scripts/ci_train.py | cut -c1-16)  ga_train.py $(sha256sum scripts/ga_train.py | cut -c1-16)  python $(which $PY)"
[ -z "${AG_NO_ENV:-}" ] && $PY -c "import torch; print('torch', torch.__version__, 'devices', torch.cuda.device_count(), torch.cuda.get_device_name(0))"
run_lane() {
  local lane=$1 L LN TAG OUT ARGS LOG RC
  for L in "${LINES[@]}"; do
    IFS=$'\t' read -r LN TAG OUT ARGS <<< "$L"
    [ "$LN" = "$lane" ] || continue
    mkdir -p $G3/$OUT
    LOG=$G3/e5/logs/${TAG}_${SLURM_JOB_ID}.log
    # wait until every --ds file of the line is complete: its ag_subset.py manifest (<stem>.manifest.json, written after
    # the npz and its sha256) exists; gives up after AG_WAIT_S seconds (default 2 h) and runs anyway (ci_train then fails)
    read -ra TOK <<< "$ARGS"; W0=$(date +%s)
    for ((i=0; i<${#TOK[@]}; i++)); do
      if [ "${TOK[$i]}" = "--ds" ]; then
        M="${TOK[$((i+1))]%.npz}.manifest.json"
        while [ ! -f "$M" ] && [ $(( $(date +%s) - W0 )) -lt ${AG_WAIT_S:-7200} ]; do sleep 20; done
        [ -f "$M" ] || echo "$(date +%T) lane $lane WARNING $TAG: no manifest $M after ${AG_WAIT_S:-7200} s"
      fi
    done
    [ $(( $(date +%s) - W0 )) -gt 5 ] && echo "$(date +%T) lane $lane $TAG waited $(( $(date +%s) - W0 )) s for its data"
    local DEV=$(( (lane - 1) % NGPU ))
    echo "$(date +%T) lane $lane start $TAG on GPU $DEV -> $G3/$OUT : $ARGS"
    HIP_VISIBLE_DEVICES=$DEV OMP_NUM_THREADS=$THR $PY -u scripts/ci_train.py $ARGS --out $G3/$OUT --tag $TAG > $LOG 2>&1; RC=$?
    echo "exit: $RC" >> $LOG
    echo "$(date +%T) lane $lane end $TAG exit: $RC"
  done
}
PIDS=()
for ln in $LANES; do run_lane $ln & PIDS+=($!); done
for p in "${PIDS[@]}"; do wait $p; done
RC=0
for L in "${LINES[@]}"; do
  IFS=$'\t' read -r LN TAG OUT ARGS <<< "$L"
  E=$(tail -1 $G3/e5/logs/${TAG}_${SLURM_JOB_ID}.log 2>/dev/null)
  echo "$TAG $E"
  [ "$E" = "exit: 0" ] || RC=1
done
echo "all done rc=$RC"
exit $RC
