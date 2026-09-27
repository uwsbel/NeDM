#!/bin/bash
# arena_gator_20260925 soil stage 1 (S1): ci_train.py runs on one multi-GPU node, one LANE per GPU.
# Same job list format, environment and trainer as scripts/ag_train_lanes.sbatch (G3/source trainer, pytorch/2.10.0,
# venv nedm, python3.12, NEDM_VEHICLE unset); the difference: lane k runs on GPU (k-1) mod n_gpu only
# (HIP_VISIBLE_DEVICES), so e.g. 8 lanes on an mi2508x node (8 MI250X GCDs) do not share a device.
# (pytorch/2.10.0 fails on MI210 compute nodes, works on MI250X: probe job 436350.)
# Job list (AG_JOBS, tab-separated, '#' lines ignored):
#   lane <TAB> tag <TAB> output folder (relative to G3) <TAB> ci_train arguments (without --out/--tag)
# A line waits until the ag_subset.py manifest of each of its --ds files exists (AG_WAIT_S, default 2 h).
#   sbatch -p mi2508x -c 128 -t 03:00:00 -J ag_s1_train_x --export=ALL,AG_JOBS=$G3/e5/jobs/<list>.tsv $G3/tools/s1/scripts/ag_train_gpus.sbatch
#SBATCH -A dannegrut
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o /work1/dannegrut/harry/experiments/arena_gator_20260925/e5/logs/%x_%j.out
set -o pipefail
G3=${AG_G3:-/work1/dannegrut/harry/experiments/arena_gator_20260925}
: "${AG_JOBS:?export AG_JOBS=<job list tsv>}"
unset NEDM_VEHICLE
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID} MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${SLURM_JOB_ID}
cd $G3/source
export PYTHONPATH=$G3/source/src:$G3/source/scripts:${PYTHONPATH:-}
PY=${AG_PY:-python3.12}
NGPU=$(rocminfo | grep -c -E "^\s+Name:\s+gfx")
NCPU=${SLURM_CPUS_PER_TASK:-$(nproc)}
mapfile -t LINES < <(grep -v '^#' "$AG_JOBS" | grep -v '^[[:space:]]*$')
LANES=$(printf '%s\n' "${LINES[@]}" | cut -f1 | sort -un)
NL=$(echo "$LANES" | wc -w); THR=$(( NCPU / (NL > 0 ? NL : 1) )); [ $THR -gt 16 ] && THR=16; [ $THR -lt 1 ] && THR=1
echo "host=$(hostname) job=$SLURM_JOB_ID part=$SLURM_JOB_PARTITION gpus=$NGPU cpus=$NCPU runs=${#LINES[@]} lanes=$NL omp=$THR jobs=$AG_JOBS sha256=$(sha256sum $AG_JOBS | cut -c1-16)"
echo "ci_train.py $(sha256sum scripts/ci_train.py | cut -c1-16)  ga_train.py $(sha256sum scripts/ga_train.py | cut -c1-16)  python $(which $PY)"
$PY -c "import torch; print('torch', torch.__version__, 'devices', torch.cuda.device_count(), torch.cuda.get_device_name(0))"
run_lane() {
  local lane=$1 gpu=$2 L LN TAG OUT ARGS LOG RC
  for L in "${LINES[@]}"; do
    IFS=$'\t' read -r LN TAG OUT ARGS <<< "$L"
    [ "$LN" = "$lane" ] || continue
    mkdir -p $G3/$OUT
    LOG=$G3/e5/logs/${TAG}_${SLURM_JOB_ID}.log
    read -ra TOK <<< "$ARGS"; W0=$(date +%s)
    for ((i=0; i<${#TOK[@]}; i++)); do
      if [ "${TOK[$i]}" = "--ds" ]; then
        M="${TOK[$((i+1))]%.npz}.manifest.json"
        while [ ! -f "$M" ] && [ $(( $(date +%s) - W0 )) -lt ${AG_WAIT_S:-7200} ]; do sleep 20; done
        [ -f "$M" ] || echo "$(date +%T) lane $lane WARNING $TAG: no manifest $M after ${AG_WAIT_S:-7200} s"
      fi
    done
    [ $(( $(date +%s) - W0 )) -gt 5 ] && echo "$(date +%T) lane $lane $TAG waited $(( $(date +%s) - W0 )) s for its data"
    echo "$(date +%T) lane $lane gpu $gpu start $TAG -> $G3/$OUT : $ARGS"
    HIP_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$THR $PY -u scripts/ci_train.py $ARGS --out $G3/$OUT --tag $TAG > $LOG 2>&1; RC=$?
    echo "exit: $RC" >> $LOG
    echo "$(date +%T) lane $lane end $TAG exit: $RC"
  done
}
PIDS=()
for ln in $LANES; do run_lane $ln $(( (ln - 1) % NGPU )) & PIDS+=($!); done
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
