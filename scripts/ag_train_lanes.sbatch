#!/bin/bash
# arena_gator_20260925 E5a: ci_train.py runs on one MI350X node (mi3501x, 4 h cap) in LANES. The lines of one lane run
# one after the other; the lanes run at the same time on the one GPU. Two lanes is the fast setting: in job 436135
# two ci_train processes on one MI350X ran at ~0.03 s per step each, three to five at 0.16-0.22 s per step each.
# Job list (AG_JOBS, tab-separated, '#' lines ignored):
#   lane <TAB> tag <TAB> output folder (relative to G3) <TAB> ci_train arguments (without --out/--tag)
# Each run writes <out>/<tag>*.pt|json|_logits.npz and <G3>/e5/logs/<tag>_<job>.log (last line 'exit: <code>'); a
# failed run does not stop its lane. A line waits until the manifest of each of its --ds files exists (ag_subset.py
# writes it after the npz), so lines whose data is still being built can be queued behind lines that are ready. Same environment and trainer as scripts/ag_train.sbatch (G3/source).
#   sbatch -J ag_train_x --export=ALL,AG_JOBS=$G3/e5/jobs/<list>.tsv $G3/tools/e5a/scripts/ag_train_lanes.sbatch
#SBATCH -A dannegrut
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p mi3501x
#SBATCH -c 24
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
NL=$(echo "$LANES" | wc -w); THR=$(( 24 / (NL > 0 ? NL : 1) )); [ $THR -lt 1 ] && THR=1
echo "host=$(hostname) job=$SLURM_JOB_ID runs=${#LINES[@]} lanes=$NL omp=$THR jobs=$AG_JOBS sha256=$(sha256sum $AG_JOBS | cut -c1-16)"
echo "ci_train.py $(sha256sum scripts/ci_train.py | cut -c1-16)  ga_train.py $(sha256sum scripts/ga_train.py | cut -c1-16)  python $(which $PY)"
[ -z "${AG_NO_ENV:-}" ] && $PY -c "import torch; print('torch', torch.__version__, 'device', torch.cuda.get_device_name(0))"
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
    echo "$(date +%T) lane $lane start $TAG -> $G3/$OUT : $ARGS"
    OMP_NUM_THREADS=$THR $PY -u scripts/ci_train.py $ARGS --out $G3/$OUT --tag $TAG > $LOG 2>&1; RC=$?
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
