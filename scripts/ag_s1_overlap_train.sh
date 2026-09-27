#!/bin/bash
# arena_gator_20260925 soil stage 1: run a training job list (scripts/ag_train_gpus.sbatch, one lane per GPU) as an
# 'srun --overlap' step inside one of OUR running soil allocations (the MI250X soil job on mi2508x) when no GPU node is
# free for a training job. The soil workers keep running; each training lane shares one GCD with one soil worker.
# No new job, no queue slot, no extra billing. Run ON THE LOGIN NODE (the srun client stays there):
#   bash ag_s1_overlap_train.sh <raw job id of the soil task> <job list tsv> <cpus for the step>
set -uo pipefail
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
JID=$1; JOBS=$2; CPUS=${3:-32}
test -f "$JOBS" || { echo "no job list $JOBS"; exit 2; }
squeue -h -j "$JID" -t R -o "%i %P %N" | grep -q . || { echo "job $JID not running"; exit 2; }
NAME=$(basename "$JOBS" .tsv)
LOG=$G3/e5/logs/${NAME}_overlap_${JID}.out
setsid nohup env -u NEDM_VEHICLE srun --overlap --jobid "$JID" -N1 -n1 -c "$CPUS" --export=ALL,AG_JOBS=$JOBS \
  bash $G3/tools/s1/scripts/ag_train_gpus.sbatch > "$LOG" 2>&1 < /dev/null &
echo "started overlap training step in $JID ($(squeue -h -j $JID -o %N)), job list $JOBS, log $LOG"
printf '%s\t%s\tstep\t%s.<next step> (srun --overlap inside soil job %s on %s, S1 training %s, %s cpus; no new job, no extra billing)\t%s:%s\te5/train/soil_s1\ttools/s1/scripts/ag_train_gpus.sbatch:%s\t-\n' \
  "$(date +%F_%T)" "$(squeue -h -j $JID -o %P)" "$JID" "$JID" "$(squeue -h -j $JID -o %N)" "$NAME" "$CPUS" "$JOBS" "$(sha256sum $JOBS | cut -c1-16)" \
  "$(sha256sum $G3/tools/s1/scripts/ag_train_gpus.sbatch | cut -c1-16)" >> $G3/e3/submissions.tsv
