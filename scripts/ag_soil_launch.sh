#!/bin/bash
# Sized soil (CRM) launch for arena_gator_20260925 (replaces crm_launch.sh: explicit collector/config, one sbatch per
# partition with the array size given here, every submission must return a job id, queue count checked first).
# Run ON THE CLUSTER LOGIN NODE:
#   ag_soil_launch.sh <tasks.json> <out dir> <collector path> <partition>:<cpus>:<hours>:<n tasks> [...]
# e.g. ag_soil_launch.sh $G3/tasks/soil_v1.json $G3/soil_v1 $G3/source/scripts/crm_collect.py mi2101x:16:12:20 mi3501x:24:4:4
# Records every submission in $G3/e3/submissions.tsv (time, partition, array, job id, tasks, out, collector sha256).
set -uo pipefail
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
TASKS=$1; OUT=$2; COLLECTOR=$3; shift 3
CFG=configs/crm_main.json
S=$G3/source/scripts/ag_soil.sbatch
CAP=${AG_QUEUE_CAP:-50}; RESERVE=${AG_QUEUE_RESERVE:-5}
test -f "$TASKS" && test -f "$COLLECTOR" && test -f "$G3/$CFG" && test -f "$S" || { echo "missing input"; exit 2; }
case "$COLLECTOR" in *crm_collect*) ;; *) echo "collector name must contain crm_collect"; exit 2;; esac
want=0; for spec in "$@"; do want=$(( want + ${spec##*:} )); done
have=$(squeue -h -r -u "$USER" | wc -l)
echo "queued/running array tasks now: $have; this launch adds $want; cap $CAP, keep $RESERVE free"
if (( have + want > CAP - RESERVE )); then echo "REFUSED: would exceed cap-reserve"; exit 3; fi
mkdir -p "$OUT/logs" "$G3/e3"
csha=$(sha256sum "$COLLECTOR" | cut -c1-16); tsha=$(sha256sum "$TASKS" | cut -c1-16)
rc=0
for spec in "$@"; do
  IFS=: read -r part cpus hrs n <<< "$spec"
  budget=$(( hrs * 3600 - 1000 ))
  jid=$(sbatch --parsable -p "$part" -c "$cpus" -t "$(printf '%02d:00:00' "$hrs")" --array=0-$(( n - 1 )) \
        -J "ag_soil_$(basename "$OUT")" -o "$OUT/logs/%x_%A_%a.out" \
        --export=ALL,CRM_TASKS=$TASKS,CRM_OUT=$OUT,CRM_CONFIG=$CFG,CRM_COLLECTOR=$COLLECTOR,CRM_BUDGET_S=$budget \
        "$S" 2>/tmp/ag_soil_launch_$$.err)
  if [[ "$jid" =~ ^[0-9]+$ ]]; then
    echo "$part array 0-$(( n - 1 )) -> job $jid"
    printf '%s\t%s\t0-%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%F_%T)" "$part" "$(( n - 1 ))" "$jid" "$TASKS:$tsha" "$OUT" "$COLLECTOR:$csha" "${hrs}h" >> "$G3/e3/submissions.tsv"
  else
    echo "SUBMIT FAILED on $part:"; grep -v '^sbatch: ' /tmp/ag_soil_launch_$$.err; tail -3 /tmp/ag_soil_launch_$$.err; rc=1
  fi
done
rm -f /tmp/ag_soil_launch_$$.err
exit $rc
