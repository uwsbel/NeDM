#!/bin/bash
# arena_gator_20260925 soil track step 2: start soil_v3 worker steps (scripts/ag_s2_soil_step.sh) inside running soil
# allocations of this effort (srun --overlap, step name ag_s2_soil; no new job, no queue slot, no extra billing).
# Run ON THE CLUSTER LOGIN NODE (the srun clients stay there):
#   bash ag_s2_soil_overlap.sh <tasks.json> <cpus per step> <raw job id> [<raw job id> ...]
# For each job: skipped unless RUNNING and without an ag_s2_soil step; budget = job end - 30 min - now (no claims after it);
# every step is recorded in $G3/e3/submissions.tsv.
set -uo pipefail
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
TASKS=$1; CPUS=$2; shift 2
OUT=$G3/soil_v1
STEP=$G3/tools/s2/scripts/ag_s2_soil_step.sh
test -f "$TASKS" && test -f "$STEP" || { echo "missing input"; exit 2; }
mkdir -p $G3/tools/s2/logs
tsha=$(sha256sum "$TASKS" | cut -c1-16); ssha=$(sha256sum "$STEP" | cut -c1-16)
for jid in "$@"; do
  aid=$(squeue -h -j "$jid" -o "%A %i" 2>/dev/null | awk -v j="$jid" '$1 == j {print $2}' | head -1)
  [[ -n "$aid" ]] || aid=$jid
  read -r state end node part <<< "$(squeue -h -j "$aid" -o "%T %e %N %P" 2>/dev/null | head -1)"
  if [[ "$state" != RUNNING ]]; then echo "$jid ($aid): not running, skipped"; continue; fi
  if squeue -h -s -u "$USER" -o "%i %j" | grep -qE "^${aid}\.[0-9]+ ag_s2_soil$"; then echo "$jid ($aid): already has an ag_s2_soil step, skipped"; continue; fi
  budget=$(( $(date -d "$end" +%s) - 1800 - $(date +%s) ))
  (( budget > 1200 )) || { echo "$jid ($aid): less than 50 min left, skipped"; continue; }
  log=$G3/tools/s2/logs/soilstep_${jid}_$(date +%H%M%S).out
  setsid nohup env -u NEDM_VEHICLE srun --jobid="$jid" --overlap -J ag_s2_soil -N1 -n1 -c "$CPUS" bash "$STEP" "$TASKS" "$OUT" "$budget" \
    > "$log" 2>&1 < /dev/null &
  disown
  echo "$jid ($aid, $part $node): soil_v3 step started, $CPUS cpus, budget ${budget} s (no claims after $(date -d @$(( $(date +%s) + budget )) +%T)), log $log"
  printf '%s\t%s\tstep\t%s\t%s\t%s\t%s\t%s\n' "$(date +%F_%T)" "$part" \
    "$aid.<next step> (srun --overlap -J ag_s2_soil inside soil_v2 job $aid on $node: soil_v3 workers, one per GPU next to the soil_v2 worker; S2 soil evaluation; no new job, no extra billing)" \
    "$TASKS:$tsha" "$OUT" "$STEP:$ssha" "budget ${budget}s" >> "$G3/e3/submissions.tsv"
done
