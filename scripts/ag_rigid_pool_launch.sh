#!/bin/bash
# Start rigid pool steps (scripts/ag_rigid_pool.sh) inside running soil allocations (arena_gator_20260925, E3b2).
# Run ON THE CLUSTER LOGIN NODE:
#   ag_rigid_pool_launch.sh <workers per step> <raw job id> [<raw job id> ...]
# For each job: skips it unless it is RUNNING and has no pool step yet (step name "bash" other than the batch step);
# deadline = job end time - 30 min (no new shard is claimed after it); records the step in $G3/e3/submissions.tsv.
# Raw job ids (not array ids): squeue -u $USER -h -t R -o "%i %A %P %N %e". Workers: 12 on mi2101x (16 cores),
# 16 on mi3501x (24 cores), 100 on mi2104x (128 cores, 4 soil workers).
set -uo pipefail
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
TASKS=${AG_POOL_TASKS_FILE:-$G3/tasks/rigid_v2.json}
OUT=${AG_POOL_OUT_DIR:-$G3/rigid_v1}
SHARDS=${AG_POOL_SHARD_SPEC:-1000-1299,2000-2124}
W=$1; shift
mkdir -p "$OUT/pool/logs"
tsha=$(sha256sum "$TASKS" | cut -c1-16)
for jid in "$@"; do
  st=$(squeue -h -j "$jid" -o "%T %e %N %P" 2>/dev/null | head -1)
  read -r state end node part <<< "$st"
  if [[ "$state" != RUNNING ]]; then echo "$jid: not running ($st), skipped"; continue; fi
  if squeue -h -s -j "$jid" -o "%i %j" | grep -v batch | grep -q " bash"; then echo "$jid: already has a pool step, skipped"; continue; fi
  dl=$(( $(date -d "$end" +%s) - 1800 ))
  log=$OUT/pool/logs/step_${jid}_$(date +%H%M%S).out
  setsid nohup srun --jobid="$jid" --overlap -N1 -n1 -c "$W" bash "$G3/source/scripts/ag_rigid_pool.sh" "$TASKS" "$OUT" "$SHARDS" "$W" "$dl" \
    > "$log" 2>&1 < /dev/null &
  disown
  echo "$jid ($part $node): pool step started, $W workers, deadline $(date -d @$dl +%T), log $log"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%F_%T)" "$part" step \
    "$jid.<next step> (srun --overlap inside soil job $jid on $node, E3b2 rigid pool, $W workers, shards $SHARDS, deadline $(date -d @$dl +%T); no new job, no extra billing)" \
    "$TASKS:$tsha" "$OUT" "$G3/source/scripts/ag_rigid_pool.sh" "-" >> "$G3/e3/submissions.tsv"
done
