#!/bin/bash
# Start rigid EVALUATION pool steps (ag_eval_pool.sh) inside running soil allocations (arena_gator_20260925, E6b).
# Run ON THE CLUSTER LOGIN NODE:  ag_eval_pool_launch.sh <workers per step>[:<concurrent shards>] <raw job id> [...]
# Skips a job unless it is RUNNING and has no step named "bash" (a pool step) yet; deadline = job end - 30 min;
# records the step in $G3/e3/submissions.tsv. Workers: 12 on mi2101x (16 cores), 16 on mi3501x, 96 on mi2104x (128 cores,
# 4 soil workers; measured 09-25: no soil slowdown with 96 rigid workers on these nodes).
set -uo pipefail
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
T=$G3/tools/e6b
LIST=${AG_EVAL_TASKLIST_FILE:-$G3/rigid_eval/TASKLIST.txt}
OUT=${AG_EVAL_OUT_DIR:-$G3/rigid_eval}
W=${1%%:*}; K=1; [[ $1 == *:* ]] && K=${1##*:}; shift
mkdir -p "$OUT/pool/logs"
lsha=$(sha256sum "$LIST" | cut -c1-16)
for jid in "$@"; do
  # the array task id of this raw id (squeue -j <array master id> lists every task of the array and their steps)
  aid=$(squeue -h -j "$jid" -o "%A %i" 2>/dev/null | awk -v j="$jid" '$1 == j {print $2}' | head -1)
  [[ -n $aid ]] || { echo "$jid: not found, skipped"; continue; }
  st=$(squeue -h -j "$aid" -o "%T %e %N %P" 2>/dev/null | head -1)
  read -r state end node part <<< "$st"
  if [[ "$state" != RUNNING ]]; then echo "$jid ($aid): not running ($st), skipped"; continue; fi
  if squeue -h -s -u "$USER" -o "%i %j" | grep -qE "^${aid}\.[0-9]+ bash$"; then echo "$jid ($aid): already has a pool step, skipped"; continue; fi
  dl=$(( $(date -d "$end" +%s) - 1800 ))
  log=$OUT/pool/logs/step_${jid}_$(date +%H%M%S).out
  setsid nohup srun --jobid="$jid" --overlap -N1 -n1 -c "$W" bash "$T/ag_eval_pool.sh" "$LIST" "$OUT" "$W" "$dl" "$K" \
    > "$log" 2>&1 < /dev/null &
  disown
  echo "$jid ($part $node): eval pool step started, $W workers ($K concurrent shards), deadline $(date -d @$dl +%T), log $log"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%F_%T)" "$part" step \
    "$jid.<next step> (srun --overlap inside soil job $jid on $node, E6b rigid evaluation pool, $W workers in $K concurrent shards, deadline $(date -d @$dl +%T); no new job, no extra billing)" \
    "$LIST:$lsha" "$OUT" "$T/ag_eval_pool.sh" "-" >> "$G3/e3/submissions.tsv"
done
