#!/bin/bash
# Login-node helper (arena_gator_20260925, E6b): every 5 min, while some shard of the evaluation task list has no done
# record, start an evaluation pool step (ag_eval_pool_launch.sh) in every RUNNING soil job of this effort (job name
# ag_soil_soil_v1) that has no pool step yet and more than 60 min left: 12 workers on mi2101x, 16 on mi3501x, 96 on
# mi2104x in 4 concurrent shards of 24 (mi2508x and devel are left alone). Exits when $G3/rigid_eval/pool/STOP exists. Start once:
#   setsid nohup bash $G3/tools/e6b/ag_eval_pool_autostep.sh >> $G3/rigid_eval/pool/logs/autostep.out 2>&1 < /dev/null &
set -uo pipefail
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
T=$G3/tools/e6b
OUT=$G3/rigid_eval
POOL=$OUT/pool
LIST=$OUT/TASKLIST.txt
while true; do
  [[ -e $POOL/STOP ]] && { echo "$(date +%T) STOP present, exiting"; exit 0; }
  open=$(python3 - "$LIST" "$POOL/done" <<'PY'
import json, os, sys
shards = set()
for f in [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith('#')]:
    shards |= {int(t['shard']) for t in json.load(open(f)) if t.get('run', True)}
print(sum(1 for s in shards if not os.path.exists(f'{sys.argv[2]}/{s}.json')))
PY
)
  if (( open > 0 )); then
    now=$(date +%s)
    while read -r raw aid part endt name; do
      [[ "$name" == ag_soil_soil_v1 ]] || continue
      left=$(( $(date -d "$endt" +%s) - now ))
      (( left > 3600 )) || continue
      squeue -h -s -u "$USER" -o "%i %j" | grep -qE "^${aid}\.[0-9]+ bash$" && continue
      case $part in mi2101x) w=12:1;; mi3501x) w=16:1;; mi2104x) w=96:4;; *) continue;; esac
      echo "$(date +%T) $open open shards; job $raw ($part) has no pool step, $(( left / 60 )) min left: starting one with $w workers"
      bash $T/ag_eval_pool_launch.sh "$w" "$raw"
    done < <(squeue -u "$USER" -h -t R -o "%A %i %P %e %j")
  fi
  sleep 300
done
