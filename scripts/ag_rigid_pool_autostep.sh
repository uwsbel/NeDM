#!/bin/bash
# Login-node helper (arena_gator_20260925, E3b2): every 5 min, start a rigid pool step (ag_rigid_pool_launch.sh) in
# every RUNNING soil job of this effort (job name ag_soil_soil_v1, not devel) that has no pool step yet and more than
# 60 min left, with 12 workers on mi2101x, 16 on mi3501x, 96 on mi2104x. Exits when all shards of the spec have a done
# record or when $G3/rigid_v1/pool/STOP exists. Start once:
#   setsid nohup bash $G3/source/scripts/ag_rigid_pool_autostep.sh > $G3/rigid_v1/pool/logs/autostep.out 2>&1 < /dev/null &
# Stop: touch $G3/rigid_v1/pool/STOP (also stops new shard claims), or kill the loop: pkill -f ag_rigid_pool_autostep.sh
set -uo pipefail
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
POOL=$G3/rigid_v1/pool
NSHARDS=${AG_POOL_NSHARDS:-425}
while true; do
  [[ -e $POOL/STOP ]] && { echo "$(date +%T) STOP present, exiting"; exit 0; }
  nd=$(ls $POOL/done 2>/dev/null | wc -l)
  (( nd >= NSHARDS )) && { echo "$(date +%T) all $nd shards done, exiting"; exit 0; }
  now=$(date +%s)
  while read -r raw part endt name; do
    [[ "$name" == ag_soil_soil_v1 ]] || continue
    left=$(( $(date -d "$endt" +%s) - now ))
    (( left > 3600 )) || continue
    squeue -h -s -j "$raw" -o "%i %j" | grep -v batch | grep -q " bash" && continue
    case $part in mi2101x) w=12;; mi3501x) w=16;; mi2104x) w=96;; *) continue;; esac
    echo "$(date +%T) job $raw ($part) has no pool step, $(( left / 60 )) min left: starting one with $w workers"
    bash $G3/source/scripts/ag_rigid_pool_launch.sh "$w" "$raw"
  done < <(squeue -u "$USER" -h -t R -o "%A %P %e %j")
  sleep 300
done
