#!/bin/bash
# E2 task 4, soil: Gator (ag_crm_collect.py --vehicle gator, calibrated cylinders) on f104 designed routes with a
# known HMMWV collect_v1 outcome; production config crm_main.json (0.08 m, 1 ms), 120 s horizon; local CRM lock.
cd /home/harry/NeDM-traverse_mppi
E=artifacts/traverse/arena_gator_20260925/e2/soil_f104
C=$PWD/artifacts/traverse/fdm_f104_50h_20260909/cases_night2/cases
CFG=$PWD/artifacts/traverse/crm_f104_v1/configs/crm_main.json
export PYTHONPATH=/home/harry/chrono/build/bin:src:scripts OMP_NUM_THREADS=4
for id in "$@"; do
  g=${id%_route_*}; r=${id##*_route_}
  [ -f $E/gator/$id/episode_complete.json ] && continue
  mkdir -p $E/gator
  flock /tmp/luffy_crm.lock /usr/bin/python3.12 -P -u scripts/ag_crm_collect.py --vehicle gator --source-root . \
    --case $C/$g.json --route $C/routes/$g/route_$r.json --chrono-data /home/harry/chrono/data --crm-config $CFG \
    --horizon-s 120 --episode-seed 1 --out $E/gator/$id > $E/gator/$id.log 2>&1 || echo "FAIL $id"
done
