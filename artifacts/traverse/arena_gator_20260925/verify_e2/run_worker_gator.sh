#!/bin/bash
# VERIFY_E2: a Gator soil row through the UNMODIFIED crm_worker.py with CRM_COLLECTOR=ag_crm_collect.py, as E3 will
# run it on the cluster: row extra = ["--vehicle", "gator", "--horizon-s", "10"] (the worker hard-codes 120 s; the
# later flag wins, a local-smoke shortcut only), config from CRM_CONFIG.  Checks: the worker appended --crm-config
# (collection_request.json crm_config == crm_main.json merged), the run is a Gator run, episode_complete.json written.
# Temporary CRM_ROOT under verify_e2/worker_root (source -> this worktree, cases/configs copied).
set -e
cd /home/harry/NeDM-traverse_mppi
V=$PWD/artifacts/traverse/arena_gator_20260925/verify_e2
R=$V/worker_root
C=$PWD/artifacts/traverse/fdm_f104_50h_20260909/cases_night2/cases
rm -rf $R; mkdir -p $R/cases/routes/f104_v2_group_0005 $R/configs
ln -s /home/harry/NeDM-traverse_mppi $R/source
cp $C/f104_v2_group_0005.json $R/cases/
cp $C/routes/f104_v2_group_0005/route_02.json $R/cases/routes/f104_v2_group_0005/
cp $PWD/artifacts/traverse/crm_f104_v1/configs/crm_main.json $R/configs/
cat > $R/tasks.json <<EOF
[{"id": "gator__f104_v2_group_0005_route_02", "pair_id": "f104_v2_group_0005_route_02",
  "case": "cases/f104_v2_group_0005.json", "route": "cases/routes/f104_v2_group_0005/route_02.json",
  "episode_seed": 12345, "tier": 0, "extra": ["--vehicle", "gator", "--horizon-s", "10"]}]
EOF
unset NEDM_VEHICLE
export CRM_ROOT=$R CRM_TASKS=$R/tasks.json CRM_OUT=$R/out CRM_CHRONO_DATA=/home/harry/chrono/data \
       CRM_CONFIG=configs/crm_main.json CRM_GPUS=1 CRM_CUDA=1 CRM_SWEEPS=1 \
       CRM_COLLECTOR=/home/harry/NeDM-traverse_mppi/scripts/ag_crm_collect.py NRD_PYTHON=/usr/bin/python3.12 \
       PYTHONPATH=/home/harry/chrono/build/bin
flock /tmp/luffy_crm.lock /usr/bin/python3.12 -u scripts/crm_worker.py > $V/worker_gator.log 2>&1
