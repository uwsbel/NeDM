#!/bin/bash
# S3 continuation data: wait for the twin pass-1 drives (both worlds), sync, read the decision states at L = 3 s and
# 1 s, and build the continuation task files (6 per decision state; CEM picks by the K1 H ensemble). Does NOT submit.
set -u
cd /home/harry/NeDM-traverse_mppi
PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts OMP_NUM_THREADS=6
K2=artifacts/traverse/crm_improve_20260922; A=$K2/a5data; S3=$K2/s3
C=/work1/dannegrut/harry/experiments/crm_f104_20260916; G2=/work1/dannegrut/harry/experiments/crm_improve_20260922
log() { echo "$(date +%H:%M:%S) $*"; }
while true; do
  n=$(ssh -o ConnectTimeout=20 amd "ls $C/crm_improve/pass1_twin/out/runs/*/episode_complete.json 2>/dev/null | wc -l" || echo 0)
  q=$(ssh -o ConnectTimeout=20 amd "squeue -u \$USER -h -n ci_p1_twin | wc -l" || echo 1)
  [ "$n" -ge 3600 ] && break
  if [ "$q" -eq 0 ]; then log "twin pass 1 queue empty at $n/3600"; break; fi
  log "waiting for twin pass 1: $n/3600"; sleep 240
done
mkdir -p $A/pass1_twin_crm/runs $A/pass1_twin_rigid/runs
INC="--include=*/ --include=trajectory.npz --include=outcome.json --include=episode_complete.json --include=command_reference.npz --include=crm_extra.npz --exclude=*"
rsync -az $INC amd:$C/crm_improve/pass1_twin/out/runs/ $A/pass1_twin_crm/runs/
rsync -az $INC amd:$G2/rigid_pass1_twin/runs/ $A/pass1_twin_rigid/runs/
log "synced twin pass 1: crm $(ls $A/pass1_twin_crm/runs | wc -l), rigid $(ls $A/pass1_twin_rigid/runs | wc -l)"
for L in 3 1; do T=${L/./p}
  $PY scripts/ci_a5data.py --stage analyze --runs $A/pass1_twin_crm/runs $A/pass1_twin_rigid/runs --worlds crm rigid --length $L \
      --approach-dir $A/approach_twin --out $A/decision_twin_L$T > $S3/analyze_L$T.log 2>&1
  log "analyze twin L=$L rc=$? $(tail -1 $S3/analyze_L$T.log | cut -c1-200)"
done
for L in 3 1; do T=${L/./p}
  ( $PY scripts/ci_a5data.py --stage continuations --decision $A/decision_twin_L$T --world crm > $S3/cont_crm_L$T.log 2>&1; log "continuations crm L=$L rc=$?" ) &
  ( $PY scripts/ci_a5data.py --stage continuations --decision $A/decision_twin_L$T --world rigid > $S3/cont_rigid_L$T.log 2>&1; log "continuations rigid L=$L rc=$?" ) &
  wait
done
log "S3 build done; outputs:"; ls -d $A/cont_twin_* 2>/dev/null; find $A -maxdepth 2 -name 'SHIP_cont*' 2>/dev/null
