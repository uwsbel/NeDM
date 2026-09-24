#!/bin/bash
# S4: drive the gradient-refined picks of the joint-transformer ensemble at both short approaches (soil).
# Waits for the pick sets, builds pass-2 rows, ships and submits when the queue has room.
set -u
cd /home/harry/NeDM-traverse_mppi
PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts OMP_NUM_THREADS=6
K1=artifacts/traverse/generalist_20260921/A_adapt; K2=artifacts/traverse/crm_improve_20260922; S4=$K2/s4
C=/work1/dannegrut/harry/experiments/crm_f104_20260916; G1=/work1/dannegrut/harry/experiments/generalist_20260921
log() { echo "$(date +%H:%M:%S) $*"; }
while true; do miss=0; for d in L0p5/picks_crm_HnG L0p5/picks_crm_XG; do [ -f $S4/$d/summary.json ] || miss=$((miss+1)); done
  [ $miss -eq 0 ] && break; log "waiting for $miss gradient pick sets"; sleep 180; done
log "pick sets ready"
for A in HnG XG; do L=0p5; F=10
  $PY scripts/ga_a5_pass2_tasks.py --world crm --arms $A --picks-dir $S4/L$L --branch-frame $F --cluster-case-prefix generalist/suite/cases \
      --cluster-approach-prefix crm_improve/pass1_suite/approach --cluster-route-prefix crm_improve/s4/L$L/routes_crm --out $S4/L$L/tasks_${A}_crm.json > /dev/null
  ssh amd "mkdir -p $C/crm_improve/s4/L$L/routes_crm"
  rsync -az $S4/L$L/routes_pass2_crm/ amd:$C/crm_improve/s4/L$L/routes_crm/
done
$PY - <<EOF
import json
C='$C'
rows=[]
for A in ('HnG','XG'):
    T='L0p5'; t=json.load(open(f'$S4/{T}/tasks_{A}_crm.json'))
    t=[r for r in t if r['arm']==A]
    for r in t:
        r['id']=r['id'].replace('__', f'__{T}_', 1); r['protocol']=T
        if r.get('ref_id'): r['ref_id']=r['ref_id'].replace('__', f'__{T}_', 1)
        ex=r['extra']; i=ex.index('--branch-route'); ex[i+1]=C+'/'+ex[i+1]
    rows+=t
for i,r in enumerate(rows): r['tier']=i
ids=[r['id'] for r in rows]; assert len(ids)==len(set(ids))
json.dump(rows, open('$S4/tasks_s4_crm.json','w'), indent=1); print(len(rows), 'rows', sum(r['run'] for r in rows), 'run')
EOF
while true; do n=$(ssh -o ConnectTimeout=20 amd "squeue -u \$USER -h -r | wc -l" || echo 99); [ "$n" -le 36 ] && break; log "queue $n tasks; waiting"; sleep 180; done
ssh amd "mkdir -p $C/crm_improve/s4/out/logs"
rsync -az $S4/tasks_s4_crm.json amd:$C/crm_improve/s4/
ssh amd "C=$C; G1=$G1; S=\$C/source/scripts/crm_collect.sbatch
X=\"--export=ALL,CRM_TASKS=\$C/crm_improve/s4/tasks_s4_crm.json,CRM_OUT=\$C/crm_improve/s4/out,CRM_CONFIG=configs/crm_main.json,CRM_COLLECTOR=\$G1/source/scripts/crm_collect_ext.py,CRM_BUDGET_S=13400\"
j1=\$(sbatch --parsable -p mi2104x -c 128 -t 04:00:00 --array=0-5 -J ci_s4 \$X -o \$C/crm_improve/s4/out/logs/%x_%A_%a.out \$S 2>/dev/null | tail -1)
j2=\$(sbatch --parsable -p mi2101x -c 16 -t 04:00:00 --array=0-5 -J ci_s4 \$X -o \$C/crm_improve/s4/out/logs/%x_%A_%a.out \$S 2>/dev/null | tail -1)
echo submitted ci_s4 \$j1 \$j2"
log "gradient drives launched"
