#!/bin/bash
# S2 launch step (split off run_s2_existing.sh to respect the 50-task queue cap): waits for the planning lanes, builds
# the pass-2 rows, ships them and submits once the queue has room (<= 34 tasks). Smaller arrays than the original.
set -u
cd /home/harry/NeDM-traverse_mppi
PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts OMP_NUM_THREADS=6
K1=artifacts/traverse/generalist_20260921/A_adapt; K2=artifacts/traverse/crm_improve_20260922; A=$K2/a5data; S2=$K2/s2
C=/work1/dannegrut/harry/experiments/crm_f104_20260916; G1=/work1/dannegrut/harry/experiments/generalist_20260921
G2=/work1/dannegrut/harry/experiments/crm_improve_20260922
log() { echo "$(date +%H:%M:%S) $*"; }
need="L1/picks_crm_T L1/picks_crm_P L1/picks_crm_H L1/picks_crm_Spcrm L0p5/picks_crm_T L0p5/picks_crm_P L0p5/picks_crm_H L1/picks_rigid_T L1/picks_rigid_P L1/picks_rigid_H L1/picks_rigid_Sprigid L0p5/picks_rigid_T L0p5/picks_rigid_P L0p5/picks_rigid_H L0p5/picks_rigid_Sprigid"
while true; do miss=0; for d in $need; do [ -f $S2/$d/summary.json ] || miss=$((miss+1)); done; [ $miss -eq 0 ] && break
  if ! pgrep -f "ci_planner.py --family free --cases $K1/suite/cases" > /dev/null; then
    for d in $need; do if [ ! -f $S2/$d/summary.json ]; then L=${d%%/*}; a=${d#*/picks_}; W=${a%%_*}; arm=${a#*_}; LL=$( [ "$L" = "L1" ] && echo 1 || echo 0.5 )
      model=$arm; [ "$arm" = "Spcrm" ] && model=Sp_crm; [ "$arm" = "Sprigid" ] && model=Sp_rigid
      log "replanning missing $d"; $PY scripts/ci_planner.py --family free --cases $K1/suite/cases --map-root artifacts/traverse/crm_f104_v1/map_root \
        --models "$K1/train/deploy_v1/${model}_deploy_s*.pt" --world $W --domain $W --arms B --poses $A/decision_suite_$L/poses_${W}_all.json \
        --out $S2/$d --task-root $K2 --verify 0 > $S2/$d.log 2>&1; fi; done
  fi
  log "waiting for $miss pick sets"; sleep 60; done
log "all pick sets present"
# 5. pass-2 rows, ship, submit
ssh amd "mkdir -p $G2/a5/cases && cp -rn $G1/a5/cases/. $G2/a5/cases/ && mkdir -p $C/crm_improve/s2/out/logs $G2/rigid_s2/logs"
rm -f $S2/tasks_s2_crm.json $S2/tasks_s2_rigid.json
for L in 1 0.5; do T=${L/./p}; F=$( [ "$L" = "1" ] && echo 20 || echo 10 )
  crm_arms=$( [ "$L" = "1" ] && echo T,P,H,Spcrm || echo T,P,H )
  $PY scripts/ga_a5_pass2_tasks.py --world crm --arms $crm_arms --picks-dir $S2/L$T --branch-frame $F --cluster-case-prefix generalist/suite/cases \
      --cluster-approach-prefix crm_improve/pass1_suite/approach --cluster-route-prefix crm_improve/s2/L$T/routes_crm --out $S2/L$T/tasks_crm.json > /dev/null
  $PY scripts/ga_a5_pass2_tasks.py --world rigid --arms T,P,H,Sprigid --picks-dir $S2/L$T --branch-frame $F --cluster-case-prefix a5/cases \
      --cluster-approach-prefix pass1_suite/approach --cluster-route-prefix s2/L$T/routes_rigid --rigid-abs-root $G2 --out $S2/L$T/tasks_rigid.json > /dev/null
  ssh amd "mkdir -p $C/crm_improve/s2/L$T/routes_crm $G2/s2/L$T/routes_rigid"
  rsync -az $S2/L$T/routes_pass2_crm/ amd:$C/crm_improve/s2/L$T/routes_crm/
  rsync -az $S2/L$T/routes_pass2_rigid/ amd:$G2/s2/L$T/routes_rigid/
done
$PY - <<EOF
import json
C='$C'
for w in ('crm','rigid'):
    rows=[]
    for T in ('L1','L0p5'):
        t=json.load(open(f'$S2/{T}/tasks_{w}.json'))
        for r in t:
            r['id']=r['id'].replace('__', f'__{T}_', 1); r['protocol']=T
            if r.get('ref_id'): r['ref_id']=r['ref_id'].replace('__', f'__{T}_', 1)
            if w=='crm':
                ex=r['extra']; i=ex.index('--branch-route'); ex[i+1]=C+'/'+ex[i+1]
        rows+=t
    for i,r in enumerate(rows): r['tier']=i
    ids=[r['id'] for r in rows]; assert len(ids)==len(set(ids))
    json.dump(rows, open(f'$S2/tasks_s2_{w}.json','w'), indent=1); print(w, len(rows), 'rows', sum(r['run'] for r in rows), 'run')
EOF
while true; do n=$(ssh -o ConnectTimeout=20 amd "squeue -u \$USER -h -r | wc -l" || echo 99); [ "$n" -le 34 ] && break; log "queue $n tasks; waiting"; sleep 120; done
rsync -az $S2/tasks_s2_crm.json amd:$C/crm_improve/s2/ && rsync -az $S2/tasks_s2_rigid.json amd:$G2/tasks/
ssh amd "C=$C; G1=$G1; G2=$G2; S=\$C/source/scripts/crm_collect.sbatch
X=\"--export=ALL,CRM_TASKS=\$C/crm_improve/s2/tasks_s2_crm.json,CRM_OUT=\$C/crm_improve/s2/out,CRM_CONFIG=configs/crm_main.json,CRM_COLLECTOR=\$G1/source/scripts/crm_collect_ext.py,CRM_BUDGET_S=13400\"
j1=\$(sbatch --parsable -p mi2101x -c 16 -t 04:00:00 --array=0-9 -J ci_s2 \$X -o \$C/crm_improve/s2/out/logs/%x_%A_%a.out \$S 2>/dev/null | tail -1)
j2=\$(sbatch --parsable -p mi2508x -c 128 -t 04:00:00 --array=0-1 -J ci_s2 \$X -o \$C/crm_improve/s2/out/logs/%x_%A_%a.out \$S 2>/dev/null | tail -1)
j3=\$(sbatch --parsable -p mi3001x -c 16 -t 04:00:00 --array=0-1 -J ci_s2 \$X -o \$C/crm_improve/s2/out/logs/%x_%A_%a.out \$S 2>/dev/null | tail -1)
j4=\$(sbatch --parsable -p mi2104x -c 128 -t 04:00:00 --array=0-5 -J ci_rs2 --export=ALL,GEN_ROOT=\$G2,GEN_TASKS=\$G2/tasks/tasks_s2_rigid.json,GEN_OUT=\$G2/rigid_s2,GEN_COLLECTOR=\$G2/source/scripts/gen_collect_ext.py -o \$G2/rigid_s2/logs/%x_%A_%a.out \$G2/source/scripts/gen_array_g.sbatch 2>/dev/null | tail -1)
echo submitted ci_s2 \$j1 \$j2 \$j3 ci_rs2 \$j4"
log "S2 existing-ensemble drives launched"
