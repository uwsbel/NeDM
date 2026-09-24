#!/bin/bash
# S2 with the NEW deploy ensembles trained with short-anchor rows (a1 GRU history, a3 joint transformer): picks at the
# 1 s and 0.5 s approaches in both worlds, pass-2 rows, ship, submit (queue-aware). Usage: bash run_s2_new.sh
set -u
cd /home/harry/NeDM-traverse_mppi
PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts OMP_NUM_THREADS=6
K1=artifacts/traverse/generalist_20260921/A_adapt; K2=artifacts/traverse/crm_improve_20260922; A=$K2/a5data; S2=$K2/s2; DN=$K2/deploy_v1
C=/work1/dannegrut/harry/experiments/crm_f104_20260916; G1=/work1/dannegrut/harry/experiments/generalist_20260921
G2=/work1/dannegrut/harry/experiments/crm_improve_20260922
log() { echo "$(date +%H:%M:%S) $*"; }
# 1. wait for 5 + 5 deploy checkpoints on the cluster, sync them
while true; do n=$(ssh -o ConnectTimeout=20 amd "ls $G2/train/deploy_v1/deploy_a1_haux_gru_s*.pt $G2/train/deploy_v1/deploy_a3_haux_txjoint_s*.pt 2>/dev/null | wc -l" || echo 0)
  [ "$n" -ge 10 ] && break; log "deploy checkpoints $n/10"; sleep 180; done
mkdir -p $DN && rsync -az amd:$G2/train/deploy_v1/ $DN/ && log "synced deploy: $(ls $DN/*.pt | wc -l) checkpoints"
# 2. picks
plan() { local W=$1 L=$2 arm=$3 glob=$4; local T=${L/./p}; local out=$S2/L$T/picks_${W}_${arm}
  [ -f $out/summary.json ] && return
  $PY scripts/ci_planner.py --family free --cases $K1/suite/cases --map-root artifacts/traverse/crm_f104_v1/map_root \
     --models "$DN/$glob" --world $W --domain $W --arms B --poses $A/decision_suite_L$T/poses_${W}_all.json \
     --out $out --task-root $K2 --verify 0 > $out.log 2>&1
  log "picks $W L=$L $arm rc=$? $(grep -m1 -o 'z_mean [-0-9.]*' $out.log)"; }
( for L in 1 0.5; do plan crm $L X 'deploy_a3_haux_txjoint_s*.pt'; plan crm $L Hn 'deploy_a1_haux_gru_s*.pt'; done ) > $S2/plan_new_crm.log 2>&1 &
( for L in 1 0.5; do plan rigid $L X 'deploy_a3_haux_txjoint_s*.pt'; plan rigid $L Hn 'deploy_a1_haux_gru_s*.pt'; done ) > $S2/plan_new_rigid.log 2>&1 &
wait; log "planning done"
# 3. rows (ids <g>__<L>_<ARM>_B), ship, submit
for L in 1 0.5; do T=${L/./p}; F=$( [ "$L" = "1" ] && echo 20 || echo 10 )
  $PY scripts/ga_a5_pass2_tasks.py --world crm --arms X,Hn --picks-dir $S2/L$T --branch-frame $F --cluster-case-prefix generalist/suite/cases \
      --cluster-approach-prefix crm_improve/pass1_suite/approach --cluster-route-prefix crm_improve/s2n/L$T/routes_crm --out $S2/L$T/tasks_new_crm.json > /dev/null
  $PY scripts/ga_a5_pass2_tasks.py --world rigid --arms X,Hn --picks-dir $S2/L$T --branch-frame $F --cluster-case-prefix a5/cases \
      --cluster-approach-prefix pass1_suite/approach --cluster-route-prefix s2n/L$T/routes_rigid --rigid-abs-root $G2 --out $S2/L$T/tasks_new_rigid.json > /dev/null
  ssh amd "mkdir -p $C/crm_improve/s2n/L$T/routes_crm $G2/s2n/L$T/routes_rigid"
  rsync -az $S2/L$T/routes_pass2_crm/ amd:$C/crm_improve/s2n/L$T/routes_crm/
  rsync -az $S2/L$T/routes_pass2_rigid/ amd:$G2/s2n/L$T/routes_rigid/
done
$PY - <<EOF
import json
C='$C'
for w in ('crm','rigid'):
    rows=[]
    for T in ('L1','L0p5'):
        t=json.load(open(f'$S2/{T}/tasks_new_{w}.json'))
        t=[r for r in t if r['arm'] in ('X','Hn')]
        for r in t:
            r['id']=r['id'].replace('__', f'__{T}_', 1); r['protocol']=T
            if r.get('ref_id'): r['ref_id']=r['ref_id'].replace('__', f'__{T}_', 1)
            if w=='crm':
                ex=r['extra']; i=ex.index('--branch-route'); ex[i+1]=C+'/'+ex[i+1]
        rows+=t
    for i,r in enumerate(rows): r['tier']=i
    ids=[r['id'] for r in rows]; assert len(ids)==len(set(ids))
    json.dump(rows, open(f'$S2/tasks_s2n_{w}.json','w'), indent=1); print(w, len(rows), 'rows', sum(r['run'] for r in rows), 'run')
EOF
while true; do n=$(ssh -o ConnectTimeout=20 amd "squeue -u \$USER -h -r | wc -l" || echo 99); [ "$n" -le 36 ] && break; log "queue $n tasks; waiting"; sleep 120; done
rsync -az $S2/tasks_s2n_crm.json amd:$C/crm_improve/s2n/ && rsync -az $S2/tasks_s2n_rigid.json amd:$G2/tasks/
ssh amd "C=$C; G1=$G1; G2=$G2; S=\$C/source/scripts/crm_collect.sbatch; mkdir -p \$C/crm_improve/s2n/out/logs \$G2/rigid_s2n/logs
X=\"--export=ALL,CRM_TASKS=\$C/crm_improve/s2n/tasks_s2n_crm.json,CRM_OUT=\$C/crm_improve/s2n/out,CRM_CONFIG=configs/crm_main.json,CRM_COLLECTOR=\$G1/source/scripts/crm_collect_ext.py,CRM_BUDGET_S=13400\"
j1=\$(sbatch --parsable -p mi2101x -c 16 -t 04:00:00 --array=0-3 -J ci_s2n \$X -o \$C/crm_improve/s2n/out/logs/%x_%A_%a.out \$S 2>/dev/null | tail -1)
j2=\$(sbatch --parsable -p mi3501x -c 24 -t 04:00:00 --array=0-3 -J ci_s2n \$X -o \$C/crm_improve/s2n/out/logs/%x_%A_%a.out \$S 2>/dev/null | tail -1)
j3=\$(sbatch --parsable -p mi2104x -c 128 -t 04:00:00 --array=0-5 -J ci_rs2n --export=ALL,GEN_ROOT=\$G2,GEN_TASKS=\$G2/tasks/tasks_s2n_rigid.json,GEN_OUT=\$G2/rigid_s2n,GEN_COLLECTOR=\$G2/source/scripts/gen_collect_ext.py -o \$G2/rigid_s2n/logs/%x_%A_%a.out \$G2/source/scripts/gen_array_g.sbatch 2>/dev/null | tail -1)
echo submitted ci_s2n \$j1 \$j2 ci_rs2n \$j3"
log "S2 new-ensemble drives launched"
