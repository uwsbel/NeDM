#!/bin/bash
# S2 with the EXISTING K1 deploy ensembles (no retraining): short approach 0.5 s / 1 s on the 800 suite pairs, both
# worlds. Waits for the CRM pass-1 suite drives, syncs pass 1 (both worlds), reads the decision states, plans CEM 4x64
# picks (family free, as K1 A5), builds pass-2 rows, ships them and submits. Usage: bash run_s2_existing.sh
set -u
cd /home/harry/NeDM-traverse_mppi
PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts OMP_NUM_THREADS=6
K1=artifacts/traverse/generalist_20260921/A_adapt; K2=artifacts/traverse/crm_improve_20260922; A=$K2/a5data; S2=$K2/s2
D=$K1/train/deploy_v1
C=/work1/dannegrut/harry/experiments/crm_f104_20260916; G1=/work1/dannegrut/harry/experiments/generalist_20260921
G2=/work1/dannegrut/harry/experiments/crm_improve_20260922
log() { echo "$(date +%H:%M:%S) $*"; }

# 1. wait for CRM pass 1 (suite, 1,600 drives)
while true; do
  n=$(ssh -o ConnectTimeout=20 amd "ls $C/crm_improve/pass1_suite/out/runs/*/episode_complete.json 2>/dev/null | wc -l" || echo 0)
  f=$(ssh -o ConnectTimeout=20 amd "ls $C/crm_improve/pass1_suite/out/failed 2>/dev/null | wc -l" || echo 0)
  [ "$n" -ge 1600 ] && break
  q=$(ssh -o ConnectTimeout=20 amd "squeue -u \$USER -h -n ci_p1_suite | wc -l" || echo 1)
  if [ "$q" -eq 0 ] && [ $((n + f)) -ge 1590 ]; then log "pass 1 ended with $n complete, $f failed"; break; fi
  log "waiting for CRM pass 1: $n/1600 complete, $f failed"; sleep 240
done
# 2. sync pass 1 (both worlds)
mkdir -p $A/pass1_suite_crm/runs $A/pass1_suite_rigid/runs
INC="--include=*/ --include=trajectory.npz --include=outcome.json --include=episode_complete.json --include=command_reference.npz --include=crm_extra.npz --exclude=*"
rsync -az $INC amd:$C/crm_improve/pass1_suite/out/runs/ $A/pass1_suite_crm/runs/
rsync -az $INC amd:$G2/rigid_pass1_suite/runs/ $A/pass1_suite_rigid/runs/
log "synced pass 1: crm $(ls $A/pass1_suite_crm/runs | wc -l), rigid $(ls $A/pass1_suite_rigid/runs | wc -l)"
# 3. decision states per length
for L in 0.5 1; do T=${L/./p}
  $PY scripts/ci_a5data.py --stage analyze --runs $A/pass1_suite_crm/runs $A/pass1_suite_rigid/runs --worlds crm rigid --length $L \
      --approach-dir $A/approach_suite --out $A/decision_suite_L$T > $S2/analyze_L$T.log 2>&1
  log "analyze L=$L rc=$? $(tail -1 $S2/analyze_L$T.log | cut -c1-160)"
done
# 4. picks (CEM 4x64, family free) for the existing ensembles
plan() { local W=$1 L=$2 arm=$3 model=$4; local T=${L/./p}; local out=$S2/L$T/picks_${W}_${arm}
  [ -f $out/summary.json ] && return
  $PY scripts/ci_planner.py --family free --cases $K1/suite/cases --map-root artifacts/traverse/crm_f104_v1/map_root \
     --models "$D/${model}_deploy_s*.pt" --world $W --domain $W --arms B --poses $A/decision_suite_L$T/poses_${W}_all.json \
     --out $out --task-root $K2 --verify 0 > $out.log 2>&1
  log "picks $W L=$L $arm rc=$?"; }
mkdir -p $S2/L0p5 $S2/L1
( for L in 1 0.5; do plan crm $L T T; plan crm $L P P; plan crm $L H H; done; plan crm 1 Spcrm Sp_crm ) > $S2/plan_crm.log 2>&1 &
( for L in 1 0.5; do plan rigid $L T T; plan rigid $L P P; plan rigid $L H H; plan rigid $L Sprigid Sp_rigid; done ) > $S2/plan_rigid.log 2>&1 &
wait
log "planning done"
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
rsync -az $S2/tasks_s2_crm.json amd:$C/crm_improve/s2/ && rsync -az $S2/tasks_s2_rigid.json amd:$G2/tasks/
ssh amd "C=$C; G1=$G1; G2=$G2; S=\$C/source/scripts/crm_collect.sbatch
X=\"--export=ALL,CRM_TASKS=\$C/crm_improve/s2/tasks_s2_crm.json,CRM_OUT=\$C/crm_improve/s2/out,CRM_CONFIG=configs/crm_main.json,CRM_COLLECTOR=\$G1/source/scripts/crm_collect_ext.py,CRM_BUDGET_S=13400\"
j1=\$(sbatch --parsable -p mi2101x -c 16 -t 04:00:00 --array=0-11 -J ci_s2 \$X -o \$C/crm_improve/s2/out/logs/%x_%A_%a.out \$S 2>/dev/null | tail -1)
j2=\$(sbatch --parsable -p mi2508x -c 128 -t 04:00:00 --array=0-2 -J ci_s2 \$X -o \$C/crm_improve/s2/out/logs/%x_%A_%a.out \$S 2>/dev/null | tail -1)
j3=\$(sbatch --parsable -p mi3001x -c 16 -t 04:00:00 --array=0-4 -J ci_s2 \$X -o \$C/crm_improve/s2/out/logs/%x_%A_%a.out \$S 2>/dev/null | tail -1)
j4=\$(sbatch --parsable -p mi2104x -c 128 -t 04:00:00 --array=0-5 -J ci_rs2 --export=ALL,GEN_ROOT=\$G2,GEN_TASKS=\$G2/tasks/tasks_s2_rigid.json,GEN_OUT=\$G2/rigid_s2,GEN_COLLECTOR=\$G2/source/scripts/gen_collect_ext.py -o \$G2/rigid_s2/logs/%x_%A_%a.out \$G2/source/scripts/gen_array_g.sbatch 2>/dev/null | tail -1)
echo submitted ci_s2 \$j1 \$j2 \$j3 ci_rs2 \$j4"
log "S2 existing-ensemble drives launched"
