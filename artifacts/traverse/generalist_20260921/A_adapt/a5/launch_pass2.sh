#!/bin/bash
# A5 pass 2: wait for the planning lanes, build the task rows for both worlds, ship routes, submit.
set -u
cd /home/harry/NeDM-traverse_mppi
PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts
K=artifacts/traverse/generalist_20260921/A_adapt; A5=$K/a5
C=/work1/dannegrut/harry/experiments/crm_f104_20260916; G=/work1/dannegrut/harry/experiments/generalist_20260921
while ! grep -q "finished" $A5/planning_crm.log || ! grep -q "finished" $A5/planning_rigid.log; do sleep 30; done
echo "$(date +%H:%M:%S) planning done"
ARMS=Spcrm,Sprigid,H,Hmask,P,T
$PY scripts/ga_a5_pass2_tasks.py --world crm --arms $ARMS --picks-dir $A5 --cluster-case-prefix generalist/suite/cases \
   --cluster-approach-prefix generalist/a5/approach --cluster-route-prefix generalist/a5/pass2/routes_crm --out $A5/tasks_pass2_crm.json | tail -4
$PY scripts/ga_a5_pass2_tasks.py --world rigid --arms $ARMS --picks-dir $A5 --cluster-case-prefix a5/cases \
   --cluster-approach-prefix a5/approach --cluster-route-prefix a5/pass2/routes_rigid --rigid-abs-root $G --out $A5/tasks_pass2_rigid.json | tail -4
ssh amd "mkdir -p $C/generalist/a5/pass2/routes_crm $C/generalist/a5/pass2/out/logs $G/a5/pass2/routes_rigid $G/rigid_a5p2/logs"
rsync -az $A5/routes_pass2_crm/ amd:$C/generalist/a5/pass2/routes_crm/
rsync -az $A5/tasks_pass2_crm.json amd:$C/generalist/a5/pass2/
rsync -az $A5/routes_pass2_rigid/ amd:$G/a5/pass2/routes_rigid/
rsync -az $A5/tasks_pass2_rigid.json amd:$G/tasks/
ssh amd "C=$C; G=$G; echo crm routes \$(ls \$C/generalist/a5/pass2/routes_crm | wc -l) rigid routes \$(ls \$G/a5/pass2/routes_rigid | wc -l); S=\$C/source/scripts/crm_collect.sbatch; X=\"--export=ALL,CRM_TASKS=\$C/generalist/a5/pass2/tasks_pass2_crm.json,CRM_OUT=\$C/generalist/a5/pass2/out,CRM_CONFIG=configs/crm_main.json,CRM_COLLECTOR=\$G/source/scripts/crm_collect_ext.py,CRM_BUDGET_S=13400\"; j1=\$(sbatch --parsable -p mi2101x -c 16 -t 04:00:00 --array=0-11 -J a5p2 \$X -o \$C/generalist/a5/pass2/out/logs/%x_%A_%a.out \$S 2>/dev/null | tail -1); j2=\$(sbatch --parsable -p mi3001x -c 16 -t 04:00:00 --array=0-4 -J a5p2 \$X -o \$C/generalist/a5/pass2/out/logs/%x_%A_%a.out \$S 2>/dev/null | tail -1); j3=\$(sbatch --parsable -p mi2104x -c 128 -t 06:00:00 --array=0-5 -J rigid_a5p2 --export=ALL,GEN_ROOT=\$G,GEN_TASKS=\$G/tasks/tasks_pass2_rigid.json,GEN_OUT=\$G/rigid_a5p2,GEN_COLLECTOR=\$G/source/scripts/gen_collect_ext.py -o \$G/rigid_a5p2/logs/%x_%A_%a.out \$G/source/scripts/gen_array_g.sbatch 2>/dev/null | tail -1); echo submitted a5p2 \$j1 \$j2 rigid_a5p2 \$j3; squeue -u \$USER -h -o '%j %T' | sort | uniq -c"
echo "$(date +%H:%M:%S) pass 2 launched"
