#!/bin/bash
# Launch the 7,182 soil continuation drives (3 s approach, twin training groups) once the queue has room (cap 50 tasks
# per user): waits until my queued/running task count is <= 30, then submits 15 tasks over mi2101x and mi3001x.
set -u
C=/work1/dannegrut/harry/experiments/crm_f104_20260916; G2=/work1/dannegrut/harry/experiments/crm_improve_20260922
while true; do
  n=$(ssh -o ConnectTimeout=20 amd "squeue -u \$USER -h -r | wc -l" || echo 99)
  [ "$n" -le 30 ] && break
  echo "$(date +%H:%M:%S) queue has $n tasks; waiting"; sleep 180
done
ssh amd "C=$C; G2=$G2; S=\$C/source/scripts/crm_collect.sbatch
X=\"--export=ALL,CRM_TASKS=\$C/crm_improve/cont_twin_L3/tasks_cont_twin_crm_L3.json,CRM_OUT=\$C/crm_improve/cont_twin_L3/out,CRM_CONFIG=configs/crm_main.json,CRM_COLLECTOR=\$G2/source/scripts/crm_collect_ext.py,CRM_BUDGET_S=13400\"
mkdir -p \$C/crm_improve/cont_twin_L3/out/logs
j1=\$(sbatch --parsable -p mi2101x -c 16 -t 04:00:00 --array=0-9 -J ci_cont3 \$X -o \$C/crm_improve/cont_twin_L3/out/logs/%x_%A_%a.out \$S 2>/dev/null | tail -1)
j2=\$(sbatch --parsable -p mi3001x -c 16 -t 04:00:00 --array=0-4 -J ci_cont3 \$X -o \$C/crm_improve/cont_twin_L3/out/logs/%x_%A_%a.out \$S 2>/dev/null | tail -1)
echo \"\$(date +%H:%M:%S) submitted soil continuations \$j1 \$j2\""
