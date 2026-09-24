#!/bin/bash
# Rigid confirmation of the final configuration: the short-window GRU history ensemble with gradient refinement,
# deciding after the 0.5 s approach. Waits for the pick set, builds rows, ships and submits.
set -u
cd /home/harry/NeDM-traverse_mppi
PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts OMP_NUM_THREADS=6
K1=artifacts/traverse/generalist_20260921/A_adapt; K2=artifacts/traverse/crm_improve_20260922; S4=$K2/s4
G2=/work1/dannegrut/harry/experiments/crm_improve_20260922
log() { echo "$(date +%H:%M:%S) $*"; }
while [ ! -f $S4/L0p5/picks_rigid_HnG/summary.json ]; do log "waiting for the rigid gradient picks"; sleep 180; done
$PY scripts/ga_a5_pass2_tasks.py --world rigid --arms HnG --picks-dir $S4/L0p5 --branch-frame 10 --cluster-case-prefix a5/cases \
    --cluster-approach-prefix pass1_suite/approach --cluster-route-prefix s4/L0p5/routes_rigid --rigid-abs-root $G2 --out $S4/L0p5/tasks_HnG_rigid.json | tail -3
ssh amd "mkdir -p $G2/s4/L0p5/routes_rigid $G2/rigid_s4/logs"
rsync -az $S4/L0p5/routes_pass2_rigid/ amd:$G2/s4/L0p5/routes_rigid/
rsync -az $S4/L0p5/tasks_HnG_rigid.json amd:$G2/tasks/
while true; do n=$(ssh -o ConnectTimeout=20 amd "squeue -u \$USER -h -r | wc -l" || echo 99); [ "$n" -le 40 ] && break; log "queue $n tasks; waiting"; sleep 180; done
ssh amd "G2=$G2; j=\$(sbatch --parsable -p mi2104x -c 128 -t 03:00:00 --array=0-5 -J ci_rs4 --export=ALL,GEN_ROOT=\$G2,GEN_TASKS=\$G2/tasks/tasks_HnG_rigid.json,GEN_OUT=\$G2/rigid_s4,GEN_COLLECTOR=\$G2/source/scripts/gen_collect_ext.py -o \$G2/rigid_s4/logs/%x_%A_%a.out \$G2/source/scripts/gen_array_g.sbatch 2>/dev/null | tail -1); echo submitted ci_rs4 \$j"
log "rigid gradient drives launched"
