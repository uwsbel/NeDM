#!/bin/bash
# Exact submit lines for the rigid A4 / B3 drives and the CRM B3 twin (written 2026-09-21 23:10; NOT executed).
# Preconditions: balance check (519.4/1500 used at 23:10); for the CRM twin the crm_collect_ext.py cluster smoke gate of
# PLAN R2 (first smoke's outcome.json shows physics_dt_s == 0.001) must have been passed by the CRM side.
set -euo pipefail
G=/work1/dannegrut/harry/experiments/generalist_20260921
C=/work1/dannegrut/harry/experiments/crm_f104_20260916

# ---- rigid A4: 800 branch_auto rows (replay + 3 continuations each, all on one node per group), 6 shards on mi2104x
mkdir -p $G/rigid_a4/logs
sbatch --parsable -p mi2104x -c 128 -t 06:00:00 --array=0-5 -J rigid_a4 \
  --export=ALL,GEN_ROOT=$G,GEN_TASKS=$G/tasks/tasks_a4_rigid.json,GEN_OUT=$G/rigid_a4,GEN_COLLECTOR=$G/source/scripts/gen_collect_ext.py \
  -o $G/rigid_a4/logs/%x_%A_%a.out $G/source/scripts/gen_array_g.sbatch

# ---- rigid B3: 1,500 pid_perturbed rows on train-group designed routes, 6 shards on mi2104x
mkdir -p $G/rigid_b3/logs
sbatch --parsable -p mi2104x -c 128 -t 06:00:00 --array=0-5 -J rigid_b3 \
  --export=ALL,GEN_ROOT=$G,GEN_TASKS=$G/tasks/tasks_b3_rigid.json,GEN_OUT=$G/rigid_b3,GEN_COLLECTOR=$G/source/scripts/gen_collect_ext.py \
  -o $G/rigid_b3/logs/%x_%A_%a.out $G/source/scripts/gen_array_g.sbatch

# ---- CRM B3 twin: the same 1,500 (group, route) pairs and seeds through crm_worker.py (C/source sbatch, CRM_ROOT=C hard-coded),
#      collector = G/source/scripts/crm_collect_ext.py, MI350X partitions only; resumable (resubmit = continue)
S=$C/source/scripts/crm_collect.sbatch
X="--export=ALL,CRM_TASKS=$G/tasks/tasks_b3_crm.json,CRM_OUT=$G/crm_b3,CRM_CONFIG=configs/crm_main.json,CRM_COLLECTOR=$G/source/scripts/crm_collect_ext.py,CRM_BUDGET_S=13400"
mkdir -p $G/crm_b3/logs
sbatch --parsable -p mi3501x -c 24  -t 04:00:00 --array=0-5 -J crm_b3 $X -o $G/crm_b3/logs/%x_%A_%a.out $S
sbatch --parsable -p mi3508x -c 256 -t 04:00:00 --array=0-1 -J crm_b3 $X -o $G/crm_b3/logs/%x_%A_%a.out $S

# progress:  ssh amd "ls $G/rigid_a4/runs | grep -vc __ ; ls $G/rigid_b3/runs | wc -l; ls $G/crm_b3/runs | wc -l; cat $G/crm_b3/workers/*.json"
# A4 drops:  ssh amd "cat $G/rigid_a4/runs/*/skipped.json | grep -o '\"reason\": \"[a-z_]*' | sort | uniq -c"
