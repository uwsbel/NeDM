#!/bin/bash
# Upload night-2 closed-loop picks and launch the paired Chrono test (all arms of a group on one node).
set -eo pipefail
L=artifacts/traverse/fdm_f104_50h_20260909/night2_v1/closed
R=/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909/night2_closed_v1
rsync -az "$L/routes/" "amd:$R/routes/"
scp -q "$L/tasks_cluster.json" "amd:$R/tasks_cluster.json"
ssh amd "cd $R && sbatch -p ${1:-mi2101x} --array=0-39 n2_closed.sbatch" | tail -1
