#!/bin/bash
# Submit CRM collection workers on every GPU partition (one sbatch per partition: submit-filter rule).
# usage: crm_launch.sh <tasks.json> <out dir> <config json relative to CRM root> [hours]
set -uo pipefail
CRM=/work1/dannegrut/harry/experiments/crm_f104_20260916
TASKS=$1; OUT=$2; CFG=$3; H=${4:-6}
S=$CRM/source/scripts/crm_collect.sbatch
mkdir -p $OUT/logs
sub() {  # partition cpus hours array
  local hrs=$3; local budget=$(( hrs * 3600 - 1000 ))
  sbatch --parsable -p $1 -c $2 -t $(printf "%02d:00:00" $hrs) --array=$4 -J crm_$(basename $OUT) -o $OUT/logs/%x_%A_%a.out \
    --export=ALL,CRM_TASKS=$TASKS,CRM_OUT=$OUT,CRM_CONFIG=$CFG,CRM_BUDGET_S=$budget $S 2>&1 | grep -E '^[0-9]+$|error|rror' || echo "submit failed: $1"
}
H4=$(( H < 4 ? H : 4 ))
sub mi2101x 16  $H  0-23
sub mi2104x 128 $H  0-5
sub mi2508x 128 $H  0-2
sub mi3508x 256 $H  0-1
sub mi3008x 192 $H  0-0
sub mi3001x 16  $H4 0-4
sub mi3501x 24  $H4 0-5
