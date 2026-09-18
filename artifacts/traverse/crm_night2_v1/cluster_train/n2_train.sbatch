#!/bin/bash
#SBATCH -A dannegrut
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 03:30:00
#SBATCH -p mi3501x
#SBATCH -J n2_train
#SBATCH --array=0-6
#SBATCH -o /work1/dannegrut/harry/experiments/crm_f104_20260916/night2/train/logs/%A_%a.out
# each array task runs 4 ensembles concurrently on its MI350X (idx = task, task+7, task+14, task+21)
set -o pipefail
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $MIOPEN_USER_DB_PATH
CRM=/work1/dannegrut/harry/experiments/crm_f104_20260916; N2=$CRM/night2
cd $CRM/source/scripts
run_one() {
  local i=$1
  local line; line=$(python3.12 -c "import json;t=json.load(open('$N2/train/tasks.json'))[$i];print(t['stage'],t['ds'],t['world'],t['arch'],t['lr'],t['extra'])")
  [ -z "$line" ] && return
  set -- $line; local stage=$1 ds=$2 world=$3 arch=$4 lr=$5; shift 5; local extra="$*"
  mkdir -p $N2/train/$stage
  echo "[$(date +%T)] start $i: $stage $ds $arch lr$lr $extra"
  python3.12 -u n2_arch_train.py --ds $N2/datasets/$ds.npz --out $N2/train/$stage --world $world --arch $arch --lr $lr --seeds 5 $extra 2>&1 | grep -v Warn | grep "ENSEMBLE\|Error\|error\|seed" | cut -c1-220
  echo "[$(date +%T)] done $i"
}
NT=$(python3.12 -c "import json;print(len(json.load(open('$N2/train/tasks.json'))))")
pids=()
for k in 0 1 2 3; do i=$((SLURM_ARRAY_TASK_ID + 7*k)); [ $i -lt $NT ] && { run_one $i & pids+=($!); }; done
wait "${pids[@]}"
echo ALL_DONE
