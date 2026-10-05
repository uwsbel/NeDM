#!/bin/bash
#SBATCH -J cn-v5-pack8
#SBATCH -A dannegrut
#SBATCH -p mi3508x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 96
#SBATCH -t 04:00:00
# Up to ~10 trainings on one 8-GPU node. CN_JOBS = "data|config|run|module|gpu;..." (module and gpu optional).
set -euo pipefail
set +u
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
set -u
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONFAULTHANDLER=1
export PYTHONPATH="$CN_CODE/src:${PYTHONPATH:-}"
cd "$CN_CODE"
pids=()
IFS=';' read -ra jobs <<< "$CN_JOBS"
i=0
for job in "${jobs[@]}"; do
  IFS='|' read -r data config run module gpu <<< "$job"
  export HIP_VISIBLE_DEVICES=${gpu:-$(( i % 8 ))}
  i=$(( i + 1 ))
  python3.12 -m ${module:-nedm.contact_nrd.train_v5} --data "$data" --config "$config" --output-dir "$run" >> "$run.stdout.log" 2>&1 &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit $status
