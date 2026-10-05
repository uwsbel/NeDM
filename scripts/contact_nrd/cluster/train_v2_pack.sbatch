#!/bin/bash
#SBATCH -J cn-pack
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 04:00:00
# Several contact-stage trainings share one MI350. CN_JOBS = "data|config|run;data|config|run;..."
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
  IFS='|' read -r data config run <<< "$job"
  if [[ ${CN_NGPU:-1} -gt 1 ]]; then export HIP_VISIBLE_DEVICES=$(( i % CN_NGPU )); fi   # one GPU per run on multi-GPU nodes
  i=$(( i + 1 ))
  python3.12 -m ${CN_MODULE:-nedm.contact_nrd.train_v2} --data "$data" --config "$config" --output-dir "$run" > "$run.stdout.log" 2>&1 &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit $status
