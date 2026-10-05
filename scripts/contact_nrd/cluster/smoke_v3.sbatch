#!/bin/bash
#SBATCH -J cn-v3-smoke
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 01:00:00
# Smoke every version-3 config (small data subset, a few updates per stage).
set -euo pipefail
set +u
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
set -u
export OMP_NUM_THREADS=1 PYTHONFAULTHANDLER=1
export PYTHONPATH="$CN_CODE/src:${PYTHONPATH:-}"
cd "$CN_CODE"
stamp=$(date +%H%M%S)
for cfg in ${CN_SMOKE_CONFIGS:-configs/contact_nrd/v3/*.json}; do
  name=$(basename $cfg .json); sys=$(echo $name | cut -d_ -f2)
  python3.12 -c "import json,sys; c=json.load(open('$cfg')); c['max_episodes']=300; json.dump(c,open('/tmp/$name.json','w'))"
  python3.12 -m nedm.contact_nrd.train_v3 --data "$CN_ROOT/data/${sys}_train" --config /tmp/$name.json --output-dir "$CN_ROOT/smoke/${name}_$stamp" --smoke > "$CN_ROOT/smoke/${name}_$stamp.log" 2>&1 && echo "OK $name" || { echo "FAIL $name"; tail -n 20 "$CN_ROOT/smoke/${name}_$stamp.log"; }
done
