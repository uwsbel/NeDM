#!/bin/bash
#SBATCH -A dannegrut
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 01:15:00
#SBATCH -J rlat2101L
#SBATCH -o /work1/dannegrut/harry/experiments/fdm_f104_50h_20260909/render_latency_v1/logs/%x_%A_%a.out
#SBATCH --array=8-9
set -o pipefail
source /work1/dannegrut/harry/experiments/fdm_f104_50h_20260909/render_latency_v1/scripts/common.sh
P=2101
case $SLURM_ARRAY_TASK_ID in
  8) capture_env t8_lp1
     bench lvp_${P}_lp1_512 1 512; probe probe_${P}_lp1_512 1 512 5
     # decide 1 thread at 1024: 4x the no-rebuild (tracing) time + the rebuild overhead, from the 512 probe
     EST=$("$NRD_PYTHON" - <<'PY'
import json
r = json.load(open('/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909/render_latency_v1/probe/probe_2101_lp1_512.json'))['sizes']['512']['regimes']
fr, mv = r['frozen']['total_median_s'], r['moving']['total_median_s']
print(round(4 * fr + max(0., mv - fr), 2))
PY
)
     echo "estimated 1-thread 1024 moving frame: ${EST}s"
     if "$NRD_PYTHON" -c "import sys; sys.exit(0 if float('${EST:-1e9}') <= 60 else 1)"; then
       echo "{\"label\": \"lvp_${P}_lp1_1024\", \"skipped\": false, \"estimate_s\": ${EST}}" > $R/results/lvp_${P}_lp1_1024_decision.json
       bench lvp_${P}_lp1_1024 1 1024; probe probe_${P}_lp1_1024 1 1024 5
     else
       echo "{\"label\": \"lvp_${P}_lp1_1024\", \"skipped\": true, \"estimate_s\": \"${EST}\", \"rule\": \"skip if > 60 s per frame\"}" > $R/results/lvp_${P}_lp1_1024_decision.json
     fi
     perfprobe perf_${P}_lp1_512 1 512 3 ;;
  9) capture_env t9_sweep512
     # same-node thread sweep (guards the cross-task thread comparison against node-to-node variance)
     for T in 1 2 4 8 16 32; do probe probe_${P}_samenode_lp${T}_512 $T 512 3; done ;;
esac
echo "[$(date +%T)] done"
