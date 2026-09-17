#!/bin/bash
#SBATCH -A dannegrut
#SBATCH -p mi2104x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH -t 00:50:00
#SBATCH -J rlat2104
#SBATCH -o /work1/dannegrut/harry/experiments/fdm_f104_50h_20260909/render_latency_v1/logs/%x_%j.out
set -o pipefail
source /work1/dannegrut/harry/experiments/fdm_f104_50h_20260909/render_latency_v1/scripts/common.sh
P=2104
capture_env mi2104x
# same node, sequential: 16 (cross-node-type reference), 32, 64, 128 lavapipe threads at 1024
for T in 16 32 64 128; do
  bench lvp_${P}_lp${T}_1024 $T 1024
  probe probe_${P}_lp${T}_1024 $T 1024 5
done
perfprobe perf_${P}_lp64_1024 64 1024 3
echo "[$(date +%T)] done"
