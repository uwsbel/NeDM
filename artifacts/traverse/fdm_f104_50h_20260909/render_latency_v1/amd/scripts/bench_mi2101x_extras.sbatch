#!/bin/bash
#SBATCH -A dannegrut
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:40:00
#SBATCH -J rlat2101X
#SBATCH -o /work1/dannegrut/harry/experiments/fdm_f104_50h_20260909/render_latency_v1/logs/%x_%j.out
set -o pipefail
source /work1/dannegrut/harry/experiments/fdm_f104_50h_20260909/render_latency_v1/scripts/common.sh
P=2101
capture_env extras
# same node: pixel scaling 256 -> 2048 at 16 threads, then two "why is a ray so expensive" controls at 1024
probe probe_${P}_samenode_lp16_sizes 16 "256 512 1024 2048" 3
probe probe_${P}_lp16_1024_novehicle 16 1024 3 --hide-vehicle
LP_NATIVE_VECTOR_WIDTH=128 probe probe_${P}_lp16_1024_vw128 16 1024 3
echo "[$(date +%T)] done"
