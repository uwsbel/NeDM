#!/bin/bash
#SBATCH -A dannegrut
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:40:00
#SBATCH -J rlat2101
#SBATCH -o /work1/dannegrut/harry/experiments/fdm_f104_50h_20260909/render_latency_v1/logs/%x_%A_%a.out
#SBATCH --array=0-7
set -o pipefail
source /work1/dannegrut/harry/experiments/fdm_f104_50h_20260909/render_latency_v1/scripts/common.sh
P=2101
case $SLURM_ARRAY_TASK_ID in
  0) capture_env t0_lp16_256;  bench lvp_${P}_lp16_256 16 256;   probe probe_${P}_lp16_256 16 256 5 ;;
  1) capture_env t1_lp16_512;  bench lvp_${P}_lp16_512 16 512;   probe probe_${P}_lp16_512 16 512 5 ;;
  2) capture_env t2_lp16_1024; bench lvp_${P}_lp16_1024 16 1024; probe probe_${P}_lp16_1024 16 1024 5
     perfprobe perf_${P}_lp16_1024 16 1024 3 ;;
  3) capture_env t3_lp16_1024_rgb; bench lvp_${P}_lp16_1024_rgb 16 1024 rgb; probe probe_${P}_lp16_1024_rgb 16 1024 5 rgb ;;
  4) capture_env t4_lp8_512;   bench lvp_${P}_lp8_512 8 512;     probe probe_${P}_lp8_512 8 512 5 ;;
  5) capture_env t5_lp8_1024;  bench lvp_${P}_lp8_1024 8 1024;   probe probe_${P}_lp8_1024 8 1024 5 ;;
  6) capture_env t6_lp4_512;   bench lvp_${P}_lp4_512 4 512;     probe probe_${P}_lp4_512 4 512 5 ;;
  7) capture_env t7_lp4_1024;  bench lvp_${P}_lp4_1024 4 1024;   probe probe_${P}_lp4_1024 4 1024 5 ;;
esac
echo "[$(date +%T)] done"
