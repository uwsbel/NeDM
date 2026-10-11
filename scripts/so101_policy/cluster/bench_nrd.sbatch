#!/bin/bash
#SBATCH -J sp-nrd-bench
#SBATCH -A dannegrut
#SBATCH -p mi3508x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:20:00
#SBATCH -o /work1/dannegrut/harry/experiments/so101_policy_20261011/logs/%x_%j.out
# Throughput of the final SO101 grasp NRD on one MI350X: the critic's dense-step harness (gpu_ratio.py: code + core +
# collision logits + contact network on ALL (world, pair) + assemble/finish + advance, random small states, 3 warm-up
# and 20 timed repetitions, inference mode) at 4,096 / 16,384 / 65,536 worlds. One process per GPU, each on its own
# GPU (HIP_VISIBLE_DEVICES only, see the cluster memory note): GPU 0 runs seed 61, GPU 1 seed 62 at the same time
# (two independent measurements of the same architecture). Default partition mi3508x (whole node; mi3501x works the
# same: set -p mi3501x and SP_GPUS=1).
#   sbatch scripts/so101_policy/cluster/bench_nrd.sbatch            (from the code dir; SP_CODE, SP_GPUS optional)
# Output: $B/bench/nrd_dense_mi350x_<model>_<job>.json, one row per world count: ms per step, NRD steps/s, peak GB.
set -euo pipefail
set +u; source /etc/profile; module load pytorch/2.10.0; source /work1/dannegrut/harry/venvs/nedm/bin/activate; set -u
B=/work1/dannegrut/harry/experiments/so101_policy_20261011
C=${SP_CODE:-$B/code_nrd_round}
RUNS=/work1/dannegrut/harry/experiments/so101_nrd_fix_20261009/runs
HARNESS=${SP_HARNESS:-$B/bench/gpu_ratio.py}
export OMP_NUM_THREADS=2 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
mkdir -p "$B/bench"
cd "$B/bench"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} start $(date -Is)"
rocm-smi --showproductname 2>/dev/null | grep -m 2 -i "card series\|gfx" || true
pids=()
for g in $(seq 0 $(( ${SP_GPUS:-2} - 1 ))); do
  seed=$(( 61 + g % 2 ))
  HIP_VISIBLE_DEVICES=$g python3.12 -P "$HARNESS" --src "$C/src" --ckpt "grasp_s$seed=$RUNS/short_rel_rk_cc_fix_s${seed}_plus/best.pt" \
    --batches 4096 16384 65536 --reps 20 --out "$B/bench/nrd_dense_mi350x_s${seed}_gpu${g}_${SLURM_JOB_ID:-none}.json" &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "done $(date -Is) exit $status"
exit $status
