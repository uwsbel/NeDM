#!/bin/bash
# Score CRM policies on UW-Madison Euler: ONE POLICY PER ARRAY TASK.
#
#   sbatch --array=0-0 scripts/cluster/euler/score_policy.sbatch          # base only
#   sbatch --array=0-$(( $(wc -l < ~/nedm/policies.txt) - 1 ))%2 \
#          scripts/cluster/euler/score_policy.sbatch                      # the lot
#
# Each task scores one policy on all 80 val episodes and writes
# $NEDM_OUT/crmtrack_<arm>_euler.json, the same record schema and the same
# episode_id keys the desktop fleet writes, so the two pair directly.
#
# PREFER score_policy_pergpu.sbatch WHEN THE NODE HAS SEVERAL GPUs. A single CRM
# episode saturates one GPU, so the useful unit of parallelism on Euler is a GPU,
# not a thread (see the CONC comment below for the measurement). This script is
# the right shape for a one-GPU-at-a-time job and for sweeps where turnaround per
# policy does not matter.
#
# WHY THE REQUEST IS SMALL AND THE ARRAY THROTTLE IS LOW, and why this differs
# from the hpcfund version of this script. On hpcfund a node is exclusive and
# billed whole, so the right move is to fill it. Euler is the opposite: the
# `sbel` partition is TWO nodes, OverSubscribe=YES:4, ExclusiveUser=NO, shared
# with the rest of the lab, and there is no accounting, no QOS limit and no
# fairshare to stop a sweep from taking the whole tier. Nothing here is
# self-limiting except this file. So: request the cores actually used, take one
# GPU, and throttle the array with %2 rather than letting 45 tasks queue up
# behind the lab's own work.
#
# -w euler19 is deliberate. `sbel` is euler16 (2080 Ti, sm_75) and euler19
# (A100, sm_80). chrono-build-parsers runs on both -- verified, and to identical
# digits: the same episode scored 0.19542514273658756 on each -- but the 2080 Ti
# takes 112 s to the A100's 65 s, and has 11 GB against 40 GB. Drop the -w to let
# a job land on either; the build supports it.
#
#SBATCH -J kyle-crmscore
#SBATCH -p sbel
#SBATCH -w euler19
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -t 03:00:00
#SBATCH -o /srv/home/kasha2/logs/%x-%A_%a.out
#SBATCH -e /srv/home/kasha2/logs/%x-%A_%a.err

set -o pipefail
module load gcc/12.2.0 nvidia/cuda/12.9.1
source /srv/home/kasha2/nedm/env-scoring.sh

# Concurrent episode subprocesses. DEFAULT 1, AND THAT IS A MEASUREMENT, NOT
# CAUTION.
#
# hpcfund benchmarked this curve flat from concurrency 4 to 16 on an MI210 and
# concluded the GPU saturates at about 4. On an A100 it is worse than flat: it
# is actively downhill. Measured on euler19, same 80-episode val index, same
# base policy, episodes taken in the same index order:
#
#     concurrency 6, 1 GPU    10 episodes in 44 min 39 s   (job 51573)
#     concurrency 1, 2 GPUs    8 episodes in  8 min 06 s   (job 51744)
#
# A single episode already saturates the device -- nvidia-smi read 100%
# utilization with 6.0 GB in use, about 1 GB per episode -- so stacking six of
# them onto one GPU does not overlap anything. It time-slices work that was
# already serial and adds six CUDA contexts' worth of switching on top, and the
# result is roughly half the throughput per GPU of running one at a time.
#
# So on Euler the knob that matters is GPUS, not concurrency, and the script for
# that is score_policy_pergpu.sbatch: one episode per GPU, each pinned with
# CUDA_VISIBLE_DEVICES, --concurrency 1 each. This script remains the right shape
# for a single-GPU node, and for the case where one policy's turnaround does not
# matter. Raise CONC only with a measurement that says so.
CONC=${CONC:-1}

POLICIES=${POLICIES:-$NEDM_ROOT/policies.txt}
LINE=$(sed -n "$((${SLURM_ARRAY_TASK_ID:-0} + 1))p" "$POLICIES")
[ -z "$LINE" ] && { echo "FATAL: no policy at line $((${SLURM_ARRAY_TASK_ID:-0} + 1)) of $POLICIES"; exit 1; }

ARM=$(basename "$LINE" .pt)
CKPT=$NEDM_CKPTS/$(basename "$LINE")
[ -f "$CKPT" ] || { echo "FATAL: missing checkpoint $CKPT"; exit 1; }
OUT=$NEDM_OUT/crmtrack_${ARM}_euler.json

echo "== host $(hostname)  arm $ARM  task ${SLURM_ARRAY_TASK_ID:-0}  conc $CONC =="
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
echo "== ckpt   $CKPT"
echo "== chrono $NEDM_CHRONO"
echo "== start  $(date) =="
t0=$(date +%s)

cd "$NEDM_REPO"
python scripts/evaluation/score_crm_tracking.py \
  --policy "$CKPT" \
  --index  "$NEDM_INDEX" \
  --out    "$OUT" \
  --concurrency "$CONC"
rc=$?

t1=$(date +%s)
echo "== rc=$rc  elapsed $((t1 - t0))s  end $(date) =="
exit $rc
