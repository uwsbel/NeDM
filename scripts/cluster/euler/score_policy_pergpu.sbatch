#!/bin/bash
# Score ONE policy across the GPUs of a single Euler node: one episode per GPU,
# not N episodes time-slicing one.
#
#   sbatch --export=ALL,ARM=go2_cts_150k scripts/cluster/euler/score_policy_pergpu.sbatch
#
# WHY THIS EXISTS, and why score_policy.sbatch's --concurrency knob is not it.
# A single CRM episode SATURATES the GPU. Measured on euler19's A100 with
# --concurrency 6: nvidia-smi reported 100% utilization and 6.0 GB in use, about
# 1 GB per episode, and the episode processes ran roughly 6x their solo wall
# time. hpcfund measured the same shape on an MI210 -- episodes per charged
# node-hour flat from concurrency 4 to 16 -- and concluded the GPU was already
# saturated. So raising --concurrency on one GPU buys nothing: it time-slices
# work that was already serial.
#
# That conclusion is load-bearing on hpcfund, where a node is 1 MI210 and is
# billed whole, so one policy per node is the only sensible unit. It is NOT
# load-bearing on Euler, where the GPU-dense nodes carry EIGHT GPUs each
# (euler01-04 and 09-10 are 8x RTX 4000 Ada, euler05-08 are 8x RTX A4500). There
# the unit that matches the hardware is one episode per GPU, which is what this
# script does: M shards, M scorer processes, each pinned to its own device with
# CUDA_VISIBLE_DEVICES and each running --concurrency 1.
#
# Memory is not the constraint -- an episode holds about 1 GB and the smallest
# card here has 20 GB -- so the GPU count is the only thing to size against.
#
# NOTE this requires a Chrono built for the target architecture. The RTX 4000 Ada
# is sm_89 and the A4500 is sm_86, and NEITHER is in Chrono's default
# CHRONO_CUDA_ARCHITECTURES (all-major: 60;70;80;90;100;120). ~/chrono-build
# cannot run on these nodes at all; ~/chrono-build-parsers can, because
# build_parsers.sbatch sets the variable Chrono actually reads. See
# docs/state/machines/euler.md.
#
# Partition is `research`, not `sbel`: the 8-GPU nodes are not in `sbel`. That
# means tier 200 and preemption WITH REQUEUE by anyone's `sbel` job. A requeue
# restarts this from zero -- the shards are not checkpointed -- so keep the
# request short and re-run rather than trying to make it survive.
#
#SBATCH -J kyle-crmscore-pergpu
#SBATCH -p research
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -o /srv/home/kasha2/logs/%x-%j.out
#SBATCH -e /srv/home/kasha2/logs/%x-%j.err

set -o pipefail
module load gcc/12.2.0 nvidia/cuda/12.9.1
source /srv/home/kasha2/nedm/env-scoring.sh

ARM=${ARM:-go2_cts_150k}
CKPT=$NEDM_CKPTS/$ARM.pt
[ -f "$CKPT" ] || { echo "FATAL: missing checkpoint $CKPT"; exit 1; }

# One shard per allocated GPU. SLURM remaps CUDA_VISIBLE_DEVICES to the GPUs it
# gave this job, so indices 0..M-1 here are this job's own devices whatever their
# physical numbering.
NGPU=$(nvidia-smi --list-gpus | wc -l)
M=${M:-$NGPU}
SHARDDIR=$NEDM_ROOT/idx/pergpu-$SLURM_JOB_ID
OUTDIR=$NEDM_OUT/pergpu-$SLURM_JOB_ID
# OUT is overridable so a throughput trial on a cut-down index cannot land on
# top of a real scoring run's record file and quietly replace it.
OUT=${OUT:-$NEDM_OUT/crmtrack_${ARM}_euler.json}
mkdir -p "$SHARDDIR" "$OUTDIR"

echo "== host $(hostname)  arm $ARM  gpus $NGPU  shards $M =="
nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader
echo "== start $(date) =="
t0=$(date +%s)

cd "$NEDM_REPO"
for k in $(seq 0 $((M - 1))); do
  python scripts/cluster/mkindex.py --index "$NEDM_INDEX" \
      --out "$SHARDDIR/shard_$k.json" --shard "$k" --of "$M" || exit 1
done

for k in $(seq 0 $((M - 1))); do
  CUDA_VISIBLE_DEVICES=$k python scripts/evaluation/score_crm_tracking.py \
      --policy "$CKPT" \
      --index  "$SHARDDIR/shard_$k.json" \
      --out    "$OUTDIR/shard_$k.json" \
      --concurrency 1 > "$OUTDIR/shard_$k.log" 2>&1 &
done
wait
rc=$?

t1=$(date +%s)
echo "== all shards done in $((t1 - t0))s =="
for k in $(seq 0 $((M - 1))); do
  echo "-- shard $k: $(tail -3 "$OUTDIR/shard_$k.log" | tr '\n' ' ')"
done

NEPS=$(python -c "import json,sys; print(len(json.load(open(sys.argv[1]))['episodes']))" "$NEDM_INDEX")
python scripts/cluster/merge_score_shards.py \
    --out "$OUT" --expect "$NEPS" \
    "$OUTDIR"/shard_*.json
rc=$?
echo "== rc=$rc  total $((`date +%s` - t0))s  end $(date) =="
exit $rc
