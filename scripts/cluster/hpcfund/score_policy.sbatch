#!/bin/bash
# Score CRM policies on the AMD HPC Fund cluster: ONE POLICY PER ARRAY TASK.
#
#   sbatch --array=0-$(( $(wc -l < $KWORK/nedm/policies.txt) - 1 ))%24 \
#          /work1/dannegrut/kyle/nedm/score_policy.sbatch
#
# Each task scores one policy on all 80 val episodes and writes
# $KWORK/nedm/out/crmtrack_<arm>_hpcfund.json, the same record schema the
# desktop fleet writes, with the same episode_id keys, so the two pair directly.
#
# WHY ONE POLICY PER NODE rather than splitting a policy's 80 episodes across
# several nodes. Nodes here are exclusive and billed whole, and an mi2101x node
# is exactly 1 MI210 + 16 cores, so the charged cost of a policy is
# (80 / concurrency) x per-episode-wall x 0.1, and that product does not change
# when the episodes are spread over K nodes: splitting divides the wall time by
# K and multiplies the billed nodes by K. What splitting does change is
# overhead, and only for the worse. Every node pays a fixed startup (module
# load, pychrono import, torch import, SLURM dispatch), so K shards pay it K
# times for zero throughput gain, and because per-episode cost varies by command
# family by roughly 10x (the reference run has episodes from 110 to 1075 rows)
# the shards finish raggedly. Splitting only buys latency on a single policy,
# which is the case the sweep is not in: with 45 arms and 24 nodes the queue is
# the throughput limit, not any one policy's wall time.
#
# mkindex.py --shard/--of exists for the other case, when one policy's turnaround
# matters more than its efficiency.
#
#SBATCH -J kyle-crmscore
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -t 02:30:00                # measured: 01:29:27 for 80 episodes at conc 8
#SBATCH -o /work1/dannegrut/kyle/logs/%x-%A_%a.out
#SBATCH -e /work1/dannegrut/kyle/logs/%x-%A_%a.err

set -o pipefail
source /work1/dannegrut/kyle/env.sh
module purge
module load gnu12 rocm
source /work1/dannegrut/kyle/nedm/env-scoring.sh

# Concurrent episode processes per node. MEASURED with bench_concurrency.sbatch
# on an MI210 (job 414035), episodes per charged node-hour:
#
#     conc  4 -> 522.8      conc  8 -> 515.3
#     conc 12 -> 513.2      conc 16 -> 514.3
#
# Flat. The single MI210 is already saturated at 4, so packing harder buys
# nothing and past 8 it costs slightly. The node-level counters agree: the
# 80-episode run held 16 cores at 8.6% mean utilization and 13.8% memory, so
# this workload is GPU-bound and the 16 cores are not the constraint. Anything
# in 4-16 performs the same; 8 is the middle of the flat region.
CONC=${CONC:-8}

POLICIES=${POLICIES:-$NEDM_ROOT/policies.txt}
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$POLICIES")
[ -z "$LINE" ] && { echo "FATAL: no policy at line $((SLURM_ARRAY_TASK_ID + 1)) of $POLICIES"; exit 1; }

ARM=$(basename "$LINE" .pt)
CKPT=$NEDM_CKPTS/$(basename "$LINE")
[ -f "$CKPT" ] || { echo "FATAL: missing checkpoint $CKPT"; exit 1; }
OUT=$NEDM_OUT/crmtrack_${ARM}_hpcfund.json

echo "== host $(hostname)  arm $ARM  task $SLURM_ARRAY_TASK_ID  conc $CONC =="
echo "== ckpt $CKPT"
echo "== start $(date) =="
t0=$(date +%s)

cd "$NEDM_REPO"
python scripts/evaluation/score_crm_tracking.py \
  --policy "$CKPT" \
  --index  "$NEDM_INDEX" \
  --out    "$OUT" \
  --concurrency "$CONC"
rc=$?

t1=$(date +%s)
el=$((t1 - t0))
# Charged node-hours: 1 node x elapsed x the mi2101x 0.1 factor. Printed per task
# so the sweep's real cost is measured from the logs, not estimated.
echo "== rc=$rc  elapsed ${el}s  charged $(awk -v s=$el 'BEGIN{printf "%.4f", s/3600*0.1}') node-hours  end $(date) =="
exit $rc
