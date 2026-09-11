#!/bin/bash
# How many concurrent CRM episodes fit on one MI210, measured rather than guessed.
#
#   sbatch /work1/dannegrut/kyle/nedm/bench_concurrency.sbatch
#
# All concurrent episodes share the node's single GPU, so the ceiling is GPU
# memory and occupancy, not the 16 cores. Nodes are billed whole, so the number
# that decides everything is episodes per charged node-hour: raising concurrency
# is free until the GPU saturates and then it stops helping. MI210 is a
# different part from the desktop fleet's card, so the fleet's concurrency 8 is
# not evidence about this machine.
#
# The per-episode wall time at concurrency 1 comes from the devel smoke test, so
# this sweeps only the packed settings: 4, 8, 12, 16.
#
# ONE EPISODE PER COMMAND FAMILY, taken from scenario_family. Per-episode cost
# varies by roughly 10x across families (the reference run has episodes from 110
# to 1075 rows), so a bench on an arbitrary slice would measure the slice, not
# the workload. Do NOT derive the family from the filename: the stem is
# go2_crm_s9300000_arc_019, so splitting on the last underscore yields
# "go2_crm_s9300000_arc" and silently gives one episode per (seed, family) pair.
#
#SBATCH -J kyle-crmbench
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -o /work1/dannegrut/kyle/logs/%x-%j.out
#SBATCH -e /work1/dannegrut/kyle/logs/%x-%j.err

set -o pipefail
source /work1/dannegrut/kyle/env.sh
module purge
module load gnu12 rocm
source /work1/dannegrut/kyle/nedm/env-scoring.sh

echo "== host $(hostname)  start $(date) =="
echo "== chrono: $NEDM_CHRONO"
rocm-smi --showproductname 2>&1 | grep -i "card series" | head -2

python - <<'PY'
import json, os
d = json.load(open(os.environ["NEDM_INDEX"]))
eps = [e for e in d["episodes"] if e.get("split") == "val"]
seen, pick = set(), []
for e in eps:
    fam = e.get("scenario_family") or "unknown"
    if fam not in seen:
        seen.add(fam); pick.append(e)
out = os.environ["NEDM_ROOT"] + "/idx/bench.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"episodes": pick}, open(out, "w"), indent=1)
print("bench set: %d episodes" % len(pick))
for f in sorted(seen):
    print("   %s" % f)
PY

N=$(python -c "import json,os;print(len(json.load(open(os.environ['NEDM_ROOT']+'/idx/bench.json'))['episodes']))")
cd "$NEDM_REPO"
echo "conc,wall_s,scored,of,ep_per_charged_node_hour" > "$NEDM_OUT/bench.csv"
for C in 4 8 12 16; do
  echo "---- concurrency $C ----"
  t0=$(date +%s)
  python scripts/evaluation/score_crm_tracking.py \
    --policy "$NEDM_CKPTS/go2_cts_150k.pt" \
    --index  "$NEDM_ROOT/idx/bench.json" \
    --out    "$NEDM_OUT/bench_c${C}.json" \
    --concurrency "$C" 2>&1 | tail -3
  t1=$(date +%s); el=$((t1 - t0))
  n=$(python -c "import json;print(sum(1 for r in json.load(open('$NEDM_OUT/bench_c${C}.json')) if 'mae_vx' in r))")
  rate=$(awk -v s=$el -v n=$n 'BEGIN{if(s>0)printf "%.1f", n/(s/3600*0.1); else print "na"}')
  echo "conc=$C  wall=${el}s  scored=$n/$N  ep_per_charged_node_hour=$rate"
  echo "$C,$el,$n,$N,$rate" >> "$NEDM_OUT/bench.csv"
done
echo "== summary =="
cat "$NEDM_OUT/bench.csv"
echo "== end $(date) =="
