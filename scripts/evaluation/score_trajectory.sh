#!/bin/bash
# Score a fine-tune's CHECKPOINT TRAJECTORY in Chrono, not just its selected point.
#
#   score_trajectory.sh <run_dir> [n_iterates] [n_episodes]
#
# Every stop rule in this codebase -- fixed ||dW||, best surrogate-internal reward,
# fixed budget -- picks one iterate using an instrument that cannot see Chrono. This
# scores a spread of iterates so the stop criterion becomes a measurement: if the curve
# is still falling at the last one the budget was too short, if it bottoms early the
# stop rule is late, and if it is flat the rule never mattered.
#
# Episodes default to 24, not the full 80: this is a SHAPE measurement across many
# iterates, and the winner is then re-scored on the full split. Spending 80 episodes
# per iterate here would cost more than the fine-tune it is auditing by two orders.
set -u
RUN=${1:?run dir, e.g. /home/kyle/sbel-artifacts/finetune_crm_t_traj}
N=${2:-8}
EPS=${3:-24}
REPO=$(for r in /home/kyle/Documents/sbel/NeDM /home/kyle/sbel/NeDM; do [ -d "$r/scripts" ] && echo "$r" && break; done)
PY=/home/kyle/miniconda3/envs/nedm/bin/python
BASE=/home/kyle/sbel-artifacts/checkpoints/go2_cts_150k.pt
cd "$REPO"; export PYTHONPATH=$PWD/src NEDM_REPO=$PWD

mapfile -t ALL < <(ls "$RUN"/traj/u*.pt 2>/dev/null | sort)
[ ${#ALL[@]} -eq 0 ] && { echo "FATAL: no iterates in $RUN/traj"; exit 1; }
# even spread over the run, always including the first and last
PICK=()
for i in $(seq 0 $((N-1))); do
  idx=$(( i * (${#ALL[@]}-1) / (N>1 ? N-1 : 1) ))
  [ "${PICK[*]-}" = "" ] || [ "${PICK[-1]}" != "${ALL[$idx]}" ] && PICK+=("${ALL[$idx]}")
done
echo "trajectory: ${#ALL[@]} iterates saved, scoring ${#PICK[@]} on $EPS episodes each"

OUT=$RUN/traj_scores; mkdir -p "$OUT"
for ck in "${PICK[@]}"; do
  u=$(basename "$ck" .pt)
  ts=$OUT/$u.ts.pt
  res=$OUT/crmtrack_$u.json
  [ -f "$res" ] && { echo "  $u already scored"; continue; }
  $PY scripts/evaluation/export_finetuned_policy.py --base "$BASE" --ckpt "$ck" --out "$ts" >/dev/null || { echo "  $u EXPORT FAILED"; continue; }
  echo "  scoring $u ..."
  $PY scripts/evaluation/score_crm_tracking.py --policy "$ts" --out "$res" --limit "$EPS" 2>&1 | tail -2
done

echo; echo "=== trajectory curve ==="
$PY - "$OUT" <<'PY'
import json, glob, os, sys, statistics as st
d = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(d, "crmtrack_u*.json"))):
    recs = [r for r in json.load(open(f)) if r.get("mae_vx") is not None]
    if not recs: continue
    u = int(os.path.basename(f)[len("crmtrack_u"):-len(".json")])
    rows.append((u, st.mean(r["mae_vx"] for r in recs), len(recs)))
if not rows:
    print("  nothing scored"); raise SystemExit
b = max(r[1] for r in rows)
print(f"  {'update':>7} {'mae_vx':>8} {'n':>4}")
for u, m, n in rows:
    print(f"  {u:7d} {m:8.4f} {n:4d}  {'#' * max(1, round(46 * m / b))}")
lo = min(rows, key=lambda r: r[1])
print(f"\n  best iterate: update {lo[0]}  mae_vx {lo[1]:.4f}")
print("  last iterate:  update {}  mae_vx {:.4f}".format(*rows[-1][:2]))
print("  => budget was TOO SHORT" if lo[0] == rows[-1][0] else
      "  => run peaked at {} of {} updates".format(lo[0], rows[-1][0]))
PY
