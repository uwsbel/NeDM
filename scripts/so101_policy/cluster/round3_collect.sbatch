#!/bin/bash
#SBATCH -J r3-collect
#SBATCH -p mi2104x
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 02:30:00
#SBATCH -A dannegrut
#SBATCH -o /work1/dannegrut/harry/experiments/so101_policy_20261011/logs/%x_%j.out
# NRD data round 3 of the SO101 policy study (campaign policy_v3: MuJoCo rollouts of 7 PPO policies) on one AMD CPU
# node (mi2104x, 128 EPYC cores): task tables -> closed-loop policy rollouts in CPU MuJoCo (pass 1) -> the collector's
# labels (pass 2, round 2's code) -> the data directories -> the policies' closed loop on the first 256 pgate tasks
# (eval_policy.py, CPU MuJoCo) -> family shares and the merge production_plus + noisy_v1 + expert_v2 + policy_v3.
# Design: src/nedm/so101_policy/data_round.py (section "round 3"); report ~/so101_policy/docs/build6_round3.md.
#   rsync the code snapshot to $C (default $B/round3/code: git HEAD + the round-3 files + contact_nrd/so101_grasp)
#   and the checkpoints to $POL (default $B/round3/policies), then: sbatch scripts/so101_policy/cluster/round3_collect.sbatch
# Each step whose output exists is skipped (the same command resumes). Environment (optional): R3_CODE, R3_WORK
# (default $B/round3/work), R3_OUT (default $B/data/policy_v3), R3_POLICIES, N_CAMPAIGN (13000), N_PGATE (512), MJW_C
# (MuJoCo workers per campaign process, one process per policy: 10), MJW_G (per pgate process: 6), HELD_MIN (0.40).
set -u
B=/work1/dannegrut/harry/experiments/so101_policy_20261011
C=${R3_CODE:-$B/round3/code}
W=${R3_WORK:-$B/round3/work}
D=${R3_OUT:-$B/data/policy_v3}
POL=${R3_POLICIES:-$B/round3/policies}
NC=${N_CAMPAIGN:-13000}
NG=${N_PGATE:-512}
MJW_C=${MJW_C:-10}
MJW_G=${MJW_G:-6}
HELD_MIN=${HELD_MIN:-0.40}
BASE=/work1/dannegrut/harry/experiments/so101_nrd_fix_20261009/production_plus
NOISY=$B/data/noisy_v1
EXPERT=$B/data/expert_v2
module load pytorch/2.10.0
PY=/work1/dannegrut/harry/venvs/simsrc_mj/bin/python
export PYTHONPATH=$C/src:$C:${PYTHONPATH:-}
export NEDM_SO101_OVERLAP_CACHE=$B/mujoco_env/overlap_cache
export NEDM_SO101_POLICY_DIR=$POL
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
CR="$PY -P scripts/so101_policy/collect_round.py"
GEO="--geom $B/ppo2/data/task_geom_v2.npz --kin-points $B/ppo2/data/kin_points_v1.npz"
IDS="w5a w5b kCa kCb kCc nRc nRa"
cd "$C" || exit 1
mkdir -p "$W/ro_campaign" "$W/ro_pgate" "$W/logs" "$W/evalmj"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} start $(date -Is)"; lscpu | grep -E "Model name|^CPU\(s\)"; free -g | head -2
$PY -P -c "import sys, mujoco, numpy, scipy, torch; print(sys.version.split()[0], 'mujoco', mujoco.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'torch', torch.__version__)"
cat CODE_COMMIT 2>/dev/null
sha256sum src/nedm/so101_policy/data_round.py scripts/so101_policy/collect_round.py src/nedm/so101_policy/decoder.py \
  src/nedm/so101_policy/vec_env.py src/nedm/so101_policy/ppo.py scripts/so101_policy/eval_policy.py \
  src/nedm/so101_policy/mujoco_env.py "$POL"/*.pt $B/ppo2/data/task_geom_v2.npz $B/ppo2/data/kin_points_v1.npz
ls -la ~/so101_policy/ckpt/ 2>&1 | head -5     # EnvConfig defaults (eval_policy where a checkpoint's path is missing)
$PY -P -m unittest tests.so101_policy.test_data_round > "$W/logs/unittest.log" 2>&1 || { tail -n 30 "$W/logs/unittest.log"; echo "unit tests failed"; exit 1; }
tail -n 3 "$W/logs/unittest.log"
# 1. task tables (deterministic per row; sharded over a process pool)
[ -f "$W/tasks_campaign.npz" ] || { echo "tasks campaign $(date -Is)"; $CR ptasks --set campaign --n "$NC" --workers 64 --out "$W/tasks_campaign.npz"; }
[ -f "$W/tasks_pgate.npz" ] || { echo "tasks pgate $(date -Is)"; $CR ptasks --set pgate --n "$NG" --workers 32 --out "$W/tasks_pgate.npz"; }
# 2. pass 1: one campaign process and one pgate process per policy
echo "pass 1 $(date -Is)"
pids=()
for id in $IDS; do
  $CR prun --tasks "$W/tasks_campaign.npz" --policies "$id" --tag c --workers "$MJW_C" --threads 1 --batch 975 \
      --python "$PY" $GEO --out "$W/ro_campaign" > "$W/logs/prun_c_$id.log" 2>&1 &
  pids+=($!)
  $CR prun --tasks "$W/tasks_pgate.npz" --expand-pgate --policies "$id" --tag g --workers "$MJW_G" --threads 1 \
      --batch 1024 --python "$PY" $GEO --out "$W/ro_pgate" > "$W/logs/prun_g_$id.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
grep -h "episodes," "$W"/logs/prun_*.log
[ $status = 0 ] || { echo "pass 1 failed"; tail -n 5 "$W"/logs/prun_*.log; exit 1; }
# 3. pass 2: the collector's labels on one pool (pass 1 vs pass 2 checked per segment)
echo "pass 2 $(date -Is)"
$CR label --rollouts "$W/ro_campaign/ro_*.npz" "$W/ro_pgate/ro_*.npz" --workers 120 > "$W/logs/label.log" 2>&1 || { tail "$W/logs/label.log"; exit 1; }
grep "segments" "$W/logs/label.log"
# 4. the data directories (validated)
echo "write $(date -Is)"
if [ ! -f "$D/manifest.json" ]; then
  rm -rf "$D.tmp"
  $CR write --campaign policy_v3 --labelled "$W/ro_campaign/lab_*.pkl" --gate2 "$W/ro_pgate/lab_*.pkl" --gate-name pgate \
      --out "$D.tmp" --n-val 1000 || exit 1
  cp "$W/tasks_campaign.npz" "$W/tasks_pgate.npz" "$D.tmp/"
  mv -T "$D.tmp" "$D"
fi
du -sh "$D" "$D"/val_full "$D"/pgate
# 5. the policies' closed loop on the first 256 pgate tasks in CPU MuJoCo (eval_policy.py: the gate's MuJoCo side)
echo "eval mujoco $(date -Is)"
pids=()
for f in "$POL"/*.pt; do
  id=$(basename "$f" | cut -d_ -f1)
  [ -f "$W/evalmj/${id}_mj256.npz" ] && continue
  $PY -P scripts/so101_policy/eval_policy.py run --policy "$f" --tasks "$W/tasks_pgate.npz" --n 256 --world mujoco \
      --workers 16 --mj-python "$PY" --threads 1 --out "$W/evalmj/${id}_mj256.npz" > "$W/logs/evalmj_$id.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
tail -q -n 1 "$W"/logs/evalmj_*.log
# 6. sampling shares of the merge and its repeat factor for policy_v3 (smallest of 1, 1.5, 2, 2.5, 3 with >= HELD_MIN
#    of the held-cube draws), then the merge (collect_round.py merge: merge_data.py's rules + repeated split-0 rows)
echo "shares $(date -Is)"
[ -f "$W/pick_repeat.json" ] || $CR pick_repeat --folders "$BASE" "$NOISY" "$EXPERT" "$D" --vary "$D" \
    --held-min "$HELD_MIN" --out "$W/pick_repeat.json" | tee "$W/logs/pick_repeat.log"
K=$($PY -c "import json; print(json.load(open('$W/pick_repeat.json'))['choice'])")
M=$B/data/merged_r3_pp_noisy_expert_policy_x$K
echo "merge with policy_v3 x$K -> $M $(date -Is)"
[ -f "$M/system.json" ] || $CR merge --base "$BASE" --extra "$NOISY" "$EXPERT" "$D" --repeat "$D=$K" --out "$M" || exit 1
$CR shares --folders "$BASE" "$NOISY" "$EXPERT" "$D" --repeat "$D=$K" --out "$W/shares_r3_x$K.json"
du -sh "$M"
echo "done $(date -Is)"
