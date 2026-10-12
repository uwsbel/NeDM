#!/bin/bash
#SBATCH -J r2-collect
#SBATCH -p mi2104x
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 01:30:00
#SBATCH -A dannegrut
#SBATCH -o /work1/dannegrut/harry/experiments/so101_policy_20261011/logs/%x_%j.out
# NRD data round 2 of the SO101 policy study (campaign expert_v2) on one AMD CPU node (mi2104x, 128 EPYC cores):
# task tables -> closed-loop expert rollouts in CPU MuJoCo (pass 1) -> the collector's labels (pass 2) -> the data
# directories. Design: src/nedm/so101_policy/data_round.py; report ~/so101_policy/docs/build3_round2.md.
#   rsync the code snapshot to $C (default $B/round2/code: git HEAD + the round-2 files + contact_nrd/so101_grasp),
#   then: sbatch scripts/so101_policy/cluster/round2_collect.sbatch
# Each step whose output exists is skipped (the same command resumes). Environment: R2_CODE (code dir), R2_WORK
# (scratch, default $B/round2/work), R2_OUT (data dir, default $B/data/expert_v2), N_CAMPAIGN (13000), N_GATE2 (1024),
# NPROC (pass-1 processes for the campaign, default 10; one more runs gate2), MJW (MuJoCo workers per process, 10).
# Outputs: $R2_OUT (training folder: unified_data.npz, system.json, manifest.json, episodes.npz, validate.json),
# $R2_OUT/val_full, $R2_OUT/gate2 (full-length evaluation folders), $R2_OUT/tasks_{campaign,gate2}.npz.
set -u
B=/work1/dannegrut/harry/experiments/so101_policy_20261011
C=${R2_CODE:-$B/round2/code}
W=${R2_WORK:-$B/round2/work}
D=${R2_OUT:-$B/data/expert_v2}
NC=${N_CAMPAIGN:-13000}
NG=${N_GATE2:-1024}
NPROC=${NPROC:-10}
MJW=${MJW:-10}
module load pytorch/2.10.0
PY=/work1/dannegrut/harry/venvs/simsrc_mj/bin/python
export PYTHONPATH=$C/src:$C:${PYTHONPATH:-}
export NEDM_SO101_OVERLAP_CACHE=$B/mujoco_env/overlap_cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
CR="$PY -P scripts/so101_policy/collect_round.py"
cd "$C" || exit 1
mkdir -p "$W/ro_campaign" "$W/ro_gate2" "$W/logs"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} start $(date -Is)"; lscpu | grep -E "Model name|^CPU\(s\)"; free -g | head -2
$PY -P -c "import sys, mujoco, numpy, scipy, torch; print(sys.version.split()[0], 'mujoco', mujoco.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'torch', torch.__version__)"
cat CODE_COMMIT 2>/dev/null
sha256sum src/nedm/so101_policy/data_round.py scripts/so101_policy/collect_round.py src/nedm/so101_policy/decoder.py \
  src/nedm/so101_policy/expert.py scripts/so101_policy/gate_expert.py src/nedm/so101_policy/mujoco_env.py
$PY -P -m unittest tests.so101_policy.test_data_round 2>&1 | tail -3
# 1. task tables (deterministic per parent; sharded over a process pool)
[ -f "$W/tasks_campaign.npz" ] || { echo "tasks campaign $(date -Is)"; $CR tasks --set campaign --n "$NC" --workers 64 --out "$W/tasks_campaign.npz"; }
[ -f "$W/tasks_gate2.npz" ] || { echo "tasks gate2 $(date -Is)"; $CR tasks --set gate2 --n "$NG" --workers 32 --out "$W/tasks_gate2.npz"; }
# 2. pass 1: NPROC strided campaign slices (every family in each) + the gate2 set, MJW MuJoCo workers each
echo "pass 1 $(date -Is)"
pids=()
for k in $(seq 0 $((NPROC - 1))); do
  $CR run --tasks "$W/tasks_campaign.npz" --stride "$k/$NPROC" --tag c --workers "$MJW" --threads 1 \
      --out "$W/ro_campaign" > "$W/logs/run_c$k.log" 2>&1 &
  pids+=($!)
done
$CR run --tasks "$W/tasks_gate2.npz" --tag g --workers "$MJW" --threads 1 --out "$W/ro_gate2" > "$W/logs/run_g.log" 2>&1 &
pids+=($!)
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
tail -n 2 "$W"/logs/run_*.log
[ $status = 0 ] || { echo "pass 1 failed"; exit 1; }
# 3. pass 2: the collector's labels on one pool (pass 1 vs pass 2 checked per segment)
echo "pass 2 $(date -Is)"
$CR label --rollouts "$W/ro_campaign/ro_*.npz" "$W/ro_gate2/ro_*.npz" --workers 120 > "$W/logs/label.log" 2>&1 || { tail "$W/logs/label.log"; exit 1; }
grep "segments" "$W/logs/label.log" | tail -n 40
# 4. the data directories (validated)
echo "write $(date -Is)"
if [ ! -f "$D/manifest.json" ]; then
  rm -rf "$D.tmp"
  $CR write --labelled "$W/ro_campaign/lab_*.pkl" --gate2 "$W/ro_gate2/lab_*.pkl" --out "$D.tmp" --n-val 1000 || exit 1
  cp "$W/tasks_campaign.npz" "$W/tasks_gate2.npz" "$D.tmp/"
  mv -T "$D.tmp" "$D"
fi
du -sh "$D" "$D"/val_full "$D"/gate2
echo "done $(date -Is)"
