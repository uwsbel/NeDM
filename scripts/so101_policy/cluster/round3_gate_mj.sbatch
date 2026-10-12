#!/bin/bash
#SBATCH -J r3-gate-mj
#SBATCH -p mi2104x
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 01:00:00
#SBATCH -A dannegrut
#SBATCH -o /work1/dannegrut/harry/experiments/so101_policy_20261011/logs/%x_%j.out
# NRD data round 3 of the SO101 policy study, policy-in-NRD gate: the CPU MuJoCo side (ground truth) on one AMD CPU
# node (mi2104x, 128 EPYC cores). Deterministic, eval_policy.py run --world mujoco (the decoder and observation from
# each checkpoint, evaluation rules, 15 mm / 15 deg):
#   the 3 competent teachers (w5a = w5_dc model_00121, kCa = kC model_00060, w5b = w5_dc_s2 model_00123) on all 512
#   pgate tasks (policy_v3/tasks_pgate.npz, seeds 410,000-410,511) and on the first 256 gate2 tasks
#   (expert_v2/tasks_gate2.npz, seeds 400,000-400,255);
#   the FSM expert (gate_expert.py run, config base, decoder v2 default) on the 1,024 gate tasks (seeds 1000-2023).
# Outputs $B/round3/work/gate/mj/. A step whose output exists is skipped. Report ~/so101_policy/docs/build6_round3.md.
#   cd $B/round3/code && sbatch scripts/so101_policy/cluster/round3_gate_mj.sbatch
set -u
B=/work1/dannegrut/harry/experiments/so101_policy_20261011
C=${R3_CODE:-$B/round3/code}
G=$B/round3/work/gate
POL=${R3_POLICIES:-$B/round3/policies}
TP=$B/data/policy_v3/tasks_pgate.npz
T2=$B/data/expert_v2/tasks_gate2.npz
TG=$B/gate/tasks_1024.npz
module load pytorch/2.10.0
PY=/work1/dannegrut/harry/venvs/simsrc_mj/bin/python
export PYTHONPATH=$C/src:$C:${PYTHONPATH:-}
export NEDM_SO101_OVERLAP_CACHE=$B/mujoco_env/overlap_cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$C" || exit 1
mkdir -p "$G/mj" "$G/logs"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} start $(date -Is)"; lscpu | grep -E "Model name|^CPU\(s\)"
$PY -P -c "import sys, mujoco, numpy, torch; print(sys.version.split()[0], 'mujoco', mujoco.__version__, 'numpy', numpy.__version__, 'torch', torch.__version__)"
sha256sum scripts/so101_policy/eval_policy.py scripts/so101_policy/gate_expert.py src/nedm/so101_policy/decoder.py \
  src/nedm/so101_policy/vec_env.py src/nedm/so101_policy/mujoco_env.py src/nedm/so101_policy/expert.py \
  "$POL"/w5a_*.pt "$POL"/kCa_*.pt "$POL"/w5b_*.pt "$TP" "$T2" "$TG"
pids=()
for id in w5a kCa w5b; do
  f=$(ls "$POL"/${id}_*.pt)
  out=$G/mj/${id}_pgate512_mj.npz
  [ -f "$out" ] || { $PY -P scripts/so101_policy/eval_policy.py run --policy "$f" --tasks "$TP" --world mujoco \
      --workers 20 --mj-python "$PY" --threads 1 --out "$out" > "$G/logs/mj_${id}_pgate512.log" 2>&1 & pids+=($!); }
  out=$G/mj/${id}_gate2_256_mj.npz
  [ -f "$out" ] || { $PY -P scripts/so101_policy/eval_policy.py run --policy "$f" --tasks "$T2" --n 256 --world mujoco \
      --workers 12 --mj-python "$PY" --threads 1 --out "$out" > "$G/logs/mj_${id}_gate2_256.log" 2>&1 & pids+=($!); }
done
out=$G/mj/fsm/v2_base_mujoco.npz
mkdir -p "$G/mj/fsm"
[ -f "$out" ] || { $PY -P scripts/so101_policy/gate_expert.py run --tasks "$TG" --world mujoco --workers 20 --threads 1 \
    --config base --decoder v2 --decoder-cfg '{}' --decoder-label v2 --out "$out" > "$G/logs/mj_fsm_v2_base.log" 2>&1 & pids+=($!); }
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
tail -q -n 1 "$G"/logs/mj_*.log
echo "done $(date -Is) exit $status"
exit $status
