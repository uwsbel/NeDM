#!/bin/bash
#SBATCH -J r3-gate-nrd
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 02:00:00
#SBATCH -o /work1/dannegrut/harry/experiments/so101_policy_20261011/logs/%x_%j.out
# NRD data round 3 of the SO101 policy study, policy-in-NRD gate: the NRD side on one MI350X, one seed per job
# (SEED=61 or 62), after the seed's fine-tune (runs/nrd_ft_r3_s<seed>/complete.json). Models: r3best = best.pt,
# r3last = last.pt, r2 = runs/nrd_ft_r2_s<seed>/best.pt (the init, again on this GPU for paired before / after numbers);
# r0 = the original final model (forgetting rows only).
#   1. the 3 competent teachers (w5a, kCa, w5b; deterministic, eval_policy.py run --world nrd) on the first 256 gate2
#      tasks and on all 512 pgate tasks;
#   2. the FSM expert gate (gate_expert.py run, config base, decoder v2 default) on the 1,024 gate tasks;
#   3. no-forgetting: eval_nrd_noisy.py on production val (7 task families), production probes (5 families) and
#      noisy_v1 val, models r0 / r2 / r3best / r3last (one file each, model name s<seed>), and --compare r2 -> r3;
#   4. reports against the CPU MuJoCo runs of round3_gate_mj.sbatch ($B/round3/work/gate/mj) where they exist.
# Outputs $B/round3/work/gate/s<seed>/. A step whose output exists is skipped (the same command resumes).
#   cd $B/round3/code && sbatch --export=ALL,SEED=61 scripts/so101_policy/cluster/round3_gate_nrd.sbatch
set -euo pipefail
set +u; source /etc/profile; module load pytorch/2.10.0; source /work1/dannegrut/harry/venvs/nedm/bin/activate; set -u
B=/work1/dannegrut/harry/experiments/so101_policy_20261011
C=${R3_CODE:-$B/round3/code}
SEED=${SEED:-61}
RUN=$B/runs/nrd_ft_r3_s$SEED
R2=$B/runs/nrd_ft_r2_s$SEED
R0=/work1/dannegrut/harry/experiments/so101_nrd_fix_20261009/runs/short_rel_rk_cc_fix_s${SEED}_plus
POL=${R3_POLICIES:-$B/round3/policies}
TP=$B/data/policy_v3/tasks_pgate.npz
T2=$B/data/expert_v2/tasks_gate2.npz
TG=$B/gate/tasks_1024.npz
NOISY=$B/data/noisy_v1
BASE=/work1/dannegrut/harry/experiments/so101_nrd_fix_20261009/production_plus
MJ=$B/round3/work/gate/mj
G=$B/round3/work/gate/s$SEED
export OMP_NUM_THREADS=4 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 PYTHONPATH="$C/src:${PYTHONPATH:-}"
cd "$C"
[ -f "$RUN/complete.json" ] || { echo "no $RUN/complete.json"; exit 1; }
mkdir -p "$G/fsm" "$G/forget" "$G/logs"
declare -A CK=([r3best]=$RUN/best.pt [r3last]=$RUN/last.pt [r2]=$R2/best.pt [r0]=$R0/best.pt)
echo "host $(hostname) job ${SLURM_JOB_ID:-none} seed $SEED start $(date -Is)"
sha256sum scripts/so101_policy/eval_policy.py scripts/so101_policy/gate_expert.py scripts/so101_policy/eval_nrd_noisy.py \
  "${CK[r3best]}" "${CK[r3last]}" "${CK[r2]}" "${CK[r0]}" "$POL"/w5a_*.pt "$POL"/kCa_*.pt "$POL"/w5b_*.pt "$TP" "$T2" "$TG"
EP="python3.12 -P scripts/so101_policy/eval_policy.py"
for m in r3best r3last r2; do
  for id in w5a kCa w5b; do
    f=$(ls "$POL"/${id}_*.pt)
    out=$G/${id}_gate2_256_$m.npz
    [ -f "$out" ] || $EP run --policy "$f" --tasks "$T2" --n 256 --world nrd --ckpt "${CK[$m]}" --device cuda --out "$out" | tail -n 1
  done
  out=$G/fsm/v2_base_nrd_${m}_s$SEED.npz
  [ -f "$out" ] || python3.12 -P scripts/so101_policy/gate_expert.py run --tasks "$TG" --world nrd --ckpt "${CK[$m]}" \
      --config base --decoder v2 --decoder-cfg '{}' --decoder-label v2 --device cuda --threads 2 --out "$out" \
      > "$G/logs/fsm_$m.log" 2>&1
  tail -n 1 "$G/logs/fsm_$m.log"
  for id in w5a kCa w5b; do
    f=$(ls "$POL"/${id}_*.pt)
    out=$G/${id}_pgate512_$m.npz
    [ -f "$out" ] || $EP run --policy "$f" --tasks "$TP" --world nrd --ckpt "${CK[$m]}" --device cuda --out "$out" | tail -n 1
  done
done
# no-forgetting sets (round 1 / 2: eval_nrd_noisy.py, split 1)
for m in r3best r3last r2 r0; do
  o=$G/forget/${m}_noisy_v1_val.json
  [ -f "$o" ] || python3.12 -P scripts/so101_policy/eval_nrd_noisy.py --data "$NOISY" --model "s$SEED=${CK[$m]}" --out "$o" > "$G/logs/forget_${m}_noisy.log" 2>&1
  o=$G/forget/${m}_production_val.json
  [ -f "$o" ] || python3.12 -P scripts/so101_policy/eval_nrd_noisy.py --data "$BASE" \
      --families nominal off_centre one_sided near_miss weak_closure drop push \
      --model "s$SEED=${CK[$m]}" --out "$o" > "$G/logs/forget_${m}_production.log" 2>&1
  o=$G/forget/${m}_production_probes_val.json
  [ -f "$o" ] || python3.12 -P scripts/so101_policy/eval_nrd_noisy.py --data "$BASE" \
      --families free_arm cube_flight cube_table jaw_fixed jaw_moving \
      --model "s$SEED=${CK[$m]}" --out "$o" > "$G/logs/forget_${m}_probes.log" 2>&1
done
for set in noisy_v1_val production_val production_probes_val; do
  for m in r3best r3last; do
    python3.12 -P scripts/so101_policy/eval_nrd_noisy.py --compare "r2=$G/forget/r2_$set.json" "$m=$G/forget/${m}_$set.json" \
        --out "$G/forget/compare_r2_${m}_$set.json" > "$G/forget/compare_r2_${m}_$set.txt" 2>&1 || true
  done
  python3.12 -P scripts/so101_policy/eval_nrd_noisy.py --compare "r0=$G/forget/r0_$set.json" "r3best=$G/forget/r3best_$set.json" \
      --out "$G/forget/compare_r0_r3best_$set.json" > "$G/forget/compare_r0_r3best_$set.txt" 2>&1 || true
done
grep -h "FAIL" "$G"/forget/compare_r2_r3best_*.txt | head -n 20 || true
# reports against CPU MuJoCo (where the mi2104x job has written them)
for id in w5a kCa w5b; do
  for s in gate2_256 pgate512; do
    ref=$MJ/${id}_${s}_mj.npz
    [ -f "$ref" ] || { echo "no $ref yet"; continue; }
    $EP report --runs "$ref" "$G/${id}_${s}_r3best.npz" "$G/${id}_${s}_r3last.npz" "$G/${id}_${s}_r2.npz" --ref "$ref" \
        --out "$G/report_${id}_$s.json" --md "$G/report_${id}_$s.md" > /dev/null || true
  done
done
if [ -f "$MJ/fsm/v2_base_mujoco.npz" ]; then
  ln -sf "$MJ/fsm/v2_base_mujoco.npz" "$G/fsm/v2_base_mujoco.npz"
  python3.12 -P scripts/so101_policy/gate_expert.py report --tasks "$TG" --runs "$G/fsm" --out "$G/fsm/report.json" > /dev/null || true
fi
echo "done $(date -Is)"
