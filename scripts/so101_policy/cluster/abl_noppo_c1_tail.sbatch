#!/bin/bash
#SBATCH -J abl-c1-tail
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 02:30:00
#SBATCH -o /work1/dannegrut/harry/experiments/so101_policy_20261011/abl_noppo/baseline/logs/%x_%j.out
# No-PPO-data ablation, baseline stage (plan ~/so101_policy/docs/abl_noppo_plan.md sec. 1): the evaluation tail of
# c1_ft.sbatch for the EXISTING c1 NRD (its own tail was cancelled on 10-12 at 19:36 before any output), so that the
# c1np tail has a reference measured the same way. Same code snapshot ($B/c1/code), same commands, same models
# (c0 = sibling geo_cs best.pt, c1best, c1last), same node type (mi3501x). Only the output folder differs: the c1 run
# folders are read-only, so everything goes to $B/abl_noppo/baseline/nrd_tail/c1_s<seed>/.
# Not here: `eval_policy.py report` (needs mujoco, which the AMD nedm venv lacks: round-3 job 460875 died there with
# exit 1); the reports are made on luffy from the copied npz files (same script, same evalmj references).
#   sbatch --export=ALL,SEED=62 abl_noppo_c1_tail.sbatch
set -euo pipefail
set +u; source /etc/profile; module load pytorch/2.10.0; source /work1/dannegrut/harry/venvs/nedm/bin/activate; set -u
B=/work1/dannegrut/harry/experiments/so101_policy_20261011
C=${SP_CODE:-$B/c1/code}
P=$B/data/policy_v3
EXPERT=$B/data/expert_v2
NOISY=$B/data/noisy_v1
BASE=/work1/dannegrut/harry/experiments/so101_nrd_fix_20261009/production_plus
POL=$B/round3/policies
seed=${SEED:?SEED}
RUN=$B/runs/nrd_ft_c1_s$seed                       # read-only input
OUT=$B/abl_noppo/baseline/nrd_tail/c1_s$seed
R2=/work1/dannegrut/harry/experiments/so101_nrd_fix_20261009/runs/full_ns_rk_cc_geo_cs_s${seed}_plus
export OMP_NUM_THREADS=4 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 PYTHONPATH="$C/src:${PYTHONPATH:-}"
export HIP_VISIBLE_DEVICES=0
mkdir -p "$OUT/gate_closed"
cd "$C"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} seed $seed code $C start $(date -Is)"
sha256sum "$R2/best.pt" "$RUN/best.pt" "$RUN/last.pt" scripts/so101_policy/collect_round.py \
  scripts/so101_policy/eval_nrd_noisy.py scripts/so101_policy/eval_policy.py "$POL"/*.pt | tee "$OUT/sha256.txt"
ls -la --time-style=full-iso "$RUN" "$R2" > "$OUT/inputs_before.txt"
MODELS="c0=$R2/best.pt c1best=$RUN/best.pt c1last=$RUN/last.pt"
for set in pgate val_full; do
  if [ ! -f "$OUT/c1_policy_$set.json" ]; then
    echo "policy_v3 $set s$seed $(date -Is)"
    python3.12 -P scripts/so101_policy/collect_round.py eval --data "$P/$set" --device cuda --carry \
      --model $MODELS --out "$OUT/c1_policy_$set.json"
  fi
done
for set in val_full gate2; do
  if [ ! -f "$OUT/c1_expert_$set.json" ]; then
    echo "expert_v2 $set s$seed $(date -Is)"
    python3.12 -P scripts/so101_policy/collect_round.py eval --data "$EXPERT/$set" --device cuda --carry \
      --model $MODELS --out "$OUT/c1_expert_$set.json"
  fi
done
if [ ! -f "$OUT/after_noisy_v1_val.json" ]; then
  echo "after noisy_v1 $(date -Is)"
  python3.12 -P scripts/so101_policy/eval_nrd_noisy.py --data "$NOISY" --model "s$seed=$RUN" --out "$OUT/after_noisy_v1_val.json"
fi
if [ ! -f "$OUT/after_production_val.json" ]; then
  echo "after production $(date -Is)"
  python3.12 -P scripts/so101_policy/eval_nrd_noisy.py --data "$BASE" \
    --families nominal off_centre one_sided near_miss weak_closure drop push \
    --model "s$seed=$RUN" --out "$OUT/after_production_val.json"
fi
if [ ! -f "$OUT/after_production_probes_val.json" ]; then
  echo "after probes $(date -Is)"
  python3.12 -P scripts/so101_policy/eval_nrd_noisy.py --data "$BASE" \
    --families free_arm cube_flight cube_table jaw_fixed jaw_moving \
    --model "s$seed=$RUN" --out "$OUT/after_production_probes_val.json"
fi
# the policy-in-NRD gate (closed loop, first 256 pgate tasks); the MuJoCo side is $B/round3/work/evalmj/<id>_mj256.npz
for m in best last; do
  for f in "$POL"/*.pt; do
    id=$(basename "$f" | cut -d_ -f1)
    out="$OUT/gate_closed/${id}_c1${m}_nrd256.npz"
    [ -f "$out" ] || python3.12 -P scripts/so101_policy/eval_policy.py run --policy "$f" --tasks "$P/tasks_pgate.npz" \
      --n 256 --world nrd --ckpt "$RUN/$m.pt" --device cuda --out "$out" | tail -n 1
  done
done
ls -la --time-style=full-iso "$RUN" "$R2" > "$OUT/inputs_after.txt"
cmp "$OUT/inputs_before.txt" "$OUT/inputs_after.txt" && echo "inputs unchanged" || echo "WARNING: input folders changed"
echo "done $(date -Is)"
