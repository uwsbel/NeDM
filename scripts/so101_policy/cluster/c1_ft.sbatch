#!/bin/bash
#SBATCH -J sp-nrd-c1
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 04:00:00
#SBATCH -o /work1/dannegrut/harry/experiments/so101_policy_20261011/logs/%x_%j.out
# Converged-state data round c1 (fine-tune of the converged-state geo_cs NRD on the round-3 merge). Derived from round3_ft.sbatch.
# NRD data round 3 of the SO101 policy study on one MI350X (NOT submitted by the round-3 collect builder; the next stage
# submits it): fine-tune the ROUND-2 grasp NRD (configs/so101_policy/nrd_round/ft_r3_s<seed>.json: init_from
# runs/nrd_ft_r2_s<seed>/best.pt, 20,000 co-training updates at lr 1e-4, the round-2 recipe and checks) on the merge
# production_plus + noisy_v1 + expert_v2 + policy_v3 (x K: round3_collect.sbatch, step 6) -> evaluations.
#   cd $B/round3/code && sbatch --export=ALL,SEED=61 scripts/so101_policy/cluster/round3_ft.sbatch
#   cd $B/round3/code && sbatch --export=ALL,SEED=62 scripts/so101_policy/cluster/round3_ft.sbatch
#   (on mi3508x: sbatch -p mi3508x --export=ALL,SEEDS=61+62 ... runs both seeds on GPUs 0 and 1 of the node.)
# A step whose output exists is skipped, so the same command resumes (train.py resumes from contact_state.pt, saved at
# every check). Resubmission chain for the 4 h cap: sbatch --dependency=afternotok:<job> --kill-on-invalid-dep=yes ...
# Environment (optional): SP_CODE (default $B/round3/code), SEED / SEEDS, R3_DATA (merged data dir; default the
# folder named by $B/round3/work/pick_repeat.json), R3_POLICIES ($B/round3/policies).
# Outputs in $B/runs/nrd_ft_r3_s<seed> (best.pt, last.pt, validation.json, complete.json, ...) and there:
#   c1_policy_pgate.json, c1_policy_val_full.json (collect_round.py eval --carry on policy_v3/pgate and val_full: models
#     r2 = the init, r3best, r3last: lift / drop agreement, end error, cube error while MuJoCo holds the cube, per
#     policy and noise level);
#   c1_expert_val_full.json, c1_expert_gate2.json (the same models on expert_v2's evaluation folders);
#   after_noisy_v1_val.json, after_production_val.json, after_production_probes_val.json (eval_nrd_noisy.py, best.pt);
#   gate_closed/<policy>_<model>_nrd256.npz + gate_closed/report_<model>.json: the 7 policies closed loop in this NRD
#     (eval_policy.py run --world nrd) on the first 256 pgate tasks vs CPU MuJoCo ($B/round3/work/evalmj, same tasks).
# Cost estimate (round 2: 20,000 updates + 9 checks = 6,473 s on one MI350X, s61): ~1.8-2.2 h training + ~30 min
# evaluations: ~2.5 h, ~0.45 ledger-h per seed (0.175 per GPU-hour).
set -euo pipefail
set +u; source /etc/profile; module load pytorch/2.10.0; source /work1/dannegrut/harry/venvs/nedm/bin/activate; set -u
B=/work1/dannegrut/harry/experiments/so101_policy_20261011
C=${SP_CODE:-$B/c1/code}
W=$B/round3/work
P=$B/data/policy_v3
EXPERT=$B/data/expert_v2
NOISY=$B/data/noisy_v1
BASE=/work1/dannegrut/harry/experiments/so101_nrd_fix_20261009/production_plus
POL=${R3_POLICIES:-$B/round3/policies}
if [ -z "${R3_DATA:-}" ]; then
  K=$(python3.12 -c "import json; print(json.load(open('$W/pick_repeat.json'))['choice'])")
  R3_DATA=$B/data/merged_r3_pp_noisy_expert_policy_x$K
fi
D=$R3_DATA
SEEDS=${SEEDS:-${SEED:-61}}
SEEDS=${SEEDS//+/ }
export OMP_NUM_THREADS=4 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 PYTHONPATH="$C/src:${PYTHONPATH:-}"
cd "$C"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} seeds $SEEDS data $D start $(date -Is)"
sha256sum src/nedm/contact_nrd/train.py src/nedm/contact_nrd/data.py configs/so101_policy/nrd_round/ft_c1_s6*.json
[ -f "$D/system.json" ] || { echo "no merged data $D (round3_collect.sbatch step 6)"; exit 1; }
cat "$D/manifest.json" | python3.12 -c "import json, sys; m = json.load(sys.stdin); print({k: m[k] for k in ('episodes', 'splits', 'repeat')})"
mkdir -p "$B/runs"
g=0
pids=()
for seed in $SEEDS; do
  RUN=$B/runs/nrd_ft_c1_s$seed
  R2=/work1/dannegrut/harry/experiments/so101_nrd_fix_20261009/runs/full_ns_rk_cc_geo_cs_s${seed}_plus
  CFG=configs/so101_policy/nrd_round/ft_c1_s$seed.json
  (
    export HIP_VISIBLE_DEVICES=$g
    if [ ! -f "$RUN/complete.json" ]; then
      echo "fine-tune s$seed $(date -Is)"
      python3.12 -P -m nedm.contact_nrd.train --data "$D" --output "$RUN" --config "$CFG" --device cuda >> "$RUN.stdout.log" 2>&1
    fi
    MODELS="c0=$R2/best.pt c1best=$RUN/best.pt c1last=$RUN/last.pt"
    for set in pgate val_full; do
      if [ ! -f "$RUN/c1_policy_$set.json" ]; then
        echo "policy_v3 $set s$seed $(date -Is)"
        python3.12 -P scripts/so101_policy/collect_round.py eval --data "$P/$set" --device cuda --carry \
          --model $MODELS --out "$RUN/c1_policy_$set.json"
      fi
    done
    for set in val_full gate2; do
      if [ ! -f "$RUN/c1_expert_$set.json" ]; then
        echo "expert_v2 $set s$seed $(date -Is)"
        python3.12 -P scripts/so101_policy/collect_round.py eval --data "$EXPERT/$set" --device cuda --carry \
          --model $MODELS --out "$RUN/c1_expert_$set.json"
      fi
    done
    if [ ! -f "$RUN/after_noisy_v1_val.json" ]; then
      python3.12 -P scripts/so101_policy/eval_nrd_noisy.py --data "$NOISY" --model "s$seed=$RUN" --out "$RUN/after_noisy_v1_val.json"
    fi
    if [ ! -f "$RUN/after_production_val.json" ]; then
      python3.12 -P scripts/so101_policy/eval_nrd_noisy.py --data "$BASE" \
        --families nominal off_centre one_sided near_miss weak_closure drop push \
        --model "s$seed=$RUN" --out "$RUN/after_production_val.json"
    fi
    if [ ! -f "$RUN/after_production_probes_val.json" ]; then
      python3.12 -P scripts/so101_policy/eval_nrd_noisy.py --data "$BASE" \
        --families free_arm cube_flight cube_table jaw_fixed jaw_moving \
        --model "s$seed=$RUN" --out "$RUN/after_production_probes_val.json"
    fi
    # the policy-in-NRD gate: each policy closed loop in this NRD vs CPU MuJoCo on the first 256 pgate tasks
    mkdir -p "$RUN/gate_closed"
    for m in best last; do
      for f in "$POL"/*.pt; do
        id=$(basename "$f" | cut -d_ -f1)
        out="$RUN/gate_closed/${id}_c1${m}_nrd256.npz"
        [ -f "$out" ] || python3.12 -P scripts/so101_policy/eval_policy.py run --policy "$f" --tasks "$P/tasks_pgate.npz" \
          --n 256 --world nrd --ckpt "$RUN/$m.pt" --device cuda --out "$out" | tail -n 1
        python3.12 -P scripts/so101_policy/eval_policy.py report --runs "$W/evalmj/${id}_mj256.npz" "$out" \
          --ref "$W/evalmj/${id}_mj256.npz" --out "$RUN/gate_closed/report_${id}_c1${m}.json" | tail -n 3
      done
    done
  ) &
  pids+=($!)
  g=$((g + 1))
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "done $(date -Is) exit $status"
exit $status
