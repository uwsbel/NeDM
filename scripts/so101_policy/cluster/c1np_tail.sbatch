#!/bin/bash
#SBATCH -J sp-nrd-c1np-tail
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 01:30:00
#SBATCH -o /work1/dannegrut/harry/experiments/so101_policy_20261011/logs/%x_%j.out
# abl_noppo stage nrd, RESUBMISSION of the c1np_ft.sbatch evaluation tail (2026-10-18 14:35 CDT). Identical to
# c1np_ft.sbatch except: (1) no `eval_policy.py report` call: it imports mujoco, which the AMD nedm venv lacks, so
# the original jobs 468935 / 468936 stop with exit 1 at the first report (as round-3 job 460875 did); the gate npz
# files are made here, the reports on luffy (same script, same evalmj references); (2) no c1 reference rows: the
# baseline stage (abl_noppo_c1_tail.sbatch, jobs 468938 / 468939) measures c0 / c1best / c1last with the same
# commands. Training is skipped (complete.json exists); every finished step is skipped (resume).
#   sbatch --dependency=afternotok:<c1np_ft job> --kill-on-invalid-dep=yes --export=ALL,SEED=<seed> c1np_tail.sbatch
# ---- original header of c1np_ft.sbatch ----
# abl_noppo ablation (2026-10-18, plan ~/so101_policy/docs/abl_noppo_plan.md sec. 1): NRD "c1np" = c1 with the
# PPO-policy rows (policy_v3) left out of the training data. Copy of c1_ft.sbatch; the only changes (old -> new):
#   code     $B/c1/code                                    -> $B/abl_noppo/code (byte copy of $B/c1/code + 2 configs)
#   data     $B/data/merged_r3_pp_noisy_expert_policy_x2.5 -> $B/data/merged_plus_noisy_v1_expert_v2 (exact subset:
#            the same production_plus + noisy_v1 + expert_v2 rows, no policy_v3; ~/so101_policy/runs/abl_noppo/nrd/notes.md)
#   config   ft_c1_s<seed>.json                            -> ft_c1np_s<seed>.json (only "note" differs)
#   run dir  $B/runs/nrd_ft_c1_s<seed>                     -> $B/runs/nrd_ft_c1np_s<seed>
#   outputs  c1_*.json, gate_closed/*_c1<m>*               -> c1np_*.json, gate_closed/*_c1np<m>*
#   models   c0 c1best c1last                              -> c0 c1npbest c1nplast
#   manifest print: m[k] -> m.get(k) (the no-PPO manifest, written by merge_data.py, has no "repeat" key)
# Added AFTER the unchanged tail (reference rows; c1's own tail was cancelled on 10-12 19:40, so no c1_*.json exists in
# $B/runs/nrd_ft_c1_s<seed>): c1 best/last on the same sets, read-only from $B/runs/nrd_ft_c1_s<seed>, written into the
# c1np run dir as c1ref_*.json, gate_closed/*_c1ref<m>*, c1ref_after_*.json (gate and after_*: c1 best only).
#   cd $B/abl_noppo/code && sbatch --export=ALL,SEED=61 scripts/so101_policy/cluster/c1np_ft.sbatch
#   cd $B/abl_noppo/code && sbatch --export=ALL,SEED=62 scripts/so101_policy/cluster/c1np_ft.sbatch
#   (on mi3508x: sbatch -p mi3508x --export=ALL,SEEDS=61+62 ... runs both seeds on GPUs 0 and 1 of the node.)
# A step whose output exists is skipped, so the same command resumes (train.py resumes from contact_state.pt).
# Environment (optional): SP_CODE (default $B/abl_noppo/code), SEED / SEEDS, C1NP_DATA (default the no-PPO merge),
# R3_POLICIES ($B/round3/policies), C1_REF (1 = run the c1 reference rows after the tail; default 1).
set -euo pipefail
set +u; source /etc/profile; module load pytorch/2.10.0; source /work1/dannegrut/harry/venvs/nedm/bin/activate; set -u
B=/work1/dannegrut/harry/experiments/so101_policy_20261011
C=${SP_CODE:-$B/abl_noppo/code}
W=$B/round3/work
P=$B/data/policy_v3
EXPERT=$B/data/expert_v2
NOISY=$B/data/noisy_v1
BASE=/work1/dannegrut/harry/experiments/so101_nrd_fix_20261009/production_plus
POL=${R3_POLICIES:-$B/round3/policies}
D=${C1NP_DATA:-$B/data/merged_plus_noisy_v1_expert_v2}
C1_REF=0   # the baseline stage covers c1 (see header); the block below is not run
SEEDS=${SEEDS:-${SEED:-61}}
SEEDS=${SEEDS//+/ }
export OMP_NUM_THREADS=4 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 PYTHONPATH="$C/src:${PYTHONPATH:-}"
cd "$C"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} seeds $SEEDS code $C data $D start $(date -Is)"
cat GIT_HEAD
sha256sum src/nedm/contact_nrd/train.py src/nedm/contact_nrd/data.py configs/so101_policy/nrd_round/ft_c1np_s6*.json
[ -f "$D/system.json" ] || { echo "no merged data $D"; exit 1; }
cat "$D/manifest.json" | python3.12 -c "import json, sys; m = json.load(sys.stdin); print({k: m.get(k) for k in ('merged_from', 'episodes', 'splits', 'repeat')})"
mkdir -p "$B/runs"
g=0
pids=()
for seed in $SEEDS; do
  RUN=$B/runs/nrd_ft_c1np_s$seed
  R2=/work1/dannegrut/harry/experiments/so101_nrd_fix_20261009/runs/full_ns_rk_cc_geo_cs_s${seed}_plus
  C1=$B/runs/nrd_ft_c1_s$seed
  CFG=configs/so101_policy/nrd_round/ft_c1np_s$seed.json
  (
    export HIP_VISIBLE_DEVICES=$g
    echo "s$seed init (c0) checkpoint: $(sha256sum "$R2/best.pt")"
    if [ ! -f "$RUN/complete.json" ]; then
      echo "fine-tune s$seed $(date -Is) config $CFG output $RUN"
      python3.12 -P -m nedm.contact_nrd.train --data "$D" --output "$RUN" --config "$CFG" --device cuda >> "$RUN.stdout.log" 2>&1
    fi
    echo "s$seed trained $(date -Is):"; sha256sum "$RUN/best.pt" "$RUN/last.pt"
    MODELS="c0=$R2/best.pt c1npbest=$RUN/best.pt c1nplast=$RUN/last.pt"
    for set in pgate val_full; do
      if [ ! -f "$RUN/c1np_policy_$set.json" ]; then
        echo "policy_v3 $set s$seed $(date -Is)"
        python3.12 -P scripts/so101_policy/collect_round.py eval --data "$P/$set" --device cuda --carry \
          --model $MODELS --out "$RUN/c1np_policy_$set.json"
      fi
    done
    for set in val_full gate2; do
      if [ ! -f "$RUN/c1np_expert_$set.json" ]; then
        echo "expert_v2 $set s$seed $(date -Is)"
        python3.12 -P scripts/so101_policy/collect_round.py eval --data "$EXPERT/$set" --device cuda --carry \
          --model $MODELS --out "$RUN/c1np_expert_$set.json"
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
    # (held-out for c1np: none of the 7 policies' episodes is in its training data)
    mkdir -p "$RUN/gate_closed"
    for m in best last; do
      for f in "$POL"/*.pt; do
        id=$(basename "$f" | cut -d_ -f1)
        out="$RUN/gate_closed/${id}_c1np${m}_nrd256.npz"
        [ -f "$out" ] || python3.12 -P scripts/so101_policy/eval_policy.py run --policy "$f" --tasks "$P/tasks_pgate.npz" \
          --n 256 --world nrd --ckpt "$RUN/$m.pt" --device cuda --out "$out" | tail -n 1
        echo "gate npz $out (report on luffy: no mujoco on AMD)"
      done
    done
    echo "c1np tail done s$seed $(date -Is)"
    if [ "$C1_REF" = 1 ]; then
      # reference rows for c1 (its tail was cancelled 10-12): c1 checkpoints read-only, outputs in the c1np run dir
      echo "c1 reference s$seed: $(sha256sum "$C1/best.pt" "$C1/last.pt" | tr '\n' ' ')"
      REF="c1best=$C1/best.pt c1last=$C1/last.pt"
      for set in pgate val_full; do
        [ -f "$RUN/c1ref_policy_$set.json" ] || python3.12 -P scripts/so101_policy/collect_round.py eval --data "$P/$set" \
          --device cuda --carry --model $REF --out "$RUN/c1ref_policy_$set.json"
      done
      for set in val_full gate2; do
        [ -f "$RUN/c1ref_expert_$set.json" ] || python3.12 -P scripts/so101_policy/collect_round.py eval --data "$EXPERT/$set" \
          --device cuda --carry --model $REF --out "$RUN/c1ref_expert_$set.json"
      done
      [ -f "$RUN/c1ref_after_noisy_v1_val.json" ] || python3.12 -P scripts/so101_policy/eval_nrd_noisy.py --data "$NOISY" \
        --model "s$seed=$C1/best.pt" --out "$RUN/c1ref_after_noisy_v1_val.json"
      [ -f "$RUN/c1ref_after_production_val.json" ] || python3.12 -P scripts/so101_policy/eval_nrd_noisy.py --data "$BASE" \
        --families nominal off_centre one_sided near_miss weak_closure drop push \
        --model "s$seed=$C1/best.pt" --out "$RUN/c1ref_after_production_val.json"
      [ -f "$RUN/c1ref_after_production_probes_val.json" ] || python3.12 -P scripts/so101_policy/eval_nrd_noisy.py --data "$BASE" \
        --families free_arm cube_flight cube_table jaw_fixed jaw_moving \
        --model "s$seed=$C1/best.pt" --out "$RUN/c1ref_after_production_probes_val.json"
      for f in "$POL"/*.pt; do
        id=$(basename "$f" | cut -d_ -f1)
        out="$RUN/gate_closed/${id}_c1refbest_nrd256.npz"
        [ -f "$out" ] || python3.12 -P scripts/so101_policy/eval_policy.py run --policy "$f" --tasks "$P/tasks_pgate.npz" \
          --n 256 --world nrd --ckpt "$C1/best.pt" --device cuda --out "$out" | tail -n 1
        python3.12 -P scripts/so101_policy/eval_policy.py report --runs "$W/evalmj/${id}_mj256.npz" "$out" \
          --ref "$W/evalmj/${id}_mj256.npz" --out "$RUN/gate_closed/report_${id}_c1refbest.json" | tail -n 3
      done
      echo "c1 reference done s$seed $(date -Is)"
    fi
  ) &
  pids+=($!)
  g=$((g + 1))
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "done $(date -Is) exit $status"
exit $status
