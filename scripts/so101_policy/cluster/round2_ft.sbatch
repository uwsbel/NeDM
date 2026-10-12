#!/bin/bash
#SBATCH -J sp-nrd-r2
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 04:00:00
#SBATCH -o /work1/dannegrut/harry/experiments/so101_policy_20261011/logs/%x_%j.out
# NRD data round 2 of the SO101 policy study on one MI350X: merge production_plus + noisy_v1 + expert_v2 -> fine-tune
# from the original final checkpoint (configs/so101_policy/nrd_round/ft_r2_s<seed>.json: init_from
# short_rel_rk_cc_fix_s<seed>_plus/best.pt, 20,000 co-training updates at lr 1e-4, the round-1 recipe) -> evaluations. A step whose output exists is skipped, so the same command resumes an
# interrupted round (train.py resumes from contact_state.pt, saved at every check).
#   cd $B/round2/code_ft && sbatch --export=ALL,SEED=61 scripts/so101_policy/cluster/round2_ft.sbatch
#   cd $B/round2/code_ft && sbatch --export=ALL,SEED=62 scripts/so101_policy/cluster/round2_ft.sbatch
#   (on mi3508x: -p mi3508x --export=ALL,SEEDS=61+62 runs both seeds on GPUs 0 and 1 of the node.)
# Resubmission chain for the 4 h cap: sbatch --dependency=afternotok:<job> --kill-on-invalid-dep=yes ... (same command).
# Environment (optional): SP_CODE (code dir, default $B/round2/code_ft = git HEAD b75ac1098 + the round-2 files; its
# contact_nrd code is the round-1 code bit for bit), SEED / SEEDS.
# Inputs: production_plus (read only), $B/data/noisy_v1, $B/data/expert_v2 (round2_collect.sbatch; its val_full and
# gate2 subfolders are evaluation sets, not merged), init_from = the original final run (config).
# Outputs: $B/data/merged_plus_noisy_v1_expert_v2 (flock-guarded, written once), $B/runs/nrd_ft_r2_s<seed> (run
# directory: best.pt, last.pt, validation.json, complete.json, ...), and in it:
#   after_noisy_v1_val.json, after_production_val.json, after_production_probes_val.json (eval_nrd_noisy.py, best.pt);
#   r2_expert_val_full.json, r2_expert_gate2.json (collect_round.py eval: the original final checkpoint 'r0' (the
#   init), the round-1 best.pt 'r1', this run's best.pt 'r2best' and last.pt 'r2last' on the same GPU: lift agreement, 1 s window, 4 s and whole-episode
#   cube error per family / noise phase / segment).
# Cost estimate (round 1: 10,000 updates + 5 checks = 2,925 s on one MI350X): 20,000 updates + 9 checks ~1.7 h,
# merge ~10 min, data load ~5 min, final validation and evaluations ~20 min: ~2.3 h; ~0.4 ledger-h per seed.
set -euo pipefail
set +u; source /etc/profile; module load pytorch/2.10.0; source /work1/dannegrut/harry/venvs/nedm/bin/activate; set -u
B=/work1/dannegrut/harry/experiments/so101_policy_20261011
C=${SP_CODE:-$B/round2/code_ft}
BASE=/work1/dannegrut/harry/experiments/so101_nrd_fix_20261009/production_plus
NOISY=$B/data/noisy_v1
EXPERT=$B/data/expert_v2
D=$B/data/merged_plus_noisy_v1_expert_v2
SEEDS=${SEEDS:-${SEED:-61}}
SEEDS=${SEEDS//+/ }
export OMP_NUM_THREADS=4 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 PYTHONPATH="$C/src:${PYTHONPATH:-}"
cd "$C"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} seeds $SEEDS start $(date -Is)"
sha256sum src/nedm/contact_nrd/train.py src/nedm/contact_nrd/data.py configs/so101_policy/nrd_round/ft_r2_s6*.json
[ -f "$EXPERT/manifest.json" ] || { echo "no $EXPERT"; exit 1; }
mkdir -p "$B/data" "$B/runs"
(
  flock 9
  if [ ! -f "$D/system.json" ]; then
    echo "merge $(date -Is)"
    rm -rf "$D.tmp"
    python3.12 -P scripts/contact_nrd/merge_data.py "$D.tmp" "$BASE" "$NOISY" "$EXPERT"
    mv -T "$D.tmp" "$D"
  fi
) 9> "$D.lock"
g=0
pids=()
for seed in $SEEDS; do
  RUN=$B/runs/nrd_ft_r2_s$seed
  R1=$B/runs/nrd_ft_noisy_v1_s$seed
  R0=/work1/dannegrut/harry/experiments/so101_nrd_fix_20261009/runs/short_rel_rk_cc_fix_s${seed}_plus
  CFG=configs/so101_policy/nrd_round/ft_r2_s$seed.json
  (
    export HIP_VISIBLE_DEVICES=$g
    if [ ! -f "$RUN/complete.json" ]; then
      echo "fine-tune s$seed $(date -Is)"
      python3.12 -P -m nedm.contact_nrd.train --data "$D" --output "$RUN" --config "$CFG" --device cuda >> "$RUN.stdout.log" 2>&1
    fi
    for set in val_full gate2; do
      if [ ! -f "$RUN/r2_expert_$set.json" ]; then
        echo "expert_v2 $set s$seed $(date -Is)"
        python3.12 -P scripts/so101_policy/collect_round.py eval --data "$EXPERT/$set" --device cuda \
          --model "r0=$R0/best.pt" "r1=$R1/best.pt" "r2best=$RUN/best.pt" "r2last=$RUN/last.pt" --out "$RUN/r2_expert_$set.json"
      fi
    done
    if [ ! -f "$RUN/after_noisy_v1_val.json" ]; then
      echo "after-numbers noisy s$seed $(date -Is)"
      python3.12 -P scripts/so101_policy/eval_nrd_noisy.py --data "$NOISY" --model "s$seed=$RUN" --out "$RUN/after_noisy_v1_val.json"
    fi
    if [ ! -f "$RUN/after_production_val.json" ]; then
      echo "after-numbers production task families s$seed $(date -Is)"
      python3.12 -P scripts/so101_policy/eval_nrd_noisy.py --data "$BASE" \
        --families nominal off_centre one_sided near_miss weak_closure drop push \
        --model "s$seed=$RUN" --out "$RUN/after_production_val.json"
    fi
    if [ ! -f "$RUN/after_production_probes_val.json" ]; then
      echo "after-numbers production probe families s$seed $(date -Is)"
      python3.12 -P scripts/so101_policy/eval_nrd_noisy.py --data "$BASE" \
        --families free_arm cube_flight cube_table jaw_fixed jaw_moving \
        --model "s$seed=$RUN" --out "$RUN/after_production_probes_val.json"
    fi
  ) &
  pids+=($!)
  g=$((g + 1))
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "done $(date -Is) exit $status"
exit $status
