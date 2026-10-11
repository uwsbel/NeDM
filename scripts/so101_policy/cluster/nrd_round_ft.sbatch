#!/bin/bash
#SBATCH -J sp-nrd-ft
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 04:00:00
#SBATCH -o /work1/dannegrut/harry/experiments/so101_policy_20261011/logs/%x_%j.out
# NRD data round 1 of the SO101 policy study on one MI350X: merge -> fine-tune -> after-numbers. A step whose output
# exists is skipped, so the same command resumes an interrupted round (train.py resumes from contact_state.pt).
#   cd $B/code_nrd_round && sbatch --export=ALL,SEED=61 scripts/so101_policy/cluster/nrd_round_ft.sbatch
#   cd $B/code_nrd_round && sbatch --export=ALL,SEED=62 scripts/so101_policy/cluster/nrd_round_ft.sbatch
# (one job per seed on mi3501x; on mi3508x set SEEDS="61 62" and both run on GPUs 0 and 1 of the node.)
# Environment (optional): SP_CODE code dir (default $B/code_nrd_round: sync the worktree's src/, scripts/, configs/
# there first), SEED / SEEDS, ROUND (default noisy_v1), NEW (default $B/data/$ROUND).
# Inputs: production_plus (the final model's data, read only) and $B/data/noisy_v1 (collect --campaign noisy_v1);
# configs/so101_policy/nrd_round/ft_<ROUND>_s<seed>.json (init_from = the final run of that seed, read only).
# Outputs: $B/data/merged_plus_<ROUND> (merge; flock-guarded, written once), $B/runs/nrd_ft_<ROUND>_s<seed> (fine-tune
# run directory: run_config.json, train_log.jsonl, best.pt, validation.json = every validation episode of both
# campaigns by family, complete.json), $B/runs/nrd_ft_<ROUND>_s<seed>/after_<ROUND>_val.json and after_production_val.json
# (scripts/so101_policy/eval_nrd_noisy.py on the noisy validation split and on the production validation episodes: the
# 'after' numbers next to the 'before' ones in docs/build_nrd_round.md).
#
# Cost and memory (estimates from the final short run: 15,000 co-training updates + checks in about 65 min on one
# MI350X; UNVERIFIED for this run): merge reads 4.2 + 1.1 GB and writes 5.4 GB (a few minutes on /work1, host ~11 GB);
# fine-tune holds 47,500 x 601 records x 2 bodies x 27 channels in float64 on the GPU (~12 GB + actions, labels and a
# transient copy) of the MI350X's 288 GB; 10,000 updates at the 1 s rollout horizon + 5 checks + the final validation
# of ~3,850 episodes: about 1 h; the two evaluations about 10 min.
set -euo pipefail
set +u; source /etc/profile; module load pytorch/2.10.0; source /work1/dannegrut/harry/venvs/nedm/bin/activate; set -u
B=/work1/dannegrut/harry/experiments/so101_policy_20261011
C=${SP_CODE:-$B/code_nrd_round}
ROUND=${ROUND:-noisy_v1}
NEW=${NEW:-$B/data/$ROUND}
BASE=/work1/dannegrut/harry/experiments/so101_nrd_fix_20261009/production_plus
D=$B/data/merged_plus_$ROUND
SEEDS=${SEEDS:-${SEED:-61}}
export OMP_NUM_THREADS=4 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 PYTHONPATH="$C/src:${PYTHONPATH:-}"
cd "$C"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} round $ROUND seeds $SEEDS start $(date -Is)"
mkdir -p "$B/data" "$B/runs"
(
  flock 9
  if [ ! -f "$D/system.json" ]; then
    echo "merge $(date -Is)"
    rm -rf "$D.tmp"
    python3.12 -P scripts/contact_nrd/merge_data.py "$D.tmp" "$BASE" "$NEW"
    mv -T "$D.tmp" "$D"
  fi
) 9> "$D.lock"
g=0
pids=()
for seed in $SEEDS; do
  RUN=$B/runs/nrd_ft_${ROUND}_s$seed
  CFG=configs/so101_policy/nrd_round/ft_${ROUND}_s$seed.json
  (
    export HIP_VISIBLE_DEVICES=$g
    if [ ! -f "$RUN/complete.json" ]; then
      echo "fine-tune s$seed $(date -Is)"
      python3.12 -P -m nedm.contact_nrd.train --data "$D" --output "$RUN" --config "$CFG" --device cuda >> "$RUN.stdout.log" 2>&1
    fi
    if [ ! -f "$RUN/after_${ROUND}_val.json" ]; then
      echo "after-numbers s$seed $(date -Is)"
      python3.12 -P scripts/so101_policy/eval_nrd_noisy.py --data "$NEW" --model "s$seed=$RUN" --out "$RUN/after_${ROUND}_val.json"
      python3.12 -P scripts/so101_policy/eval_nrd_noisy.py --data "$BASE" \
        --families nominal off_centre one_sided near_miss weak_closure drop push \
        --model "s$seed=$RUN" --out "$RUN/after_production_val.json"
    fi
  ) &
  pids+=($!)
  g=$((g + 1))
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "done $(date -Is) exit $status"
exit $status
