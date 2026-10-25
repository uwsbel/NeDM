#!/bin/bash
# Real2sim teacher: PPO fine-tune of jt_h110 in the NRD under the robot dynamics stage (configs/so101_policy/ppo/r2s/).
#   ENVCFG=<env json> RUN=<dir> [ITERS=200 SEED=0] sbatch -p mi3001x scripts/so101_policy/cluster/r2s_teacher_ppo.sbatch
#   wrist-roll cap fine-tune (nominal NRD worlds): ENVCFG=configs/so101_policy/ppo/r2s/teacher_roll_v8.json
#   TRAINCFG=configs/so101_policy/ppo/r2s/joint_ppo_v3_kl02.json (the env config's roll_hi_deg also caps the dev
#   evaluations; the log adds "roll over cap" = share of the episodes that commanded the roll above it)
#   DEV_TASKS=<task file> (default dev100), DEV_EXTRA="<task files>" (more dev sets: log + RUN/dev_evals.jsonl),
#   EXTRA=--dev-match-env (dev evaluations with the env config's horizon / hold cap), ENVS, EVAL_EVERY, SAVE_EVERY
#   REPO=<another synced copy> (e.g. $R2S/repo_v9): runs that copy's code and sources its r2s_env.sh (unset: $R2S/repo,
#   as before). Round v9 restart teacher (rest starts + widened pause): ENVCFG=configs/so101_policy/ppo/r2s/
#   teacher_restart_v9a.json (nominal NRD) or _v9b.json (+ the student recipe's slow-arm dynamics), INIT=<D's teacher>
#SBATCH -J r2s-tppo
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -o /work1/dannegrut/harry/so101_r2s_claude/logs/%x_%j.out
set -uo pipefail
set +u; source "${REPO:-/work1/dannegrut/harry/so101_r2s_claude/repo}/scripts/so101_policy/cluster/r2s_env.sh"; set -u
cd "$REPO"
ENVCFG=${ENVCFG:?set ENVCFG=<env json>}
RUN=${RUN:?set RUN=<dir>}
[ -e "$RUN/ckpt" ] && { echo "$RUN exists, refusing to overwrite"; exit 1; }
echo "host $(hostname) job ${SLURM_JOB_ID:-none} start $(date -Is) gpu $(rocminfo | grep -m1 -o 'gfx9[0-9a-z]*') env $ENVCFG run $RUN"
echo "train ${TRAINCFG:-configs/so101_policy/ppo/joint_ppo_v3.json} init ${INIT:-$R2S_JT} dev ${DEV_TASKS:-$R2S_DATA/tasks/dev100.npz} ${DEV_EXTRA:-} ${EXTRA:-}"
[ -n "${TRAINCFG:-}" ] || echo "NOTE: TRAINCFG unset: joint_ppo_v3.json (desired_kl 0.005; the r2s fine-tunes use r2s/joint_ppo_v3_kl02.json)"
unset NEDM_SO101_DYN
python3.12 -P scripts/so101_policy/vision_viral_teacher.py ppo --nrd-ckpt "$R2S_NRD" --dev-tasks "${DEV_TASKS:-$R2S_DATA/tasks/dev100.npz}" \
    ${DEV_EXTRA:+--dev-tasks-extra $DEV_EXTRA} \
    --critic-warmup 20 --eval-every "${EVAL_EVERY:-20}" --save-every "${SAVE_EVERY:-20}" --iterations "${ITERS:-200}" --envs "${ENVS:-8192}" \
    --train-config "${TRAINCFG:-configs/so101_policy/ppo/joint_ppo_v3.json}" --init "${INIT:-$R2S_JT}" --env-config "$ENVCFG" --out "$RUN" \
    --seed "${SEED:-0}" ${EXTRA:-}
status=$?
echo "exit $status $(date -Is)"
exit $status
