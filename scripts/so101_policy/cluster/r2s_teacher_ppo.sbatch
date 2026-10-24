#!/bin/bash
# Real2sim teacher: PPO fine-tune of jt_h110 in the NRD under the robot dynamics stage (configs/so101_policy/ppo/r2s/).
#   ENVCFG=<env json> RUN=<dir> [ITERS=200 SEED=0] sbatch -p mi3001x scripts/so101_policy/cluster/r2s_teacher_ppo.sbatch
#   wrist-roll cap fine-tune (nominal NRD worlds): ENVCFG=configs/so101_policy/ppo/r2s/teacher_roll_v8.json
#   TRAINCFG=configs/so101_policy/ppo/r2s/joint_ppo_v3_kl02.json (the env config's roll_hi_deg also caps the dev
#   evaluations; the log adds "roll over cap" = share of the episodes that commanded the roll above it)
#SBATCH -J r2s-tppo
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -o /work1/dannegrut/harry/so101_r2s_claude/logs/%x_%j.out
set -uo pipefail
set +u; source /work1/dannegrut/harry/so101_r2s_claude/repo/scripts/so101_policy/cluster/r2s_env.sh; set -u
cd "$REPO"
ENVCFG=${ENVCFG:?set ENVCFG=<env json>}
RUN=${RUN:?set RUN=<dir>}
[ -e "$RUN/ckpt" ] && { echo "$RUN exists, refusing to overwrite"; exit 1; }
echo "host $(hostname) job ${SLURM_JOB_ID:-none} start $(date -Is) gpu $(rocminfo | grep -m1 -o 'gfx9[0-9a-z]*') env $ENVCFG run $RUN"
unset NEDM_SO101_DYN
python3.12 -P scripts/so101_policy/vision_viral_teacher.py ppo --nrd-ckpt "$R2S_NRD" --dev-tasks "$R2S_DATA/tasks/dev100.npz" \
    --critic-warmup 20 --eval-every 20 --save-every 20 --iterations "${ITERS:-200}" --envs 8192 \
    --train-config "${TRAINCFG:-configs/so101_policy/ppo/joint_ppo_v3.json}" --init "${INIT:-$R2S_JT}" --env-config "$ENVCFG" --out "$RUN" \
    --seed "${SEED:-0}"
status=$?
echo "exit $status $(date -Is)"
exit $status
