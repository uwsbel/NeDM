#!/bin/bash
# Real2sim student, step (b): the online loop (build9's o16 recipe: jt_h110, jaw lead 0, 3,000 decisions, 1,024 NRD
# worlds, CPU MuJoCo evaluations every 1,000 decisions on the C and HOME development sets). About 65-75 min on one
# MI350X with supersample 1 (o16: 60.5 min for 3,000; the real scene renders ~15 % slower, the lag adds no render);
# estimated 2.5-3 h with the scene's supersample 2 (4x rays; smoke: 0.19 s render per decision at 64 worlds).
#
#   sbatch scripts/so101_policy/cluster/r2s_online.sbatch                # INIT, RUN, SEED overridable
#   TEACHER_JT=<teacher .pt> (default $R2S_JT = jt_h110): the labelling teacher, e.g. a roll-capped fine-tune
#   (configs/so101_policy/ppo/r2s/teacher_roll_v8.json) with ONLINE_CFG=viral_r2s_online_relabel_roll_v8.json
#SBATCH -J r2s-online
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -t 03:45:00
#SBATCH -o /work1/dannegrut/harry/so101_r2s_claude/logs/%x_%j.out
set -uo pipefail
set +u; source /work1/dannegrut/harry/so101_r2s_claude/repo/scripts/so101_policy/cluster/r2s_env.sh; set -u
cd "$REPO"
INIT=${INIT:-$R2S/runs/v1_start_s0/r2/last.pt}
RUN=${RUN:-$R2S/runs/v1_online_s16}
SEED=${SEED:-16}
# no resume in cmd_online: a second job with the same RUN would mix curve.csv and overwrite the checkpoints
[ -e "$RUN/last.pt" ] || [ -e "$RUN/curve.csv" ] && { echo "$RUN exists, refusing to overwrite"; exit 1; }
echo "host $(hostname) job ${SLURM_JOB_ID:-none} start $(date -Is) gpu $(rocminfo | grep -m1 -o 'gfx9[0-9a-z]*') init $INIT run $RUN jt ${TEACHER_JT:-$R2S_JT}"
python3.12 -P scripts/so101_policy/vision_viral.py online --init "$INIT" --out "$RUN" --jt "${TEACHER_JT:-$R2S_JT}" \
    --nrd-ckpt "$R2S_NRD" --mj-python "$R2S_PY" --tasks "$R2S_TRAIN" --eval-tasks $R2S_DEV --guard-tasks $R2S_FRESH \
    --config "${STUDENT_CFG:-configs/so101_policy/vision/viral_r2s_v1.json}" --set jaw_lead=0 "scene=\"$R2S_SCENE\"" "calib=\"$R2S_CALIB\"" \
    --online-config "${ONLINE_CFG:-configs/so101_policy/vision/viral_r2s_online_v1.json}" --eval-randomise --eval-workers 16 \
    --render-worlds 128 --seed "$SEED" ${EXTRA:-}
status=$?
echo "exit $status $(date -Is)"
exit $status
