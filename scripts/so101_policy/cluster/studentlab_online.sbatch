#!/bin/bash
# Studentlab trial arms A0-A4 (2026-10-22; ~/so101_studentlab_claude/trials/BUILD.md): online DAgger from the robot
# student s28 (3,000 decisions, 1,024 NRD worlds, jt_h110, the s28 recipe: r2s_online.sbatch) with the arm's student
# config (new inputs: zero-init expansion of s28) and online config (goal jumps, hold cap 6 s). Everything this job
# writes is under $R2S = /work1/dannegrut/harry/so101_studentlab_claude (repo copy, data copies, caches, runs, logs).
#
#   ARM=a1 SEED=211 sbatch --export=ALL scripts/so101_policy/cluster/studentlab_online.sbatch
#   (smoke: ARM=a4 SEED=1 SMOKE=1 sbatch -t 00:25:00 --export=ALL ...; 64 worlds, 40 decisions, small buffer, no eval)
#SBATCH -J sl-online
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -t 03:45:00
#SBATCH -o /work1/dannegrut/harry/so101_studentlab_claude/logs/%x_%j.out
set -uo pipefail
export R2S=/work1/dannegrut/harry/so101_studentlab_claude
export REPO=$R2S/repo
unset R2S_SCENE R2S_CALIB R2S_TRAIN R2S_DEV R2S_FRESH     # data always from $R2S/data (never another study's)
set +u; source $REPO/scripts/so101_policy/cluster/r2s_env.sh; set -u
cd "$REPO"
ARM=${ARM:?set ARM=a0..a4}
SEED=${SEED:?set SEED}
INIT=${INIT:-$R2S/data/init/robot_v2_s28_it03000.pt}
RUN=${RUN:-$R2S/runs/${ARM}_s${SEED}${SMOKE:+_smoke}}
STUDENT_CFG=configs/so101_policy/vision/studentlab/${ARM}_student.json
ONLINE_CFG=configs/so101_policy/vision/studentlab/${ARM}_online.json
[ -e "$RUN/last.pt" ] || [ -e "$RUN/curve.csv" ] && { echo "$RUN exists, refusing to overwrite"; exit 1; }
SMOKE_ARGS=()
if [ -n "${SMOKE:-}" ]; then
  SMOKE_ARGS=(--oset envs=64 iterations=40 buffer=4096 eval_every=0 save_every=40 log_every=5)
fi
echo "host $(hostname) job ${SLURM_JOB_ID:-none} start $(date -Is) gpu $(rocminfo | grep -m1 -o 'gfx9[0-9a-z]*') arm $ARM init $INIT run $RUN head $(head -c 12 $REPO/SYNC_HEAD.txt 2>/dev/null)"
python3.12 -P scripts/so101_policy/vision_viral.py online --init "$INIT" --out "$RUN" --jt "$R2S_JT" \
    --nrd-ckpt "$R2S_NRD" --mj-python "$R2S_PY" --tasks "$R2S_TRAIN" --eval-tasks $R2S_DEV --guard-tasks $R2S_FRESH \
    --config "$STUDENT_CFG" --set jaw_lead=0 "scene=\"$R2S_SCENE\"" "calib=\"$R2S_CALIB\"" \
    --online-config "$ONLINE_CFG" "${SMOKE_ARGS[@]}" --eval-randomise --eval-workers 16 \
    --render-worlds 128 --seed "$SEED" ${EXTRA:-}
status=$?
echo "exit $status $(date -Is)"
exit $status
