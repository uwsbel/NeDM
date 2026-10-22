#!/bin/bash
# Real2sim student, step (a): the round-based DAgger start (build9's d2 recipe, rounds 0-2, teacher jt_h110 only):
# round 0 = 4,096 JT-driven tasks (half with 1 x std noise), rounds 1-2 = 2,048 tasks each, half driven by the previous
# round's student with randomised visuals; training 10,000 / 4,000 / 4,000 steps; CPU MuJoCo evaluations on the C and
# HOME development sets (nominal + randomised). About 35-45 min on one MI350X with supersample 1 (d2: 33 min for rounds 0-2); the renderer part x4 with supersample 2.
#
#   sbatch scripts/so101_policy/cluster/r2s_start.sbatch                 # RUN, SEED, R2S_SCENE overridable
#SBATCH -J r2s-start
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -t 02:30:00
#SBATCH -o /work1/dannegrut/harry/so101_r2s_claude/logs/%x_%j.out
set -uo pipefail
set +u; source /work1/dannegrut/harry/so101_r2s_claude/repo/scripts/so101_policy/cluster/r2s_env.sh; set -u
cd "$REPO"
RUN=${RUN:-$R2S/runs/v1_start_s0}
SEED=${SEED:-0}
echo "host $(hostname) job ${SLURM_JOB_ID:-none} start $(date -Is) gpu $(rocminfo | grep -m1 -o 'gfx9[0-9a-z]*') run $RUN"
python3.12 -P scripts/so101_policy/vision_viral.py run --out "$RUN" --jt "$R2S_JT" --nrd-ckpt "$R2S_NRD" \
    --mj-python "$R2S_PY" --tasks "$R2S_TRAIN" --eval-tasks $R2S_DEV --guard-tasks $R2S_FRESH \
    --config configs/so101_policy/vision/viral_r2s_v1.json --set "scene=\"$R2S_SCENE\"" "calib=\"$R2S_CALIB\"" \
    --r0-tasks 4096 --rounds 2 --round-tasks 2048 --teacher-share 0.5 --noise 1.0 --r0-noise-frac 0.5 \
    --train-steps0 10000 --train-steps 4000 --collect-randomise --eval-randomise --eval-workers 16 \
    --render-worlds 256 --seed "$SEED" ${EXTRA:-}
status=$?
echo "exit $status $(date -Is)"
exit $status
