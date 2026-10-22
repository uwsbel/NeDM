#!/bin/bash
# Real2sim student, step (c): CPU MuJoCo evaluation of one student checkpoint (the student's real scene and nominal
# image lag; nominal and randomised visuals) on the task sets $SETS (default: the C and HOME development sets; the
# fresh sets only for the final evaluation of the chosen checkpoint). About 1 min per 100 tasks and visual mode.
#
#   STUDENT=$R2S/runs/v1_online_s16/ckpt/it_03000.pt sbatch scripts/so101_policy/cluster/r2s_eval.sbatch
#SBATCH -J r2s-eval
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -o /work1/dannegrut/harry/so101_r2s_claude/logs/%x_%j.out
set -uo pipefail
set +u; source /work1/dannegrut/harry/so101_r2s_claude/repo/scripts/so101_policy/cluster/r2s_env.sh; set -u
cd "$REPO"
STUDENT=${STUDENT:?set STUDENT=<checkpoint>}
SETS=${SETS:-$R2S_DEV}
OUT=${OUT:-$(dirname "$STUDENT")/eval_$(basename "$STUDENT" .pt)}
mkdir -p "$OUT"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} start $(date -Is) student $STUDENT out $OUT"
status=0
for t in $SETS; do
  for mode in nominal randomised; do
    flag=""; [ $mode = randomised ] && flag="--randomise"
    python3.12 -P scripts/so101_policy/vision_viral.py eval --student "$STUDENT" --jt "$R2S_JT" --nrd-ckpt "$R2S_NRD" \
        --mj-python "$R2S_PY" --tasks "$t" --world mujoco --workers 16 --render-worlds 128 $flag ${EXTRA:-} \
        --out "$OUT/$(basename "$t" .npz)_mujoco_${mode}.npz" || status=$?
  done
done
echo "exit $status $(date -Is)"
exit $status
