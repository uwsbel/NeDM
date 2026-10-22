#!/bin/bash
# Real2sim student: CPU MuJoCo evaluation of one checkpoint on the lowered / tilted real table (~/so101_r2s_claude/
# table_test): dev100 + home45_dev100, nominal visuals, conditions: measured table plane, the plane with the -2.09 deg
# shoulder_lift joint-map correction, 10 mm lower flat table. REPO may point to a frozen code copy.
#
#   STUDENT=<ckpt> OUT=<dir> REPO=$R2S/repo_tt sbatch scripts/so101_policy/cluster/r2s_table.sbatch
#SBATCH -J r2s-table
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -t 00:45:00
#SBATCH -o /work1/dannegrut/harry/so101_r2s_claude/logs/%x_%j.out
set -uo pipefail
set +u; source /work1/dannegrut/harry/so101_r2s_claude/repo/scripts/so101_policy/cluster/r2s_env.sh; set -u
cd "$REPO"
STUDENT=${STUDENT:?set STUDENT=<checkpoint>}
OUT=${OUT:?set OUT=<dir>}
mkdir -p "$OUT"
PLANE=-0.00468,-0.03641,0.01836
echo "host $(hostname) job ${SLURM_JOB_ID:-none} start $(date -Is) repo $REPO student $STUDENT out $OUT"
status=0
SETS=${SETS:-"$R2S_DATA/tasks/dev100.npz $R2S_DATA/tasks/home45_dev100.npz"}
for t in $SETS; do
  for c in "plane|--table-plane=$PLANE" "plane_slm2|--table-plane=$PLANE --q-offset-deg 0,-2.086,0,0,0,0" "dz10|--table-dz -0.010"; do
    name=${c%%|*}; opt=${c#*|}
    python3.12 -P scripts/so101_policy/vision_viral.py eval --student "$STUDENT" --jt "$R2S_JT" --nrd-ckpt "$R2S_NRD" \
        --mj-python "$R2S_PY" --tasks "$t" --world mujoco --workers 16 --render-worlds 128 $opt \
        --out "$OUT/$(basename "$t" .npz)_${name}.npz" || status=$?
  done
done
echo "exit $status $(date -Is)"
exit $status
