#!/bin/bash
# ROBUSTNESS: shove the trunk once per episode on CRM and score the recovery, 4 arms per
# node (one per MI210). 8 directions x 2 lattice positions = 32 episodes per arm, identical
# pushes for every arm, so the pairing is on the disturbance as well as the path.
#   sbatch --array=0-N --export=ALL,LIST=<file of "label policy_path" lines>,FORCES="120 240"
# Each arm runs every force in turn, so the result is a dose-response curve: falls and
# recovery against how hard the shove was, which one level cannot give.
# Records go to eval_push_f<FORCE>/, scored with paired_eval.py --metrics push.
# COST: ~1 h x 0.4 per node = ~0.4 node-hours per 4 arms.
#SBATCH -J kyle-evalpush
#SBATCH -p mi2104x
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH -o /work1/dannegrut/kyle/logs/%x-%A_%a.out
#SBATCH -e /work1/dannegrut/kyle/logs/%x-%A_%a.err
set -uo pipefail
K=/work1/dannegrut/kyle
module purge
module load gnu12 rocm
source $K/nedm/env-scoring.sh
export PYTHONPATH=$K/qrun:$K/qrun/src:$PYTHONPATH
cd $K/qrun/quadruped
A=${SLURM_ARRAY_TASK_ID}
FORCES=${FORCES:-"120 180 240 300"}
U=$K/qrun/assets/data/robot/go2_irrvis/urdf/go2_description.urdf
mkdir -p $K/logs/eval_push
echo "task $A host $(hostname) job ${SLURM_JOB_ID} forces $FORCES list $LIST code $(cat $K/qrun/.source_commit)"
# A GPU fault in Chrono's FSI kills the process (thrust throws from a destructor), so
# evaluate.py writes what it has and exits 90 before teardown. Resume rather than lose the
# rest of the arm, and make the failure VISIBLE: job 432517 reported COMPLETED while two
# arms had died, and their truncated records were nearly compared against full ones.
run_eval() {  # run_eval LOG CMD...
  local log=$1; shift
  local rc=0
  for attempt in 1 2 3; do
    if [ $attempt -eq 1 ]; then "$@" > "$log" 2>&1; else "$@" --resume >> "$log" 2>&1; fi
    rc=$?
    [ $rc -ne 90 ] && break
    echo "  GPU fault; resuming (attempt $((attempt + 1)))" >> "$log"
  done
  return $rc
}
G=0
while read -r LABEL POLICY; do
  [ -z "$LABEL" ] && continue
  ( export ROCR_VISIBLE_DEVICES=$G HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8
    for F in $FORCES; do
      mkdir -p $K/qrun/eval_push_f$F
      run_eval $K/logs/eval_push/${LABEL}_f$F.log \
        python -u evaluate.py --policy $POLICY --urdf $U --corpus $K/qrun/data/go2_crm_v2 \
        --out $K/qrun/eval_push_f$F/$LABEL --label $LABEL --terrain crm --seconds 8 \
        --push-force $F --push-dirs 8 --push-reps 2 --push-at 4 --spawn-spread 1.0
      rc=$?
      echo "PUSH $LABEL f$F exit $rc on GPU $G $(date +%H:%M)"
      [ $rc -ne 0 ] && echo "ARM_FAILED $LABEL f$F rc=$rc"
    done ) &
  G=$((G + 1))
done < <(sed -n "$((4 * A + 1)),$((4 * A + 4))p" $LIST)
wait
echo "TASK_${A}_DONE"
