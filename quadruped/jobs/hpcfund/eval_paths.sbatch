#!/bin/bash
# MULTI-PATH paired Chrono evaluation: 4 held-out paths x 10 command families, 15 s each.
# evaluation is no longer bottlenecked on one workstation. Every arm scored for a paired
# comparison must come from this same partition and build, base included; paired_eval.py
# refuses records from different Chrono builds.
#   sbatch --array=0-N --export=ALL,LIST=<file of "label policy_path" lines> eval_packed.sbatch
# Task A takes lines 4A..4A+3. Settings match north's evaluations exactly.
# COST: ~15 min x 0.4 per node = ~0.1 node-hours per 4 arms.
#SBATCH -J kyle-evalpaths
#SBATCH -p mi2104x
#SBATCH -N 1
#SBATCH -t 03:00:00
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
mkdir -p $K/qrun/eval_paths $K/logs/eval_paths
echo "task $A  host $(hostname)  job ${SLURM_JOB_ID}  list $LIST  code $(cat $K/qrun/.source_commit)"
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
    run_eval $K/logs/eval_paths/$LABEL.log \
      python -u evaluate.py --policy $POLICY \
      --urdf $K/qrun/assets/data/robot/go2_irrvis/urdf/go2_description.urdf \
      --corpus $K/qrun/data/go2_crm_v2 --out $K/qrun/eval_paths/$LABEL --label $LABEL \
      --terrain crm --paths 4 --seconds 15 --spawn-spread 1.0
    rc=$?
    echo "EVAL $LABEL exit $rc on GPU $G"
    [ $rc -ne 0 ] && echo "ARM_FAILED $LABEL rc=$rc" ) &
  G=$((G + 1))
done < <(sed -n "$((4 * A + 1)),$((4 * A + 4))p" $LIST)
wait
echo "TASK_${A}_DONE"
