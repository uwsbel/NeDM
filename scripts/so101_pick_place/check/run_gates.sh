#!/usr/bin/env bash
# The sim checks of a camera student (the v12 selection gates, results/sim_checks/SELECTION_v12.md on the Hub): G1
# nominal success on the sim table, G2 the robot-like check, G3 the table plane -3 / +3 mm, the B3 release veto and
# the G4 numbers (reported only). Every run is one check_eval.py evaluation (CPU MuJoCo + GPU renders).
#
#   run_gates.sh run <name> <student.pt> <root> [ref]    the evaluations one after the other -> <root>/runs/gates/local/<name>/
#   run_gates.sh summary <name> <root> [ref_name]        -> <root>/out/gates/local/<name>.json (gates_summary.py, against
#                                                        D's runs in <root>/runs/gates/local/<ref_name>, default D)
# "ref" adds D's extra reference runs (G2 repeats b / c, the H68 grey repeat): run D once with "ref" before the
# summary of another student (name D, checkpoint student/s4_v8_roll_s801/it_02000.pt). Student names for the summary:
# v12_<arm>_s<seed>_it<1000|1500> (the released student: v12_tb10_s1210_it1000).
#
# Per checkpoint (9 evaluations; ref: + 3):
#   g1_dev100, g1_homedev100   sim table, nominal, no servo model, roll cap 98, 1,500 steps, hold cap 20 s, own look
#   g2_a                       J224, measured plane, flat floor, fit_d2q, 40 s, own look (file seeds)
#   g2_b, g2_c (ref)           the same, seeds 613,600 / 613,900
#   g3_m3, g3_p3               g2_a with the plane height -3 / +3 mm
#   g4_h68_RL_a, g4_h68_RL_b   H68 x 4 draws, 8 s, real look (arm RL), seeds 613,000 / 613,300 (B3 veto: both blocks)
#   g4_h68_GR_a                H68 grey look, seeds 613,000 (G4 (d) contrast, reported)
#   g4_h68_GR_b (ref)          seeds 613,300
#   g4_j224_RL                 J224 in arm RL (G4 (c), reported)
#
# Environment: PY (Python with torch, mujoco, warp-lang, newton; default python3), WORKERS (CPU MuJoCo workers,
# default 6), RUNNER (a command prefix for every evaluation, e.g. a memory cap), ONLY (comma list of run names: run
# only these), HORIZON / TASK_ROWS (smoke only: shorter episodes / the first rows of the task file; the gate rules
# need the full sets).
# Ported from the study's runner ~/so101_r2s_claude/v12/check/run_gates.sh + lib/run_one.sh (local mode; the AMD
# mode, the GPU lock and the memory cap of the study machine are left out). The evaluation list is the same.
set -uo pipefail
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CK="$CODE/scripts/so101_pick_place/check"
A="$CODE/artifacts/so101_pick_place"
J=$A/rl_inputs/check_tasks/jit224.npz
H=$A/rl_inputs/check_tasks/h68.npz
DEV=$A/rl_inputs/tasks/dev100.npz
HDEV=$A/rl_inputs/tasks/home_dev100.npz
DYN_FILE=$CODE/configs/so101_pick_place/dyn_fit_d2q.json
PY=${PY:-python3}
MODE=${1:?mode: run | summary}

list() {   # <ref>: one line per evaluation: <dyn> <run> <check_eval args>
  local REF=$1
  echo "fit_d2q g2_a --arm own --tasks $J --horizon 2000"
  echo "fit_d2q g4_j224_RL --arm RL --tasks $J --horizon 2000"
  echo "fit_d2q g3_m3 --arm own --tasks $J --horizon 2000 --plane-dz -0.003"
  echo "fit_d2q g3_p3 --arm own --tasks $J --horizon 2000 --plane-dz 0.003"
  [ "$REF" = ref ] && echo "fit_d2q g2_b --arm own --tasks $J --horizon 2000 --seed-base 613600"
  [ "$REF" = ref ] && echo "fit_d2q g2_c --arm own --tasks $J --horizon 2000 --seed-base 613900"
  echo "none g1_dev100 --arm own --tasks $DEV --plane sim --horizon 1500 --hold-cap-s 20 --roll-hi-deg 98"
  echo "none g1_homedev100 --arm own --tasks $HDEV --plane sim --horizon 1500 --hold-cap-s 20 --roll-hi-deg 98"
  echo "fit_d2q g4_h68_RL_a --arm RL --tasks $H --draws 4 --seed-base 613000 --horizon 400"
  echo "fit_d2q g4_h68_RL_b --arm RL --tasks $H --draws 4 --seed-base 613300 --horizon 400"
  echo "fit_d2q g4_h68_GR_a --arm GR --tasks $H --draws 4 --seed-base 613000 --horizon 400"
  [ "$REF" = ref ] && echo "fit_d2q g4_h68_GR_b --arm GR --tasks $H --draws 4 --seed-base 613300 --horizon 400"
  return 0
}

export PYTHONPATH="$CODE/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
cd "$CODE"
case $MODE in
run)
  NAME=${2:?name}; CKPT=$(readlink -f "${3:?student checkpoint}"); ROOT=${4:?output root}; REF=${5:-}
  [ -f "$CKPT" ] || { echo "run_gates: no checkpoint $CKPT"; exit 2; }
  O=$ROOT/runs/gates/local/$NAME; mkdir -p "$O" "$ROOT/logs"
  T0=$(date +%s)
  echo "[$(date +%T)] run_gates $NAME $CKPT $REF" | tee -a "$O/gates.log"
  list "$REF" | while read -r dyn run args; do
    if [ -n "${ONLY:-}" ] && [[ ",$ONLY," != *",$run,"* ]]; then continue; fi
    [ -f "$O/$run.npz" ] && { echo "skip $run (exists)" >> "$O/gates.log"; continue; }
    if [ -n "${TASK_ROWS:-}" ]; then        # smoke: the first TASK_ROWS rows of the task file
      set -- $args; tf=""; rest=()
      while [ $# -gt 0 ]; do if [ "$1" = --tasks ]; then tf=$2; shift 2; else rest+=("$1"); shift; fi; done
      sub=$O/_rows_${run}.npz
      "$PY" -c "import numpy as np, sys; t = dict(np.load(sys.argv[1], allow_pickle=True)); n = len(t['seeds']); k = int(sys.argv[3]); np.savez(sys.argv[2], **{a: (v[:k] if np.ndim(v) and len(v) == n else v) for a, v in t.items()})" "$tf" "$sub" "$TASK_ROWS"
      args="${rest[*]} --tasks $sub"
    fi
    [ -n "${HORIZON:-}" ] && args="$args --horizon $HORIZON"
    if [ "$dyn" = none ]; then export NEDM_SO101_DYN=""; else export NEDM_SO101_DYN="$(cat "$DYN_FILE")"; fi
    t=$(date +%s)
    ${RUNNER:-} "$PY" -B "$CK/check_eval.py" eval --tag "gates_${NAME}_$run" --workers "${WORKERS:-6}" --student "$CKPT" \
        --out "$O/$run.npz" $args > "$ROOT/logs/gates_${NAME}_$run.log" 2>&1 < /dev/null
    rc=$?
    rm -f "$O/_rows_${run}.npz"
    echo "[$(date +%T)] $run rc $rc wall $(( $(date +%s) - t )) s" | tee -a "$O/gates.log"
  done
  echo "[$(date +%T)] run_gates $NAME done, wall $(( $(date +%s) - T0 )) s" | tee -a "$O/gates.log";;
summary)
  NAME=${2:?name}; ROOT=${3:?root}; REFN=${4:-D}
  mkdir -p "$ROOT/out/gates/local"
  ${RUNNER:-} "$PY" -B "$CK/gates_summary.py" --platform local --name "$NAME" --ref "$REFN" --root "$ROOT" \
      --out "$ROOT/out/gates/local/$NAME.json";;
*) sed -n 2,26p "$0"; exit 2;;
esac
