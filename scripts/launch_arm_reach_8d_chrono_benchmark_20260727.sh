#!/usr/bin/env bash
# 100-goal seeded Chrono transfer battery for the 8-D-ROM arm reaching policy.
#
# The 8-D analog of the 12-D battery at
#   artifacts/rl_runs_arm_goal_reach/arm_reach_..._noee12d_rom_20260724/chrono_reach_benchmark_N100_seed12345
# and deliberately byte-comparable to it: same harness (scripts/benchmark_arm_reach_chrono.py),
# same N=100, same seed=12345, same 5 cm tolerance. The two runs' env_cfg goal blocks are
# identical (q_lo/q_hi/max_sample_attempts), so the seeded sampler yields the SAME 100 EE
# goals -- the script asserts this against the 12-D goals.json before spending ~45 min of
# Chrono time, so a config drift fails fast instead of producing a silently unmatched pair.
#
# Notes carried over from the earlier batteries:
#   * ONE fresh Chrono process per goal -- the arm env's reset_idx rebuilds the whole
#     M113+arm scene, and repeated in-process sim re-creation has stack-smashed before.
#   * NEVER run two Chrono batteries concurrently (machine-freeze history); this script
#     refuses to start if another arm/tracked battery is already running.
#   * The success metric uses the REAL Chrono gripper position, not FK -- for the 8-D ROM
#     ArmReachingChronoEnv.current_ee_base reads ee_base_buf captured from the sim, exactly
#     as the 12-D battery did, so the two success rates measure the same thing.
#   * pychrono needs the conda env's libstdc++ ahead of the system one, else importing it
#     dies with `CXXABI_1.3.15 not found`. Set explicitly so this works from any shell.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/harry/anaconda3/envs/nedm/bin/python}"
CONDA_LIB="${CONDA_LIB:-/home/harry/anaconda3/envs/nedm/lib}"
RUN_DIR="${RUN_DIR:-artifacts/rl_runs_arm_goal_reach/arm_reach_adaptivekl005_lr1e4_tol005_ep150_bonus150_sigma015_8d_rom_20260727}"
REF_DIR="${REF_DIR:-artifacts/rl_runs_arm_goal_reach/arm_reach_adaptivekl005_lr1e4_tol005_ep150_bonus150_sigma015_noee12d_rom_20260724}"
NUM_GOALS="${NUM_GOALS:-100}"
SEED="${SEED:-12345}"
SESSION_NAME="${SESSION_NAME:-arm_reach_8d_chrono_bench}"
OUT_DIR="$RUN_DIR/chrono_reach_benchmark_N${NUM_GOALS}_seed${SEED}"
LAUNCH_LOG="$RUN_DIR/chrono_benchmark_N${NUM_GOALS}_seed${SEED}.launch.log"

export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

# --- serialize: never two Chrono batteries at once -------------------------------------
# Match the python interpreter running one of the eval scripts, NOT any shell whose
# command line merely mentions them (a bare name match hits this launcher itself).
CHRONO_PAT='python[^ ]* .*(eval_arm_rl_chrono_reaching|benchmark_arm_reach_chrono|benchmark_tracked_goal_chrono|eval_tracked_rl_goal_chrono)\.py'
if pgrep -f "$CHRONO_PAT" >/dev/null 2>&1; then
  echo "ERROR: a Chrono battery/eval is already running -- refusing to start a second one."
  echo "       (machine-freeze history; run them one at a time)"
  pgrep -af "$CHRONO_PAT" | head
  exit 1
fi

if [[ ! -f "$RUN_DIR/env_cfg.json" ]]; then
  echo "ERROR: $RUN_DIR/env_cfg.json not found"; exit 1
fi

echo "run dir : $RUN_DIR"
echo "output  : $OUT_DIR"
echo "goals   : N=$NUM_GOALS seed=$SEED (must match the 12-D battery)"

bench_cmd=(
  "$PYTHON_BIN" scripts/benchmark_arm_reach_chrono.py
  --run-dir "$RUN_DIR"
  --num-goals "$NUM_GOALS"
  --seed "$SEED"
  --device cuda
)

# --- goal-set guard: fail fast if the sampled goals would not match the 12-D battery ----
REF_GOALS="$REF_DIR/chrono_reach_benchmark_N${NUM_GOALS}_seed${SEED}/goals.json"
if [[ -f "$REF_GOALS" ]]; then
  # Reuse the benchmark's own sampler so the guard exercises the identical code path.
  "$PYTHON_BIN" -c "
import json, sys, numpy as np
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from benchmark_arm_reach_chrono import sample_goals
from pathlib import Path
goals, _ = sample_goals(Path('$RUN_DIR'), $NUM_GOALS, $SEED, 'cuda')
ref = np.asarray(json.load(open('$REF_GOALS'))['goals_base'], dtype=np.float32)
d = np.abs(goals - ref).max()
print(f'[goal-guard] max |8D goals - 12D goals| = {d:.3e}')
assert d < 1e-5, 'goal sets differ -- the two batteries would not be comparable'
print('[goal-guard] goal sets are identical; batteries are directly comparable')
" || { echo "ERROR: goal-set guard failed"; exit 1; }
else
  echo "WARN: reference goals.json not found at $REF_GOALS -- skipping the comparability guard"
fi

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "battery already running in tmux session $SESSION_NAME"; exit 0
  fi
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" \
    "$(printf '%q ' "${bench_cmd[@]}") > $(printf '%q' "$REPO_ROOT/$LAUNCH_LOG") 2>&1"
  echo "started 8-D arm Chrono battery in tmux session $SESSION_NAME"
  echo "log: $LAUNCH_LOG"
  echo "attach: tmux attach -t $SESSION_NAME"
  exit 0
fi

"${bench_cmd[@]}" 2>&1 | tee "$LAUNCH_LOG"
