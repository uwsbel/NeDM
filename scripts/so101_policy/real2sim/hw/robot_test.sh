#!/usr/bin/env bash
# One command for each robot test on the Orin (make_orin_bundle.sh copies it to the bundle root; README.md).
#
#   ./robot_test.sh check                 files, Python env, robot port, cameras, calibration (opens no device)
#   ./robot_test.sh replay                3 recordings, no robot (model time on this computer)
#   ./robot_test.sh dry [s] [args]        robot + cameras, reads only, no motor writes (default 30 s)
#   ./robot_test.sh live [s] [args]       MOVES THE ROBOT (default 15 s)
#   ./robot_test.sh live-tilt [s] [args]  live + the table-tilt correction --q-offset-deg 0,-2.086,0,0,0 (README)
#   ./robot_test.sh eval [n] [s] [args]   MOVES THE ROBOT: n episodes in a row (default 10 x 40 s), servo correction
#
# [args] go to run_student_real.py as they are. Outputs: ~/so101_hw_runs/<mode>_<time>/ (RUNS overrides).
set -eo pipefail
SELF="$(readlink -f "${BASH_SOURCE[0]}")"
cd "$(dirname "$SELF")"
MODE="${1:-}"
[ $# -gt 0 ] && shift
RUN=scripts/so101_policy/real2sim/hw/run_student_real.py
S=(--student assets/student.pt --calib assets/calib_v1.json)
RUNS="${RUNS:-$HOME/so101_hw_runs}"
DATA_ROOT="${DATA_ROOT:-$HOME/lerobot/data}"
TILT=(--q-offset-deg 0,-2.086,0,0,0)
# eval: floor stops below the real table (it is 4-17 mm below the sim table), integral correction of the servo error
EVAL=(--amp 0 --guard '{"cmd_floor_m":-0.015,"hard_floor_m":-0.020,"i_time_s":0.2,"i_max_deg":3.0}')
export HF_HUB_OFFLINE=1

usage() { sed -n '2,11p' "$SELF" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

activate() {
  if [ "${CONDA_DEFAULT_ENV:-}" != lerobot ]; then
    # shellcheck disable=SC1091
    source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}" && conda activate lerobot \
      || { echo "cannot activate the conda env lerobot"; exit 1; }
  fi
}

# the bundle files are the ones that were built (MANIFEST.txt lines 4-)
files_ok() { sed -n '4,$p' MANIFEST.txt | sha256sum -c --quiet; }

need_tty() {
  [ -t 0 ] && [ -t 1 ] || { echo "run this in an interactive terminal (desktop or ssh -t): the stop keys need a TTY"; exit 1; }
}

# the launcher sets these itself (the banner must tell what the run does): refuse them in [args]
no_overrides() {
  local a
  for a in "$@"; do
    case "$a" in
      --time-limit|--time-limit=*|--q-offset-deg|--q-offset-deg=*|--out|--out=*|--backend|--backend=*|--auto-confirm|\
      --episodes-n|--episodes-n=*)
        echo "do not give $a here: use [n] [s] for the episodes and the time, the mode live-tilt for the correction"
        exit 1 ;;
    esac
  done
}

secs() { [[ "${1:-}" =~ ^[0-9]+([.][0-9]+)?$ ]]; }

out_dir() { echo "$RUNS/${1}_$(date +%Y%m%d_%H%M%S)"; }

case "$MODE" in
  check)
    activate
    st=0
    echo "== bundle files"
    if files_ok; then echo "ok ($(head -1 MANIFEST.txt))"; else echo "CHANGED OR MISSING (above)"; st=1; fi
    echo "== Python env ($CONDA_DEFAULT_ENV)"
    python - <<'EOF' || st=1
import importlib.metadata as m
import cv2, numpy, pyarrow, torch
import lerobot.robots.so_follower, lerobot.cameras.opencv  # import only, opens nothing
print(f"lerobot {m.version('lerobot')}, torch {torch.__version__}, cuda {torch.cuda.is_available()}, "
      f"cv2 {cv2.__version__}, numpy {numpy.__version__}, pyarrow {pyarrow.__version__}")
assert torch.cuda.is_available(), "no CUDA: the student would run on the CPU"
EOF
    echo "== robot port, external camera, wrist camera (the defaults of run_student_real.py)"
    python - <<'EOF' || st=1
import os, sys
sys.path.insert(0, "scripts/so101_policy/real2sim/hw")
import run_student_real as R  # opens no device
paths = [R.FOLLOWER_PORT, R.CAM_PATHS["external"], R.CAM_PATHS["wrist"]]
bad = 0
for p in paths:
    if os.path.exists(p):
        print(f"ok  {p} -> {os.path.realpath(p)}")
    else:
        print(f"MISSING  {p} (unplugged? another USB port? README)"); bad = 1
for b in R.device_users(paths):
    print("IN USE (dry / live will refuse):", b); bad = 1
sys.exit(bad)
EOF
    echo "== LeRobot calibration of the follower"
    c="$HOME/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json"
    if [ -f "$c" ]; then echo "ok  $c"; else echo "MISSING  $c"; st=1; fi
    echo "== power mode"
    nvpmodel -q 2>/dev/null | head -1 || true
    [ $st -eq 0 ] && echo "CHECK OK" || echo "CHECK FOUND PROBLEMS (above)"
    exit $st
    ;;
  replay)
    activate
    O="$(out_dir replay)"
    echo "replay (no robot) -> $O"
    exec python "$RUN" replay "${S[@]}" --data-root "$DATA_ROOT" \
      --episodes green_cube_red_plate:0,green_cube_red_plate:2,green_cube_red_plate_1:5 --out "$O" "$@"
    ;;
  dry|live|live-tilt|eval)
    need_tty
    activate
    files_ok || { echo "bundle files changed or missing (MANIFEST.txt): refusing"; exit 1; }
    T=30; [ "$MODE" != dry ] && T=15
    N=1
    if [ "$MODE" = eval ]; then
      N=10; T=40
      if [[ "${1:-}" =~ ^[0-9]+$ ]]; then N="$1"; shift; fi
    fi
    if secs "${1:-}"; then T="$1"; shift; fi
    no_overrides "$@"
    EXTRA=(); [ "$MODE" = live-tilt ] && EXTRA=("${TILT[@]}")
    [ "$MODE" = eval ] && EXTRA=("${EVAL[@]}" --episodes-n "$N")
    O="$(out_dir "$MODE")"
    # run_student_real.py refuses dry / live while another program can use the port or a camera (device_users)
    if [ "$MODE" = dry ]; then
      echo "DRY run, $T s: the robot and the cameras are read only; no motor writes. -> $O"
      exec python "$RUN" dry "${S[@]}" --out "$O" --time-limit "$T" "$@"
    fi
    if [ "$MODE" = eval ]; then
      echo "EVAL session, $N episodes of up to $T s: THIS MOVES THE ROBOT. -> $O"
      echo "Settings: ${EVAL[*]}"
      cat <<EOF
  1. Arm in the folded rest pose, workspace clear, one hand near the arm.
  2. Type MOVE: the arm goes to HOME (wrist roll about 120 deg; watch the wrist camera cable).
  3. Each episode: put the cube and the plate 0.15-0.27 m from the base, type RUN. Push space when the episode is
     done (or it ends at $T s). Type the result (g grasp, l lift, p placed; for example glp, or -).
  4. Type h: the arm goes back to HOME (the gripper opens), then reset the scene for the next episode.
  5. Stop at any time: q / space / Esc / Ctrl+C. Type s to end the session: menu r return, t torque off.
EOF
    else
      echo "LIVE run, $T s: THIS MOVES THE ROBOT. -> $O"
      if [ "$MODE" = live-tilt ]; then echo "Table-tilt correction ON: ${TILT[*]}"; fi
      cat <<EOF
  1. Arm in the folded rest pose, workspace clear, one hand near the arm.
  2. Type MOVE: the arm goes to HOME (11-13 s, <= 15 deg/s, the wrist roll turns about 120 deg).
  3. Put the cube and the plate 0.15-0.27 m from the shoulder pan axis, then type RUN.
  4. Stop at any time: q / space / Esc / Ctrl+C. At the end the torque stays ON (menu: r return, t torque off).
EOF
    fi
    exec python "$RUN" live --i-am-supervising "${S[@]}" --out "$O" --time-limit "$T" "${EXTRA[@]}" "$@"
    ;;
  ""|-h|--help|help) usage 0 ;;
  *) echo "unknown mode: $MODE"; usage 1 ;;
esac
