#!/usr/bin/env bash
# One command for each robot test on the Orin (make_orin_bundle.sh copies it to the bundle root; README.md).
#
#   ./robot_test.sh check                 files, Python env, robot port, cameras, calibration (opens no device)
#   ./robot_test.sh replay                3 recordings, no robot (model time on this computer)
#   ./robot_test.sh dry [s] [args]        robot + cameras, reads only, no motor writes (default 30 s)
#   ./robot_test.sh live [s] [args]       MOVES THE ROBOT (default 15 s)
#   ./robot_test.sh live-tilt [s] [args]  live + the table-tilt correction --q-offset-deg 0,-2.086,0,0,0 (README)
#
# [args] go to run_student_real.py as they are. Outputs: ~/so101_hw_runs/<mode>_<time>/ (RUNS overrides).
set -eo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
MODE="${1:-}"
[ $# -gt 0 ] && shift
RUN=scripts/so101_policy/real2sim/hw/run_student_real.py
S=(--student assets/student.pt --calib assets/calib_v1.json)
RUNS="${RUNS:-$HOME/so101_hw_runs}"
DATA_ROOT="${DATA_ROOT:-$HOME/lerobot/data}"
TILT=(--q-offset-deg 0,-2.086,0,0,0)
export HF_HUB_OFFLINE=1

usage() { sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

activate() {
  if [ "${CONDA_DEFAULT_ENV:-}" != lerobot ]; then
    # shellcheck disable=SC1091
    source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}" && conda activate lerobot \
      || { echo "cannot activate the conda env lerobot"; exit 1; }
  fi
}

# robot port + camera paths = the defaults of run_student_real.py (no device is opened) -> DEVS
load_devices() {
  mapfile -t DEVS < <(devices)
  [ "${#DEVS[@]}" -eq 3 ] || { echo "cannot read the robot port and camera paths from $RUN"; exit 1; }
}
devices() {
  python - <<'EOF'
import json, sys
sys.path.insert(0, "scripts/so101_policy/real2sim/hw")
import run_student_real as R
print(R.FOLLOWER_PORT)
for k in ("external", "wrist"):
    print(R.CAM_PATHS[k])
EOF
}

# 0 = no other process has the robot port or a camera open (fuser reads /proc only)
devices_free() {
  local ok=0 p dev pids
  for p in "${DEVS[@]}"; do
    dev="$(readlink -f "$p")"
    pids="$(fuser "$dev" 2>/dev/null | xargs || true)"
    if [ -n "$pids" ]; then
      echo "IN USE: $p -> $dev by PID $pids:"
      ps -o pid=,cmd= -p "${pids// /,}" | cut -c1-150
      ok=1
    fi
  done
  return $ok
}

# the bundle files are the ones that were built (MANIFEST.txt lines 4-)
files_ok() { sed -n '4,$p' MANIFEST.txt | sha256sum -c --quiet; }

need_tty() {
  [ -t 0 ] && [ -t 1 ] || { echo "run this in an interactive terminal (desktop or ssh -t): the stop keys need a TTY"; exit 1; }
}

secs() { [[ "${1:-}" =~ ^[0-9]+([.][0-9]+)?$ ]]; }

out_dir() { echo "$RUNS/${1}_$(date +%Y%m%d_%H%M%S)"; }

case "$MODE" in
  check)
    activate
    load_devices
    st=0
    echo "== bundle files"
    if files_ok; then echo "ok ($(head -1 MANIFEST.txt))"; else st=1; fi
    echo "== Python env ($CONDA_DEFAULT_ENV)"
    python - <<'EOF' || st=1
import importlib.metadata as m
import cv2, numpy, pyarrow, torch
import lerobot.robots.so_follower, lerobot.cameras.opencv  # import only, opens nothing
print(f"lerobot {m.version('lerobot')}, torch {torch.__version__}, cuda {torch.cuda.is_available()}, "
      f"cv2 {cv2.__version__}, numpy {numpy.__version__}, pyarrow {pyarrow.__version__}")
assert torch.cuda.is_available(), "no CUDA: the student would run on the CPU"
EOF
    echo "== robot port, external camera, wrist camera"
    for p in "${DEVS[@]}"; do
      if [ -e "$p" ]; then echo "ok  $p -> $(readlink -f "$p")"; else echo "MISSING  $p (unplugged? see README)"; st=1; fi
    done
    devices_free || st=1
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
  dry|live|live-tilt)
    need_tty
    activate
    load_devices
    files_ok || { echo "bundle files changed or missing (MANIFEST.txt): refusing"; exit 1; }
    T=30; [ "$MODE" != dry ] && T=15
    if secs "${1:-}"; then T="$1"; shift; fi
    devices_free || { echo "stop the other process first (teleop / record / another run), then start again"; exit 1; }
    EXTRA=(); [ "$MODE" = live-tilt ] && EXTRA=("${TILT[@]}")
    O="$(out_dir "$MODE")"
    if [ "$MODE" = dry ]; then
      echo "DRY run, $T s: the robot and the cameras are read only; no motor writes. -> $O"
      exec python "$RUN" dry "${S[@]}" --out "$O" --time-limit "$T" "$@"
    fi
    cat <<EOF
LIVE run, $T s: THIS MOVES THE ROBOT. -> $O
$( [ "$MODE" = live-tilt ] && echo "Table-tilt correction ON: ${TILT[*]}" )
  1. Arm in the folded rest pose, workspace clear, one hand near the arm.
  2. Type MOVE: the arm goes to HOME (11-13 s, <= 15 deg/s, the wrist roll turns about 120 deg).
  3. Put the cube and the plate 0.15-0.27 m from the shoulder pan axis, then type RUN.
  4. Stop at any time: q / space / Esc / Ctrl+C. At the end the torque stays ON (menu: r return, t torque off).
EOF
    exec python "$RUN" live --i-am-supervising "${S[@]}" --out "$O" --time-limit "$T" "${EXTRA[@]}" "$@"
    ;;
  ""|-h|--help|help) usage 0 ;;
  *) echo "unknown mode: $MODE"; usage 1 ;;
esac
