#!/usr/bin/env bash
# Build the folder (and .tar.gz) that run_student_real.py needs on the robot's Jetson AGX Orin: the nedm.so101_policy
# Python sources, this folder's scripts, the SO101 collision points (table floor without mujoco), the student
# checkpoint, calib_v1.json and assets/deploy.json (the floor mode of the student, read from its checkpoint, the table
# plane source, HOME and the eval guard: robot_test.sh passes them to every run). It writes locally only; copying it to
# the Orin is the user's step (README.md).
#
#   scripts/so101_policy/real2sim/hw/make_orin_bundle.sh <student.pt> [out_dir] [calib_v1.json]
#   HOME_SIM_DEG=a,b,c,d,e,f ...   another HOME (sim deg; default hw_lib.ROBOT_HOME_SIM_DEG = 0,-5,-15,92,0,30)
#   PYTHON=<python with torch> ... the Python that reads the checkpoint (default python3)
set -euo pipefail
HW="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HW/../../../.." && pwd)"
CKPT="${1:?usage: make_orin_bundle.sh <student.pt> [out_dir] [calib_v1.json]}"
OUT="${2:-$ROOT/artifacts/so101_pick_place_repro/bundle/so101_hw}"
CALIB="${3:-$ROOT/configs/so101_pick_place/calib_v1.json}"
POINTS="${SO101_KIN_POINTS:-$ROOT/artifacts/so101_pick_place/rl_inputs/geometry/kin_points_v1.npz}"
PY="${PYTHON:-python3}"

[ -f "$CKPT" ] || { echo "student checkpoint $CKPT missing"; exit 1; }
[ -f "$POINTS" ] || { echo "kin points $POINTS missing"; exit 1; }
rm -rf "$OUT"
mkdir -p "$OUT/src/nedm/so101_policy/vision" "$OUT/src/nedm/render" "$OUT/scripts/so101_policy/real2sim/hw" "$OUT/assets"
cp "$ROOT/src/nedm/__init__.py" "$OUT/src/nedm/"
cp "$ROOT/src/nedm/render/student_view.py" "$OUT/src/nedm/render/"       # the crop code shared with the renderer
cp "$ROOT"/src/nedm/so101_policy/*.py "$OUT/src/nedm/so101_policy/"
cp "$ROOT"/src/nedm/so101_policy/vision/*.py "$OUT/src/nedm/so101_policy/vision/"
cp "$HW/hw_lib.py" "$HW/run_student_real.py" "$HW/README.md" "$OUT/scripts/so101_policy/real2sim/hw/"
cp "$HW/robot_test.sh" "$HW/QDGAIN_TEST.md" "$OUT/"                     # the launcher + the qd-gain test plan (bundle root)
cp "$POINTS" "$OUT/assets/kin_points_v1.npz"
cp "$CKPT" "$OUT/assets/student.pt"
if [ -f "$CALIB" ]; then cp "$CALIB" "$OUT/assets/calib_v1.json"; else echo "NOTE: no $CALIB (live mode will refuse)"; fi
# the deploy settings, from the bundle's own copies (the bundle's sources: the same rules as on the robot)
HOME_ARG=(); [ -n "${HOME_SIM_DEG:-}" ] && HOME_ARG=(--home "$HOME_SIM_DEG")
NEDM_SRC="$OUT/src" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" "$PY" \
  "$OUT/scripts/so101_policy/real2sim/hw/run_student_real.py" deploy-json --student "$OUT/assets/student.pt" \
  --calib "$OUT/assets/calib_v1.json" --out "$OUT/assets/deploy.json" "${HOME_ARG[@]}"
find "$OUT" -name __pycache__ -prune -exec rm -rf {} +                 # only the built files in the manifest
FLOOR_MODE="$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["floor_mode"])' "$OUT/assets/deploy.json")"
{
  echo "built $(date -Iseconds) from $ROOT (git $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo ?)"\
       "$(git -C "$ROOT" status --porcelain -- src/nedm/so101_policy scripts/so101_policy/real2sim/hw 2>/dev/null | wc -l)"\
       "changed files under the copied paths)"
  echo "student $CKPT (floor mode $FLOOR_MODE, assets/deploy.json)"      # header: 3 lines (robot_test.sh: lines 4-)
  echo "calib $CALIB"
  (cd "$OUT" && find . -type f ! -name MANIFEST.txt -print0 | sort -z | xargs -0 sha256sum)
} > "$OUT/MANIFEST.txt"
tar -C "$(dirname "$OUT")" -czf "$OUT.tar.gz" "$(basename "$OUT")"
du -sh "$OUT" "$OUT.tar.gz"
echo "bundle: $OUT.tar.gz  floor mode $FLOOR_MODE  (copy it to the Orin yourself, e.g. scp $OUT.tar.gz sbel@proxy:~/ ;"\
     "README.md)"
