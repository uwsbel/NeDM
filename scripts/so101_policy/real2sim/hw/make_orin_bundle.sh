#!/usr/bin/env bash
# Build the folder (and .tar.gz) that run_student_real.py needs on the robot's Jetson AGX Orin: the nedm.so101_policy
# Python sources, this folder's scripts, the SO101 collision points (table floor without mujoco), the student
# checkpoint and calib_v1.json. It writes locally only; copying it to the Orin is the user's step (README.md).
#
#   scripts/so101_policy/real2sim/hw/make_orin_bundle.sh <student.pt> [out_dir] [calib_v1.json]
set -euo pipefail
HW="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HW/../../../.." && pwd)"
CKPT="${1:?usage: make_orin_bundle.sh <student.pt> [out_dir] [calib_v1.json]}"
OUT="${2:-$HOME/so101_r2s_claude/hw_bundle/so101_hw}"
CALIB="${3:-$HOME/so101_r2s_claude/calib/calib_v1.json}"
POINTS="${SO101_KIN_POINTS:-$HOME/so101_policy/ckpt/kin_points_v1.npz}"

[ -f "$CKPT" ] || { echo "student checkpoint $CKPT missing"; exit 1; }
[ -f "$POINTS" ] || { echo "kin points $POINTS missing"; exit 1; }
rm -rf "$OUT"
mkdir -p "$OUT/src/nedm/so101_policy/vision" "$OUT/scripts/so101_policy/real2sim/hw" "$OUT/assets"
cp "$ROOT/src/nedm/__init__.py" "$OUT/src/nedm/"
cp "$ROOT"/src/nedm/so101_policy/*.py "$OUT/src/nedm/so101_policy/"
cp "$ROOT"/src/nedm/so101_policy/vision/*.py "$OUT/src/nedm/so101_policy/vision/"
cp "$HW/hw_lib.py" "$HW/run_student_real.py" "$HW/README.md" "$OUT/scripts/so101_policy/real2sim/hw/"
cp "$POINTS" "$OUT/assets/kin_points_v1.npz"
cp "$CKPT" "$OUT/assets/student.pt"
if [ -f "$CALIB" ]; then cp "$CALIB" "$OUT/assets/calib_v1.json"; else echo "NOTE: no $CALIB (live mode will refuse)"; fi
{
  echo "built $(date -Iseconds) from $ROOT (git $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo ?)"\
       "$(git -C "$ROOT" status --porcelain -- src/nedm/so101_policy scripts/so101_policy/real2sim/hw 2>/dev/null | wc -l)"\
       "changed files under the copied paths)"
  echo "student $CKPT"
  echo "calib $CALIB"
  (cd "$OUT" && find . -type f ! -name MANIFEST.txt -print0 | sort -z | xargs -0 sha256sum)
} > "$OUT/MANIFEST.txt"
tar -C "$(dirname "$OUT")" -czf "$OUT.tar.gz" "$(basename "$OUT")"
du -sh "$OUT" "$OUT.tar.gz"
echo "bundle: $OUT.tar.gz  (copy it to the Orin yourself, e.g. scp $OUT.tar.gz sbel@proxy:~/ ; README.md)"
