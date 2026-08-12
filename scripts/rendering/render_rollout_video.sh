#!/usr/bin/env bash
# Render a Chrono HMMWV policy rollout to an mp4.
#
# Stage 3 of the Blender render pipeline: drives blender-render/render_hmmwv_scene.py
# over a frame range and encodes the result.
#
# Blender 5.1.2 segfaults in native code under sustained Cycles load on this box (see
# the newton-instability note), so frames are rendered in chunks -- a fresh Blender
# process per chunk plus --skip-existing -- and the whole thing is retried until every
# frame exists. A crash therefore costs one chunk, not the whole sequence.
#
# Usage:
#   scripts/rendering/render_rollout_video.sh <export-dir> <rollout-npz> <out-mp4> [label]
#
# <export-dir>  directory holding exported.assets.py + output/stateNNNNN.py, as written
#               by eval_hmmwv_rl_chrono_tracking.py --blender-output-dir
# <rollout-npz> chrono_tracking_NN.npz from the same eval (pose + ref_pose)
set -euo pipefail

EXPORT_DIR=${1:?usage: render_rollout_video.sh <export-dir> <rollout-npz> <out-mp4> [label]}
ROLLOUT_NPZ=${2:?missing rollout npz}
OUT_MP4=${3:?missing output mp4}
LABEL=${4:-}

BLENDER=${BLENDER:-$HOME/blender-5.1.2-linux-x64/blender}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RENDER_PY="$REPO_ROOT/blender-render/render_hmmwv_scene.py"

TERRAIN=${TERRAIN:-rigid}
WIDTH=${WIDTH:-1600}
HEIGHT=${HEIGHT:-900}
SAMPLES=${SAMPLES:-96}
AZIMUTH=${AZIMUTH:-180}
ELEVATION=${ELEVATION:-40}
FPS=${FPS:-20}
CHUNK=${CHUNK:-60}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-12}
GPU_FLAG=${GPU_FLAG:---gpu}
FRAMES_DIR=${FRAMES_DIR:-$EXPORT_DIR/frames}
# Extra pass-through args for the render script. CRM needs, at minimum,
#   EXTRA_ARGS="--crm-surface-dir $EXPORT_DIR/crm_surface --terrain-z-m 0.25"
# so the soil mesh is swapped per frame and the trajectory tubes sit on top of the
# soil rather than inside it.
EXTRA_ARGS=${EXTRA_ARGS:-}

NUM_STATES=$(find "$EXPORT_DIR/output" -name 'state*.py' | wc -l)
LAST=$((NUM_STATES - 1))
if [ "$NUM_STATES" -eq 0 ]; then
  echo "no state files in $EXPORT_DIR/output" >&2
  exit 1
fi
mkdir -p "$FRAMES_DIR"
echo "rendering $NUM_STATES frames from $EXPORT_DIR -> $FRAMES_DIR"

render_chunk() {
  local start=$1 end=$2
  "$BLENDER" --background --factory-startup --python "$RENDER_PY" -- \
    --assets "$EXPORT_DIR/exported.assets.py" \
    --terrain "$TERRAIN" \
    --rollout "$ROLLOUT_NPZ" \
    --animate --anim-dir "$FRAMES_DIR" \
    --anim-start "$start" --anim-end "$end" --skip-existing \
    --camera-azimuth-deg "$AZIMUTH" --camera-elevation-deg "$ELEVATION" \
    --width "$WIDTH" --height "$HEIGHT" --samples "$SAMPLES" $GPU_FLAG $EXTRA_ARGS \
    2>&1 | grep -E '^(FRAME|CYCLES_DEVICE|RENDERED_SEQUENCE|TERRAIN)' || true
}

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  missing=$(python3 - "$FRAMES_DIR" "$LAST" <<'PY'
import sys, pathlib
frames_dir, last = pathlib.Path(sys.argv[1]), int(sys.argv[2])
print(sum(1 for i in range(last + 1) if not (frames_dir / f"frame{i:04d}.png").is_file()))
PY
)
  if [ "$missing" -eq 0 ]; then
    echo "all frames present after $((attempt - 1)) attempt(s)"
    break
  fi
  echo "attempt $attempt: $missing frame(s) missing"
  for start in $(seq 0 "$CHUNK" "$LAST"); do
    end=$((start + CHUNK - 1))
    [ "$end" -gt "$LAST" ] && end=$LAST
    render_chunk "$start" "$end"
  done
done

missing=$(python3 - "$FRAMES_DIR" "$LAST" <<'PY'
import sys, pathlib
frames_dir, last = pathlib.Path(sys.argv[1]), int(sys.argv[2])
print(sum(1 for i in range(last + 1) if not (frames_dir / f"frame{i:04d}.png").is_file()))
PY
)
if [ "$missing" -ne 0 ]; then
  echo "giving up: $missing frame(s) still missing after $MAX_ATTEMPTS attempts" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_MP4")"
FILTER="format=yuv420p"
if [ -n "$LABEL" ]; then
  FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
  ESCAPED=${LABEL//:/\\:}
  FILTER="drawtext=fontfile=$FONT:text='$ESCAPED':x=32:y=28:fontsize=34:fontcolor=white:box=1:boxcolor=black@0.45:boxborderw=12,$FILTER"
fi
ffmpeg -y -loglevel error -framerate "$FPS" -i "$FRAMES_DIR/frame%04d.png" \
  -vf "$FILTER" -c:v libx264 -crf 18 -preset slow "$OUT_MP4"
echo "ENCODED $OUT_MP4 ($NUM_STATES frames @ ${FPS} fps)"
