#!/usr/bin/env bash
# Render a Chrono HMMWV policy rollout with a chase ("follow") camera.
#
# Thin wrapper over render_rollout_video.sh that swaps the fixed auto-framed ortho camera
# for a perspective camera riding with the vehicle. Use this for a *demo* clip: it shows
# the tyre/soil interaction close up, at the cost of the shared framing that makes the
# fixed-camera videos directly comparable between policies. For the paper's
# mixture-vs-experts comparison keep using render_rollout_video.sh.
#
# Usage:
#   scripts/rendering/render_follow_video.sh <export-dir> <rollout-npz> <out-mp4> [label]
#
# CRM also needs the soil sequence, same as the fixed-camera path:
#   EXTRA_ARGS="--crm-surface-dir <export-dir>/crm_surface --terrain-z-m 0.25 \
#               --crm-filler-plane --crm-filler-drop-m 0.02 --soil-shading flat"
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

# Chase geometry. Defaults aim at the brief "vehicle plus tyre-ground interaction":
# a low three-quarter rear view, close enough that the contact patch is legible and wide
# enough that the ruts behind the vehicle stay in frame.
FOLLOW_MODE=${FOLLOW_MODE:-chase}
FOLLOW_AZIMUTH=${FOLLOW_AZIMUTH:-38}
FOLLOW_ELEVATION=${FOLLOW_ELEVATION:-16}
FOLLOW_DISTANCE=${FOLLOW_DISTANCE:-9.5}
FOLLOW_TARGET_Z=${FOLLOW_TARGET_Z:--0.30}
FOLLOW_LEAD=${FOLLOW_LEAD:-0.0}
FOLLOW_SMOOTH=${FOLLOW_SMOOTH:-9}
# 50 mm at this distance frames little more than the vehicle itself; 32 mm keeps the whole
# body plus all four contact patches and the ground being crossed, without pushing the
# camera so far back that the ruts lose contrast.
LENS=${LENS:-32}
# The substituted ground plane is a disc centred on the trajectory. At 12 deg of elevation
# the camera looks out toward its rim, so it needs to be much larger than the 140 m the
# top-down fixed camera can get away with.
TERRAIN_RADIUS=${TERRAIN_RADIUS:-400}

FOLLOW_ARGS="--follow --follow-mode $FOLLOW_MODE \
--follow-azimuth-deg $FOLLOW_AZIMUTH --follow-elevation-deg $FOLLOW_ELEVATION \
--follow-distance-m $FOLLOW_DISTANCE --follow-target-z-offset-m $FOLLOW_TARGET_Z \
--follow-lead-m $FOLLOW_LEAD --follow-smooth-frames $FOLLOW_SMOOTH \
--cam-lens-mm $LENS --terrain-radius-m $TERRAIN_RADIUS"

export EXTRA_ARGS="$FOLLOW_ARGS ${EXTRA_ARGS:-}"
export SAMPLES=${SAMPLES:-128}
export ELEVATION=${ELEVATION:-24}
export AZIMUTH=${AZIMUTH:-180}

exec "$REPO_ROOT/scripts/rendering/render_rollout_video.sh" "$@"
