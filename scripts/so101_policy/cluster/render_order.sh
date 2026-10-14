#!/bin/bash
# Newton's ray-to-thread order (RenderConfig.render_order) on this GPU: pixel (Newton's default), view, tiled 8x8 /
# 16x8 / 16x16, and the shape BVH rebuilt at every call instead of refitted, at 1024 worlds, 128 and 288 px, shadows
# on and off, top + wrist (render_amd_bench.py bench). Checks that a tiled render gives the same pictures (128 px).
#   AMD: sbatch -t 01:00:00 scripts/so101_policy/cluster/render_run.sbatch bash scripts/so101_policy/cluster/render_order.sh
#   luffy: RENDER_DATA=... RENDER_OUT=... RENDER_TAG=luffy PY=~/miniconda3/envs/so101_render/bin/python bash ...
set -uo pipefail
D=${RENDER_DATA:-$RENDER_AMD/data}
O=${RENDER_OUT:-$RENDER_AMD/out}
TAG=${RENDER_TAG:-$(rocminfo | grep -m1 -o 'gfx9[0-9a-z]*')}
PY=${PY:-python3.12 -P}
B=scripts/so101_policy/cluster/render_amd_bench.py
mkdir -p "$O"
for sh in "" "--no-shadows"; do
 for res in 128 288; do
  for order in pixel view tiled:8x8 tiled:16x8 tiled:16x16; do
   o=${order%%:*}; t=${order#*:}; [ "$t" = "$order" ] && t=16x8
   timeout 900 $PY $B bench --states "$D/states_val_4x4096.npz" --assets "$D/so101_assets.npz" --worlds ${RENDER_WORLDS:-1024} \
     --res $res $sh --order $o --tile $t --out "$O/order_$TAG.jsonl" 2>&1 | grep -E '^\{|Error' || echo "FAIL $order $res $sh"
  done
  timeout 900 $PY $B bench --states "$D/states_val_4x4096.npz" --assets "$D/so101_assets.npz" --worlds ${RENDER_WORLDS:-1024} \
    --res $res $sh --rebuild --out "$O/order_$TAG.jsonl" 2>&1 | grep -E '^\{|Error' || echo "FAIL rebuild $res $sh"
 done
done
$PY $B parity --states "$D/states_val_4x4096.npz" --assets "$D/so101_assets.npz" --res 128 --order tiled --tile 8x8 \
  --out "$O/parity_${TAG}_128_tiled8x8.npz" 2>&1 | grep '^{'
$PY $B parity --states "$D/states_val_4x4096.npz" --assets "$D/so101_assets.npz" --res 128 \
  --out "$O/parity_${TAG}_128_pixel.npz" 2>&1 | grep '^{'
$PY $B compare "$O/parity_${TAG}_128_pixel.npz" "$O/parity_${TAG}_128_tiled8x8.npz" --out "$O/compare_order_$TAG.json" \
  | grep -E "hit_miss|depth_abs_mm_max|rgb_identical"
echo order-done
