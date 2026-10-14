#!/bin/bash
# Throughput sweep of the SO101 scene, Newton backend, top + wrist (run through render_run.sbatch): shadows on / off x
# 128 / 224 / 288 px x 256 / 1024 / 4096 worlds, one process per configuration, recorded validation states
# (states_val_4x4096.npz, the same as luffy's reference sweep). Rows appended to $O/bench_<gfx>.jsonl.
# RENDER_SHADOWS=on|off|both, RENDER_RES, RENDER_WORLDS: subsets. RENDER_ORDER=tiled RENDER_TILE=8x8: Newton's tiled
# ray order (up to 3x faster than the default on MI350X, build8_amd_render.md).
#   sbatch -t 01:30:00 scripts/so101_policy/cluster/render_run.sbatch bash scripts/so101_policy/cluster/render_bench.sh
set -uo pipefail
D=${RENDER_DATA:-$RENDER_AMD/data}
O=${RENDER_OUT:-$RENDER_AMD/out}
TAG=${RENDER_TAG:-$(rocminfo | grep -m1 -o 'gfx9[0-9a-z]*')}
PY=${PY:-python3.12 -P}
ORDER="--order ${RENDER_ORDER:-pixel} --tile ${RENDER_TILE:-8x8}"
mkdir -p "$O"
case ${RENDER_SHADOWS:-both} in on) SH=("");; off) SH=("--no-shadows");; *) SH=("" "--no-shadows");; esac
for sh in "${SH[@]}"; do
 for res in ${RENDER_RES:-128 224 288}; do
  for n in ${RENDER_WORLDS:-256 1024 4096}; do
   timeout 1200 $PY scripts/so101_policy/cluster/render_amd_bench.py bench $ORDER --states "$D/states_val_4x4096.npz" \
     --assets "$D/so101_assets.npz" --worlds $n --res $res $sh --out "$O/bench_$TAG.jsonl" 2>&1 | grep -E '^\{|Error|error' \
     || echo "FAIL worlds $n res $res $sh"
  done
 done
done
echo sweep-done
