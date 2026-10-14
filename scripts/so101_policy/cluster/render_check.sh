#!/bin/bash
# Renderer checks on the current GPU (run through render_run.sbatch): Warp + torch in one process, Kyle's render tests and
# the SO101 scene tests, then the cross-machine parity: the same 64 recorded states at 128 / 224 / 288 px, top + wrist,
# compared with the luffy RTX 5090 renders (parity_luffy_<res>.npz in $D, made with render_amd_bench.py parity).
#   sbatch scripts/so101_policy/cluster/render_run.sbatch bash scripts/so101_policy/cluster/render_check.sh
set -uo pipefail
D=${RENDER_DATA:-$RENDER_AMD/data}
O=${RENDER_OUT:-$RENDER_AMD/out}
TAG=${RENDER_TAG:-$(rocminfo | grep -m1 -o 'gfx9[0-9a-z]*')}
mkdir -p "$O"
echo "== gpu check"
python3.12 -P scripts/so101_policy/cluster/render_amd_gpucheck.py | tee "$O/gpucheck_$TAG.json"
echo "== render tests (Kyle's + SO101 scene); NEDM_SO101_RENDER_BACKEND=newton"
NEDM_SO101_RENDER_BACKEND=newton python3.12 -P -m unittest discover -s tests/render -v > "$O/tests_$TAG.log" 2>&1
echo "unittest exit $?"
grep -E "^(test_|Ran |OK|FAILED)|\.\.\. (ok|ERROR|FAIL|skipped)" "$O/tests_$TAG.log" | tail -70
for res in 128 224 288; do
  python3.12 -P scripts/so101_policy/cluster/render_amd_bench.py parity --states "$D/states_val_4x4096.npz" \
    --assets "$D/so101_assets.npz" --res $res --out "$O/parity_${TAG}_$res.npz" 2>&1 | grep '^{'
  python3.12 -P scripts/so101_policy/cluster/render_amd_bench.py compare "$D/parity_luffy_$res.npz" \
    "$O/parity_${TAG}_$res.npz" --out "$O/compare_luffy_${TAG}_$res.json" > /dev/null && \
    python3.12 -P -c "import json,sys; d=json.load(open(sys.argv[1])); [print(sys.argv[2], c, {k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()}) for c, r in d['cameras'].items()]" \
      "$O/compare_luffy_${TAG}_$res.json" "$res"
done
# determinism on this GPU: the same call twice
python3.12 -P scripts/so101_policy/cluster/render_amd_bench.py parity --states "$D/states_val_4x4096.npz" \
  --assets "$D/so101_assets.npz" --res 128 --out "$O/parity_${TAG}_128_again.npz" 2>&1 | grep '^{'
python3.12 -P scripts/so101_policy/cluster/render_amd_bench.py compare "$O/parity_${TAG}_128.npz" \
  "$O/parity_${TAG}_128_again.npz" | grep -E "depth_abs_mm_max|rgb_identical_frac|hit_miss"
