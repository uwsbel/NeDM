#!/bin/bash
# Build AMD's ROCm port of Warp (AMD-Ecosystem/warp, branch amd-integration, 1.13.0+rocm.0) for MI350X (gfx950), MI300X
# (gfx942) and MI210 (gfx90a), then check it on this node's GPU: Warp sees the device, Kyle's render tests pass, and the
# SO101 scene renders. Copy of Kyle's scripts/render/hpcfund/build_warp_rocm.sbatch with gfx950 added and our paths.
#
#   sbatch scripts/so101_policy/cluster/render_warp_build.sbatch          (RENDER_ARCH, RENDER_AMD optional)
#
# Kyle's build notes (2026-10-03, ROCm 7.2.0), kept: reduce.cu does not finish at -O3 (-Xarch_device -fno-inline only
# for that file brings it to ~100 s per arch); the CPU kernel compiler does not link with gcc 12.2 (--no-standalone:
# GPU-only build, all the renderer needs); packman caches outside $HOME (25 GB quota).
# The venv ($RENDER_AMD/venv: python3.12, numpy, trimesh, scipy, mujoco 3.15.0, newton 1.2.0 --no-deps) is made here if
# it is missing (2026-10-13: made on the login node with the same commands); torch comes from the cluster stack
# (render_env.sh), so the venv has no torch and no warp wheel.
#SBATCH -J r8-warp-build
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -t 01:30:00
#SBATCH -o /work1/dannegrut/harry/experiments/so101_policy_20261011/render_amd/logs/%x_%j.out
set -uo pipefail
RENDER_AMD=${RENDER_AMD:-/work1/dannegrut/harry/experiments/so101_policy_20261011/render_amd}
ARCH=${RENDER_ARCH:-"gfx90a;gfx942;gfx950"}
WARP=$RENDER_AMD/warp-rocm
export XDG_CACHE_HOME=$RENDER_AMD/cache PIP_CACHE_DIR=$RENDER_AMD/cache/pip
export PM_PACKAGES_ROOT=$RENDER_AMD/cache/packman WARP_CACHE_PATH=$RENDER_AMD/cache/warp
set +u; source /etc/profile; module purge; module load gnu12 rocm/7.2.0; set -u
echo "host $(hostname) job ${SLURM_JOB_ID:-none} start $(date -Is) rocm $(hipconfig --version | head -1) cpus $(nproc)"
rocminfo | grep -m 3 -E "Marketing Name|gfx9" || true
if [ ! -d "$RENDER_AMD/venv" ]; then        # one-time (done on the login node on 2026-10-13; same commands)
  /usr/bin/python3.12 -m venv "$RENDER_AMD/venv"
  "$RENDER_AMD/venv/bin/pip" install -q numpy pyyaml trimesh scipy imageio imageio-ffmpeg pillow mujoco==3.15.0
  "$RENDER_AMD/venv/bin/pip" install -q --no-deps newton==1.2.0     # --no-deps: keep the CUDA warp wheel out
fi
source "$RENDER_AMD/venv/bin/activate"

[ -d "$WARP" ] || GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 --branch amd-integration \
  https://github.com/AMD-Ecosystem/warp.git "$WARP"
cd "$WARP"
echo "warp $(git log -1 --format='%h %ci') arch $ARCH"
if [ ! -f warp/bin/warp.so ] || [ -n "${RENDER_REBUILD:-}" ]; then
  T0=$SECONDS
  BLOG=$RENDER_AMD/logs/warp-build-${SLURM_JOB_ID:-none}.log
  python build_lib.py --hip-arch "$ARCH" --no-use-libmathdx --no-standalone \
    --hipcc-options="-Xarch_device -fno-inline" -j "$(nproc)" > "$BLOG" 2>&1
  echo "build exit $? after $((SECONDS - T0)) s (log: $BLOG)"
fi
ls -la warp/bin/ || exit 2
[ -f warp/bin/warp.so ] || exit 2
for a in gfx90a gfx942 gfx950; do
  printf "%s code objects in warp.so: %s\n" "$a" "$(strings warp/bin/warp.so | grep -c "amdgcn-amd-amdhsa--$a" || true)"
done

source "$RENDER_AMD/repo/scripts/so101_policy/cluster/render_env.sh"
cd "$RENDER_AMD/repo"
echo "== Warp + torch on this GPU (one HIP runtime, zero-copy interop)"
export MUJOCO_GL=disable          # no EGL on the compute nodes; mujoco only exports meshes here
python3.12 -P scripts/so101_policy/cluster/render_amd_gpucheck.py
echo "== Kyle's render tests + SO101 scene tests"
# expected on the compute nodes: 1 error (test_depth_matches_mujoco needs MuJoCo's OpenGL renderer), 9 skipped (Madrona)
NEDM_SO101_RENDER_BACKEND=newton python3.12 -P -m unittest discover -s tests/render -v 2>&1 | grep -E "\.\.\. |^Ran |^OK|^FAILED"
echo "done $(date -Is)"
