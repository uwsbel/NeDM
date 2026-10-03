#!/bin/bash
# Build AMD's ROCm port of Warp for the Instinct GPUs on the AMD HPC Fund cluster, then prove
# the renderer works on it by running the repository's render tests on the GPU.
#
#   sbatch scripts/render/hpcfund/build_warp_rocm.sbatch
#
# Overrides: WORK (default /work1/dannegrut/$USER), REPO (this checkout, default $WORK/NeDM),
# VENV (default $WORK/venvs/nrd-render), ARCH (default "gfx90a;gfx942": MI210/MI250 and MI300X).
#
# What is not obvious, each found the hard way on 2026-10-03 with ROCm 7.2.0:
#   * reduce.cu does not finish compiling at -O3 (25+ min, either arch). build_dll.py applies
#     --hipcc-options to that one file only, and with -fno-inline it takes about 100 s per arch.
#   * the CPU kernel compiler (warp-clang.so) fails to link with gcc 12.2, hence --no-standalone.
#     The build is GPU-only, which is all the renderer needs.
#   * packman caches under $HOME by default. The quota there is 25 GB.
#SBATCH -J warp-rocm-build
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -t 00:40:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -uo pipefail
WORK=${WORK:-/work1/dannegrut/$USER}
REPO=${REPO:-$WORK/NeDM}
VENV=${VENV:-$WORK/venvs/nrd-render}
ARCH=${ARCH:-"gfx90a;gfx942"}
WARP=$WORK/src/warp-rocm
export XDG_CACHE_HOME=$WORK/.cache PIP_CACHE_DIR=$WORK/.cache/pip
export PM_PACKAGES_ROOT=$WORK/.cache/packman WARP_CACHE_PATH=$WORK/.cache/warp

module purge
module load gnu12 rocm
echo "host $(hostname) job $SLURM_JOB_ID rocm $(hipconfig --version | head -1)"

if [ ! -d "$VENV" ]; then
  /usr/bin/python3.12 -m venv "$VENV"
  "$VENV/bin/pip" install -q numpy pyyaml trimesh scipy imageio imageio-ffmpeg pillow
  "$VENV/bin/pip" install -q --no-deps newton==1.2.0        # --no-deps: keep the CUDA warp wheel out
fi
source "$VENV/bin/activate"

[ -d "$WARP" ] || GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 --branch amd-integration \
  https://github.com/AMD-Ecosystem/warp.git "$WARP"
cd "$WARP"
if [ ! -f warp/bin/warp.so ]; then
  T0=$SECONDS
  python build_lib.py --hip-arch "$ARCH" --no-use-libmathdx --no-standalone \
    --hipcc-options="-Xarch_device -fno-inline" -j "$(nproc)" > "$WORK/warp-rocm-build-$SLURM_JOB_ID.log" 2>&1
  echo "build exit $? after $((SECONDS - T0)) s (log: $WORK/warp-rocm-build-$SLURM_JOB_ID.log)"
fi
ls -la warp/bin/warp.so || exit 2

# The render tests work their expected values out by hand, so a pass here means the ray
# tracer returns the right distances on this GPU, not just that it ran.
source "$REPO/scripts/render/hpcfund/env.sh"
cd "$REPO"
python -m unittest discover -s tests/render -v
