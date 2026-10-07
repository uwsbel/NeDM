#!/bin/bash
# Build Madrona's batch renderer for the "madrona" backend of nedm.render. NVIDIA GPUs only.
#
#   scripts/render/madrona/build.sh [directory]        # default ~/madrona, about 5 minutes
#   source <directory>/env.sh                           # then run anything with --backend madrona
#
# It builds the C++ engine inside madrona_mjx at a pinned commit. The JAX and MuJoCo MJX
# wrapper that project ships is not installed and not used: nedm.render drives the engine
# from PyTorch. Madrona and madrona_mjx are MIT.
#
# What is not obvious, each found on 2026-10-07:
#   * Madrona needs a CUDA 12.5 or older toolkit to compile its GPU code. This script puts one
#     in a conda environment and leaves the system alone.
#   * Do NOT `conda activate` that environment while building. Its compiler flags leak into
#     Madrona's own bundled compiler and the link fails looking for /lib64/libm.so.6.
#   * PyTorch must bundle CUDA 12.5 or newer (the cu126 wheels do). With an older one the
#     import fails on a missing __nvJitLinkCreate_12_5.
#   * The viewer library is built whether or not it is used, and it needs the X11 development
#     headers. On a machine without them, install the xorg-* packages into the environment.
set -eo pipefail
DIR=$(realpath -m "${1:-$HOME/madrona}")
ENV=${MADRONA_ENV:-madrona}
MADRONA_MJX_COMMIT=1505699168fd7c0c12629356f357b05491ce9d0c

# conda is often not on PATH in a non-interactive shell, so look for it
CONDA=${CONDA_EXE:-$(command -v conda || true)}
for base in "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3" /opt/conda; do
  [ -n "$CONDA" ] || { [ -x "$base/bin/conda" ] && CONDA=$base/bin/conda; }
done
[ -n "$CONDA" ] || { echo "conda not found: set CONDA_EXE to the conda executable"; exit 1; }
BASE=$("$CONDA" info --base)
"$CONDA" env list | grep -q "^$ENV " || "$CONDA" create -y -q -n "$ENV" -c conda-forge python=3.12 cmake ninja "cuda-version=12.5" cuda-toolkit
PREFIX=$BASE/envs/$ENV

mkdir -p "$DIR" && cd "$DIR"
if [ ! -d madrona_mjx ]; then
  git clone -q https://github.com/shacklettbp/madrona_mjx.git
  git -C madrona_mjx checkout -q "$MADRONA_MJX_COMMIT"
  git -C madrona_mjx submodule update -q --init --recursive external/madrona
fi
mkdir -p madrona_mjx/build && cd madrona_mjx/build
env -u LDFLAGS -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u CC -u CXX -u LD_LIBRARY_PATH PATH="$PREFIX/bin:$PATH" \
  cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT="$PREFIX" -DPython_EXECUTABLE="$PREFIX/bin/python" > cmake.log 2>&1
env -u LDFLAGS -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u CC -u CXX -u LD_LIBRARY_PATH PATH="$PREFIX/bin:$PATH" \
  ninja > build.log 2>&1 || { grep -E "error|FAILED" build.log | head; exit 1; }
ls _madrona_mjx_batch_renderer*.so

"$PREFIX/bin/pip" install -q "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu126
"$PREFIX/bin/pip" install -q numpy trimesh scipy imageio imageio-ffmpeg pillow pyyaml

cat > "$DIR/env.sh" <<ENV
# Source this, then use $PREFIX/bin/python with --backend madrona.
export NEDM_MADRONA_BUILD=$DIR/madrona_mjx/build
# Madrona compiles its GPU code at start-up, which takes minutes. These make it a one-time cost.
export MADRONA_MWGPU_KERNEL_CACHE=$DIR/cache/mwgpu
export MADRONA_BVH_KERNEL_CACHE=$DIR/cache/bvh
export PATH=$PREFIX/bin:\$PATH
unset LD_LIBRARY_PATH
ENV
mkdir -p "$DIR/cache"
echo "built. Next: source $DIR/env.sh"
