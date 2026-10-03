# Environment for rendering on the AMD HPC Fund cluster. Source it after `module load gnu12 rocm`.
#   source scripts/render/hpcfund/env.sh
WORK=${WORK:-/work1/dannegrut/$USER}
REPO=${REPO:-$WORK/NeDM}
export WARP_CACHE_PATH=$WORK/.cache/warp
[ -n "${VIRTUAL_ENV:-}" ] || source "${VENV:-$WORK/venvs/nrd-render}/bin/activate"

# torch comes from the cluster's shared stack (2.10.0 on ROCm 7.1). numpy comes with it.
TORCH_STACK=${TORCH_STACK:-/share/sw/ai/pytorch/2.10.0}
export PYTHONPATH=$REPO/src:$WORK/src/warp-rocm:$TORCH_STACK${PYTHONPATH:+:$PYTHONPATH}

# ONE HIP RUNTIME. torch bundles the ROCm 7.1 runtime and Warp is built against 7.2. Without
# this, whichever of the two is imported second finds no GPU at all.
export LD_PRELOAD=/opt/rocm-7.2.0/lib/libamdhip64.so.7

# torch on the single-GPU partitions (devel, mi2101x) fails its first matrix multiply with
# hipErrorFileNotFound unless BOTH of these point at the system ROCm. Harmless elsewhere.
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm-7.2.0/lib/rocblas/library
export HIPBLASLT_TENSILE_LIBPATH=/opt/rocm-7.2.0/lib/hipblaslt/library
