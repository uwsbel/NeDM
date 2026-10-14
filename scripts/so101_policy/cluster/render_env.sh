# Environment for the batch renderer (Newton / Warp backend) with torch in the SAME process on the AMD HPC Fund
# cluster: MI350X (gfx950, partition mi3501x / mi3508x) and MI210 (gfx90a). Source it in a job script:
#
#   source scripts/so101_policy/cluster/render_env.sh        # then: python3.12 -P your_script.py
#
# Overrides: RENDER_AMD (the render scratch dir with venv/, warp-rocm/, cache/), REPO (code dir, default
# $RENDER_AMD/repo). Copy of Kyle's scripts/render/hpcfund/env.sh with our paths (build8_amd_render.md).
#
# What each line is for (Kyle's findings on MI210, 2026-10-03, re-checked on MI350X 2026-10-13):
#   * Warp is AMD's ROCm port (1.13.0+rocm.0), built against ROCm 7.2.0 (render_warp_build.sbatch). torch is the
#     cluster's 2.10.0+rocm7.1 stack, which bundles its own HIP runtime. Two HIP runtimes in one process do not work
#     (the one loaded second sees no GPU), so LD_PRELOAD makes both use the 7.2.0 runtime.
#   * torch comes in through PYTHONPATH (the cluster's stack); it is appended, never assigned (the venv has no torch).
#   * ROCBLAS / HIPBLASLT_TENSILE_LIBPATH: torch's first matrix multiply fails on the MI210 partitions without them
#     (hipErrorFileNotFound). They are only set for gfx90a (RENDER_TENSILE=1 forces them).
RENDER_AMD=${RENDER_AMD:-/work1/dannegrut/harry/experiments/so101_policy_20261011/render_amd}
REPO=${REPO:-$RENDER_AMD/repo}
export RENDER_AMD REPO
type module >/dev/null 2>&1 || source /etc/profile
module purge >/dev/null 2>&1
module load gnu12 rocm/7.2.0 >/dev/null 2>&1
export WARP_CACHE_PATH=$RENDER_AMD/cache/warp XDG_CACHE_HOME=$RENDER_AMD/cache TORCH_HOME=$RENDER_AMD/cache/torch
[ "${VIRTUAL_ENV:-}" = "$RENDER_AMD/venv" ] || source "$RENDER_AMD/venv/bin/activate"
TORCH_STACK=${TORCH_STACK:-/share/sw/ai/pytorch/2.10.0}
export PYTHONPATH=$REPO/src:$RENDER_AMD/warp-rocm:$TORCH_STACK${PYTHONPATH:+:$PYTHONPATH}
export LD_PRELOAD=/opt/rocm-7.2.0/lib/libamdhip64.so.7
if [ "${RENDER_TENSILE:-}" = 1 ] || rocminfo 2>/dev/null | grep -q "gfx90a"; then
  export ROCBLAS_TENSILE_LIBPATH=/opt/rocm-7.2.0/lib/rocblas/library
  export HIPBLASLT_TENSILE_LIBPATH=/opt/rocm-7.2.0/lib/hipblaslt/library
fi
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
# MIOpen (torch convolutions): one database per job on the node's local disk. A database on /work1 shared by two jobs
# that started at the same time failed with "SQLite database: unable to open database file" (2026-10-13). Cost: each job
# compiles its convolution kernels again at the first training step (10 s at 128 px, 28 s at 288 px, MI350X). Keep the
# batch shape fixed: a new input shape compiles again (the first smoke, whose batch size changed at every step,
# averaged 1.3 s per training step; with a fixed batch of 128 at 128 px: 11 ms).
export MIOPEN_USER_DB_PATH=${TMPDIR:-/tmp}/miopen-$USER-${SLURM_JOB_ID:-$$}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p "$MIOPEN_USER_DB_PATH"
