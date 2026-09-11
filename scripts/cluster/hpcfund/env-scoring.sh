# NeDM CRM policy-scoring environment on the AMD HPC Fund cluster.
#   source /work1/dannegrut/kyle/nedm/env-scoring.sh
#
# Builds on $KWORK/env.sh (cache redirects off the 25G $HOME) and the nedm venv.
# Assumes `module load gnu12 rocm` has already run; the sbatch scripts do it.

source /work1/dannegrut/kyle/env.sh
source /work1/dannegrut/kyle/venvs/nedm/bin/activate

export NEDM_ROOT=$KWORK/nedm
export NEDM_REPO=$NEDM_ROOT/NeDM

# WHICH CHRONO. Scoring needs pychrono.parsers, because the Go2 is loaded from a
# URDF through ChParserURDF, and the original $KWORK/chrono-build was configured
# with CH_ENABLE_MODULE_PARSERS=OFF. Rather than reconfigure that tree -- the only
# validated HIP pychrono we have, and the one with the working Vulkan sensor
# backend -- parsers was enabled in a SECOND tree built from the same source at
# the same fat gfx90a;gfx942;gfx950 arch set. The original is left untouched.
# Point NEDM_CHRONO at the old tree to compare the two builds.
export NEDM_CHRONO=${NEDM_CHRONO:-$KWORK/chrono-build-parsers}

# urdfdom/console_bridge/tinyxml2 are not on this system and were built into the
# toolchain; Chrono_parsers links against them, so they must be on the loader path.
export LD_LIBRARY_PATH=$TOOLCHAIN/urdf/lib:$TOOLCHAIN/urdf/lib64:$LD_LIBRARY_PATH

# pychrono was compiled against the numpy in NRD_PYSTACK (2.4.3). Do NOT
# pip-install a different numpy into the venv: the ABI must match or every array
# crossing the binding boundary is garbage. NRD_PYSTACK also supplies torch
# 2.10.0+rocm7.1, which is why this workflow needs no pip installs at all. The
# policy is a ~0.5M-param TorchScript at 50 Hz and imported_policy.py already
# loads it with map_location="cpu", so torch never touches the GPU; the GPU is
# Chrono's SPH alone.
export PYTHONPATH=$NEDM_CHRONO/bin:$NRD_PYSTACK:$NEDM_REPO/src:$PYTHONPATH
export CHRONO_DATA_DIR=$NEDM_CHRONO/data/

export NEDM_GO2_ASSETS=$NEDM_ROOT/stage/assets
export NEDM_CKPTS=$NEDM_ROOT/stage/checkpoints
export NEDM_INDEX=$NEDM_ROOT/stage/score_subset_index.cluster.json
export NEDM_OUT=$NEDM_ROOT/out

# ONE THREAD PER EPISODE PROCESS. The node is billed whole and packed with N
# concurrent episode processes; torch and any BLAS default to one thread per
# core, so without this each of N processes would claim all 16 cores and the
# node would be oversubscribed N-fold. The policy does not need threads, it
# needs to not fight the other episodes for the cores the SPH host side uses.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Per-episode scratch to job-local disk, never $WORK or $HOME: the scorer makes
# and deletes a temp dir per episode and once leaked 16,352 of them / 67 GB.
export TMPDIR=${SLURM_TMPDIR:-${TMPDIR:-$KWORK/tmp}}

mkdir -p "$NEDM_OUT" 2>/dev/null
