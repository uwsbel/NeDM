# NeDM CRM policy-scoring environment on UW-Madison Euler.
#   source /srv/home/kasha2/nedm/env-scoring.sh
#
# Builds on ~/env.sh (paths, module names, the python stack the bindings were
# compiled against). Assumes `module load gcc/12.2.0 nvidia/cuda/12.9.1` has
# already run; the sbatch scripts do it.

source /srv/home/kasha2/env.sh
source /srv/home/kasha2/venvs/nedm/bin/activate

export NEDM_ROOT=$EHOME/nedm
export NEDM_REPO=$NEDM_ROOT/NeDM

# WHICH CHRONO. Scoring needs pychrono.parsers, because src/nedm/quadruped/robot.py
# loads the Go2 through ChParserURDF and has no alternative path, and ~/chrono-build
# was configured with CH_ENABLE_MODULE_PARSERS=OFF. Rather than reconfigure that
# tree -- the only pychrono here with a working OptiX sensor backend, and expensive
# to reproduce -- parsers was enabled in a SECOND tree built from the same source.
# The original is left untouched. Point NEDM_CHRONO at it to compare the two.
#
# The second tree is also the only one that can run on most of Euler's GPUs:
# ~/chrono-build was compiled for 60/70/80/90/100-real + 120, so the 2080 Ti
# (sm_75), A4500 (sm_86) and RTX 4000 Ada (sm_89) have no cubin and no PTX they
# can JIT from. chrono-build-parsers adds 75, 86 and 89.
export NEDM_CHRONO=${NEDM_CHRONO:-$EHOME/chrono-build-parsers}

# console_bridge, urdfdom_headers and urdfdom are not on this system and were
# built into the toolchain; Chrono_parsers links against them, so they must be on
# the loader path. They install to lib64 here, not lib. tinyxml2 is NOT in this
# list: Euler ships 10.0.0 in /usr/lib64, which is already on the default path
# and is the same version hpcfund had to build from source.
export LD_LIBRARY_PATH=$TOOLCHAIN/urdf/lib:$TOOLCHAIN/urdf/lib64:$LD_LIBRARY_PATH

# pychrono was compiled against the SYSTEM numpy, 1.26.4. Do NOT pip-install a
# different numpy into the venv: the ABI must match or every array crossing the
# binding boundary is garbage. Note this is a different numpy major from the AMD
# cluster's 2.4.3 -- that is a per-machine fact, not a discrepancy to reconcile.
#
# torch 2.6.0+cpu IS pip-installed here, because Euler has no system torch and
# the only conda torch is in a module tree this project avoids. CPU is not a
# compromise: the policy is a ~0.5M-parameter TorchScript at 50 Hz and
# imported_policy.py already loads it with map_location="cpu", so torch never
# touches the GPU on any machine. The GPU is Chrono's SPH alone.
export PYTHONPATH=$NEDM_CHRONO/bin:$NEDM_REPO/src:$PYTHONPATH
export CHRONO_DATA_DIR=$NEDM_CHRONO/data/

export NEDM_GO2_ASSETS=$NEDM_ROOT/stage/assets
export NEDM_CKPTS=$NEDM_ROOT/stage/checkpoints
export NEDM_INDEX=${NEDM_INDEX:-$NEDM_ROOT/stage/score_subset_index.euler.json}
export NEDM_OUT=${NEDM_OUT:-$NEDM_ROOT/out}

# ONE THREAD PER EPISODE PROCESS. The scorer runs N concurrent episode
# subprocesses; torch and any BLAS default to one thread per visible core, so
# without this each of N processes would claim every core the job was given and
# they would fight each other for the cores the SPH host side needs. The policy
# does not want threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Per-episode scratch to NODE-LOCAL disk, never CephFS. The scorer makes and
# deletes a temp dir per episode and once leaked 16,352 of them / 67 GB; doing
# that on /srv/home would be doing it to the whole SBEL lab's shared pool, which
# has no quota to stop it. Measured on euler19: TMPDIR is already /tmp, which is
# a node-local 875 GB nvme with 812 GB free. Kept explicit so a node that sets
# TMPDIR somewhere shared does not silently redirect it back onto Ceph.
export TMPDIR=/tmp

mkdir -p "$NEDM_OUT" 2>/dev/null
