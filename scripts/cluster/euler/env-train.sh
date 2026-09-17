# NeDM surrogate-TRAINING environment on UW-Madison Euler.
#   source /srv/home/kasha2/nedm/NeDM/scripts/cluster/euler/env-train.sh
#
# Separate from env-scoring.sh because the two want opposite things from torch.
# Scoring runs a ~0.5M-parameter TorchScript policy at 50 Hz and deliberately
# keeps torch on the CPU so the GPU belongs to Chrono's SPH alone. Training is
# the other way round: the transformer wants the GPU, and nothing here needs
# pychrono at all, so this file does not touch NEDM_CHRONO, CHRONO_DATA_DIR or
# the urdf toolchain.
#
# WHY THIS FILE EXISTS. A GPU torch installed from pip ships its own CUDA
# runtime inside site-packages/nvidia/*/lib rather than using the module's, and
# the loader is not told about those directories by activating the venv. The
# symptom is an import or a first kernel launch dying on a missing
# libnvrtc.so.12 (or libcudnn, or libcublas), which looks like a broken CUDA
# module and is not. That cost three separate debugging rounds here -- euler
# scoring, soil processing, and a training batch -- each fixed by pasting the
# same LD_LIBRARY_PATH line into one more sbatch script, which is why it now
# lives in one place that training jobs source.

source /srv/home/kasha2/env.sh
source /srv/home/kasha2/venvs/nedm/bin/activate

export NEDM_ROOT=$EHOME/nedm
export NEDM_REPO=$NEDM_ROOT/NeDM
export PYTHONPATH=$NEDM_REPO/src:$PYTHONPATH

# The CUDA runtime that came with the pip torch. Globbed rather than listed,
# because the set of nvidia/* wheels changes with the torch version and a
# hard-coded list silently goes stale. Ordered before the existing path so the
# libraries torch was linked against win over anything the modules provide.
_nvlib=$("$VIRTUAL_ENV/bin/python" - <<'PY' 2>/dev/null
import glob, os, sysconfig
sp = sysconfig.get_paths()["purelib"]
d = sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
print(os.pathsep.join(d))
PY
)
if [ -n "$_nvlib" ]; then
    export LD_LIBRARY_PATH=$_nvlib:$LD_LIBRARY_PATH
else
    # Not fatal: a CPU-only or system torch legitimately has no bundled runtime.
    # Say so loudly anyway, because the alternative is discovering it as a
    # missing libnvrtc.so.12 several minutes into a queued job.
    echo "env-train.sh: no bundled NVIDIA runtime found in the venv --" \
         "if this job dies on libnvrtc/libcudnn/libcublas, that is why" >&2
fi
unset _nvlib

# Fail fast and legibly rather than at the first kernel launch. A training job
# that silently lands on the CPU wastes the whole allocation looking healthy.
"$VIRTUAL_ENV/bin/python" - <<'PY' || echo "env-train.sh: TORCH/CUDA CHECK FAILED" >&2
import torch
ok = torch.cuda.is_available()
n = torch.cuda.device_count() if ok else 0
print(f"  torch {torch.__version__}  cuda_available={ok}  devices={n}")
if ok:
    print(f"  device 0: {torch.cuda.get_device_name(0)}")
raise SystemExit(0 if ok else 1)
PY

export NEDM_RUNS=${NEDM_RUNS:-$NEDM_ROOT/runs}
export NEDM_DATASETS=${NEDM_DATASETS:-$NEDM_ROOT/datasets}
mkdir -p "$NEDM_RUNS" 2>/dev/null

# Training is one process per GPU, so unlike scoring it should NOT be pinned to a
# single thread -- the dataloader workers want the cores.
export TMPDIR=/tmp
