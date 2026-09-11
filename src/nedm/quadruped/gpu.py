"""Which GPU compute backend this box has, decided from the driver node.

Chrono FSI/SPH needs a GPU, and `import pychrono.fsi` does NOT tell you whether
one is usable: the projectchrono package bundles a GPU runtime, so the import
resolves on a GPU-less box and only the first device call fails -- with
cudaGetDeviceCount error 35 ("driver version is insufficient") -- partway into a
run that has already paid for terrain construction. Measured on d33, an AMD-GPU
box: import fine, no usable device. A guard that can only fire after the
expensive part is not a guard.

So test the driver node, before the expensive part. That is cheap, it does not
depend on torch being importable, and it does not depend on which runtime
pychrono happens to bundle. There are two such nodes, because this project now
runs against two Chrono builds:

    /dev/nvidiactl   exists iff the NVIDIA driver is loaded   -> CUDA  (desktop fleet)
    /dev/kfd         exists iff the ROCm compute driver is    -> HIP   (hpcfund MI210)

/dev/kfd is the right AMD node to test, not /dev/dri: /dev/dri appears for any
display-capable GPU including integrated graphics, whereas /dev/kfd is created
only by the ROCm compute stack, which is exactly what HIP kernels need.

THE ORDER IS LOAD-BEARING, NOT ARBITRARY. A box can have both nodes: sbel-ubuntu
carries an AMD integrated GPU with the ROCm driver loaded AND the NVIDIA card
that its Chrono is actually built against, so it presents /dev/kfd and
/dev/nvidiactl at once. Testing /dev/kfd first would label the whole desktop
fleet "hip". NVIDIA is therefore tested first, which is right for every box we
run on: the AMD path is the cluster, and the cluster has no NVIDIA driver.

The guard only needs the yes/no -- on a dual box either answer means "a GPU is
present" -- but the returned label is what callers log, so it should name the
backend that box's Chrono was built for rather than whichever node was found
first alphabetically.

Detect rather than hardcode, because the SAME checkout runs on both: the NVIDIA
desktop fleet, and the AMD HPC Fund cluster whose Chrono is built with
CHRONO_GPU_BACKEND=HIP. Hardcoding either one breaks the other silently.
"""

from __future__ import annotations

import os

CUDA = "cuda"
HIP = "hip"


def gpu_backend() -> str | None:
    """Return "cuda", "hip", or None if no GPU compute driver is loaded."""
    if os.path.exists("/dev/nvidiactl"):
        return CUDA
    if os.path.exists("/dev/kfd"):
        return HIP
    return None


def require_gpu_backend(what: str = "--terrain crm") -> str:
    """Backend name, or exit with the reason this box cannot run GPU work."""
    backend = gpu_backend()
    if backend is None:
        raise SystemExit(
            f"FATAL: {what} needs a GPU and this machine has none (no "
            "/dev/nvidiactl for CUDA, no /dev/kfd for ROCm/HIP). Chrono's "
            "FSI/SPH module is GPU-only. `import pychrono.fsi` would succeed "
            "here anyway -- the package bundles a GPU runtime -- and the run "
            "would die at the first GPU call after building the terrain. "
            "Dispatch CRM to a GPU box; this one can still run rigid "
            "collection and scoring, which are CPU-only.")
    return backend
