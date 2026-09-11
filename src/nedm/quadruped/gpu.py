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
desktop fleet, and the AMD boxes whose Chrono is built with
CHRONO_GPU_BACKEND=HIP. Hardcoding either one breaks the other silently.

TWO THINGS THE NODE ALONE DOES NOT TELL YOU, both now checked here because both
were hit for real on d33 (a desktop gfx1201 box, not the cluster):

  1. Whether the node can be OPENED. /dev/kfd is mode rw-rw---- root:render, so
     a user outside the `render` group sees the node and still cannot use it.
     Existence therefore passes while every HIP call fails, which at the first
     device call is indistinguishable from having no GPU at all.

  2. Whether the pychrono on THIS path was built for the GPU that is present.
     The node says what silicon the box has; only the .so says what the build
     can drive. d33 has both a conda CUDA pychrono and a source HIP build, so
     "a GPU is present" and "this build can use it" genuinely come apart there,
     and the wrong PYTHONPATH silently selects a pychrono that cannot run. That
     one needs the library mapped, so it is a separate call made just AFTER the
     import -- still before terrain construction, which is the part that costs.
"""

from __future__ import annotations

import os

CUDA = "cuda"
HIP = "hip"

#: Driver node that must be present AND openable for each backend.
NODE = {CUDA: "/dev/nvidiactl", HIP: "/dev/kfd"}


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
    node = NODE[backend]
    if not os.access(node, os.R_OK | os.W_OK):
        raise SystemExit(
            f"FATAL: {what} found {node} but cannot open it. This is a "
            "PERMISSION problem, not a missing GPU, and it fails at the first "
            f"device call looking exactly like one. {node} is mode rw-rw---- "
            "root:render, so add the user to the render and video groups "
            "(sudo usermod -aG render,video $USER) and then start a NEW login "
            "-- a running session keeps the group list it was created with.")
    return backend


def chrono_fsi_backend() -> str | None:
    """Which GPU runtime the LOADED Chrono::FSI::SPH was compiled against.

    Returns "cuda", "hip", or None when it cannot be told.

    Call this AFTER `import pychrono.fsi`. That import is cheap and device-free
    -- the very property that makes the node check necessary -- and once it has
    run, /proc/self/maps names the exact libChrono_fsisph.so that got loaded.
    Reading the map beats guessing a path: conda and build-tree layouts differ,
    and RPATH means the file sitting next to the wrapper is not necessarily the
    one the loader chose.

    The marker is the fat-binary registration symbol each toolchain emits,
    nvcc -> __cudaRegisterFatBinary against hipcc -> __hipRegisterFatBinary.
    Looking for a DT_NEEDED on libcudart instead does NOT work: Chrono links the
    CUDA runtime statically, so the conda build's libChrono_fsisph.so carries no
    libcudart dependency at all and the scan silently returns "unknown".
    """
    try:
        path = None
        with open("/proc/self/maps", encoding="utf-8") as fh:
            for line in fh:
                if "libChrono_fsisph" in line and ".so" in line:
                    path = line.rstrip("\n").split(" ")[-1].strip()
                    break
        if not path:
            return None
        with open(path, "rb") as fh:
            blob = fh.read()
    except Exception:
        return None                        # a probe must never break the run
    hip = b"__hipRegisterFatBinary" in blob
    cuda = b"__cudaRegisterFatBinary" in blob
    if hip and not cuda:
        return HIP
    if cuda and not hip:
        return CUDA
    return None


def verify_chrono_backend(host_backend: str, what: str = "--terrain crm") -> None:
    """Exit if the imported pychrono cannot drive the GPU this box actually has.

    Silent when the two agree, and silent when the build cannot be identified --
    an unknown build on a GPU box is the old behavior, not a reason to invent a
    failure. It only speaks up for the case it can prove wrong, which is the one
    that otherwise dies confusingly a long way downstream.
    """
    built = chrono_fsi_backend()
    if built is None or built == host_backend:
        return
    raise SystemExit(
        f"FATAL: {what} imported a {built.upper()} pychrono, but this box's GPU "
        f"is {host_backend.upper()} ({NODE[host_backend]}). The import succeeds "
        "either way, so without this check the run would build the terrain and "
        "only then die at the first device call. Point PYTHONPATH at the "
        f"{host_backend.upper()} build of pychrono for this machine.")
