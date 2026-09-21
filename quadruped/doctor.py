#!/usr/bin/env python3
"""Preflight. Refuse to start a run the host cannot correctly finish.

Every check here exists because its absence already cost something:

  chrono hash     a capacity arm read -9.5% on one box against the conda build where
                  another read -39.6% on the same policy. The binary that resolves is
                  not the binary you assume.
  torch GEMM      hpcfund's torch imports and reports cuda_avail True, then dies at the
                  first nn.Linear with missing kernels. is_available() proves nothing.
  host capability  a 15-step branch at batch 64 OOMs 8 GB and 16 GB cards. Discovering
                  that three hours in is avoidable.
  numpy ABI       pychrono is compiled against one numpy per host. Swapping it is a
                  segfault, not an ImportError.
  env name        one name, `nedm`, everywhere. Two names resolving to different builds
                  is how the trap survived.

Run standalone:   python doctor.py --action collect
Or from a script: from doctor import require; require("collect")
"""
from __future__ import annotations

import argparse
import hashlib
import os
import socket
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MACHINES = HERE / "params" / "machines.yaml"

# The conda-forge build. Identical on every desktop, which is what made it look safe.
TRAP_MD5 = "8e9e386546fe0b33"


class Failed(Exception):
    pass


def _load():
    import yaml
    if not MACHINES.exists():
        raise Failed(f"no machine registry at {MACHINES}")
    return yaml.safe_load(MACHINES.read_text())


def identify(reg) -> tuple[str, dict]:
    """Resolve this box to a registry entry.

    Keyed by ssh alias, but three of six hosts report a different hostname than their
    alias (a3-ubuntu is kyle-B650M-D3HP, sbel-ubuntu is kyle-sbel, d33-ubuntu is d33),
    so matching on hostname alone silently fails to find the entry.
    """
    hn = socket.gethostname()
    hosts = reg.get("hosts", {})
    for alias, spec in hosts.items():
        if alias == hn or spec.get("hostname", "").split(".")[0] == hn.split(".")[0]:
            return alias, spec
    raise Failed(
        f"host {hn!r} is not in {MACHINES.name}. Add it before running here -- an "
        f"unregistered host has no recorded capability, so nothing can be checked."
    )


def chrono_md5() -> tuple[str, str]:
    import pychrono
    d = Path(pychrono.__file__).parent
    so = d / "_core.so"
    if not so.exists():
        raise Failed(f"pychrono at {d} has no _core.so")
    return hashlib.md5(so.read_bytes()).hexdigest()[:16], str(d)


def check_chrono(alias, reg) -> str:
    """Refuse to run against a build that is not this host's registered one.

    WAS DEAD CODE. `_no_driver` had been inserted into the middle of this function, so
    everything below its `return` -- the trap-hash refusal and the registry md5
    comparison -- never executed. `check_chrono` fell off the end of the try/except and
    returned None, which the caller then printed as "chrono None".

    The consequence was the exact failure this function exists to prevent: the conda trap
    build has already produced one wrong verdict on this project, doctor.py was written to
    make that impossible, STATE.md records that it "refuses the trap hash", and it did
    not. Found by `namecheck.py`, which flagged `got`, `where`, `reg` and `alias` as
    unbound reads inside `_no_driver` -- names that belong to this function.
    """
    try:
        got, where = chrono_md5()
    except ImportError as e:
        raise Failed(
            f"pychrono does not import: {e}. On euler, source activate-nedm.sh first; "
            f"on a desktop, check that nedm_chrono.pth points at the build."
        )
    if got == TRAP_MD5:
        raise Failed(
            f"pychrono resolves to the CONDA TRAP build ({got}) at {where}.\n"
            f"  That binary is not the NeDM pin and lacks patches 0001/0002. It has "
            f"already produced one wrong verdict. Remove the conda package and point "
            f"nedm_chrono.pth at this host's source build."
        )
    want = (reg.get("chrono", {}).get("builds", {}).get(alias) or {}).get("md5")
    if want and got != want:
        raise Failed(
            f"pychrono md5 {got} does not match the registry entry {want} for {alias}.\n"
            f"  Resolved from {where}. Either the build changed or the wrong one is on "
            f"the path; a corpus collected now would not be comparable to anything."
        )
    return got


def _no_driver(msg: str) -> bool:
    return "libcuda.so" in msg or "libamdhip" in msg


def check_modules():
    """parsers is not optional: it is what loads the Go2 URDF."""
    missing = []
    for m in ("fsi", "vehicle", "parsers"):
        try:
            __import__("pychrono." + m)
        except Exception as e:  # noqa: BLE001 - a missing .so raises ImportError or OSError
            missing.append(f"{m} ({type(e).__name__}: {str(e)[:60]})")
    if missing:
        joined = "; ".join(missing)
        if _no_driver(joined):
            raise Failed(
                "pychrono submodules cannot load the GPU driver: " + joined + "\n"
                "  This is what a LOGIN NODE looks like. doctor must run where the work "
                "runs -- inside the job on a compute node, not on the submit host."
            )
        raise Failed("pychrono submodules failed to import: " + joined)


def check_numpy(spec):
    import numpy
    want = spec.get("numpy")
    if want and numpy.__version__ != want:
        raise Failed(
            f"numpy is {numpy.__version__}, registry says {want}. pychrono is compiled "
            f"against one numpy per host and a mismatch segfaults rather than raising."
        )


def check_torch_gpu(required: bool):
    """A real GEMM. `is_available()` returning True is not evidence the GPU computes."""
    import torch
    if not torch.cuda.is_available():
        if required:
            raise Failed(f"torch {torch.__version__} reports no usable device")
        return f"torch {torch.__version__} (cpu)"
    import torch.nn as nn
    try:
        m = nn.Linear(256, 256).cuda()
        x = torch.randn(32, 256, device="cuda", requires_grad=True)
        y = m(x).pow(2).mean()
        y.backward()
        torch.cuda.synchronize()
        if not (y.isfinite() and x.grad.isfinite().all()):
            raise Failed("GEMM produced non-finite values")
    except Failed:
        raise
    except Exception as e:  # noqa: BLE001
        raise Failed(
            f"torch reports a device but a real GEMM failed: {type(e).__name__}: "
            f"{str(e)[:90]}\n  This is the hpcfund failure mode -- is_available() is True "
            f"and the first nn.Linear dies."
        )
    name = torch.cuda.get_device_name(0)
    gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    return f"torch {torch.__version__} on {name} ({gb:.1f} GiB), GEMM verified"


def check_env_name(reg):
    want = reg.get("standard", {}).get("env_name", "nedm")
    prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    got = Path(prefix).name
    if got != want:
        return f"WARNING: env is {got!r}, standard is {want!r}"
    return f"env {got}"


def check_action(alias, spec, action):
    can = spec.get("can", [])
    cannot = spec.get("cannot", [])
    if action in cannot:
        why = (spec.get("cannot_because") or {}).get(action) or "see machines.yaml"
        raise Failed(f"{alias} is recorded as unable to {action}: {why}")
    if can and action not in can:
        raise Failed(f"{alias} is not permitted to {action}; it may {', '.join(can)}")
    return f"{alias} may {action}"


def require(action: str, verbose: bool = True) -> dict:
    """Raise Failed unless this host can correctly perform `action`."""
    reg = _load()
    alias, spec = identify(reg)
    lines = [check_action(alias, spec, action)]
    lines.append(check_env_name(reg))
    md5 = check_chrono(alias, reg)
    lines.append(f"chrono {md5}")
    check_modules()
    lines.append("pychrono fsi/vehicle/parsers OK")
    check_numpy(spec)
    needs_gpu = action in ("train", "finetune")
    lines.append(check_torch_gpu(required=needs_gpu))
    if verbose:
        for ln in lines:
            print("  ok  " + ln)
    return {"host": alias, "chrono_md5": md5, "action": action}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--action", required=True,
                    choices=["collect", "train", "finetune", "evaluate"])
    a = ap.parse_args()
    print(f"doctor: {a.action} on {socket.gethostname()}")
    try:
        require(a.action)
    except Failed as e:
        print(f"\nFAIL  {e}", file=sys.stderr)
        return 1
    print("doctor: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
