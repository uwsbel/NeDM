"""Origin stamp for a collected episode.

Recorded PER EPISODE rather than per dataset because two machines write into one
pooled dataset: if a cross-box artefact ever appears, the split has to be
recoverable from the episodes themselves.

WHY BOTH git_commit AND git_tree. A commit hash names a point in history and is
INVALIDATED BY REWRITING IT; a tree hash names the code itself and is not. Both
failed on this project the same night:

    sbel-pc  the 968 rigid episodes' commit was orphaned by a rebase
    dorm-pc  a batch split across two commit values when a commit was reworded
             mid-run -- both real, both pointing at one identical tree

So git_tree is the durable identifier and git_commit is the convenient one. Keep
both: the commit is what a human searches for, the tree is what still resolves.

TWO PROPERTIES, INDEPENDENTLY NECESSARY. Content-addressing gives stability under
rewriting and says NOTHING about survival -- an unreachable tree object is
collected just as an unreachable commit is. So: content-address the identifier
AND make the object reachable. The orphaned commit above is preserved by the tag
`provenance/go2-rigid-968`; without it the recorded hash would have named an
object nobody could fetch, which is the same defect it was recorded to prevent.

WHAT gpu_arch DOES NOT ASSERT. It reports the DEVICE capability from
torch.cuda.get_device_capability, not what Chrono was compiled for. On sbel-pc
both read sm_86 -- but only because CHRONO_CUDA_ARCHITECTURES=86 was set by hand
after an empty architecture list silently produced CUDA=FALSE and disabled
FSI_SPH and Sensor OptiX. The build architecture there is a value someone CHOSE,
and this field would not show a mismatch if there were one.

DO NOT MUTATE ANYTHING A RUNNING COLLECTION READS -- AND THAT INCLUDES git HEAD,
not only source files. dorm-pc split its own batch across two git_commit values
by rewording a commit mid-run, having correctly avoided touching any source file.
If a change looks necessary mid-run, STOP THE RUN FIRST and then decide: once the
file has changed underneath it the episodes are already split, and restarting
buys a clean second half rather than a clean record.

Observations and the original implementation are dorm-pc's (kyle-N7-B650E); the
reachability half of the two-property rule and git_tree are from sbel-pc.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True,
                             cwd=str(REPO_ROOT), timeout=10).stdout.strip()
        return out or None
    except Exception:  # noqa: BLE001 - provenance must never break a collection
        return None


def _chrono_build() -> dict[str, Any]:
    """Which pychrono actually ran, by PATH and by API generation.

    THIS BOX HAS TWO CHRONO BUILDS AND THE RECORD DID NOT SAY WHICH ONE RAN:
    conda pychrono 10.0.0 under miniconda, and a source build at
    Documents/sbel/chrono-build/bin selected only by PYTHONPATH. They differ in
    the CRM API -- the source build has SoilProperties/SetCrmSPH/
    SetFreeFlowDuration, the conda build the older names -- so they are not the
    same simulator for anything touching SPH soil.

    Reconstructing which one produced an episode after the fact cost an hour and
    was only possible because the two builds happen to disagree about an
    attribute name. git_commit and git_tree pin OUR code exactly and say nothing
    about the binary underneath it, which is half of what produced the numbers.

    Read from sys.modules rather than importing: by the time a collection calls
    this, pychrono is loaded, and a provenance call must not be the thing that
    first imports a 12 MB extension module.
    """
    module = sys.modules.get("pychrono")
    build: dict[str, Any] = {
        "pychrono_path": getattr(module, "__file__", None) if module else None,
        "api_generation": None,
    }
    try:
        from nedm import chrono_crm_compat

        build["api_generation"] = chrono_crm_compat.stamp()
    except Exception:  # noqa: BLE001 - provenance must never break a collection
        pass
    return build


def provenance() -> dict[str, Any]:
    """machine, gpu_name, gpu_arch, seed_offset, git_commit, git_tree, chrono_build.

    Every field degrades to None rather than raising: a collection must not die
    because it could not identify itself.
    """
    gpu_name = gpu_arch = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            gpu_arch = f"sm_{major}{minor}"
    except Exception:  # noqa: BLE001
        pass
    return {
        "machine": socket.gethostname(),
        "gpu_name": gpu_name,
        "gpu_arch": gpu_arch,
        "seed_offset": int(os.environ.get("NEDM_SEED_OFFSET", "0")),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_tree": _git("rev-parse", "HEAD^{tree}"),
        "chrono_build": _chrono_build(),
    }
