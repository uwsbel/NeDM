"""The four architectures by name, and the loading of a checkpoint of any of them."""
from __future__ import annotations

import torch

from nedm.contact_nrd.arch_ablation import arch_1m, arch_1t, arch_3m, arch_3t

ARCHS = {"3t": arch_3t, "3m": arch_3m, "1t": arch_1t, "1m": arch_1m}


def arch_of(packet):
    """The architecture a checkpoint holds; a contact NRD checkpoint without the entry is a 3T."""
    return packet.get("arch") or ((packet.get("runtime") or {}).get("config") or {}).get("arch", "3t")


def load_model(path, device="cpu"):
    """Checkpoint of any of the four architectures (best.pt of a run, or a released seedNN.pt) -> (model in eval
    mode, checkpoint dict)."""
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    packet = torch.load(path, map_location=device, weights_only=True)
    return ARCHS[arch_of(packet)].load(packet).to(device).eval(), packet
