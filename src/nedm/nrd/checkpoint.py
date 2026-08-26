"""Load a joint NRD checkpoint written by ``nedm.nrd.trainer`` (stage ``joint``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from nedm.nrd.model import NRDDynamicsModel


def load_nrd_model(checkpoint_path: Path | str, device: torch.device | str) -> tuple[NRDDynamicsModel, dict[str, Any]]:
    """Rebuild the model from the checkpoint's own config/metadata; returns (model in eval mode, payload)."""
    device = torch.device(device)
    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    config = payload["config"]
    metadata = payload["metadata"]
    model = NRDDynamicsModel(
        state_dim=len(metadata["state_fields"]),
        action_dim=len(metadata["action_fields"]),
        transformer_cfg=config["model"],
        vision_cfg=config.get("vision", {}),
        normalization=metadata["normalization"],
        state_fields=list(metadata["state_fields"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


__all__ = ["load_nrd_model"]
