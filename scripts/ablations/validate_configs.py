#!/usr/bin/env python3
"""Dry-build every OFAT ablation config's model (no data load) to catch dim
errors early and record parameter counts for the plan doc.

Builds HMMWVDynamicsModel exactly as the trainer does (input_dim = state+action
+ num_terrains one-hot), asserts n_embd % n_head == 0, and prints a table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from nedm.training.constants import STATE_FIELD_PRESETS, DEFAULT_ACTION_FIELDS  # noqa: E402
from nedm.training.model import HMMWVDynamicsModel  # noqa: E402

STATE_DIM = len(STATE_FIELD_PRESETS["tire_normal_force_omega"])  # 15
ACTION_DIM = len(DEFAULT_ACTION_FIELDS)  # 3


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> int:
    manifest = json.loads((REPO_ROOT / "configs/ablation_ofat/manifest.json").read_text())
    base = json.loads((REPO_ROOT / manifest["base_config"]).read_text())
    num_terrains = len(base["terrain_conditioning"]["terrains"])

    rows = []
    for e in manifest["entries"]:
        if e["config"] is not None:
            cfg = json.loads((REPO_ROOT / e["config"]).read_text())
            model_cfg = cfg["model"]
        else:  # anchor reference: reconstruct its model block
            model_cfg = {"block_size": e["block_size"], "n_layer": e["n_layer"],
                         "n_head": e["n_head"], "n_embd": e["n_embd"],
                         "dropout": 0.0, "bias": False, "head_hidden_dim": e["n_embd"]}
        assert model_cfg["n_embd"] % model_cfg["n_head"] == 0, (
            f"{e['name']}: n_embd {model_cfg['n_embd']} not divisible by n_head {model_cfg['n_head']}")
        model = HMMWVDynamicsModel(
            state_dim=STATE_DIM, action_dim=ACTION_DIM, target_dim=STATE_DIM,
            transformer_cfg=model_cfg, normalization=base_norm_stub(),
            num_terrains=num_terrains,
        )
        n = count_params(model)
        head_dim = model_cfg["n_embd"] // model_cfg["n_head"]
        rows.append((e["arm"], e["name"], model_cfg["n_layer"], model_cfg["n_head"],
                     model_cfg["n_embd"], head_dim, model_cfg["block_size"], n))
        del model

    print(f"{'arm':8s} {'name':22s} {'L':>2s} {'H':>3s} {'E':>4s} {'hd':>3s} {'ctx':>4s} {'params':>12s}")
    print("-" * 70)
    for arm, name, L, H, E, hd, ctx, n in rows:
        print(f"{arm:8s} {name:22s} {L:2d} {H:3d} {E:4d} {hd:3d} {ctx:4d} {n:12,d}")
    print(f"\nAll {len(rows)} configs build cleanly (state={STATE_DIM}, action={ACTION_DIM}, "
          f"one-hot terrains={num_terrains} -> input_dim={STATE_DIM + ACTION_DIM + num_terrains}).")
    return 0


def base_norm_stub() -> dict:
    """Zero-mean / unit-std normalization stub so model construction succeeds
    without touching the dataset (params/shape do not depend on norm values).
    The model does torch.tensor(normalization[...]) so these must be lists."""
    return {
        "state_mean": [0.0] * STATE_DIM,
        "state_std": [1.0] * STATE_DIM,
        "action_mean": [0.0] * ACTION_DIM,
        "action_std": [1.0] * ACTION_DIM,
        "target_mean": [0.0] * STATE_DIM,
        "target_std": [1.0] * STATE_DIM,
    }


if __name__ == "__main__":
    raise SystemExit(main())
