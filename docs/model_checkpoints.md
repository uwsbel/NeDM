# Model Checkpoints

Trained `.pt` checkpoints are tracked with Git LFS. Raw datasets and processed
`.npy` caches are not — see [progress.md](progress.md) for how to regenerate
them.

After cloning:

```bash
git lfs install
git lfs pull
```

## Deployed checkpoints

These are the models the manuscript reports. Each is the checkpoint of lowest
**open-loop rollout error** (`checkpoint_metric: rollout_sel`), not lowest
one-step validation loss — the file is still named `best_val.pt`, but it is the
rollout-selected epoch.

| Model | Checkpoint | Shape | Selected |
|---|---|---|---|
| Terrain-conditioned HMMWV | `artifacts/training_runs/ablation_ofat/L8_H8_E256_ctx128/checkpoints/best_val.pt` | 15-D state + 3-D action + 2-D terrain code; L8/8H/E256/ctx128, 6.40 M params | epoch 51 |
| Rigid-only specialist | `artifacts/training_runs/ablation_ofat/L8_H8_E256_ctx128_mix00/checkpoints/best_val.pt` | same architecture, flat data only | epoch 61 |
| CRM-only specialist | `artifacts/training_runs/ablation_ofat/L8_H8_E256_ctx128_mix100/checkpoints/best_val.pt` | same architecture, CRM data only | epoch 54 |
| Tracked base | `artifacts/training_runs/tracked_transformer_v1/checkpoints/best_val.pt` | 3-D `[vx, vy, r]`; 3L/4H/E96/ctx16, 0.34 M params | epoch 8 |
| Arm | `artifacts/training_runs/arm_transformer_8d_v1/checkpoints/best_val.pt` | 8-D `[q, q̇]`, action = absolute `q_cmd`; 5L/8H/E256/ctx16, 4.0 M params | epoch 76 |

The remaining runs under `artifacts/training_runs/ablation_ofat/` are the
architecture, data-quantity and feature ablation arms (Appendices C–E); the
6-layer anchor under
`hmmwv_transformer_v07_tire_normal_force_omega_300g_crm2000_mix25_rebal_rollout_onehot/`
is ablation model 10.

Policy checkpoints live beside their PPO runs in `artifacts/rl_runs/` and
`artifacts/rl_runs_arm_goal_reach/`; the transferred iterations are `model_1000`
/ `model_999` (HMMWV tracking) and `model_1499` (tracked base, arm).

## Loading a dynamics checkpoint

```python
from pathlib import Path

import torch

from nedm.training.model import HMMWVDynamicsModel

checkpoint_path = Path(
    "artifacts/training_runs/ablation_ofat/L8_H8_E256_ctx128/checkpoints/best_val.pt"
)
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
metadata = checkpoint["metadata"]
model_config = checkpoint["config"]["model"]

model = HMMWVDynamicsModel(
    state_dim=len(metadata["state_fields"]),
    action_dim=len(metadata["action_fields"]),
    target_dim=len(metadata["state_fields"]),
    transformer_cfg=model_config,
    normalization=metadata["normalization"],
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

The checkpoint carries its own model config and normalization metadata, so
inference needs nothing else. To load one as a frozen environment for RL, use
`nedm.rl.dynamics.load_frozen_dynamics`, which wraps the same file.

`.pt` and `.pth` both route through LFS. Local-only data roots:

- `artifacts/datasets/`
- `artifacts/training_datasets/`
