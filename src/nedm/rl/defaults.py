"""Default artifact paths for the RL entry points.

Each default names the artifact the manuscript actually reports, so a fresh
clone can run every entry point without passing paths. Checkpoints are the
open-loop-rollout-selected ones (``rollout_sel``), not lowest one-step loss.
"""

from __future__ import annotations

from pathlib import Path


# Study Case I: terrain-conditioned HMMWV. L8/8H/E256/ctx128, 75/25 flat/CRM,
# rollout-selected epoch 51.
DEFAULT_RL_DYNAMICS_CHECKPOINT = Path(
    "artifacts/training_runs/ablation_ofat/L8_H8_E256_ctx128/checkpoints/best_val.pt"
)
DEFAULT_RL_PROCESSED_DATASET_DIR = Path(
    "artifacts/training_datasets/hmmwv_tire_rigid_300g_normal_force_omega_seq_v1"
)
# 40 training references: 20 flat + 20 CRM, random mid-episode windows.
DEFAULT_RL_REFERENCE_PATH = Path(
    "artifacts/rl_reference_sets/"
    "hmmwv_tire_normal_force_omega_flat_crm_train_refs_40_1100_randwin_seed20260623.npz"
)

# Study Case II, reach mode: 8-D [q, qd] arm ROM with q_cmd as the action.
# 5L/8H/E256/ctx16, rollout-selected epoch 76. The checkpoint and the processed
# cache must be changed together -- the 8-D state layout is not interchangeable
# with the retired 15-D and 12-D caches.
DEFAULT_ARM_DYNAMICS_CHECKPOINT = Path(
    "artifacts/training_runs/arm_transformer_8d_v1/checkpoints/best_val.pt"
)
DEFAULT_ARM_PROCESSED_DATASET_DIR = Path(
    "artifacts/training_datasets/arm_dyn_v3_8d_seq16_v1"
)
DEFAULT_ARM_GEOMETRY_PATH = Path(
    "artifacts/arm_geometry/arm_geometry_v1.json"
)

# Study Case II, drive mode: 3-D [vx, vy, r] tracked-base ROM.
# 3L/4H/E96/ctx16, rollout-selected epoch 8.
DEFAULT_TRACKED_DYNAMICS_CHECKPOINT = Path(
    "artifacts/training_runs/tracked_transformer_v1/checkpoints/best_val.pt"
)
DEFAULT_TRACKED_PROCESSED_DATASET_DIR = Path(
    "artifacts/training_datasets/tracked_drive_v2_seq16_v1"
)

# Study Case III: terrain-conditioned Go2 quadruped on rigid + CRM. Same
# architecture and hyperparameters as the HMMWV anchor -- only the dataset paths
# and the 3-D command action differ -- rollout-selected epoch 46 (rollout_sel
# 0.0853; 0.0914 unbiased on the full val split, a +7.1% selection gap).
DEFAULT_GO2_DYNAMICS_CHECKPOINT = Path(
    "artifacts/training_runs/go2_transformer_v01_contact_mix25_onehot/checkpoints/best_val.pt"
)
# The CRM cache. Used only when references have to be rebuilt; the shipped
# reference set below already spans both domains.
DEFAULT_GO2_PROCESSED_DATASET_DIR = Path(
    "artifacts/training_datasets/go2_crm_seq_v1"
)
# 40 training references: 20 flat + 20 CRM, matching the HMMWV anchor's split,
# carrying metadata["domains"] so each env samples only its own terrain.
DEFAULT_GO2_REFERENCE_PATH = Path(
    "artifacts/rl_references/go2_flat_crm_ref40.npz"
)
