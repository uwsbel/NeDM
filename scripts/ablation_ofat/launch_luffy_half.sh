#!/usr/bin/env bash
# Launch the SECOND HALF of the OFAT sweep on luffy (RTX 5090, ~/miniconda3).
# Newton keeps running the full queue and covers the first half as it goes; luffy
# takes the queue tail in parallel. Run this ON luffy (it is synced there):
#   ssh luffy 'cd ~/NeDM && bash scripts/ablation_ofat/launch_luffy_half.sh'
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export PYTHON=/home/harry/miniconda3/envs/nedm/bin/python
export CONDA_SH=/home/harry/miniconda3/etc/profile.d/conda.sh
export CONDA_ENV=nedm
# Second half of the 12 remaining configs (the queue tail newton reaches last):
export RUN_ONLY="L6_H16_E512_ctx128 L6_H4_E256_ctx128 L6_H16_E256_ctx128 L6_H8_E256_ctx32 L6_H8_E256_ctx64 L6_H8_E256_ctx256"
export SESSION_NAME=ofat_ablation_sweep_luffy
export SWEEP_LOG=artifacts/training_runs/ablation_ofat/sweep_luffy.log

bash scripts/ablation_ofat/launch_sweep.sh
