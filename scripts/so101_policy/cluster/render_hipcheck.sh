#!/bin/bash
# Is the LD_PRELOAD of render_env.sh needed on this GPU? Warp (ROCm 7.2 build) + torch (ROCm 7.1 stack) in one process,
# both import orders, with and without the preload (run through render_run.sbatch).
#   sbatch -t 00:15:00 scripts/so101_policy/cluster/render_run.sbatch bash scripts/so101_policy/cluster/render_hipcheck.sh
for pre in "$LD_PRELOAD" ""; do
  for order in torch_first warp_first; do
    LD_PRELOAD=$pre RENDER_IMPORT_ORDER=$order timeout 300 python3.12 -P scripts/so101_policy/cluster/render_amd_gpucheck.py \
      2>&1 | grep -E '^\{|Error|error' | tail -2 || echo "FAIL preload=[$pre] $order"
  done
done
