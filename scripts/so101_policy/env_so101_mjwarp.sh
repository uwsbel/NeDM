#!/usr/bin/env bash
# Conda env `so101_mjwarp` for the MuJoCo Warp worlds of the SO101 policy study (src/nedm/so101_policy/mjwarp_env.py)
# and their tests / bench (tests/so101_policy/test_mjwarp_env.py, scripts/so101_policy/bench_mjwarp_env.py).
#
#   bash scripts/so101_policy/env_so101_mjwarp.sh clone    # how luffy's env was made (2026-10-11 00:45 + 01:09)
#   bash scripts/so101_policy/env_so101_mjwarp.sh fresh    # the same pins without the nedm env (UNVERIFIED)
#   bash scripts/so101_policy/env_so101_mjwarp.sh check    # print the versions and run the CPU unit tests
#
# clone: `conda create --clone nedm` (environment.nedm.yml + mujoco 3.15.0 + onnxruntime 1.30.0 on luffy), then the
# MuJoCo Warp packages. Validated set (pip freeze of luffy's env, 2026-10-11): python 3.12.13, torch 2.12.0 (CUDA 13.0
# wheels, nvidia-cuda-runtime 13.0.96), numpy 2.4.6, mujoco 3.15.0, mujoco-warp 3.15.0, warp-lang 1.18.0,
# absl-py 2.4.0, etils 1.14.0, pytest 9.1.1. GPU: RTX 5090 (sm_120); the first run compiles MuJoCo Warp's kernels
# (about 2 min, cached in ~/.cache/warp). Use `python -P` with PYTHONPATH=src from the repo root.
set -euo pipefail
ENV=${ENV_NAME:-so101_mjwarp}
PINS="mujoco==3.15.0 mujoco-warp==3.15.0 warp-lang==1.18.0 absl-py==2.4.0 etils==1.14.0 pytest==9.1.1"
case "${1:-check}" in
  clone)
    conda create -y -q -n "$ENV" --clone nedm
    conda run -n "$ENV" pip install $PINS ;;
  fresh)
    conda create -y -q -n "$ENV" python=3.12 pip
    conda run -n "$ENV" pip install torch==2.12.0 numpy==2.4.6 $PINS ;;
  check)
    conda run -n "$ENV" python -c "import torch, mujoco, mujoco_warp, warp, numpy; print('torch', torch.__version__, \
'cuda', torch.cuda.is_available(), '| mujoco', mujoco.__version__, '| mujoco_warp', mujoco_warp.__version__, \
'| warp', warp.__version__, '| numpy', numpy.__version__)"
    cd "$(dirname "$0")/../.."
    PYTHONPATH=src conda run -n "$ENV" python -P -m pytest -q tests/so101_policy/test_mjwarp_env.py -k "not gpu" ;;
  *) echo "usage: $0 clone|fresh|check"; exit 2 ;;
esac
