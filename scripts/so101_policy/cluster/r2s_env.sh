# Real2sim camera student (~/so101_r2s_claude/docs/integrate.md): torch + the Newton/Warp renderer on the AMD cluster
# (MI350X), like render_env.sh, but every location the job WRITES is under $R2S (isolation rule of the study): Warp /
# comgr / torch caches, Python byte code, the MuJoCo overlap cache, MIOpen's per-job database. The build9 venv and the
# Warp ROCm build are only read.
#
#   source scripts/so101_policy/cluster/r2s_env.sh            # then: python3.12 -P scripts/so101_policy/...
R2S=${R2S:-/work1/dannegrut/harry/so101_r2s_claude}
RENDER_AMD=${RENDER_AMD:-/work1/dannegrut/harry/experiments/so101_policy_20261011/render_amd}   # read only
REPO=${REPO:-$R2S/repo}
export R2S RENDER_AMD REPO
type module >/dev/null 2>&1 || source /etc/profile
module purge >/dev/null 2>&1
module load gnu12 rocm/7.2.0 >/dev/null 2>&1
export WARP_CACHE_PATH=$R2S/cache/warp XDG_CACHE_HOME=$R2S/cache TORCH_HOME=$R2S/cache/torch
export PYTHONPYCACHEPREFIX=$R2S/cache/pycache NEDM_SO101_OVERLAP_CACHE=$R2S/cache/so101_overlap
[ "${VIRTUAL_ENV:-}" = "$RENDER_AMD/venv" ] || source "$RENDER_AMD/venv/bin/activate"
TORCH_STACK=${TORCH_STACK:-/share/sw/ai/pytorch/2.10.0}
export PYTHONPATH=$REPO/src:$RENDER_AMD/warp-rocm:$TORCH_STACK
export LD_PRELOAD=/opt/rocm-7.2.0/lib/libamdhip64.so.7
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 MUJOCO_GL=disable OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MIOPEN_USER_DB_PATH=$R2S/cache/miopen/${SLURM_JOB_ID:-$$}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p "$MIOPEN_USER_DB_PATH" "$PYTHONPYCACHEPREFIX"
# data of the study on AMD (copies; see integrate.md)
export R2S_DATA=$R2S/data R2S_PY=$RENDER_AMD/venv/bin/python3.12
export R2S_JT=$R2S/data/jt_h110.pt R2S_NRD=$R2S/data/nrd_c1_s62.pt
export R2S_SCENE=${R2S_SCENE:-$R2S/data/scene/real_scene_v2.json} R2S_CALIB=${R2S_CALIB:-$R2S/data/calib/calib_v1.json}
# v2 (2026-10-22): HOME with wrist roll 45 +- 15 deg (tasks_h45); the roll-0 HOME files stay in tasks/
export R2S_TRAIN=${R2S_TRAIN:-$R2S/data/tasks/train_mix45_16k.npz}
export R2S_DEV=${R2S_DEV:-"$R2S/data/tasks/dev100.npz $R2S/data/tasks/home45_dev100.npz"}
export R2S_FRESH=${R2S_FRESH:-"$R2S/data/tasks/fresh100.npz $R2S/data/tasks/home45_fresh100.npz"}
