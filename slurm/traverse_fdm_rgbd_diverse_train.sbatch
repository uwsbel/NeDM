#!/bin/bash
# One isolated ROCm GPU instance per RGB-D/blank x seed task. No test loading.
#SBATCH -J fdm_diverse_train
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --array=0-3%4
#SBATCH -t 02:00:00
#SBATCH -o /work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909/logs/%x_%A_%a.out
set -eo pipefail
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
: "${CODE_ROOT:?Set immutable CODE_ROOT}"
: "${DATA_ROOT:?Set frozen train/val DATA_ROOT}"
: "${OUT_ROOT:?Set a new output root for this suite}"
: "${SLURM_JOB_ID:?Training must run in an AMD Slurm allocation}"
: "${SLURM_ARRAY_TASK_ID:?Submit the four-task Slurm array}"
case "$SLURM_ARRAY_TASK_ID" in
    0) arm=rgbd; seed=11 ;;
    1) arm=blank; seed=11 ;;
    2) arm=rgbd; seed=29 ;;
    3) arm=blank; seed=29 ;;
    *) echo "Invalid array task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
STEPS=${STEPS:-5000}
BATCH=${BATCH:-32}
EVAL_EVERY=${EVAL_EVERY:-1000}
EVAL_BATCH=${EVAL_BATCH:-32}
WORK_WEIGHT=${WORK_WEIGHT:-0.2}
ATTITUDE_WEIGHT=${ATTITUDE_WEIGHT:-0.5}
PROGRESS_TARGET=${PROGRESS_TARGET:-bounded_motion}
run="$OUT_ROOT/patch_${arm}_s${seed}"
mkdir -p "$OUT_ROOT"
# No implicit resume or overwrite: smoke and full suites use different roots.
mkdir "$run"
export FDM_TRAIN_RUN="$run" FDM_TRAIN_ARM="$arm" FDM_TRAIN_SEED="$seed"
export FDM_TRAIN_STEPS="$STEPS" FDM_TRAIN_BATCH="$BATCH"
export FDM_TRAIN_EVAL_EVERY="$EVAL_EVERY" FDM_TRAIN_EVAL_BATCH="$EVAL_BATCH"
export FDM_TRAIN_WORK_WEIGHT="$WORK_WEIGHT" FDM_TRAIN_ATTITUDE_WEIGHT="$ATTITUDE_WEIGHT"
export FDM_TRAIN_PROGRESS_TARGET="$PROGRESS_TARGET"
fdm_cache=$(mktemp -d "/tmp/fdm_diverse_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_XXXXXX")
export MIOPEN_USER_DB_PATH="$fdm_cache" MIOPEN_CUSTOM_CACHE_DIR="$fdm_cache"
finish() {
    code=$?
    trap - EXIT
    set +e
    python3.12 - "$run" "$code" <<'PY'
import json, os, pathlib, sys, time
out, code = pathlib.Path(sys.argv[1]), int(sys.argv[2])
record = {"exit_code": code, "state": "complete" if code == 0 else "failed",
          "finished_unix": time.time(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
          "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
          "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID")}
(out / "array_task_result.json").write_text(json.dumps(record, indent=2) + "\n")
PY
    rm -rf -- "$fdm_cache"
    exit "$code"
}
trap finish EXIT

# mi3501x is an exclusive single-GPU VM partition, not a Slurm GRES request.
# Preserve visibility established by the allocation. Never mask an eight-GPU
# node down to one and accidentally treat that as the requested resource.
python3.12 -u - <<'PY' 2>&1 | tee "$run/preflight.log"
import hashlib, json, os, pathlib, socket, sys
import numpy as np
import torch
root, data, out = (pathlib.Path(os.environ[k]).resolve() for k in ("CODE_ROOT", "DATA_ROOT", "FDM_TRAIN_RUN"))
host = socket.gethostname()
if not host.startswith("k007-005-v"):
    raise RuntimeError(f"Expected an allocated mi3501x instance, got {host}")
if not torch.version.hip or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError(f"Expected exactly one visible ROCm GPU: HIP={torch.version.hip}, GPUs={torch.cuda.device_count()}")
props = torch.cuda.get_device_properties(0)
if not any(family in props.name for family in ("MI350", "MI355")):
    raise RuntimeError(f"Expected MI350 family on mi3501x, got {props.name}")
with np.load(data / "train.npz", allow_pickle=False) as pack:
    horizon = int(pack["nominal_pose"].shape[1])
    training_windows = len(pack["history"])
    if training_windows < 1 or horizon < 1:
        raise ValueError("Empty training pack")
with np.load(data / "val.npz", allow_pickle=False) as pack:
    if pack["nominal_pose"].shape[1] != horizon or not len(pack["history"]):
        raise ValueError("Validation horizon differs or validation is empty")
    validation_windows = len(pack["history"])
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
record = {"host": host, "python": sys.executable, "python_version": sys.version,
          "torch": str(torch.__version__), "rocm": str(torch.version.hip),
          "gpu_name": props.name, "gpu_total_memory_bytes": props.total_memory,
          "visible_gpu_count": torch.cuda.device_count(),
          "code_root": str(root), "data_root": str(data), "output_root": str(out),
          "training_windows": training_windows, "validation_windows": validation_windows,
          "horizon_steps_from_pack": horizon,
          "manifest_sha256": sha(data / "manifest.json"),
          "sbatch_sha256": sha(root / "slurm/traverse_fdm_rgbd_diverse_train.sbatch"),
          "training_script_sha256": sha(root / "scripts/traverse_fdm_rgbd_diverse_train.py"),
          "environment": {k: os.environ.get(k) for k in (
              "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_PARTITION",
              "SLURM_CPUS_PER_TASK", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
              "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONPATH",
              "FDM_TRAIN_ARM", "FDM_TRAIN_SEED", "FDM_TRAIN_STEPS", "FDM_TRAIN_BATCH",
              "FDM_TRAIN_EVAL_EVERY", "FDM_TRAIN_EVAL_BATCH", "FDM_TRAIN_WORK_WEIGHT",
              "FDM_TRAIN_ATTITUDE_WEIGHT", "FDM_TRAIN_PROGRESS_TARGET")},
          "controls": {"candidate_patches": True, "rotation_augmentation": True,
                       "protected_test_loaded": False, "training_location": "AMD cluster only"}}
(out / "array_task_provenance.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2), flush=True)
PY

python3.12 -u scripts/traverse_fdm_rgbd_diverse_train.py \
    --data "$DATA_ROOT" --out "$run" --arm "$arm" --seed "$seed" \
    --steps "$STEPS" --batch "$BATCH" --eval-every "$EVAL_EVERY" --eval-batch "$EVAL_BATCH" \
    --candidate-patches --rotation-augmentation --progress-target "$PROGRESS_TARGET" \
    --work-weight "$WORK_WEIGHT" --attitude-weight "$ATTITUDE_WEIGHT" \
    --device cuda:0 --threads 4 --save-predictions \
    2>&1 | tee "$run/job_${SLURM_JOB_ID}.log"

python3.12 - "$run" "$STEPS" <<'PY'
import json, pathlib, sys
out, expected = pathlib.Path(sys.argv[1]), int(sys.argv[2])
status = json.loads((out / "status.json").read_text())
if status.get("state") != "complete" or status.get("step") != expected:
    raise RuntimeError(f"Training did not reach the declared budget: {status}")
for name in ("last.pt", "best.pt", "provenance.json", "metrics_last.json", "last_controls.json", "best_controls.json"):
    if not (out / name).is_file():
        raise RuntimeError(f"Completed training omitted {name}")
print(f"FDM_DIVERSE_TRAIN_COMPLETE: {out} at update {expected}")
PY
