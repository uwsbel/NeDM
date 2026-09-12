#!/bin/bash
# Preprocess the soil-varied corpus and train a surrogate on it (attempt 2).
#
# Attempt 1 failed in preprocess for two reasons, both of which the baseline corpus had
# already been through and nothing recorded as prerequisites:
#
#   1. grav_body_x/y/z are NOT written by the collector. They are added afterwards by
#      scripts/collection/add_gravity_channels.py, computed from the stored quaternion
#      because reconstructing them from logged roll/pitch is measurably wrong. Every
#      existing episode carries them; a freshly collected one does not.
#   2. --action-fields takes nargs="+", i.e. SPACE-separated names. Passing the list
#      comma-joined made argparse read the whole string as one field name, and the
#      error then quoted a 12-joint string as a single missing column.
#      It also defaults to the HMMWV driver_* channels. Without it the Go2 joint
#      targets are never read and preprocess demands columns no quadruped CSV has.
#
#SBATCH -J kyle-soilsurr
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -o /work1/dannegrut/kyle/logs/%x-%A.out
#SBATCH -e /work1/dannegrut/kyle/logs/%x-%A.err
set -o pipefail
source /work1/dannegrut/kyle/env.sh
module purge; module load gnu12 rocm
source /work1/dannegrut/kyle/nedm/env-scoring.sh
cd "$NEDM_REPO"

RAW=/work1/dannegrut/kyle/nedm/collect/go2_crm_soil_consolidated/crm
PROC=/work1/dannegrut/kyle/nedm/training_datasets/go2_crm_soil
RUN=/work1/dannegrut/kyle/nedm/training_runs/go2_crm_soil

echo "== add gravity channels =="
# --apply, or it dry-runs and preprocess fails on the same missing columns. The tool
# follows the consolidator convention where writing is opt-in.
python scripts/collection/add_gravity_channels.py "$RAW" --apply 2>&1 | tail -4
f=$(ls "$RAW"/episodes/*.csv | head -1)
head -1 "$f" | tr "," "\n" | grep -c grav_body | sed "s/^/  grav cols now: /"

echo "== preprocess =="
python scripts/preprocess/build_hmmwv_training_dataset.py \
  --dataset-root "$RAW" --output-dir "$PROC" \
  --state-field-preset quadruped_crm_baseline \
  --action-fields joint_rr_hip_target_rad joint_rr_thigh_target_rad joint_rr_calf_target_rad joint_rl_hip_target_rad joint_rl_thigh_target_rad joint_rl_calf_target_rad joint_fr_hip_target_rad joint_fr_thigh_target_rad joint_fr_calf_target_rad joint_fl_hip_target_rad joint_fl_thigh_target_rad joint_fl_calf_target_rad \
  --rollout-fields pos_x_m pos_y_m yaw_rad 2>&1 | tail -6
[ -f "$PROC/metadata.json" ] || { echo "FATAL: preprocess produced no metadata"; exit 1; }
python -c "
import json; m=json.load(open('$PROC/metadata.json'))
print('  state dims:', len(m['state_fields']), ' action dims:', len(m['action_fields']))
"

echo "== config =="
python - <<'PY'
import json
base = json.load(open("configs/go2_crm_baseline_s1.json"))
P = "/work1/dannegrut/kyle/nedm/training_datasets/go2_crm_soil"
base["processed_dataset_dir"] = P
base["output_dir"] = "/work1/dannegrut/kyle/nedm/training_runs/go2_crm_soil"
OLD = "/home/kyle/sbel-artifacts/training_datasets/go2_crm_baseline"
def retarget(o):
    """Rewrite by VALUE, not by key name.

    A key-based version missed loss.channel_weight_datasets, which is a list of bare
    directory-path STRINGS rather than dicts carrying processed_dataset_dir. The config
    then looked fully retargeted -- all four processed_dataset_dir entries pointed at the
    soil dataset -- while the trainer still opened the baseline metadata for its channel
    weights and died. Matching on the path itself cannot miss a container shape.
    """
    if isinstance(o, dict):
        return {k: retarget(v) for k, v in o.items()}
    if isinstance(o, list):
        return [retarget(v) for v in o]
    if isinstance(o, str) and o.rstrip("/") == OLD:
        return P
    return o
base = retarget(base)
base["processed_dataset_dir"] = P
base["output_dir"] = "/work1/dannegrut/kyle/nedm/training_runs/go2_crm_soil"
leftover = [x for x in json.dumps(base).split('"') if "sbel-artifacts/training_datasets" in x]
if leftover:
    raise SystemExit(f"FATAL: config still references {leftover}")
print(f"  retargeted, no stale dataset paths remain")
json.dump(base, open("configs/go2_crm_soil.json", "w"), indent=1)
print("  wrote configs/go2_crm_soil.json")
PY

echo "== train =="
python scripts/training/train_hmmwv_dynamics.py --config configs/go2_crm_soil.json 2>&1 | tail -8
ls -l "$RUN/checkpoints/best_val.pt" 2>/dev/null && echo "SOIL SURROGATE OK" || echo "NO CHECKPOINT"
