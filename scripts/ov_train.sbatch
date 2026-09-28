#!/bin/bash
# offroad_vehicles_20260927 module M4 (PLAN 1.7, 4.1): ci_train.py runs for any vehicle's soil planner on one MI350X node
# (mi3501x: 1 GPU, 24 CPUs, 4 h cap), in LANES. Copy of scripts/ag_train_gpus.sbatch / ag_train_lanes.sbatch (both frozen)
# with the root G4 and a lanes-per-GPU setting:
#   - lane k runs on GPU ((k-1) / OV_LANES_PER_GPU) mod n_gpu; default OV_LANES_PER_GPU=2, i.e. the K3 setting "2 lanes per
#     MI350X" (jobs 436135, 439361/439362: two ci_train processes on one MI350X ran at ~0.03 s per step each, three to five
#     at 0.16-0.22 s per step each). On an 8-GPU node lanes 1-2 share GPU 0, lanes 3-4 GPU 1, ...
#   - the lines of one lane run one after the other; a line waits until the ag_subset.py manifest of each of its --ds files
#     exists (OV_WAIT_S, default 2 h); a failed run does not stop its lane.
#   - trainer = G4/source/scripts/ci_train.py; the job refuses to start unless its sha256 starts with OV_CI_TRAIN_SHA
#     (default 7a4f2d67, the trainer of every arena_gator soil model).
# Job list (OV_JOBS, tab-separated, '#' lines ignored): lane <TAB> tag <TAB> output folder (relative to G4) <TAB> ci_train
# arguments without --out/--tag. The PLAN 1.7 recipe, per vehicle V (subset from ov_build_ds.py -> ag_subset.py):
#   1  V_full_soil_deploy   e5/train/V_full_soil      --ds $G4/e4/subsets/V_full_f104_V_soil.npz --mode deploy --arch gru --cond none --domain-filter crm --ctx geom --split-eval val --bs 256 --epochs 30 --seeds 5 --seed0 0 --roundtrip-check
#   2  V_full_soil_holdout  e5/train/offline_soil     --ds $G4/e4/subsets/V_full_f104_V_soil.npz --mode holdout --arch gru --cond none --domain-filter crm --ctx geom --split-eval val --bs 256 --epochs 30 --seeds 5 --seed0 0
# Each run writes <out>/<tag>*.pt|json|_logits.npz and $G4/e5/logs/<tag>_<job>.log (last line 'exit: <code>').
# Submit (the -o folder must exist; never with NEDM_VEHICLE):
#   mkdir -p $G4/e5/logs && env -u NEDM_VEHICLE sbatch -p mi3501x -t 01:30:00 -J ov_train_V \
#       --export=ALL,OV_JOBS=$G4/e5/jobs/soil_V.tsv $G4/source/scripts/ov_train.sbatch
#SBATCH -A dannegrut
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p mi3501x
#SBATCH -c 24
#SBATCH -t 01:30:00
#SBATCH -o /work1/dannegrut/harry/experiments/offroad_vehicles_20260927/e5/logs/%x_%j.out
set -o pipefail
G4=${OV_G4:-/work1/dannegrut/harry/experiments/offroad_vehicles_20260927}
: "${OV_JOBS:?export OV_JOBS=<job list tsv>}"
unset NEDM_VEHICLE
LPG=${OV_LANES_PER_GPU:-2}
if [ -z "${OV_NO_ENV:-}" ]; then
  source /etc/profile
  module load pytorch/2.10.0
  source /work1/dannegrut/harry/venvs/nedm/bin/activate
fi
export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID} MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${SLURM_JOB_ID}
cd $G4/source || { echo "no $G4/source"; exit 2; }
export PYTHONPATH=$G4/source/src:$G4/source/scripts:${PYTHONPATH:-}
PY=${OV_PY:-python3.12}
SHA=$(sha256sum scripts/ci_train.py | cut -c1-16)
case "$SHA" in "${OV_CI_TRAIN_SHA:-7a4f2d67}"*) ;; *) echo "ci_train.py sha256 $SHA does not start with ${OV_CI_TRAIN_SHA:-7a4f2d67}: refusing"; exit 3;; esac
mkdir -p $G4/e5/logs
NGPU=$(rocminfo | grep -c -E "^\s+Name:\s+gfx"); [ "$NGPU" -lt 1 ] && NGPU=1
NCPU=${SLURM_CPUS_PER_TASK:-$(nproc)}
mapfile -t LINES < <(grep -v '^#' "$OV_JOBS" | grep -v '^[[:space:]]*$')
LANES=$(printf '%s\n' "${LINES[@]}" | cut -f1 | sort -un)
NL=$(echo "$LANES" | wc -w); THR=$(( NCPU / (NL > 0 ? NL : 1) )); [ $THR -gt 16 ] && THR=16; [ $THR -lt 1 ] && THR=1
echo "host=$(hostname) job=$SLURM_JOB_ID part=$SLURM_JOB_PARTITION gpus=$NGPU lanes_per_gpu=$LPG cpus=$NCPU runs=${#LINES[@]} lanes=$NL omp=$THR jobs=$OV_JOBS sha256=$(sha256sum $OV_JOBS | cut -c1-16) NEDM_VEHICLE=${NEDM_VEHICLE:-unset}"
echo "ci_train.py $SHA  ga_train.py $(sha256sum scripts/ga_train.py | cut -c1-16)  python $(which $PY)"
$PY -c "import torch; print('torch', torch.__version__, 'devices', torch.cuda.device_count(), torch.cuda.get_device_name(0))"
run_lane() {
  local lane=$1 gpu=$2 L LN TAG OUT ARGS LOG RC
  for L in "${LINES[@]}"; do
    IFS=$'\t' read -r LN TAG OUT ARGS <<< "$L"
    [ "$LN" = "$lane" ] || continue
    mkdir -p $G4/$OUT
    LOG=$G4/e5/logs/${TAG}_${SLURM_JOB_ID}.log
    read -ra TOK <<< "$ARGS"; W0=$(date +%s)
    for ((i=0; i<${#TOK[@]}; i++)); do
      if [ "${TOK[$i]}" = "--ds" ]; then
        M="${TOK[$((i+1))]%.npz}.manifest.json"
        while [ ! -f "$M" ] && [ $(( $(date +%s) - W0 )) -lt ${OV_WAIT_S:-7200} ]; do sleep 20; done
        [ -f "$M" ] || echo "$(date +%T) lane $lane WARNING $TAG: no manifest $M after ${OV_WAIT_S:-7200} s"
      fi
    done
    [ $(( $(date +%s) - W0 )) -gt 5 ] && echo "$(date +%T) lane $lane $TAG waited $(( $(date +%s) - W0 )) s for its data"
    echo "$(date +%T) lane $lane gpu $gpu start $TAG -> $G4/$OUT : $ARGS"
    HIP_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$THR $PY -u scripts/ci_train.py $ARGS --out $G4/$OUT --tag $TAG > $LOG 2>&1; RC=$?
    echo "exit: $RC" >> $LOG
    echo "$(date +%T) lane $lane end $TAG exit: $RC"
  done
}
PIDS=()
for ln in $LANES; do run_lane $ln $(( ((ln - 1) / LPG) % NGPU )) & PIDS+=($!); done
for p in "${PIDS[@]}"; do wait $p; done
RC=0
for L in "${LINES[@]}"; do
  IFS=$'\t' read -r LN TAG OUT ARGS <<< "$L"
  E=$(tail -1 $G4/e5/logs/${TAG}_${SLURM_JOB_ID}.log 2>/dev/null)
  echo "$TAG $E"
  [ "$E" = "exit: 0" ] || RC=1
done
echo "all done rc=$RC"
exit $RC
