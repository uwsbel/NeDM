#!/bin/bash
# Does the Chrono::Sensor OptiX path work on Euler, and how fast?
#
#   sbatch -p sbel -w euler16 --gres=gpu:1 -c 8 scripts/cluster/euler/render_probe.sbatch
#
# A 2-second, one-camera clip. The point is not the clip; it is (a) whether an
# OptiX context comes up at all on a compute node -- render_go2_crm_video.py
# swallows the failure and prints "render DISABLED", so a broken OptiX looks
# like a successful run that produced no frames -- and (b) seconds of wall per
# rendered frame, which extrapolates to the real 11 s clips.
#
# RUN THIS ON AN RTX NODE, NOT AN A100. GA100 has no RT cores at all, so OptiX
# there ray-traces on the SMs and measures the wrong thing. Euler's RT hardware
# is the RTX 2080 Ti on euler16 (Turing), the RTX A4500 on euler05-08 (Ampere),
# and the RTX 4000 Ada on euler01-04 and 09-10. Note that ALL THREE are
# architectures Chrono's default all-major CUDA arch list leaves out, so this
# only runs against ~/chrono-build-parsers.
#
#SBATCH -J kyle-renderprobe
#SBATCH -p sbel
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -t 00:40:00
#SBATCH -o /srv/home/kasha2/logs/%x-%j.out
#SBATCH -e /srv/home/kasha2/logs/%x-%j.err

set -o pipefail
module load gcc/12.2.0 nvidia/cuda/12.9.1
source /srv/home/kasha2/nedm/env-scoring.sh

DUR=${DUR:-2.0}
# CAMS is a COMMA-separated camera list, so it cannot be passed through
# `sbatch --export=ALL,CAMS=a,b,c` -- sbatch splits --export on commas and you
# silently get CAMS=a with b and c parsed as two more (empty) variables. Export
# it in the submitting shell instead and let --export=ALL carry it:
#
#     CAMS=chase,side,front_3q,low_close,topdown,trail,fixed_wide,fixed_ground \
#       sbatch --export=ALL -p sbel -w euler16 --gres=gpu:1 -c 8 render_probe.sbatch
CAMS=${CAMS:-chase}
RD=${RD:-/tmp/go2_render_probe_$SLURM_JOB_ID}

echo "== host $(hostname)  start $(date) =="
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
echo "== duration ${DUR}s  cameras $CAMS  render-dir $RD =="
python -c "import pychrono.sensor as s; print('pychrono.sensor:', s.__file__)" || exit 1

cd "$NEDM_REPO"
t0=$(date +%s)
python scripts/evaluation/render_go2_crm_video.py \
  --terrain crm --duration-s "$DUR" \
  --imported-ckpt "$NEDM_CKPTS/go2_cts_150k.pt" \
  --command-family constant --command-params '{"vx": 0.5, "vy": 0.0, "wz": 0.0}' \
  --episode-index 0 --seed 0 \
  --patch-x 8.0 --patch-y 4.0 --spawn-x-m 1.2 --spawn-y-m 0.0 \
  --render-cameras "$CAMS" --render-dir "$RD" \
  --output-dir "/tmp/go2_render_probe_out_$SLURM_JOB_ID" --overwrite \
  --progress-interval-s 99
rc=$?
t1=$(date +%s)

echo "== rc=$rc  wall $((t1 - t0))s =="
NF=$(find "$RD" -name "*.png" 2>/dev/null | wc -l)
echo "== frames written: $NF =="
if [ "$NF" -gt 0 ]; then
  echo "== $(awk -v s=$((t1 - t0)) -v n=$NF 'BEGIN{printf "%.2f", s/n}') s of wall per frame (INCLUDES the CRM sim, which would run anyway) =="
  # An 11 s clip at 24 fps is 264 frames per camera.
  echo "== extrapolated 11 s / 24 fps / 1 camera: $(awk -v s=$((t1 - t0)) -v n=$NF 'BEGIN{printf "%.1f", s/n*264/60}') min =="
fi
du -sh "$RD" 2>/dev/null
rm -rf "/tmp/go2_render_probe_out_$SLURM_JOB_ID"
exit $rc
