#!/bin/bash
# Follow-up runs: the AMD cluster's own lavapipe build (Mesa 24.2.8 / LLVM 19.1.7) on this CPU -- where its time goes,
# how it scales with threads and image size, and what four concurrent runs on one 16-thread machine do.
set -u
OUT=/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_f104_50h_20260909/render_latency_v1/vulkan_local
CASE=/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/cases_g216/cases/g216_v2_group_0000.json
DATA=/home/harry/chrono/data
PY="/usr/bin/time -v /usr/bin/python3.12"
export PYTHONPATH=/home/harry/chrono_build_vulkan_bench/bin
export VK_LOADER_DEBUG=driver
LVP=/usr/share/vulkan/icd.d/lvp_icd.json
CACHE=/home/harry/chrono_build_vulkan_bench/mesa_cache
AMDLVP="VK_ICD_FILENAMES=/home/harry/tools/lvp_amd/lvp_icd.json LD_LIBRARY_PATH=/home/harry/tools/lvp_amd/lib_min MESA_SHADER_CACHE_DIR=${CACHE}_amdlvp"
H="$OUT/bench_verbose.py --case $CASE --chrono-data $DATA --out $OUT"
B="$OUT/render_breakdown.py --case $CASE --chrono-data $DATA --out $OUT/breakdown"
bench() {
  local label=$1; shift
  echo "=== $label start $(date -Is)" | tee -a $OUT/logs/run_all.log
  env "$@" > $OUT/logs/$label.log 2>&1
  echo "=== $label exit $? end $(date -Is)" | tee -a $OUT/logs/run_all.log
}
bench bd_amdlvp16_1024 $AMDLVP LP_NUM_THREADS=16 $PY $B --label bd_amdlvp16_1024 --size 1024 --frames 3
bench amdlvp16_depth_small $AMDLVP LP_NUM_THREADS=16 $PY $H --label amdlvp16_depth_small --sizes 256 512 --frames 5
bench amdlvp8_depth $AMDLVP LP_NUM_THREADS=8 $PY $H --label amdlvp8_depth --sizes 1024 --frames 5
bench amdlvp4_depth $AMDLVP LP_NUM_THREADS=4 $PY $H --label amdlvp4_depth --sizes 1024 --frames 5
bench amdlvp16_rgbd $AMDLVP LP_NUM_THREADS=16 $PY $H --label amdlvp16_rgbd --sizes 1024 --frames 5 --with-rgb

# four runs at once on this 16-thread machine (the AMD "four runs share a node" situation), each LP_NUM_THREADS=16
for kind in amdlvp lvp; do
  echo "=== ${kind}16_x4 start $(date -Is)" | tee -a $OUT/logs/run_all.log
  for i in 0 1 2 3; do
    if [ $kind = amdlvp ]; then ENVS="$AMDLVP"; else ENVS="VK_ICD_FILENAMES=$LVP MESA_SHADER_CACHE_DIR=$CACHE"; fi
    env $ENVS LP_NUM_THREADS=16 $PY $H --label ${kind}16_x4_run$i --sizes 1024 --frames 5 > $OUT/logs/${kind}16_x4_run$i.log 2>&1 &
  done
  wait
  echo "=== ${kind}16_x4 end $(date -Is)" | tee -a $OUT/logs/run_all.log
done
echo "=== followup done $(date -Is)" | tee -a $OUT/logs/run_all.log
