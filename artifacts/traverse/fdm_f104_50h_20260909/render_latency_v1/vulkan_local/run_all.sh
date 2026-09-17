#!/bin/bash
# Sequential local Vulkan-RT benchmark runs (one process at a time, nothing else running).
# Build: /home/harry/chrono_build_vulkan_bench (Vulkan-RT only, no OptiX; source = AMD commit f54254fa9f + AMD's python-binding files)
set -u
OUT=/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_f104_50h_20260909/render_latency_v1/vulkan_local
CASE=/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/cases_g216/cases/g216_v2_group_0000.json
DATA=/home/harry/chrono/data
PY="/usr/bin/time -v /usr/bin/python3.12"
export PYTHONPATH=/home/harry/chrono_build_vulkan_bench/bin
export VK_LOADER_DEBUG=driver
NV=/usr/share/vulkan/icd.d/nvidia_icd.json
LVP=/usr/share/vulkan/icd.d/lvp_icd.json
CACHE=/home/harry/chrono_build_vulkan_bench/mesa_cache
mkdir -p $OUT/logs

bench() {  # label, then env assignments and harness args
  local label=$1; shift
  echo "=== $label start $(date -Is)" | tee -a $OUT/logs/run_all.log
  env "$@" > $OUT/logs/$label.log 2>&1
  echo "=== $label exit $? end $(date -Is)" | tee -a $OUT/logs/run_all.log
}
H="$OUT/bench_verbose.py --case $CASE --chrono-data $DATA --out $OUT"
B="$OUT/render_breakdown.py --case $CASE --chrono-data $DATA --out $OUT/breakdown"

# (b) same Vulkan-RT code on the RTX 5090 through the NVIDIA Vulkan driver
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -lms 500 > $OUT/logs/nvidia_smi_during_vk5090.csv &
SMI=$!
bench vk5090_depth VK_ICD_FILENAMES=$NV $PY $H --label vk5090_depth --sizes 256 512 1024 2048 --frames 30
bench vk5090_rgbd  VK_ICD_FILENAMES=$NV $PY $H --label vk5090_rgbd --sizes 1024 --frames 30 --with-rgb
bench bd_vk5090_1024 VK_ICD_FILENAMES=$NV $PY $B --label bd_vk5090_1024 --size 1024 --frames 5
kill $SMI

# (a) lavapipe (system Mesa 25.2.8 / LLVM 20) on this CPU; first run on an empty Mesa shader cache
rm -rf $CACHE; mkdir -p $CACHE
bench lvp16_coldcache VK_ICD_FILENAMES=$LVP LP_NUM_THREADS=16 MESA_SHADER_CACHE_DIR=$CACHE $PY $H --label lvp16_coldcache --sizes 256 --frames 2
bench lvp16_depth VK_ICD_FILENAMES=$LVP LP_NUM_THREADS=16 MESA_SHADER_CACHE_DIR=$CACHE $PY $H --label lvp16_depth --sizes 256 512 1024 --frames 5
bench lvp8_depth  VK_ICD_FILENAMES=$LVP LP_NUM_THREADS=8  MESA_SHADER_CACHE_DIR=$CACHE $PY $H --label lvp8_depth --sizes 1024 --frames 5
bench lvp4_depth  VK_ICD_FILENAMES=$LVP LP_NUM_THREADS=4  MESA_SHADER_CACHE_DIR=$CACHE $PY $H --label lvp4_depth --sizes 1024 --frames 5
bench lvp16_rgbd  VK_ICD_FILENAMES=$LVP LP_NUM_THREADS=16 MESA_SHADER_CACHE_DIR=$CACHE $PY $H --label lvp16_rgbd --sizes 1024 --frames 5 --with-rgb
bench bd_lvp16_1024 VK_ICD_FILENAMES=$LVP LP_NUM_THREADS=16 MESA_SHADER_CACHE_DIR=$CACHE $PY $B --label bd_lvp16_1024 --size 1024 --frames 3
bench bd_lvp16_256  VK_ICD_FILENAMES=$LVP LP_NUM_THREADS=16 MESA_SHADER_CACHE_DIR=$CACHE $PY $B --label bd_lvp16_256 --size 256 --frames 3

# control: the exact portable lavapipe the AMD cluster uses (Mesa 24.2.8 / LLVM 19.1.7), run on this CPU
bench amdlvp16_depth VK_ICD_FILENAMES=/home/harry/tools/lvp_amd/lvp_icd.json LD_LIBRARY_PATH=/home/harry/tools/lvp_amd/lib_min LP_NUM_THREADS=16 MESA_SHADER_CACHE_DIR=${CACHE}_amdlvp $PY $H --label amdlvp16_depth --sizes 1024 --frames 5
echo "=== all done $(date -Is)" | tee -a $OUT/logs/run_all.log
