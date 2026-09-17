# sourced by the sbatch scripts: environment + one function per measurement kind
source /work1/dannegrut/harry/nrd/env.sh
nrd_pychrono
nrd_use_lavapipe
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909/render_latency_v1
C=/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909
SRC=$C/nav_v1/source
CASE=$C/sensor_v2/cases_g216/cases/g216_v2_group_0000.json
DATA=/work1/dannegrut/harry/nrd/chrono-build/data
mkdir -p $R/results $R/probe $R/perf $R/env $R/logs
cd $R

capture_env() {  # $1 = tag
  {
    echo "== date"; date -Is
    echo "== host"; hostname
    echo "== slurm"; env | grep -E '^SLURM_(JOB_ID|ARRAY_JOB_ID|ARRAY_TASK_ID|CPUS_PER_TASK|JOB_CPUS_PER_NODE|JOB_NODELIST|JOB_PARTITION)=' | sort
    echo "== nproc / affinity"; nproc; taskset -cp $$
    echo "== lscpu"; lscpu
    echo "== meminfo"; head -3 /proc/meminfo
    echo "== loadavg"; cat /proc/loadavg
    echo "== simd flags"; grep -m1 -o -wE 'avx2|avx512f|avx512vl|avx512bw|avx512_vnni|fma|sse4_2' /proc/cpuinfo | sort -u | tr '\n' ' '; echo
    echo "== mesa"; strings $NRD_ROOT/toolchain/lvp/lib/libvulkan_lvp.so | grep -m3 -oE 'Mesa [0-9]+\.[0-9]+\.[0-9]+[^ ]*'
    echo "== llvm"; ls -la $NRD_ROOT/toolchain/lvp/lib/ | grep -i llvm
    echo "== vulkan devices (loader + ICD from env)"; "$NRD_PYTHON" $R/scripts/vk_device.py
    echo "== perf_event_paranoid"; cat /proc/sys/kernel/perf_event_paranoid
    echo "== top cpu users on node"; ps -eo pid,user,pcpu,comm --sort=-pcpu | head -8
    echo "== ldd lvp"; ldd $NRD_ROOT/toolchain/lvp/lib/libvulkan_lvp.so
    echo "== env"; env | grep -E '^(VK_|LP_|MESA_|NRD_PYTHON=|OMP_NUM)' | sort
  } > $R/env/$1.txt 2>&1
}

bench() {  # $1 label, $2 threads, $3 size, [$4 rgb]
  local extra=""; [ "$4" = rgb ] && extra="--with-rgb"
  echo "[$(date +%T)] bench $1 LP=$2 size=$3 $extra"
  LP_NUM_THREADS=$2 "$NRD_PYTHON" -P -u $R/scripts/render_latency_bench.py --case $CASE --label $1 --sizes $3 \
    --frames 5 $extra --chrono-data $DATA --repo-root $SRC --out $R/results || echo "FAILED bench $1"
}

probe() {  # $1 label, $2 threads, $3 size(s), $4 frames, [$5 rgb | --hide-vehicle]
  local extra=""; [ "$5" = rgb ] && extra="--with-rgb"; [ "$5" = --hide-vehicle ] && extra="--hide-vehicle"
  echo "[$(date +%T)] probe $1 LP=$2 size=$3 frames=$4 $extra"
  LP_NUM_THREADS=$2 "$NRD_PYTHON" -P -u $R/scripts/render_split_probe.py --case $CASE --label $1 --sizes $3 \
    --frames $4 $extra --chrono-data $DATA --repo-root $SRC --out $R/probe || echo "FAILED probe $1"
}

perfprobe() {  # $1 label, $2 threads, $3 size, $4 frames, [$5 rgb]
  local extra=""; [ "$5" = rgb ] && extra="--with-rgb"
  local F=250 T=/tmp/perf_${SLURM_JOB_ID}_$1
  echo "[$(date +%T)] perf-probe $1 LP=$2 size=$3 frames=$4 $extra"
  LP_NUM_THREADS=$2 perf record -e cpu-clock:u -F $F -k CLOCK_MONOTONIC -o $T.data -- \
    "$NRD_PYTHON" -P -u $R/scripts/render_split_probe.py --case $CASE --label $1 --sizes $3 \
    --frames $4 $extra --chrono-data $DATA --repo-root $SRC --out $R/perf || { echo "FAILED perf $1"; return; }
  perf script -i $T.data -F tid,time,ip,sym,dso --no-demangle 2>/dev/null > $T.script || true
  "$NRD_PYTHON" $R/scripts/perf_split.py --script $T.script --probe $R/perf/$1.json --freq $F \
    --out $R/perf/$1_split.json || echo "FAILED perf_split $1"
  perf report -i $T.data --stdio --sort dso 2>/dev/null | head -40 > $R/perf/$1_report_dso.txt || true
  gzip -c $T.script > $R/perf/$1.script.gz || true
  rm -f $T.data $T.script
}
