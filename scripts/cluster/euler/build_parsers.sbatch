#!/bin/bash
# Build a SECOND Chrono tree on UW-Madison Euler that has the Parsers module
# (URDF), leaving the existing ~/chrono-build byte-identical.
#
# Scoring needs pychrono.parsers, because src/nedm/quadruped/robot.py loads the
# Go2 through ChParserURDF and has no alternative path. The existing
# ~/chrono-build was configured with CH_ENABLE_MODULE_PARSERS=OFF, and it is the
# only pychrono on this machine with a working OptiX sensor backend -- an
# artifact that is expensive to reproduce and cheap to lose. Enabling a module in
# place means re-running its configure, and a configure that goes wrong costs the
# whole thing. A second tree costs ~2 GB of an unquota'd 941 TB pool.
#
# TWO DIFFERENCES FROM THE hpcfund VERSION of this script, both Euler facts:
#
#   1. Only THREE dependencies are built, not four. Euler ships tinyxml2 10.0.0
#      with a real CMake config (/usr/lib64/cmake/tinyxml2), which is the exact
#      version the hpcfund toolchain had to build from source. console_bridge,
#      urdfdom_headers and urdfdom are still absent and still have to be built.
#
#   2. urdfdom must be 4.x, for the same reason as on hpcfund: the 3.x series
#      parses with tinyxml v1 (urdf_parser.h includes <tinyxml.h>), while
#      chrono_parsers/CMakeLists.txt does find_package(tinyxml2 REQUIRED) and
#      ChParserURDF.h includes <tinyxml2.h>. 4.0.0 migrated to tinyxml2 and still
#      exports parseURDF(const std::string&) and the urdf::{Box,Cylinder,Mesh,
#      Sphere,Joint,Link,Pose,Rotation,Vector} types ChParserURDF.cpp calls.
#
# CUDA ARCHITECTURES: THE VARIABLE IS CHRONO_CUDA_ARCHITECTURES, NOT
# CMAKE_CUDA_ARCHITECTURES. Chrono's cmake/ChronoGPUDetect.cmake ends with
#
#     set(CMAKE_CUDA_ARCHITECTURES ${CHRONO_CUDA_ARCHITECTURES}
#         CACHE STRING "CUDA architectures" FORCE)
#
# so a -DCMAKE_CUDA_ARCHITECTURES on the command line is accepted without
# complaint and then silently overwritten. When CHRONO_CUDA_ARCHITECTURES is
# empty it defaults to CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR, which under CUDA 12.9
# is 60;70;80;90;100;120 -- MAJOR architectures only.
#
# That default is wrong for Euler specifically, and it is wrong silently. It
# covers the A100 (sm_80) and H100 (sm_90) and nothing else this cluster owns:
# the 2080 Ti (sm_75), RTX A4500 (sm_86) and RTX 4000 Ada (sm_89) get no cubin,
# and the only PTX in the fat binary is compute_120, which cannot JIT backwards.
# Those three account for most of Euler's 98 GPUs, and one of them is euler16 --
# half of the two-node `sbel` tier. ~/chrono-build has exactly this problem: its
# configure script passes -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" and its
# CMakeCache reads 60-real;70-real;80-real;90-real;100-real;120.
#
# So set the variable Chrono actually reads. Naming 80 and 90 explicitly does not
# change the code generated for them, so this tree stays comparable to the
# existing one on the A100; it only adds the three architectures that were
# missing.
#
# Compiles on the `sbel` partition because it is PriorityTier 400 and therefore
# not preemptible, and on euler16 specifically, whose 2080 Ti this workload has no
# use for -- nvcc needs no GPU. -c 32 of 96 cores, because the partition is
# OverSubscribe=YES:4 and ExclusiveUser=NO: taking the whole node here is taking
# it from the lab, not from a billing meter.
#
#SBATCH -J kyle-chrono-parsers
#SBATCH -p sbel
#SBATCH -w euler16
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 08:00:00
#SBATCH -o /srv/home/kasha2/logs/%x-%j.out
#SBATCH -e /srv/home/kasha2/logs/%x-%j.err

source /srv/home/kasha2/env.sh
module load gcc/12.2.0 nvidia/cuda/12.9.1

NEWBUILD=$EHOME/chrono-build-parsers
URDFROOT=$TOOLCHAIN/urdf
DEPSRC=$EHOME/src/urdf-deps
NPROC=${SLURM_CPUS_PER_TASK:-16}

# HARD GUARD: never write outside Kyle's home, and never at the existing tree.
case "$NEWBUILD" in
  /srv/home/kasha2/*) ;;
  *) echo "FATAL: refusing to build outside /srv/home/kasha2: $NEWBUILD"; exit 1;;
esac
if [ "$NEWBUILD" = "$CHRONO_BUILD" ]; then
  echo "FATAL: target equals the existing build tree. Refusing."; exit 1
fi

echo "== host $(hostname)  cores $NPROC  start $(date) =="
echo "== new build tree : $NEWBUILD"
echo "== untouched tree : $CHRONO_BUILD"
echo "== gcc  : $(gcc --version | head -1)"
echo "== nvcc : $(nvcc --version | tail -2 | head -1)"
CHKSUM_BEFORE=$(ls -la $CHRONO_BUILD/bin/pychrono/ | md5sum)
echo "== existing pychrono fingerprint before: $CHKSUM_BEFORE"

# Start from a clean tree. An architecture change has to invalidate every CUDA
# object, and a half-built tree from a previous attempt is the one thing that
# could carry the old ones forward.
rm -rf "$NEWBUILD"

set -e
mkdir -p "$URDFROOT"

build_dep () {   # name srcdir extra_cmake_args...
  local name=$1; shift
  local src=$1; shift
  if [ -f "$URDFROOT/.done-$name" ]; then echo "-- $name already built"; return 0; fi
  echo "-- building $name"
  cmake -S "$src" -B "$DEPSRC/build-$name" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$URDFROOT" \
        -DCMAKE_PREFIX_PATH="$URDFROOT" \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        "$@" > "$DEPSRC/$name-configure.log" 2>&1
  cmake --build "$DEPSRC/build-$name" -j "$NPROC" > "$DEPSRC/$name-build.log" 2>&1
  cmake --install "$DEPSRC/build-$name" > "$DEPSRC/$name-install.log" 2>&1
  touch "$URDFROOT/.done-$name"
  echo "   $name ok"
}

# Order matters: urdfdom needs the other two plus the system tinyxml2.
build_dep console_bridge   "$DEPSRC/console_bridge-1.0.2"
build_dep urdfdom_headers  "$DEPSRC/urdfdom_headers-1.1.1"
build_dep urdfdom          "$DEPSRC/urdfdom-4.0.1" -DBUILD_TESTING=OFF

# set -e OFF before the inventory: these deps install to lib64 on this system,
# not lib, so `ls $URDFROOT/lib` exits nonzero and would kill the job right after
# the deps had all built successfully. (Measured on hpcfund, job 414029.)
set +e
echo "-- installed urdf deps --"
ls "$URDFROOT"/lib "$URDFROOT"/lib64 2>/dev/null | head -30
find "$URDFROOT" -name "*-config.cmake" -o -name "*Config.cmake" 2>/dev/null | head -10

echo "=== configure Chrono (+PARSERS) $(date) ==="
cmake -S "$CHRONO_SRC" -B "$NEWBUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$URDFROOT" \
  -DSWIG_EXECUTABLE="$SWIG_ROOT/bin/swig" \
  -DPython3_EXECUTABLE=/usr/bin/python3.12 \
  -DPython3_NumPy_INCLUDE_DIRS="$NRD_NUMPY_INC" \
  -DCHRONO_CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90" \
  -DOptiX_INSTALL_DIR="$OPTIX_ROOT" \
  -DCH_ENABLE_MODULE_PYTHON=ON \
  -DCH_ENABLE_MODULE_VEHICLE=ON -DCH_ENABLE_MODULE_VEHICLE_MODELS=ON -DCH_ENABLE_MODULE_VEHICLE_COSIM=ON \
  -DCH_ENABLE_MODULE_FSI=ON -DCH_ENABLE_MODULE_FSI_SPH=ON \
  -DCH_ENABLE_MODULE_FEA=ON -DCH_ENABLE_MODULE_FEA_MULTIPHYSICS=ON \
  -DCH_ENABLE_MODULE_POSTPROCESS=ON \
  -DCH_ENABLE_MODULE_DEM=ON \
  -DCH_ENABLE_MODULE_SENSOR=ON \
  -DCH_USE_SENSOR_OPTIX=ON -DCH_USE_SENSOR_VULKAN_RT=OFF \
  -DCH_ENABLE_MODULE_IRRLICHT=OFF -DCH_ENABLE_MODULE_VSG=OFF \
  -DCH_ENABLE_MODULE_PARSERS=ON -DCH_ENABLE_MODULE_ROS=OFF \
  -DCH_ENABLE_MODULE_MULTICORE=OFF -DCH_ENABLE_MODULE_MODAL=OFF \
  -DCH_ENABLE_MODULE_CASCADE=OFF -DCH_ENABLE_MODULE_MATLAB=OFF
rc=$?
echo "=== configure exit $rc  $(date) ==="
[ $rc -ne 0 ] && exit $rc

echo "--- did the URDF parser actually get enabled? ---"
grep -E "^CH_ENABLE_MODULE_PARSERS:BOOL|^urdfdom_DIR|^tinyxml2_DIR|^console_bridge_DIR|^CHRONO_CUDA_ARCHITECTURES|^CMAKE_CUDA_ARCHITECTURES" "$NEWBUILD/CMakeCache.txt"
# CMAKE_CUDA_ARCHITECTURES is the one that reaches nvcc, and Chrono FORCE-writes
# it from CHRONO_CUDA_ARCHITECTURES. If the two disagree here, the -D above did
# not take and the tree will not run on half this cluster.

echo "=== build $(date) ==="
cmake --build "$NEWBUILD" -j "$NPROC"
rc=$?
echo "=== build exit $rc  $(date) ==="

echo "--- pychrono modules in the NEW tree ---"
ls "$NEWBUILD/bin/pychrono/" 2>/dev/null | tr '\n' ' '; echo
echo "--- existing tree fingerprint after: $(ls -la $CHRONO_BUILD/bin/pychrono/ | md5sum)"
echo "--- (must equal before: $CHKSUM_BEFORE) ---"
echo "== end $(date) =="
exit $rc
