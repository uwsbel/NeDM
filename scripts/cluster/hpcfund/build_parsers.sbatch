#!/bin/bash
# Build a SECOND Chrono tree that has the Parsers module (URDF), leaving the
# existing $KWORK/chrono-build byte-identical.
#
# The existing tree is the only validated HIP pychrono we have, with the Vulkan
# sensor backend working. Enabling a new module in place means re-running its
# configure, and a configure that goes wrong costs a full rebuild of a working
# artifact. A second tree costs ~2 GB of a 2 TB quota and about 0.1 charged
# node-hours, so the cheap thing and the safe thing are the same thing here.
#
# Same module set, same fat gfx90a;gfx942;gfx950 compile, PARSERS added.
#
#SBATCH -J kyle-chrono-parsers
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -o /work1/dannegrut/kyle/logs/%x-%j.out
#SBATCH -e /work1/dannegrut/kyle/logs/%x-%j.err

source /work1/dannegrut/kyle/env.sh
module purge
module load gnu12 rocm cmake

NEWBUILD=$KWORK/chrono-build-parsers
URDFROOT=$TOOLCHAIN/urdf
DEPSRC=$KWORK/src/urdf-deps
NPROC=$(nproc)

# HARD GUARD: never write outside Kyle, and never at the existing build tree.
case "$NEWBUILD" in
  /work1/dannegrut/kyle/*) ;;
  *) echo "FATAL: refusing to build outside Kyle: $NEWBUILD"; exit 1;;
esac
if [ "$NEWBUILD" = "$CHRONO_BUILD" ]; then
  echo "FATAL: target equals the existing build tree. Refusing."; exit 1
fi
echo "== host $(hostname)  cores $NPROC  start $(date) =="
echo "== new build tree : $NEWBUILD"
echo "== untouched tree : $CHRONO_BUILD"
CHKSUM_BEFORE=$(ls -la $CHRONO_BUILD/bin/pychrono/ | md5sum)
echo "== existing pychrono fingerprint before: $CHKSUM_BEFORE"

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

# Order matters: urdfdom needs the other three.
#
# urdfdom 4.0.1, NOT 3.x. The 3.x series still parses with tinyxml v1
# (urdf_parser.h includes <tinyxml.h>), which is unmaintained, absent here, and
# not what Chrono wants: chrono_parsers/CMakeLists.txt does
# find_package(tinyxml2 REQUIRED) and ChParserURDF.h includes <tinyxml2.h>.
# Building 3.1.1 against tinyxml2 fails outright -- measured, job 414027 died on
# "fatal error: tinyxml.h: No such file or directory". urdfdom 4.0.0 migrated to
# tinyxml2, so 4.x is the series that matches Chrono, and it still exports the
# parseURDF(const std::string&) -> ModelInterfaceSharedPtr entry point and the
# urdf::{Box,Cylinder,Mesh,Sphere,Joint,Link,Pose,Rotation,Vector} types that
# ChParserURDF.cpp actually calls.
build_dep console_bridge   "$DEPSRC/console_bridge-1.0.2"
build_dep urdfdom_headers  "$DEPSRC/urdfdom_headers-1.1.1"
build_dep tinyxml2         "$DEPSRC/tinyxml2-10.0.0" -Dtinyxml2_BUILD_TESTING=OFF
build_dep urdfdom          "$DEPSRC/urdfdom-4.0.1" -DBUILD_TESTING=OFF

# set -e OFF before the inventory: these deps install to lib64 on this system,
# not lib, so `ls $URDFROOT/lib` exits nonzero and under set -e that killed the
# job immediately after the deps had all built successfully (job 414029).
set +e
echo "-- installed urdf deps --"
ls "$URDFROOT"/lib "$URDFROOT"/lib64 2>/dev/null | head -30
find "$URDFROOT" -name "*-config.cmake" -o -name "*Config.cmake" 2>/dev/null | head -10

echo "=== configure Chrono (+PARSERS) $(date) ==="
cmake -S "$CHRONO_SRC" -B "$NEWBUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$URDFROOT" \
  -DEigen3_DIR="$EIGEN_ROOT/share/eigen3/cmake" \
  -DSWIG_EXECUTABLE="$SWIG_ROOT/bin/swig" \
  -DPython3_EXECUTABLE=/usr/bin/python3.12 \
  -DPython3_NumPy_INCLUDE_DIRS="$NRD_NUMPY_INC" \
  -DCHRONO_GPU_BACKEND=HIP \
  -DCHRONO_ROCM_ROOT=/opt/rocm-7.2.0 \
  -DCHRONO_HIP_ARCHITECTURES="gfx90a;gfx942;gfx950" \
  -DCH_ENABLE_MODULE_PYTHON=ON \
  -DCH_ENABLE_MODULE_VEHICLE=ON -DCH_ENABLE_MODULE_VEHICLE_MODELS=ON -DCH_ENABLE_MODULE_VEHICLE_COSIM=ON \
  -DCH_ENABLE_MODULE_FSI=ON -DCH_ENABLE_MODULE_FSI_SPH=ON \
  -DCH_ENABLE_MODULE_FEA=ON -DCH_ENABLE_MODULE_FEA_MULTIPHYSICS=ON \
  -DCH_ENABLE_MODULE_POSTPROCESS=ON \
  -DCH_ENABLE_MODULE_DEM=ON \
  -DCH_ENABLE_MODULE_SENSOR=ON \
  -DCH_USE_SENSOR_OPTIX=OFF -DCH_USE_SENSOR_VULKAN_RT=ON \
  -DVulkan_INCLUDE_DIR="$VULKAN_SDK/include" \
  -DVulkan_LIBRARY="$VULKAN_SDK/lib/libvulkan.so" \
  -DVulkan_GLSLANG_VALIDATOR_EXECUTABLE="$VULKAN_SDK/bin/glslangValidator" \
  -DCH_ENABLE_MODULE_IRRLICHT=OFF -DCH_ENABLE_MODULE_VSG=OFF \
  -DCH_ENABLE_MODULE_PARSERS=ON -DCH_ENABLE_MODULE_ROS=OFF \
  -DCH_ENABLE_MODULE_MULTICORE=OFF -DCH_ENABLE_MODULE_MODAL=OFF \
  -DCH_ENABLE_MODULE_CASCADE=OFF -DCH_ENABLE_MODULE_MATLAB=OFF
rc=$?
echo "=== configure exit $rc  $(date) ==="
[ $rc -ne 0 ] && exit $rc

echo "--- did the URDF parser actually get enabled? ---"
grep -E "^CH_ENABLE_MODULE_PARSERS:BOOL|^urdfdom_DIR|^tinyxml2_DIR|^console_bridge_DIR" "$NEWBUILD/CMakeCache.txt"
grep -rE "CH_USE_URDF" "$NEWBUILD/CMakeCache.txt" 2>/dev/null

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
