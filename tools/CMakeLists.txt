# Standalone build for the NeDM CRM-render gate.
#
#   cmake -S tools -B build-gate -DChrono_DIR=<chrono-build>/cmake
#   cmake --build build-gate -j
#
# Deliberately separate from the Chrono source tree: a gate that lives upstream
# is versioned with upstream, so re-pinning silently changes or removes it. This
# one is versioned alongside the results it produced.
cmake_minimum_required(VERSION 3.18)
project(nedm_crm_render_gate LANGUAGES CXX)

find_package(Chrono REQUIRED COMPONENTS Sensor FSI FSI_SPH Vehicle Parsers CONFIG)

add_executable(crm_render_gate crm_render_gate.cpp)
target_compile_features(crm_render_gate PRIVATE cxx_std_17)
target_link_libraries(crm_render_gate PRIVATE ${CHRONO_TARGETS})

# The Chrono build resolves tinyxml2 from a conda env (Ubuntu ships no
# tinyxml2Config.cmake), and that env ALSO contains a complete conda pychrono
# installation under include/chrono. The imported tinyxml2 target drags the whole
# env include directory in, so a consumer compiling only with -isystem paths
# picks up conda's Chrono 10.0.0 headers ahead of this source tree's. That is a
# silent version mix; here it happened to fail loudly on a missing HACD header.
# Force the source tree ahead of everything inherited.
get_target_property(_gate_incs crm_render_gate INCLUDE_DIRECTORIES)
target_include_directories(crm_render_gate BEFORE PRIVATE
    ${CHRONO_SOURCE_DIR}/src
    ${CHRONO_BUILD_DIR}
    ${CHRONO_SOURCE_DIR}/src/chrono/collision/bullet
    ${CHRONO_SOURCE_DIR}/src/chrono_thirdparty
    ${CHRONO_SOURCE_DIR}/src/chrono_thirdparty/HACDv2
    ${CHRONO_SOURCE_DIR}/src/chrono_thirdparty/yaml-cpp/include)

# Absolute data paths, so the gate does not depend on being run from
# chrono-build/bin. Upstream's relative "../data/" segfaults elsewhere.
target_compile_definitions(crm_render_gate PRIVATE
    CHRONO_DATA_DIR="${CHRONO_BUILD_DIR}/data/"
    CHRONO_VEHICLE_DATA_DIR="${CHRONO_BUILD_DIR}/data/vehicle/")
