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

find_package(Chrono REQUIRED COMPONENTS Sensor FSI Vehicle Robot CONFIG)

add_executable(crm_render_gate crm_render_gate.cpp)
target_compile_features(crm_render_gate PRIVATE cxx_std_17)
target_link_libraries(crm_render_gate PRIVATE ${CHRONO_TARGETS})
