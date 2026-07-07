cmake_minimum_required(VERSION 3.24)

project(
  QuixiCoreXPU
  VERSION 0.1.0
  DESCRIPTION "Native Intel GPU backend scaffold for QuixiCore"
  LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(CTest)
include(GNUInstallDirs)

# Build position-independent so the static libs can be linked into the PyTorch
# extension shared object (bindings/pytorch).
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(QUIXICORE_XPU_BUILD_TESTS "Build QuixiCore XPU tests" ON)
option(QUIXICORE_XPU_BUILD_EXAMPLES "Build QuixiCore XPU examples" ON)
option(QUIXICORE_XPU_ENABLE_SYCL "Build SYCL/oneAPI device-probe targets" OFF)

add_library(quixicore_xpu src/backend.cpp)
add_library(QuixiCore::XPU ALIAS quixicore_xpu)
set_target_properties(quixicore_xpu PROPERTIES EXPORT_NAME XPU)

target_compile_features(quixicore_xpu PUBLIC cxx_std_20)
target_include_directories(
  quixicore_xpu
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

if(MSVC)
  target_compile_options(quixicore_xpu PRIVATE /W4)
else()
  target_compile_options(quixicore_xpu PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(QUIXICORE_XPU_BUILD_EXAMPLES)
  add_executable(quixicore_xpu_info examples/backend_info.cpp)
  target_link_libraries(quixicore_xpu_info PRIVATE QuixiCore::XPU)
endif()

if(BUILD_TESTING AND QUIXICORE_XPU_BUILD_TESTS)
  add_executable(quixicore_xpu_backend_smoke tests/backend_smoke.cpp)
  target_link_libraries(quixicore_xpu_backend_smoke PRIVATE QuixiCore::XPU)
  add_test(NAME quixicore_xpu_backend_smoke COMMAND quixicore_xpu_backend_smoke)
endif()

if(QUIXICORE_XPU_ENABLE_SYCL)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-fsycl" QUIXICORE_XPU_HAS_FSYCL)
  if(NOT QUIXICORE_XPU_HAS_FSYCL)
    message(FATAL_ERROR "QUIXICORE_XPU_ENABLE_SYCL requires a compiler that accepts -fsycl, such as icpx.")
  endif()

  # Optional ahead-of-time compilation for a specific Intel GPU target. Default
  # is JIT (empty). For Battlemage set QUIXICORE_XPU_SYCL_TARGETS=bmg (or an
  # explicit "spir64_gen -device bmg" pair). AoT is a perf/startup lever, not a
  # correctness requirement.
  set(QUIXICORE_XPU_SYCL_TARGETS "" CACHE STRING
      "SYCL AoT device target(s) passed to -fsycl-targets (empty = JIT).")

  set(QX_SYCL_FLAGS -fsycl)
  if(QUIXICORE_XPU_SYCL_TARGETS)
    list(APPEND QX_SYCL_FLAGS -fsycl-targets=${QUIXICORE_XPU_SYCL_TARGETS})
  endif()

  # Vendor variant support. The XPU backend ships both a native SYCL variant and
  # a vendor (oneDNN) variant of each applicable op, co-equal and selectable at
  # the ABI. oneDNN is optional: when found, *.onednn.cpp translation units are
  # compiled and QUIXICORE_XPU_HAS_ONEDNN is defined; when absent, the backend
  # still builds and Variant::vendor transparently falls back to SYCL.
  option(QUIXICORE_XPU_ENABLE_ONEDNN "Build the oneDNN vendor variants" ON)
  set(QUIXICORE_XPU_HAS_ONEDNN OFF)
  if(QUIXICORE_XPU_ENABLE_ONEDNN)
    find_package(dnnl CONFIG QUIET)
    if(dnnl_FOUND)
      set(QUIXICORE_XPU_HAS_ONEDNN ON)
      message(STATUS "QuixiCore XPU: oneDNN found; vendor variants enabled.")
    else()
      message(STATUS "QuixiCore XPU: oneDNN not found; vendor variants fall back to SYCL.")
    endif()
  endif()

  # Native op/runtime/kernel translation units. Compiled with -fsycl and layered
  # on top of the (non-SYCL) status library. New kernels are picked up by naming
  # their SYCL TU *.sycl.cpp under kernels/<family>/<operation>/variants/.
  file(GLOB_RECURSE QUIXICORE_XPU_SYCL_SOURCES CONFIGURE_DEPENDS
       ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/*.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/src/dispatch/*.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.sycl.cpp)

  if(QUIXICORE_XPU_HAS_ONEDNN)
    file(GLOB_RECURSE QUIXICORE_XPU_ONEDNN_SOURCES CONFIGURE_DEPENDS
         ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.onednn.cpp)
    list(APPEND QUIXICORE_XPU_SYCL_SOURCES ${QUIXICORE_XPU_ONEDNN_SOURCES})
  endif()

  add_library(quixicore_xpu_ops ${QUIXICORE_XPU_SYCL_SOURCES})
  add_library(QuixiCore::XPUOps ALIAS quixicore_xpu_ops)
  set_target_properties(quixicore_xpu_ops PROPERTIES EXPORT_NAME XPUOps)
  target_compile_features(quixicore_xpu_ops PUBLIC cxx_std_20)
  target_include_directories(
    quixicore_xpu_ops
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      # Internal per-op headers (e.g. kernels/activations/gelu/gelu_kernel.hpp)
      # are reachable as "activations/gelu/gelu_kernel.hpp".
      ${CMAKE_CURRENT_SOURCE_DIR}/kernels)
  target_link_libraries(quixicore_xpu_ops PUBLIC QuixiCore::XPU)
  target_compile_options(quixicore_xpu_ops PUBLIC ${QX_SYCL_FLAGS})
  target_link_options(quixicore_xpu_ops PUBLIC ${QX_SYCL_FLAGS})
  if(QUIXICORE_XPU_HAS_ONEDNN)
    target_compile_definitions(quixicore_xpu_ops PUBLIC QUIXICORE_XPU_HAS_ONEDNN)
    target_link_libraries(quixicore_xpu_ops PUBLIC DNNL::dnnl)
  endif()

  add_executable(quixicore_xpu_sycl_device_probe examples/sycl_device_probe.cpp)
  target_link_libraries(quixicore_xpu_sycl_device_probe PRIVATE QuixiCore::XPU)
  target_compile_options(quixicore_xpu_sycl_device_probe PRIVATE -fsycl)
  target_link_options(quixicore_xpu_sycl_device_probe PRIVATE -fsycl)

  if(BUILD_TESTING AND QUIXICORE_XPU_BUILD_TESTS)
    add_test(NAME quixicore_xpu_sycl_device_probe COMMAND quixicore_xpu_sycl_device_probe)

    add_executable(quixicore_xpu_ops_smoke tests/xpu_ops_smoke.cpp)
    target_link_libraries(quixicore_xpu_ops_smoke PRIVATE QuixiCore::XPUOps)
    target_include_directories(quixicore_xpu_ops_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/kernels)
    add_test(NAME quixicore_xpu_ops_smoke COMMAND quixicore_xpu_ops_smoke)
  endif()

  # On-device benchmark harness (not a CTest test; driven by perf/bench_kernels.py).
  add_executable(quixicore_xpu_bench perf/harness/xpu_bench.cpp)
  target_link_libraries(quixicore_xpu_bench PRIVATE QuixiCore::XPUOps)
  target_include_directories(quixicore_xpu_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/kernels)
endif()

install(
  TARGETS quixicore_xpu
  EXPORT QuixiCoreXPUTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
  EXPORT QuixiCoreXPUTargets
  NAMESPACE QuixiCore::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/QuixiCoreXPU)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/QuixiCoreXPUConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/QuixiCoreXPUConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/QuixiCoreXPU)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/QuixiCoreXPUConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/QuixiCoreXPUConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/QuixiCoreXPUConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/QuixiCoreXPU)
