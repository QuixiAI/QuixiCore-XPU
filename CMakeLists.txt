cmake_minimum_required(VERSION 3.24)

project(
  QuixiCoreXPU
  VERSION 0.1.0
  DESCRIPTION "Native Intel GPU backend scaffold for QuixiCore"
  LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(CTest)
include(GNUInstallDirs)

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

  add_executable(quixicore_xpu_sycl_device_probe examples/sycl_device_probe.cpp)
  target_link_libraries(quixicore_xpu_sycl_device_probe PRIVATE QuixiCore::XPU)
  target_compile_options(quixicore_xpu_sycl_device_probe PRIVATE -fsycl)
  target_link_options(quixicore_xpu_sycl_device_probe PRIVATE -fsycl)

  if(BUILD_TESTING AND QUIXICORE_XPU_BUILD_TESTS)
    add_test(NAME quixicore_xpu_sycl_device_probe COMMAND quixicore_xpu_sycl_device_probe)
  endif()
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
