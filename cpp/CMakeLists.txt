cmake_minimum_required(VERSION 3.16)
project(turret_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# -O2 is plenty for this core; on a Pi 4/5 add -mcpu=native via CMAKE_CXX_FLAGS.

add_library(turret_core_headers INTERFACE)
target_include_directories(turret_core_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Plain C++ tests (no Python needed).
enable_testing()
add_executable(test_core tests/test_core.cpp)
target_link_libraries(test_core PRIVATE turret_core_headers)
add_test(NAME test_core COMMAND test_core)

# Python bindings. Point CMake at pybind11 with:
#   cmake -S cpp -B build -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
option(TURRET_BUILD_PYTHON "Build the pybind11 module" ON)
if(TURRET_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(turret_core src/bindings.cpp)
  target_link_libraries(turret_core PRIVATE turret_core_headers)
endif()
