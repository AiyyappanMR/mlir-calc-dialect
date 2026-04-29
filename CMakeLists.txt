cmake_minimum_required(VERSION 3.10)

project(mlir-calc LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
set(CALC_SOURCE_DIR ${CMAKE_SOURCE_DIR})
set(CALC_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddMLIRPython)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/calc-c)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# CMake 3.22 doesn't support Development.Module in the unversioned find_package(Python).
# Use the full Development component instead, then manually find nanobind, and skip
# the mlir_configure_python_dev_packages() macro that would re-run the failing search.
find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_DIR
  RESULT_VARIABLE _nb_status
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _nb_status EQUAL 0)
  message(FATAL_ERROR "nanobind not found. Install via: pip install nanobind")
endif()
find_package(nanobind CONFIG REQUIRED)
set(MLIR_DISABLE_CONFIGURE_PYTHON_DEV_PACKAGES ON)
include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()  # no-op due to DISABLE flag above
set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "calc_mlir")

add_subdirectory(include)
add_subdirectory(python)
add_subdirectory(lib)
add_subdirectory(calc-opt)
add_subdirectory(test)