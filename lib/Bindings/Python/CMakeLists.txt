find_package(Python COMPONENTS Interpreter Development REQUIRED)

execute_process(COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir OUTPUT_VARIABLE NANOBIND_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH ${NANOBIND_CMAKE_DIR})
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(CalcPythonPasses CalcPasses.cpp)
set_target_properties(CalcPythonPasses PROPERTIES PREFIX "" OUTPUT_NAME "_calcPasses")

find_library(MLIR_PYTHON_CAPI_LIB MLIRPythonCAPI HINTS "/home/mcw/llvm-project/build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs")

target_link_libraries(CalcPythonPasses PRIVATE MLIRCalcCAPI ${MLIR_PYTHON_CAPI_LIB})


add_custom_command(TARGET CalcPythonPasses POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_FILE:CalcPythonPasses>
    ${CMAKE_BINARY_DIR}/python/calc_dialect/_calcPasses$<TARGET_FILE_SUFFIX:CalcPythonPasses>
)
