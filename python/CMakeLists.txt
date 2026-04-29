set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)
set(CALC_MLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/calc_mlir")

# All calc_mlir Python imports will look for MLIR core under calc_mlir._mlir_libs.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=calc_mlir.")

# LLVM's HandleLLVMOptions.cmake globally sets -Wl,-z,defs (fail on undefined shared-lib
# symbols). The nanobind shared library intentionally leaves Python symbols undefined
# (resolved at runtime when dlopen'd into a Python process). Appending -Wl,-z,undefs
# HERE in CMAKE_SHARED_LINKER_FLAGS ensures it appears AFTER the global -z,defs in the
# final link command, so it wins.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,undefs")
endif()

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(CalcMLIRPythonSources)
declare_mlir_python_sources(CalcMLIRPythonExtensions)

declare_mlir_python_sources(CalcMLIRPythonSources.Dialects
  ROOT_DIR "${CALC_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT CalcMLIRPythonSources
)

# Generates dialects/_calc_ops_gen.py from CalcBinding.td and registers
# dialects/calc/__init__.py as a static source.
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CalcMLIRPythonSources.Dialects
  ROOT_DIR "${CALC_MLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/CalcBinding.td
  SOURCES dialects/calc/__init__.py
  DIALECT_NAME calc
  DEPENDS MLIRcalcOpsIncGen
)

declare_mlir_python_sources(CalcMLIRSiteInitialize
  ROOT_DIR "${CALC_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT CalcMLIRPythonSources
  SOURCES _mlir_libs/_site_initialize_0.py
)

################################################################################
# Extension (_calcMlir.so) — embeds MLIRCalcCAPI into the aggregated CAPI lib
################################################################################

declare_mlir_python_extension(CalcMLIRPythonExtensions.Main
  MODULE_NAME _calcMlir
  ADD_TO_PARENT CalcMLIRPythonExtensions
  SOURCES CalcModule.cpp
  EMBED_CAPI_LINK_LIBS MLIRCalcCAPI
  PRIVATE_LINK_LIBS LLVMSupport
)

################################################################################
# Assembled package under build/python_packages/calc_mlir/
# Bundles MLIR core Python + our dialect extension into one self-contained package.
################################################################################

set(_source_components
  MLIRPythonSources
  MLIRPythonExtension.Core
  MLIRPythonExtension.RegisterEverything
  CalcMLIRPythonSources
  CalcMLIRPythonExtensions
  CalcMLIRSiteInitialize
)

add_mlir_python_common_capi_library(CalcMLIRAggregateCAPI
  INSTALL_COMPONENT CalcMLIRPythonModules
  INSTALL_DESTINATION python_packages/calc_mlir/calc_mlir/_mlir_libs
  OUTPUT_DIRECTORY "${CALC_BINARY_DIR}/python_packages/calc_mlir/calc_mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT ".."
  DECLARED_SOURCES ${_source_components}
)

add_mlir_python_modules(CalcMLIRPythonModules
  ROOT_PREFIX "${CALC_BINARY_DIR}/python_packages/calc_mlir/calc_mlir"
  INSTALL_PREFIX "python_packages/calc_mlir/calc_mlir"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS CalcMLIRAggregateCAPI
)