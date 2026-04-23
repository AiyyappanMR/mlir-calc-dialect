# point cmake to our td file
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/CalcOps.td)

# make sure the output directory exists
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/calc_dialect/dialects")

# generate the python bindings using mlir-tblgen
mlir_tablegen(calc_dialect/dialects/_calc_ops_gen.py
  -gen-python-op-bindings -bind-dialect=calc
  EXTRA_INCLUDES ${PROJECT_SOURCE_DIR}/include/calc
)

add_public_tablegen_target(CalcPythonOpsGen)

# copy our static python files into the build dir so they sit next to the generated file
configure_file(calc_dialect/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/calc_dialect/__init__.py COPYONLY)
configure_file(calc_dialect/dialects/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/calc_dialect/dialects/__init__.py COPYONLY)
configure_file(calc_dialect/dialects/calc.py ${CMAKE_CURRENT_BINARY_DIR}/calc_dialect/dialects/calc.py COPYONLY)
configure_file(calc_dialect/dialects/_ods_common.py ${CMAKE_CURRENT_BINARY_DIR}/calc_dialect/dialects/_ods_common.py COPYONLY)
